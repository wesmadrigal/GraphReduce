#!/usr/bin/env python
"""RelBench rel-hm: user churn example aligned to the official task definition."""

from __future__ import annotations

import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from relbench.metrics import accuracy, average_precision, f1, roc_auc
from relbench_dataset_utils import materialize_relbench_dataset

from graphreduce.enum import ComputeLayerEnum, PeriodUnit
from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import DuckdbNode

TABLE_NAME_TO_FILENAME = {
    "article": "article.parquet",
    "customer": "customer.parquet",
    "transactions": "transactions.parquet",
}

LOOKBACK_START = datetime.datetime(2019, 9, 7)
VALIDATION_CUT_DATE = datetime.datetime(2020, 9, 7)
TEST_CUT_DATE = datetime.datetime(2020, 9, 14)
CUT_DATE = TEST_CUT_DATE
LABEL_DAYS = 7
LOOKBACK_DAYS = (TEST_CUT_DATE - LOOKBACK_START).days
TRAIN_CUT_DATE = VALIDATION_CUT_DATE - datetime.timedelta(days=LABEL_DAYS)


def run_rel_hm_user_churn(
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float] | None, dict[str, float] | None, int, list[str], str]:
    use_dir = data_dir or Path("tests/data/relbench/rel-hm")
    materialized = materialize_relbench_dataset("rel-hm", use_dir, TABLE_NAME_TO_FILENAME)

    con = duckdb.connect()
    split_frames: dict[str, pd.DataFrame] = {}

    try:
        con.sql(f"CREATE OR REPLACE VIEW article_src AS SELECT * FROM read_parquet('{use_dir / 'article.parquet'}')")
        con.sql(f"CREATE OR REPLACE VIEW customer_src AS SELECT * FROM read_parquet('{use_dir / 'customer.parquet'}')")
        con.sql(
            f"""
            CREATE OR REPLACE VIEW transactions_src AS
            SELECT
                row_number() OVER () AS transaction_id,
                *
            FROM read_parquet('{use_dir / 'transactions.parquet'}')
            """
        )

        article_columns = con.sql("SELECT * FROM article_src LIMIT 0").to_df().columns.tolist()
        customer_columns = con.sql("SELECT * FROM customer_src LIMIT 0").to_df().columns.tolist()
        transaction_columns = con.sql("SELECT * FROM transactions_src LIMIT 0").to_df().columns.tolist()

        article_id_col = {column.lower(): column for column in article_columns}["article_id"]
        customer_id_col = {column.lower(): column for column in customer_columns}["customer_id"]
        tx_id_col = {column.lower(): column for column in transaction_columns}["transaction_id"]
        tx_customer_col = {column.lower(): column for column in transaction_columns}["customer_id"]
        tx_article_col = {column.lower(): column for column in transaction_columns}["article_id"]
        tx_date_col = {column.lower(): column for column in transaction_columns}["t_dat"]

        for split_name, cut_date in {
            "train": TRAIN_CUT_DATE,
            "val": VALIDATION_CUT_DATE,
            "test": TEST_CUT_DATE,
        }.items():
            customer = DuckdbNode(
                fpath="customer_src",
                prefix="cust",
                pk=customer_id_col,
                date_key=None,
                columns=customer_columns,
            )
            article = DuckdbNode(
                fpath="article_src",
                prefix="art",
                pk=article_id_col,
                date_key=None,
                columns=article_columns,
            )
            transactions = DuckdbNode(
                fpath="transactions_src",
                prefix="txn",
                pk=tx_id_col,
                date_key=tx_date_col,
                columns=transaction_columns,
            )

            graph = GraphReduce(
                name=f"rel_hm_user_churn_{cut_date.date()}",
                parent_node=customer,
                compute_layer=ComputeLayerEnum.duckdb,
                sql_client=con,
                cut_date=cut_date,
                compute_period_val=(cut_date - LOOKBACK_START).days,
                compute_period_unit=PeriodUnit.day,
                auto_features=True,
                auto_labels=False,
                date_filters_on_agg=True,
                auto_feature_hops_back=3,
                auto_feature_hops_front=0,
            )

            for node in [customer, article, transactions]:
                graph.add_node(node)

            graph.add_entity_edge(customer, transactions, parent_key=customer_id_col, relation_key=tx_customer_col, reduce=True)
            graph.add_entity_edge(transactions, article, parent_key=tx_article_col, relation_key=article_id_col, reduce=True)

            graph.do_transformations_sql()
            features = con.sql(f"SELECT * FROM {graph.parent_node._cur_data_ref}").to_df().copy()

            labels = con.sql(
                f"""
                WITH timestamp_df AS (
                    SELECT TIMESTAMP '{cut_date}' AS timestamp
                )
                SELECT
                    timestamp,
                    customer_id,
                    CAST(
                        NOT EXISTS (
                            SELECT 1
                            FROM transactions_src
                            WHERE
                                transactions_src.{tx_customer_col} = customer_src.{customer_id_col}
                                AND transactions_src.{tx_date_col} > timestamp
                                AND transactions_src.{tx_date_col} <= timestamp + INTERVAL '{LABEL_DAYS} days'
                        ) AS INTEGER
                    ) AS churn
                FROM
                    timestamp_df,
                    customer_src
                WHERE
                    EXISTS (
                        SELECT 1
                        FROM transactions_src
                        WHERE
                            transactions_src.{tx_customer_col} = customer_src.{customer_id_col}
                            AND transactions_src.{tx_date_col} > timestamp - INTERVAL '{LABEL_DAYS} days'
                            AND transactions_src.{tx_date_col} <= timestamp
                    )
                """
            ).to_df()

            frame = features.merge(
                labels[["customer_id", "churn"]],
                left_on=f"cust_{customer_id_col}",
                right_on="customer_id",
                how="inner",
            ).drop(columns=["customer_id"])
            frame["churn"] = frame["churn"].astype("int8")
            split_frames[split_name] = frame
    finally:
        con.close()

    df_train = split_frames["train"]
    df_val = split_frames["val"]
    df_test = split_frames["test"]
    target = "churn"

    common_columns = set(df_train.columns) & set(df_val.columns) & set(df_test.columns)
    feature_columns = [
        column
        for column in df_train.select_dtypes(include=[np.number]).columns
        if column != target
        and "label" not in column.lower()
        and not column.lower().endswith("_id")
        and column in common_columns
    ]
    if not feature_columns or df_train[target].nunique() < 2:
        return df_train, df_val, df_test, None, None, len(feature_columns), materialized, target

    model = CatBoostClassifier(
        iterations=400,
        depth=8,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=50,
        allow_writing_files=False,
    )
    model.fit(
        df_train[feature_columns].fillna(0),
        df_train[target],
        eval_set=(df_val[feature_columns].fillna(0), df_val[target]),
        use_best_model=True,
        verbose=50,
    )

    val_predictions = model.predict_proba(df_val[feature_columns].fillna(0))[:, 1]
    test_predictions = model.predict_proba(df_test[feature_columns].fillna(0))[:, 1]

    val_metrics = {
        "average_precision": float(average_precision(df_val[target].to_numpy(), val_predictions)),
        "accuracy": float(accuracy(df_val[target].to_numpy(), val_predictions)),
        "f1": float(f1(df_val[target].to_numpy(), val_predictions)),
        "roc_auc": float(roc_auc(df_val[target].to_numpy(), val_predictions)),
    }
    test_metrics = {
        "average_precision": float(average_precision(df_test[target].to_numpy(), test_predictions)),
        "accuracy": float(accuracy(df_test[target].to_numpy(), test_predictions)),
        "f1": float(f1(df_test[target].to_numpy(), test_predictions)),
        "roc_auc": float(roc_auc(df_test[target].to_numpy(), test_predictions)),
    }

    return df_train, df_val, df_test, val_metrics, test_metrics, len(feature_columns), materialized, target


def main() -> None:
    df_train, df_val, df_test, val_metrics, test_metrics, n_features, materialized, target = run_rel_hm_user_churn()
    print("materialized_files:", materialized, flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
    print("train_timestamp:", TRAIN_CUT_DATE.date(), flush=True)
    print("validation_timestamp:", VALIDATION_CUT_DATE.date(), flush=True)
    print("test_timestamp:", TEST_CUT_DATE.date(), flush=True)
    print("label_period_days:", LABEL_DAYS, flush=True)
    print("target:", target, flush=True)
    print("train_rows:", len(df_train), flush=True)
    print("validation_rows:", len(df_val), flush=True)
    print("test_rows:", len(df_test), flush=True)
    print("columns:", len(df_train.columns), flush=True)
    print("feature_count:", n_features, flush=True)
    print("validation_metrics:", val_metrics if val_metrics is not None else "skipped", flush=True)
    print("test_metrics:", test_metrics if test_metrics is not None else "skipped", flush=True)


if __name__ == "__main__":
    main()
