#!/usr/bin/env python
"""RelBench rel-hm: item sales example aligned to the official task definition."""

from __future__ import annotations

import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from relbench.metrics import mae, r2, rmse

from graphreduce.enum import ComputeLayerEnum, PeriodUnit
from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import DuckdbNode
from relbench_dataset_utils import materialize_relbench_dataset

TABLE_NAME_TO_FILENAME = {
    "article": "article.parquet",
    "customer": "customer.parquet",
    "transactions": "transactions.parquet",
}

LOOKBACK_START = datetime.datetime(2019, 9, 7)
VALIDATION_CUT_DATE = datetime.datetime(2020, 9, 7)
TEST_CUT_DATE = datetime.datetime(2020, 9, 14)
HOLDOUT_DATE = TEST_CUT_DATE
LABEL_DAYS = 7
TRAIN_CUT_DATES = pd.date_range(
    start=pd.Timestamp(VALIDATION_CUT_DATE) - pd.Timedelta(days=LABEL_DAYS),
    end=pd.Timestamp(LOOKBACK_START),
    freq=-pd.Timedelta(days=LABEL_DAYS),
).to_pydatetime().tolist()


def run_rel_hm_item_sales(
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
        tx_price_col = {column.lower(): column for column in transaction_columns}["price"]

        for split_name, cut_dates in {
            "train": TRAIN_CUT_DATES,
            "val": [VALIDATION_CUT_DATE],
            "test": [TEST_CUT_DATE],
        }.items():
            frames_for_split: list[pd.DataFrame] = []

            for cut_date in cut_dates:
                feature_cut_date = cut_date + datetime.timedelta(days=1)

                article = DuckdbNode(
                    fpath="article_src",
                    prefix="art",
                    pk=article_id_col,
                    date_key=None,
                    columns=article_columns,
                )
                customer = DuckdbNode(
                    fpath="customer_src",
                    prefix="cust",
                    pk=customer_id_col,
                    date_key=None,
                    columns=customer_columns,
                )
                transactions = DuckdbNode(
                    fpath="transactions_src",
                    prefix="txn",
                    pk=tx_id_col,
                    date_key=tx_date_col,
                    columns=transaction_columns,
                )

                graph = GraphReduce(
                    name=f"rel_hm_item_sales_{cut_date.date()}",
                    parent_node=article,
                    compute_layer=ComputeLayerEnum.duckdb,
                    sql_client=con,
                    cut_date=feature_cut_date,
                    compute_period_val=(feature_cut_date - LOOKBACK_START).days + 1,
                    compute_period_unit=PeriodUnit.day,
                    auto_features=True,
                    auto_labels=False,
                    date_filters_on_agg=True,
                    auto_feature_hops_back=3,
                    auto_feature_hops_front=0,
                )

                for node in [article, customer, transactions]:
                    graph.add_node(node)

                graph.add_entity_edge(article, transactions, parent_key=article_id_col, relation_key=tx_article_col, reduce=True)
                graph.add_entity_edge(customer, transactions, parent_key=customer_id_col, relation_key=tx_customer_col, reduce=True)

                graph.do_transformations_sql()
                features = con.sql(f"SELECT * FROM {graph.parent_node._cur_data_ref}").to_df().copy()
                features["timestamp"] = pd.Timestamp(cut_date)

                labels = con.sql(
                    f"""
                    WITH timestamp_df AS (
                        SELECT TIMESTAMP '{cut_date}' AS timestamp
                    )
                    SELECT
                        timestamp,
                        article_id,
                        sales
                    FROM
                        timestamp_df,
                        article_src,
                        (
                            SELECT
                                COALESCE(SUM({tx_price_col}), 0) AS sales
                            FROM
                                transactions_src
                            WHERE
                                transactions_src.{tx_article_col} = article_src.{article_id_col}
                                AND transactions_src.{tx_date_col} > timestamp
                                AND transactions_src.{tx_date_col} <= timestamp + INTERVAL '{LABEL_DAYS} days'
                        )
                    """
                ).to_df()

                frame = features.merge(
                    labels[["timestamp", "article_id", "sales"]],
                    left_on=["timestamp", f"art_{article_id_col}"],
                    right_on=["timestamp", "article_id"],
                    how="inner",
                ).drop(columns=["article_id"])
                frame["sales"] = frame["sales"].fillna(0).astype("float64")
                frames_for_split.append(frame)

            split_frames[split_name] = pd.concat(frames_for_split, ignore_index=True)
    finally:
        con.close()

    df_train = split_frames["train"]
    df_val = split_frames["val"]
    df_test = split_frames["test"]
    target = "sales"

    common_columns = set(df_train.columns) & set(df_val.columns) & set(df_test.columns)
    feature_columns = [
        column
        for column in df_train.select_dtypes(include=[np.number]).columns
        if column != target
        and "label" not in column.lower()
        and not column.lower().endswith("_id")
        and column in common_columns
    ]
    if not feature_columns:
        return df_train, df_val, df_test, None, None, 0, materialized, target

    model = CatBoostRegressor(
        iterations=700,
        depth=8,
        learning_rate=0.05,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=42,
        verbose=50,
        allow_writing_files=False,
    )
    model.fit(
        df_train[feature_columns].fillna(0),
        df_train[target].fillna(0).astype("float64"),
    )

    val_predictions = model.predict(df_val[feature_columns].fillna(0))
    test_predictions = model.predict(df_test[feature_columns].fillna(0))

    val_metrics = {
        "r2": float(r2(df_val[target].fillna(0).astype("float64").to_numpy(), np.asarray(val_predictions, dtype="float64"))),
        "mae": float(mae(df_val[target].fillna(0).astype("float64").to_numpy(), np.asarray(val_predictions, dtype="float64"))),
        "rmse": float(rmse(df_val[target].fillna(0).astype("float64").to_numpy(), np.asarray(val_predictions, dtype="float64"))),
    }
    test_metrics = {
        "r2": float(r2(df_test[target].fillna(0).astype("float64").to_numpy(), np.asarray(test_predictions, dtype="float64"))),
        "mae": float(mae(df_test[target].fillna(0).astype("float64").to_numpy(), np.asarray(test_predictions, dtype="float64"))),
        "rmse": float(rmse(df_test[target].fillna(0).astype("float64").to_numpy(), np.asarray(test_predictions, dtype="float64"))),
    }

    return df_train, df_val, df_test, val_metrics, test_metrics, len(feature_columns), materialized, target


def main() -> None:
    df_train, df_val, df_test, val_metrics, test_metrics, n_features, materialized, target = run_rel_hm_item_sales()
    print("materialized_files:", materialized, flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
    print("train_cut_dates:", [cut_date.date() for cut_date in TRAIN_CUT_DATES], flush=True)
    print("train_timestamps:", df_train["timestamp"].nunique(), flush=True)
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
