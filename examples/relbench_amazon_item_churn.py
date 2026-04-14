#!/usr/bin/env python
"""RelBench rel-amazon: item churn example aligned to the official task definition."""

from __future__ import annotations

import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from relbench.metrics import accuracy, average_precision, f1, roc_auc

from graphreduce.enum import ComputeLayerEnum, PeriodUnit
from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import DuckdbNode
from relbench_dataset_utils import materialize_relbench_dataset

TABLE_NAME_TO_FILENAME = {
    "customer": "customer.parquet",
    "product": "product.parquet",
    "review": "review.parquet",
}

VALIDATION_CUT_DATE = datetime.datetime(2015, 10, 1)
TEST_CUT_DATE = datetime.datetime(2016, 1, 1)
CUT_DATE = TEST_CUT_DATE
LOOKBACK_START = datetime.datetime(2008, 1, 1)
LABEL_PERIOD_DAYS = 365 // 4
LOOKBACK_DAYS = (TEST_CUT_DATE - LOOKBACK_START).days + 1
TRAIN_CUT_DATE = VALIDATION_CUT_DATE - datetime.timedelta(days=LABEL_PERIOD_DAYS)


def run_rel_amazon_item_churn(
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float] | None, dict[str, float] | None, int, list[str], str]:
    use_dir = data_dir or Path("tests/data/relbench/rel-amazon")
    materialized = materialize_relbench_dataset("rel-amazon", use_dir, TABLE_NAME_TO_FILENAME)

    con = duckdb.connect()
    split_frames: dict[str, pd.DataFrame] = {}

    try:
        con.sql(f"CREATE OR REPLACE VIEW customer_src AS SELECT * FROM read_parquet('{use_dir / 'customer.parquet'}')")
        con.sql(f"CREATE OR REPLACE VIEW product_src AS SELECT * FROM read_parquet('{use_dir / 'product.parquet'}')")
        con.sql(
            f"""
            CREATE OR REPLACE VIEW review_src AS
            SELECT
                row_number() OVER () AS review_id,
                *
            FROM read_parquet('{use_dir / 'review.parquet'}')
            """
        )

        customer_columns = con.sql("SELECT * FROM customer_src LIMIT 0").to_df().columns.tolist()
        product_columns = con.sql("SELECT * FROM product_src LIMIT 0").to_df().columns.tolist()
        review_columns = con.sql("SELECT * FROM review_src LIMIT 0").to_df().columns.tolist()

        customer_id = {column.lower(): column for column in customer_columns}["customer_id"]
        product_id = {column.lower(): column for column in product_columns}["product_id"]
        review_id = {column.lower(): column for column in review_columns}["review_id"]
        review_customer_id = {column.lower(): column for column in review_columns}["customer_id"]
        review_product_id = {column.lower(): column for column in review_columns}["product_id"]
        review_time = {column.lower(): column for column in review_columns}["review_time"]

        for split_name, cut_date in {
            "train": TRAIN_CUT_DATE,
            "val": VALIDATION_CUT_DATE,
            "test": TEST_CUT_DATE,
        }.items():
            feature_cut_date = cut_date + datetime.timedelta(days=1)

            customer_node = DuckdbNode(
                fpath="customer_src",
                prefix="cust",
                pk=customer_id,
                date_key=None,
                columns=customer_columns,
            )
            product_node = DuckdbNode(
                fpath="product_src",
                prefix="prod",
                pk=product_id,
                date_key=None,
                columns=product_columns,
            )
            review_node = DuckdbNode(
                fpath="review_src",
                prefix="rev",
                pk=review_id,
                date_key=review_time,
                columns=review_columns,
            )

            graph = GraphReduce(
                name=f"rel_amazon_item_churn_{cut_date.date()}",
                parent_node=product_node,
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
                use_temp_tables=True,
            )

            for node in [customer_node, product_node, review_node]:
                graph.add_node(node)

            graph.add_entity_edge(customer_node, review_node, parent_key=customer_id, relation_key=review_customer_id, reduce=True)
            graph.add_entity_edge(product_node, review_node, parent_key=product_id, relation_key=review_product_id, reduce=True)

            graph.do_transformations_sql()
            features = con.sql(f"SELECT * FROM {graph.parent_node._cur_data_ref}").to_df().copy()

            labels = con.sql(
                f"""
                WITH timestamp_df AS (
                    SELECT TIMESTAMP '{cut_date}' AS timestamp
                )
                SELECT
                    timestamp,
                    product_id,
                    CAST(
                        NOT EXISTS (
                            SELECT 1
                            FROM review_src
                            WHERE
                                review_src.{review_product_id} = product_src.{product_id}
                                AND review_src.{review_time} > timestamp
                                AND review_src.{review_time} <= timestamp + INTERVAL '{LABEL_PERIOD_DAYS} days'
                        ) AS INTEGER
                    ) AS churn
                FROM
                    timestamp_df,
                    product_src
                WHERE
                    EXISTS (
                        SELECT 1
                        FROM review_src
                        WHERE
                            review_src.{review_product_id} = product_src.{product_id}
                            AND review_src.{review_time} > timestamp - INTERVAL '{LABEL_PERIOD_DAYS} days'
                            AND review_src.{review_time} <= timestamp
                    )
                """
            ).to_df()

            frame = features.merge(
                labels[["product_id", "churn"]],
                left_on=f"prod_{product_id}",
                right_on="product_id",
                how="inner",
            ).drop(columns=["product_id"])
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
        iterations=2000,
        depth=6,
        learning_rate=0.02,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        l2_leaf_reg=10.0,
        random_strength=1.0,
        bagging_temperature=1.0,
        border_count=254,
        od_type="Iter",
        od_wait=200,
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
    df_train, df_val, df_test, val_metrics, test_metrics, n_features, materialized, target = run_rel_amazon_item_churn()
    print("materialized_files:", materialized, flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
    print("train_cut_date:", TRAIN_CUT_DATE.date(), flush=True)
    print("validation_cut_date:", VALIDATION_CUT_DATE.date(), flush=True)
    print("test_cut_date:", TEST_CUT_DATE.date(), flush=True)
    print("label_period_days:", LABEL_PERIOD_DAYS, flush=True)
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
