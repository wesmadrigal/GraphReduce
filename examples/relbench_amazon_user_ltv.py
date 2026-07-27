#!/usr/bin/env python
"""RelBench rel-amazon: user LTV end-to-end example."""

from __future__ import annotations

import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from graphreduce.enum import ComputeLayerEnum, PeriodUnit
from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import DuckdbNode
from relbench_dataset_utils import (
    RelBenchFrameStore,
    get_relbench_dataset_db,
    get_relbench_split_task_table,
    get_relbench_task,
    iter_training_frames,
    register_relbench_db_views,
    target_table_from_frame,
)
from relbench_regression_metrics import add_nmae
from relbench_catboost_utils import fit_tuned_regressor_incremental, set_feature_families

VALIDATION_CUT_DATE = datetime.datetime(2015, 10, 1)
TEST_CUT_DATE = datetime.datetime(2016, 1, 1)
HOLDOUT_CUT_DATE = TEST_CUT_DATE
LOOKBACK_START = datetime.datetime(2008, 1, 1)
LABEL_PERIOD_DAYS = 365 // 4
LABEL_PERIOD = datetime.timedelta(days=LABEL_PERIOD_DAYS)

TABLE_NAME_TO_FILENAME = {
    "customer": "customer.parquet",
    "product": "product.parquet",
    "review": "review.parquet",
}


def _split_timestamps() -> dict[str, list[pd.Timestamp]]:
    train = pd.date_range(
        start=pd.Timestamp(VALIDATION_CUT_DATE) - pd.Timedelta(days=LABEL_PERIOD_DAYS),
        end=pd.Timestamp(LOOKBACK_START),
        freq=-pd.Timedelta(days=LABEL_PERIOD_DAYS),
    ).tolist()
    return {
        "train": train,
        "val": [pd.Timestamp(VALIDATION_CUT_DATE)],
        "test": [pd.Timestamp(TEST_CUT_DATE)],
    }


def _feature_cut_date(task_timestamp: pd.Timestamp) -> datetime.datetime:
    # RelBench allows rows up to the task timestamp inclusive. GraphReduce's
    # feature prep uses a strict `< cut_date` filter, so we shift by one day to
    # include the full labeled day in this midnight-granularity dataset.
    return task_timestamp.to_pydatetime() + datetime.timedelta(days=1)


def _build_labels(
    con: duckdb.DuckDBPyConnection,
    split_timestamps: list[pd.Timestamp],
    customer_id: str,
    product_id: str,
    product_price: str,
    review_customer_id: str,
    review_product_id: str,
    review_time: str,
) -> pd.DataFrame:
    timestamp_df = pd.DataFrame({"timestamp": pd.to_datetime(split_timestamps)})
    con.register("timestamp_df", timestamp_df)
    try:
        return con.sql(
            f"""
            SELECT
                timestamp,
                customer_id,
                ltv
            FROM
                timestamp_df,
                customer_src,
                (
                    SELECT
                        COALESCE(SUM(product_src.{product_price}), 0) AS ltv
                    FROM
                        review_src,
                        product_src
                    WHERE
                        review_src.{review_customer_id} = customer_src.{customer_id}
                        AND review_src.{review_product_id} = product_src.{product_id}
                        AND review_src.{review_time} > timestamp
                        AND review_src.{review_time} <= timestamp + INTERVAL '{LABEL_PERIOD_DAYS} days'
                )
            WHERE
                EXISTS (
                    SELECT 1
                    FROM review_src
                    WHERE
                        review_src.{review_customer_id} = customer_src.{customer_id}
                        AND review_src.{review_time} > timestamp - INTERVAL '{LABEL_PERIOD_DAYS} days'
                        AND review_src.{review_time} <= timestamp
                )
            """
        ).to_df()
    finally:
        con.unregister("timestamp_df")


def _build_feature_frames(
    con: duckdb.DuckDBPyConnection,
    split_timestamps: list[pd.Timestamp],
    customer_columns: list[str],
    product_columns: list[str],
    review_columns: list[str],
    customer_id: str,
    product_id: str,
    review_id: str,
    review_customer_id: str,
    review_product_id: str,
    review_time: str,
) -> RelBenchFrameStore:
    frame_store = RelBenchFrameStore("rel-amazon-user-ltv-features", persist_each_frame=True)

    def build_frame(frame_con: duckdb.DuckDBPyConnection, task_timestamp: pd.Timestamp) -> pd.DataFrame:
        feature_cut_date = _feature_cut_date(task_timestamp)

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
            auto_text_features=False,
        )

        graph = GraphReduce(
            name=f"rel_amazon_user_ltv_{task_timestamp.date()}",
            parent_node=customer_node,
            compute_layer=ComputeLayerEnum.duckdb,
            sql_client=frame_con,
            cut_date=feature_cut_date,
            compute_period_val=(feature_cut_date - LOOKBACK_START).days + 1,
            compute_period_unit=PeriodUnit.day,
            auto_features=True,
            date_filters_on_agg=True,
            auto_feature_hops_back=1,
            auto_feature_hops_front=0,
            use_temp_tables=True,
        )

        nodes = [customer_node, product_node, review_node]
        review_node.feature_family_max_columns = 4
        review_node.categorical_top_k = 5
        set_feature_families([review_node], ("base",))
        for node in nodes:
            graph.add_node(node)

        graph.add_entity_edge(
            customer_node,
            review_node,
            parent_key=customer_id,
            relation_key=review_customer_id,
            reduce=True,
        )
        graph.add_entity_edge(
            product_node,
            review_node,
            parent_key=product_id,
            relation_key=review_product_id,
            reduce=True,
        )

        graph.do_transformations_sql()
        frame = frame_con.sql(f"SELECT * FROM {graph.parent_node._cur_data_ref}").to_df().copy()
        graph._clean_refs()
        frame["timestamp"] = task_timestamp
        return frame

    for frame in iter_training_frames(con, split_timestamps, build_frame):
        frame_store.append(frame)
    return frame_store


def run_rel_amazon_user_ltv(
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float] | None, dict[str, float] | None, int, list[str], str]:
    _, db = get_relbench_dataset_db("rel-amazon", download=True, upto_test_timestamp=False)
    task = get_relbench_task("rel-amazon", "user-ltv", download=True)
    materialized: list[str] = []

    con = duckdb.connect()
    split_stores: dict[str, RelBenchFrameStore] = {}

    try:
        register_relbench_db_views(
            con,
            db,
            {"customer": "customer_src", "product": "product_src", "review": "review_src"},
            {"review": "review_id"},
        )

        customer_columns = con.sql("SELECT * FROM customer_src LIMIT 0").to_df().columns.tolist()
        product_columns = con.sql("SELECT * FROM product_src LIMIT 0").to_df().columns.tolist()
        review_columns = con.sql("SELECT * FROM review_src LIMIT 0").to_df().columns.tolist()
        review_feature_columns = [
            column for column in review_columns
            if column.lower() not in {"review_text", "summary"}
        ]

        customer_id = {column.lower(): column for column in customer_columns}["customer_id"]
        product_id = {column.lower(): column for column in product_columns}["product_id"]
        product_price = {column.lower(): column for column in product_columns}["price"]
        review_id = {column.lower(): column for column in review_columns}["review_id"]
        review_customer_id = {column.lower(): column for column in review_columns}["customer_id"]
        review_product_id = {column.lower(): column for column in review_columns}["product_id"]
        review_time = {column.lower(): column for column in review_columns}["review_time"]

        split_tasks = {}
        for split_name in ["train", "val", "test"]:
            task, task_table, cut_timestamps = get_relbench_split_task_table(
                "rel-amazon",
                "user-ltv",
                split_name,
                download=True,
                task=task,
                db=db,
            )
            split_tasks[split_name] = task
            feature_store = _build_feature_frames(
                con,
                cut_timestamps,
                customer_columns,
                product_columns,
                review_feature_columns,
                customer_id,
                product_id,
                review_id,
                review_customer_id,
                review_product_id,
                review_time,
            )
            labels = task_table.df.copy()
            train_store = RelBenchFrameStore(
                f"rel-amazon-user-ltv-{split_name}", persist_each_frame=True
            )
            for features in feature_store.iter_batches():
                timestamp = pd.Timestamp(features["timestamp"].iloc[0])
                timestamp_labels = labels[pd.to_datetime(labels[task.time_col]) == timestamp]
                frame = features.merge(
                    timestamp_labels[[task.time_col, task.entity_col, task.target_col]],
                    left_on=["timestamp", f"cust_{customer_id}"],
                    right_on=[task.time_col, task.entity_col],
                    how="right",
                    validate="one_to_one",
                )
                frame[task.target_col] = frame[task.target_col].fillna(0).astype("float64")
                train_store.append(frame)
            feature_store.close()
            split_stores[split_name] = train_store
    finally:
        con.close()

    train_store = split_stores["train"]
    df_val = split_stores["val"].to_dataframe()
    df_test = split_stores["test"].to_dataframe()
    split_stores["val"].close()
    split_stores["test"].close()
    target = split_tasks["train"].target_col

    train_sample = train_store.sample_frame()
    common_columns = set(train_sample.columns) & set(df_val.columns) & set(df_test.columns)
    feature_columns = [
        column
        for column in train_sample.select_dtypes(include=[np.number]).columns
        if column != target
        and "label" not in column.lower()
        and not column.lower().endswith("_id")
        and column in common_columns
    ]
    if not feature_columns:
        return train_store, df_val, df_test, None, None, 0, materialized, target

    model, best_config, best_val_mae = fit_tuned_regressor_incremental(
        lambda: train_store.iter_batches(),
        feature_columns,
        target,
        df_val[feature_columns].fillna(0),
        df_val[target].fillna(0).astype("float64"),
        batch_count=len(train_store.part_paths),
    )
    print("catboost_config:", best_config, flush=True)
    print("catboost_validation_mae:", best_val_mae, flush=True)
    print("catboost_best_iteration:", model.get_best_iteration(), flush=True)

    val_predictions = model.predict(df_val[feature_columns].fillna(0))
    test_predictions = model.predict(df_test[feature_columns].fillna(0))
    val_metrics = add_nmae(
        split_tasks["val"].evaluate(
            val_predictions,
            target_table=target_table_from_frame(split_tasks["val"], df_val),
        ),
        df_val[target],
        val_predictions,
        train_store.target_std(target),
    )
    test_metrics = add_nmae(
        split_tasks["test"].evaluate(
            test_predictions,
            target_table=target_table_from_frame(split_tasks["test"], df_test),
        ),
        df_test[target],
        test_predictions,
        train_store.target_std(target),
    )

    return train_store, df_val, df_test, val_metrics, test_metrics, len(feature_columns), materialized, target


def main() -> None:
    df_train, df_val, df_test, val_metrics, test_metrics, n_features, materialized, target = run_rel_amazon_user_ltv()
    print("materialized_files:", materialized, flush=True)
    print("validation_cut_date:", VALIDATION_CUT_DATE.date(), flush=True)
    print("test_cut_date:", TEST_CUT_DATE.date(), flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
    print("train_timestamps:", df_train.column_nunique("timestamp"), flush=True)
    print("label_period_days:", LABEL_PERIOD_DAYS, flush=True)
    print("target:", target, flush=True)
    print("train_rows:", df_train.row_count, flush=True)
    print("validation_rows:", len(df_val), flush=True)
    print("test_rows:", len(df_test), flush=True)
    print("columns:", len(df_train.columns), flush=True)
    print("feature_count:", n_features, flush=True)
    print("validation_nmae:", val_metrics["nmae"] if val_metrics is not None else "skipped", flush=True)
    print("test_nmae:", test_metrics["nmae"] if test_metrics is not None else "skipped", flush=True)
    print("validation_metrics:", val_metrics if val_metrics is not None else "skipped", flush=True)
    print("test_metrics:", test_metrics if test_metrics is not None else "skipped", flush=True)
    df_train.close()


if __name__ == "__main__":
    main()
