#!/usr/bin/env python
"""RelBench rel-amazon: user churn example aligned to the official task definition."""

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
from relbench_catboost_utils import fit_incremental_classifier, set_feature_families

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


def run_rel_amazon_user_churn(
    data_dir: Path | None = None,
) -> tuple[RelBenchFrameStore, pd.DataFrame, pd.DataFrame, dict[str, float] | None, dict[str, float] | None, int, list[str], str]:
    _, db = get_relbench_dataset_db("rel-amazon", download=True, upto_test_timestamp=False)
    official_task = get_relbench_task("rel-amazon", "user-churn", download=True)
    materialized: list[str] = []

    con = duckdb.connect()
    split_frames: dict[str, RelBenchFrameStore] = {}

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
        review_id = {column.lower(): column for column in review_columns}["review_id"]
        review_customer_id = {column.lower(): column for column in review_columns}["customer_id"]
        review_product_id = {column.lower(): column for column in review_columns}["product_id"]
        review_time = {column.lower(): column for column in review_columns}["review_time"]

        split_tasks = {}
        split_specs = {}
        for split_name in ("train", "val", "test"):
            task, task_table, cut_timestamps = get_relbench_split_task_table(
                "rel-amazon",
                "user-churn",
                split_name,
                download=True,
                task=official_task,
                db=db,
            )
            split_tasks[split_name] = task
            split_specs[split_name] = (
                task,
                task_table,
                [timestamp.to_pydatetime() for timestamp in cut_timestamps],
            )

        for split_name, (task, task_table, cut_dates) in split_specs.items():
            frame_store = RelBenchFrameStore(
                f"rel-amazon-user-churn-{split_name}", persist_each_frame=True
            )
            def build_frame(frame_con: duckdb.DuckDBPyConnection, cut_date: datetime.datetime) -> pd.DataFrame:
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
                    columns=review_feature_columns,
                    auto_text_features=False,
                )
                review_node.feature_family_max_columns = 4
                review_node.categorical_top_k = 5
                set_feature_families([review_node], ("base",))

                graph = GraphReduce(
                    name=f"rel_amazon_user_churn_{cut_date.date()}",
                    parent_node=customer_node,
                    compute_layer=ComputeLayerEnum.duckdb,
                    sql_client=frame_con,
                    cut_date=feature_cut_date,
                    compute_period_val=(feature_cut_date - LOOKBACK_START).days + 1,
                    compute_period_unit=PeriodUnit.day,
                    auto_features=True,
                    auto_labels=False,
                    date_filters_on_agg=True,
                    auto_feature_hops_back=1,
                    auto_feature_hops_front=0,
                    use_temp_tables=True,
                )

                for node in [customer_node, product_node, review_node]:
                    graph.add_node(node)

                graph.add_entity_edge(customer_node, review_node, parent_key=customer_id, relation_key=review_customer_id, reduce=True)
                graph.add_entity_edge(product_node, review_node, parent_key=product_id, relation_key=review_product_id, reduce=True)

                graph.do_transformations_sql()
                features = frame_con.sql(f"SELECT * FROM {graph.parent_node._cur_data_ref}").to_df().copy()
                graph._clean_refs()
                features["timestamp"] = pd.Timestamp(cut_date)

                labels = task_table.df.copy()
                labels[task.time_col] = pd.to_datetime(labels[task.time_col])
                labels = labels[
                    labels[task.time_col] == pd.Timestamp(cut_date)
                ].copy()

                frame = features.merge(
                    labels[[task.time_col, task.entity_col, task.target_col]],
                    left_on=["timestamp", f"cust_{customer_id}"],
                    right_on=[task.time_col, task.entity_col],
                    how="right",
                    validate="one_to_one",
                )
                frame[task.target_col] = frame[task.target_col].astype("int8")
                return frame

            frame_workers = None if split_name == "train" else 1
            for frame in iter_training_frames(con, cut_dates, build_frame, workers=frame_workers):
                frame_store.append(frame)
            split_frames[split_name] = frame_store
    finally:
        con.close()

    train_store = split_frames["train"]
    df_val = split_frames["val"].to_dataframe()
    df_test = split_frames["test"].to_dataframe()
    split_frames["val"].close()
    split_frames["test"].close()
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
    if not feature_columns or train_store.target_nunique(target) < 2:
        return train_store, df_val, df_test, None, None, len(feature_columns), materialized, target

    model, val_auc = fit_incremental_classifier(
        lambda: train_store.iter_batches(),
        feature_columns,
        target,
        df_val[feature_columns].fillna(0),
        df_val[target],
        batch_count=len(train_store.part_paths),
        config={
            "name": "amazon_user_churn_incremental",
            "iterations": 2000,
            "depth": 6,
            "learning_rate": 0.02,
            "l2_leaf_reg": 10.0,
        },
        auto_class_weights="Balanced",
    )
    print("catboost_validation_auc:", val_auc, flush=True)

    val_predictions = model.predict_proba(df_val[feature_columns].fillna(0))[:, 1]
    test_predictions = model.predict_proba(df_test[feature_columns].fillna(0))[:, 1]

    val_metrics = split_tasks["val"].evaluate(
        val_predictions,
        target_table=target_table_from_frame(split_tasks["val"], df_val),
    )
    test_metrics = split_tasks["test"].evaluate(
        test_predictions,
        target_table=target_table_from_frame(split_tasks["test"], df_test),
    )

    return train_store, df_val, df_test, val_metrics, test_metrics, len(feature_columns), materialized, target


def main() -> None:
    df_train, df_val, df_test, val_metrics, test_metrics, n_features, materialized, target = run_rel_amazon_user_churn()
    print("materialized_files:", materialized, flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
    print("train_timestamps:", df_train.column_nunique("timestamp"), flush=True)
    print("validation_cut_date:", VALIDATION_CUT_DATE.date(), flush=True)
    print("test_cut_date:", TEST_CUT_DATE.date(), flush=True)
    print("label_period_days:", LABEL_PERIOD_DAYS, flush=True)
    print("target:", target, flush=True)
    print("train_rows:", df_train.row_count, flush=True)
    print("validation_rows:", len(df_val), flush=True)
    print("test_rows:", len(df_test), flush=True)
    print("columns:", len(df_train.columns), flush=True)
    print("feature_count:", n_features, flush=True)
    print("validation_metrics:", val_metrics if val_metrics is not None else "skipped", flush=True)
    print("test_metrics:", test_metrics if test_metrics is not None else "skipped", flush=True)
    df_train.close()


if __name__ == "__main__":
    main()
