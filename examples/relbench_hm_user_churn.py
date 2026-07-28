#!/usr/bin/env python
"""RelBench rel-hm: user churn example aligned to the official task definition."""

from __future__ import annotations

import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from relbench_dataset_utils import (
    RelBenchFrameStore,
    get_relbench_dataset_db,
    get_relbench_split_task_table,
    get_relbench_task,
    iter_training_frames,
    register_relbench_db_views,
    target_table_from_frame,
)

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


def run_rel_hm_user_churn(
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float] | None, dict[str, float] | None, int, list[str], str]:
    _, db = get_relbench_dataset_db("rel-hm", download=True, upto_test_timestamp=False)
    official_task = get_relbench_task("rel-hm", "user-churn", download=True)
    materialized: list[str] = []

    con = duckdb.connect()
    split_frames: dict[str, pd.DataFrame] = {}

    try:
        register_relbench_db_views(
            con,
            db,
            {
                "article": "article_src",
                "customer": "customer_src",
                "transactions": "transactions_src",
            },
            {"transactions": "transaction_id"},
        )

        article_columns = con.sql("SELECT * FROM article_src LIMIT 0").to_df().columns.tolist()
        customer_columns = con.sql("SELECT * FROM customer_src LIMIT 0").to_df().columns.tolist()
        transaction_columns = con.sql("SELECT * FROM transactions_src LIMIT 0").to_df().columns.tolist()

        article_columns_by_lower = {column.lower(): column for column in article_columns}
        article_feature_names = [
            "article_id",
            "product_group_name",
            "department_name",
            "index_group_name",
            "section_name",
            "garment_group_name",
        ]
        article_columns = [
            article_columns_by_lower[name]
            for name in article_feature_names
            if name in article_columns_by_lower
        ]

        article_id_col = {column.lower(): column for column in article_columns}["article_id"]
        customer_id_col = {column.lower(): column for column in customer_columns}["customer_id"]
        tx_id_col = {column.lower(): column for column in transaction_columns}["transaction_id"]
        tx_customer_col = {column.lower(): column for column in transaction_columns}["customer_id"]
        tx_article_col = {column.lower(): column for column in transaction_columns}["article_id"]
        tx_date_col = {column.lower(): column for column in transaction_columns}["t_dat"]

        split_tasks = {}
        split_specs = {}
        for split_name in ("train", "val", "test"):
            task, task_table, cut_timestamps = get_relbench_split_task_table(
                "rel-hm",
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
            frame_store = RelBenchFrameStore(f"rel-hm-user-churn-{split_name}")
            def build_frame(frame_con, cut_date):
                con = frame_con
                feature_cut_date = cut_date + datetime.timedelta(days=1)
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
                feature_families=("base",),
                categorical_cardinality_threshold=5,
                categorical_top_k=2,
                auto_text_features=False,
            )
                transactions = DuckdbNode(
                fpath="transactions_src",
                prefix="txn",
                pk=tx_id_col,
                date_key=tx_date_col,
                columns=transaction_columns,
                feature_families=("base", "temporal"),
                ts_periods=[1, 7, 30, 90, 365],
                categorical_cardinality_threshold=5,
                categorical_top_k=2,
                auto_text_features=False,
                feature_family_max_columns=8,
            )

                graph = GraphReduce(
                name=f"rel_hm_user_churn_{cut_date.date()}",
                parent_node=customer,
                compute_layer=ComputeLayerEnum.duckdb,
                sql_client=con,
                cut_date=feature_cut_date,
                compute_period_val=(feature_cut_date - LOOKBACK_START).days,
                compute_period_unit=PeriodUnit.day,
                auto_features=True,
                auto_labels=False,
                date_filters_on_agg=True,
                # Customer churn needs transaction history and its immediate
                # article attributes. Deeper reverse walks revisit the
                # 15M-row transaction table and create an unbounded join.
                auto_feature_hops_back=2,
                auto_feature_hops_front=0,
            )

                for node in [customer, article, transactions]:
                    graph.add_node(node)

                graph.add_entity_edge(customer, transactions, parent_key=customer_id_col, relation_key=tx_customer_col, reduce=True)
                graph.add_entity_edge(transactions, article, parent_key=tx_article_col, relation_key=article_id_col, reduce=True)

                graph.do_transformations_sql()
                features = con.sql(f"SELECT * FROM {graph.parent_node._cur_data_ref}").to_df().copy()
                graph._clean_refs()
                features["timestamp"] = pd.Timestamp(cut_date)

                labels = task_table.df.copy()
                labels[task.time_col] = pd.to_datetime(labels[task.time_col])
                labels = labels[
                    labels[task.time_col] == pd.Timestamp(cut_date)
                ].copy()

                frame = features.merge(
                    labels[[task.time_col, task.entity_col, task.target_col]],
                    left_on=["timestamp", f"cust_{customer_id_col}"],
                    right_on=[task.time_col, task.entity_col],
                    how="right",
                    validate="one_to_one",
                )
                frame[task.target_col] = frame[task.target_col].astype("int8")
                return frame

            frame_workers = None if split_name == "train" else 1
            for frame in iter_training_frames(con, cut_dates, build_frame, workers=frame_workers):
                frame_store.append(frame)
            split_frames[split_name] = frame_store.to_dataframe()
            frame_store.close()
    finally:
        con.close()

    df_train = split_frames["train"]
    df_val = split_frames["val"]
    df_test = split_frames["test"]
    target = split_tasks["train"].target_col

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

    val_metrics = split_tasks["val"].evaluate(
        val_predictions,
        target_table=target_table_from_frame(split_tasks["val"], df_val),
    )
    test_metrics = split_tasks["test"].evaluate(
        test_predictions,
        target_table=target_table_from_frame(split_tasks["test"], df_test),
    )

    return df_train, df_val, df_test, val_metrics, test_metrics, len(feature_columns), materialized, target


def main() -> None:
    df_train, df_val, df_test, val_metrics, test_metrics, n_features, materialized, target = run_rel_hm_user_churn()
    print("materialized_files:", materialized, flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
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
