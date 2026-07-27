#!/usr/bin/env python
"""RelBench rel-hm: item sales example aligned to the official task definition."""

from __future__ import annotations

import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from graphreduce.enum import ComputeLayerEnum, PeriodUnit
from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import DuckdbNode
from relbench_dataset_utils import (
    RelBenchFrameStore,
    get_relbench_dataset_db,
    get_relbench_split_task_table,
    iter_training_frames,
    register_relbench_db_views,
    target_table_from_frame,
)
from relbench_regression_metrics import add_nmae

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
def run_rel_hm_item_sales(
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float] | None, dict[str, float] | None, int, list[str], str]:
    _, db = get_relbench_dataset_db("rel-hm", download=True, upto_test_timestamp=False)
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

        article_id_col = {column.lower(): column for column in article_columns}["article_id"]
        customer_id_col = {column.lower(): column for column in customer_columns}["customer_id"]
        tx_id_col = {column.lower(): column for column in transaction_columns}["transaction_id"]
        tx_customer_col = {column.lower(): column for column in transaction_columns}["customer_id"]
        tx_article_col = {column.lower(): column for column in transaction_columns}["article_id"]
        tx_date_col = {column.lower(): column for column in transaction_columns}["t_dat"]
        tx_price_col = {column.lower(): column for column in transaction_columns}["price"]

        split_tasks = {}
        for split_name in ["train", "val", "test"]:
            task, task_table, cut_timestamps = get_relbench_split_task_table(
                "rel-hm", "item-sales", split_name, download=True, db=db
            )
            split_tasks[split_name] = task
            cut_dates = [timestamp.to_pydatetime() for timestamp in cut_timestamps]
            frame_store = RelBenchFrameStore(f"rel-hm-item-sales-{split_name}")

            def build_frame(frame_con, cut_date):
                con = frame_con
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
                graph._clean_refs()
                features["timestamp"] = pd.Timestamp(cut_date)

                labels = task_table.df.copy()

                frame = features.merge(
                    labels[[task.time_col, task.entity_col, task.target_col]],
                    left_on=["timestamp", f"art_{article_id_col}"],
                    right_on=[task.time_col, task.entity_col],
                    how="right",
                    validate="one_to_one",
                )
                frame[task.target_col] = frame[task.target_col].fillna(0).astype("float64")
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
        eval_set=(df_val[feature_columns].fillna(0), df_val[target].fillna(0).astype("float64")),
        use_best_model=True,
    )

    val_predictions = model.predict(df_val[feature_columns].fillna(0))
    test_predictions = model.predict(df_test[feature_columns].fillna(0))

    val_predictions = np.asarray(val_predictions, dtype="float64")
    test_predictions = np.asarray(test_predictions, dtype="float64")
    val_metrics = add_nmae(
        split_tasks["val"].evaluate(
            val_predictions,
            target_table=target_table_from_frame(split_tasks["val"], df_val),
        ),
        df_val[target],
        val_predictions,
        df_train[target],
    )
    test_metrics = add_nmae(
        split_tasks["test"].evaluate(
            test_predictions,
            target_table=target_table_from_frame(split_tasks["test"], df_test),
        ),
        df_test[target],
        test_predictions,
        df_train[target],
    )

    return df_train, df_val, df_test, val_metrics, test_metrics, len(feature_columns), materialized, target


def main() -> None:
    df_train, df_val, df_test, val_metrics, test_metrics, n_features, materialized, target = run_rel_hm_item_sales()
    print("materialized_files:", materialized, flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
    print("train_cut_dates:", sorted(df_train["timestamp"].drop_duplicates().dt.date.astype(str).tolist()), flush=True)
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
    print("validation_nmae:", val_metrics["nmae"] if val_metrics is not None else "skipped", flush=True)
    print("test_nmae:", test_metrics["nmae"] if test_metrics is not None else "skipped", flush=True)
    print("validation_metrics:", val_metrics if val_metrics is not None else "skipped", flush=True)
    print("test_metrics:", test_metrics if test_metrics is not None else "skipped", flush=True)


if __name__ == "__main__":
    main()
