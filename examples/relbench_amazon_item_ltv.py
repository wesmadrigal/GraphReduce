#!/usr/bin/env python
"""RelBench rel-amazon: item LTV end-to-end example."""

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

VALIDATION_CUT_DATE = datetime.datetime(2015, 10, 1)
TEST_CUT_DATE = datetime.datetime(2016, 1, 1)
HOLDOUT_CUT_DATE = TEST_CUT_DATE
LOOKBACK_START = datetime.datetime(2008, 1, 1)
LABEL_PERIOD_DAYS = 365 // 4

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
    product_id: str,
    product_price: str,
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
                product_src.{product_id} AS product_id,
                COALESCE(SUM(product_src.{product_price}), 0) AS ltv
            FROM
                timestamp_df,
                product_src,
                review_src
            WHERE
                review_src.{review_product_id} = product_src.{product_id}
                AND review_src.{review_time} > timestamp
                AND review_src.{review_time} <= timestamp + INTERVAL '{LABEL_PERIOD_DAYS} days'
            GROUP BY
                timestamp,
                product_src.{product_id}
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
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for task_timestamp in split_timestamps:
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
        )

        graph = GraphReduce(
            name=f"rel_amazon_item_ltv_{task_timestamp.date()}",
            parent_node=product_node,
            compute_layer=ComputeLayerEnum.duckdb,
            sql_client=con,
            cut_date=feature_cut_date,
            compute_period_val=(feature_cut_date - LOOKBACK_START).days + 1,
            compute_period_unit=PeriodUnit.day,
            auto_features=True,
            date_filters_on_agg=True,
            auto_feature_hops_back=3,
            auto_feature_hops_front=0,
            use_temp_tables=True,
        )

        for node in [customer_node, product_node, review_node]:
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
        frame = con.sql(f"SELECT * FROM {graph.parent_node._cur_data_ref}").to_df().copy()
        frame["timestamp"] = task_timestamp
        frames.append(frame)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _relbench_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    truth = y_true.fillna(0).astype("float64").to_numpy()
    pred = np.asarray(y_pred, dtype="float64")
    return {
        "r2": float(r2(truth, pred)),
        "mae": float(mae(truth, pred)),
        "rmse": float(rmse(truth, pred)),
    }


def run_rel_amazon_item_ltv(
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float] | None, dict[str, float] | None, int, list[str], str]:
    use_dir = data_dir or Path("tests/data/relbench/rel-amazon")
    materialized = materialize_relbench_dataset("rel-amazon", use_dir, TABLE_NAME_TO_FILENAME)

    con = duckdb.connect()
    split_frames: dict[str, pd.DataFrame] = {}

    try:
        for table_name, filename in TABLE_NAME_TO_FILENAME.items():
            if table_name == "review":
                con.sql(
                    f"""
                    CREATE OR REPLACE VIEW review_src AS
                    SELECT
                        row_number() OVER () AS review_id,
                        *
                    FROM read_parquet('{use_dir / filename}')
                    """
                )
            else:
                con.sql(f"CREATE OR REPLACE VIEW {table_name}_src AS SELECT * FROM read_parquet('{use_dir / filename}')")

        customer_columns = con.sql("SELECT * FROM customer_src LIMIT 0").to_df().columns.tolist()
        product_columns = con.sql("SELECT * FROM product_src LIMIT 0").to_df().columns.tolist()
        review_columns = con.sql("SELECT * FROM review_src LIMIT 0").to_df().columns.tolist()

        customer_id = {column.lower(): column for column in customer_columns}["customer_id"]
        product_id = {column.lower(): column for column in product_columns}["product_id"]
        product_price = {column.lower(): column for column in product_columns}["price"]
        review_id = {column.lower(): column for column in review_columns}["review_id"]
        review_customer_id = {column.lower(): column for column in review_columns}["customer_id"]
        review_product_id = {column.lower(): column for column in review_columns}["product_id"]
        review_time = {column.lower(): column for column in review_columns}["review_time"]

        for split_name, split_timestamps in _split_timestamps().items():
            labels = _build_labels(
                con,
                split_timestamps,
                product_id,
                product_price,
                review_product_id,
                review_time,
            )
            features = _build_feature_frames(
                con,
                split_timestamps,
                customer_columns,
                product_columns,
                review_columns,
                customer_id,
                product_id,
                review_id,
                review_customer_id,
                review_product_id,
                review_time,
            )
            frame = features.merge(
                labels[["timestamp", "product_id", "ltv"]],
                left_on=["timestamp", f"prod_{product_id}"],
                right_on=["timestamp", "product_id"],
                how="inner",
            ).drop(columns=["product_id"])
            frame["ltv"] = frame["ltv"].fillna(0).astype("float64")
            split_frames[split_name] = frame
    finally:
        con.close()

    df_train = split_frames["train"]
    df_val = split_frames["val"]
    df_test = split_frames["test"]
    target = "ltv"

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
    val_metrics = _relbench_metrics(df_val[target], val_predictions)
    test_metrics = _relbench_metrics(df_test[target], test_predictions)

    return df_train, df_val, df_test, val_metrics, test_metrics, len(feature_columns), materialized, target


def main() -> None:
    df_train, df_val, df_test, val_metrics, test_metrics, n_features, materialized, target = run_rel_amazon_item_ltv()
    print("materialized_files:", materialized, flush=True)
    print("validation_cut_date:", VALIDATION_CUT_DATE.date(), flush=True)
    print("test_cut_date:", TEST_CUT_DATE.date(), flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
    print("train_timestamps:", df_train["timestamp"].nunique(), flush=True)
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
