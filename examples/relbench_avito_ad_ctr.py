#!/usr/bin/env python
"""RelBench rel-avito: ad CTR example aligned to the official task definition."""

from __future__ import annotations

import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from relbench.metrics import mae, r2, rmse
from relbench_dataset_utils import materialize_relbench_dataset

from graphreduce.enum import ComputeLayerEnum, PeriodUnit
from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import DuckdbNode

TABLE_NAME_TO_FILENAME = {
    "AdsInfo": "AdsInfo.parquet",
    "Category": "Category.parquet",
    "Location": "Location.parquet",
    "PhoneRequestsStream": "PhoneRequestsStream.parquet",
    "SearchInfo": "SearchInfo.parquet",
    "SearchStream": "SearchStream.parquet",
    "UserInfo": "UserInfo.parquet",
    "VisitStream": "VisitsStream.parquet",
}

LOOKBACK_START = datetime.datetime(2015, 4, 25)
VALIDATION_CUT_DATE = datetime.datetime(2015, 5, 8)
TEST_CUT_DATE = datetime.datetime(2015, 5, 14)
CUT_DATE = TEST_CUT_DATE
LABEL_PERIOD_DAYS = 4
LOOKBACK_DAYS = (TEST_CUT_DATE - LOOKBACK_START).days + 1
TRAIN_CUT_DATES = [
    datetime.datetime(2015, 5, 4),
    datetime.datetime(2015, 4, 30),
    datetime.datetime(2015, 4, 26),
]


def run_rel_avito_ad_ctr(
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float] | None, dict[str, float] | None, int, list[str], str]:
    use_dir = data_dir or Path("tests/data/relbench/rel-avito")
    materialized = materialize_relbench_dataset("rel-avito", use_dir, TABLE_NAME_TO_FILENAME)

    con = duckdb.connect()
    split_frames: dict[str, pd.DataFrame] = {}

    try:
        con.sql(f"CREATE OR REPLACE VIEW ads_src AS SELECT * FROM read_parquet('{use_dir / 'AdsInfo.parquet'}')")
        con.sql(f"CREATE OR REPLACE VIEW category_src AS SELECT * FROM read_parquet('{use_dir / 'Category.parquet'}')")
        con.sql(f"CREATE OR REPLACE VIEW location_src AS SELECT * FROM read_parquet('{use_dir / 'Location.parquet'}')")
        con.sql(
            f"""
            CREATE OR REPLACE VIEW search_stream_src AS
            SELECT
                row_number() OVER () AS search_stream_id,
                *
            FROM read_parquet('{use_dir / 'SearchStream.parquet'}')
            """
        )
        ads_columns = con.sql("SELECT * FROM ads_src LIMIT 0").to_df().columns.tolist()
        category_columns = con.sql("SELECT * FROM category_src LIMIT 0").to_df().columns.tolist()
        location_columns = con.sql("SELECT * FROM location_src LIMIT 0").to_df().columns.tolist()
        search_stream_columns = con.sql("SELECT * FROM search_stream_src LIMIT 0").to_df().columns.tolist()

        ad_id = {column.lower(): column for column in ads_columns}["adid"]
        category_id = {column.lower(): column for column in category_columns}["categoryid"]
        location_id = {column.lower(): column for column in location_columns}["locationid"]
        stream_id = {column.lower(): column for column in search_stream_columns}["search_stream_id"]
        stream_search_id = {column.lower(): column for column in search_stream_columns}["searchid"]
        stream_ad_id = {column.lower(): column for column in search_stream_columns}["adid"]
        stream_date = {column.lower(): column for column in search_stream_columns}["searchdate"]
        stream_is_click = {column.lower(): column for column in search_stream_columns}["isclick"]

        for split_name, cut_dates in {
            "train": TRAIN_CUT_DATES,
            "val": [VALIDATION_CUT_DATE],
            "test": [TEST_CUT_DATE],
        }.items():
            frames_for_split: list[pd.DataFrame] = []

            for cut_date in cut_dates:
                feature_cut_date = cut_date + datetime.timedelta(days=1)

                ads_node = DuckdbNode(
                    fpath="ads_src",
                    prefix="ad",
                    pk=ad_id,
                    date_key=None,
                    columns=ads_columns,
                )
                category_node = DuckdbNode(
                    fpath="category_src",
                    prefix="cat",
                    pk=category_id,
                    date_key=None,
                    columns=category_columns,
                )
                location_node = DuckdbNode(
                    fpath="location_src",
                    prefix="loc",
                    pk=location_id,
                    date_key=None,
                    columns=location_columns,
                )
                search_stream_node = DuckdbNode(
                    fpath="search_stream_src",
                    prefix="ss",
                    pk=stream_id,
                    date_key=stream_date,
                    columns=search_stream_columns,
                )
                graph = GraphReduce(
                    name=f"rel_avito_ad_ctr_{cut_date.date()}",
                    parent_node=ads_node,
                    compute_layer=ComputeLayerEnum.duckdb,
                    sql_client=con,
                    cut_date=feature_cut_date,
                    compute_period_val=(feature_cut_date - LOOKBACK_START).days + 1,
                    compute_period_unit=PeriodUnit.day,
                    auto_features=True,
                    auto_labels=False,
                    date_filters_on_agg=True,
                    auto_feature_hops_back=2,
                    auto_feature_hops_front=0,
                    use_temp_tables=True,
                )

                for node in [
                    ads_node,
                    category_node,
                    location_node,
                    search_stream_node,
                ]:
                    graph.add_node(node)

                graph.add_entity_edge(ads_node, search_stream_node, parent_key=ad_id, relation_key=stream_ad_id, reduce=True)
                graph.add_entity_edge(ads_node, category_node, parent_key="CategoryID", relation_key=category_id, reduce=False)
                graph.add_entity_edge(ads_node, location_node, parent_key="LocationID", relation_key=location_id, reduce=False)

                graph.do_transformations_sql()
                features = con.sql(f"SELECT * FROM {graph.parent_node._cur_data_ref}").to_df().copy()
                features["timestamp"] = pd.Timestamp(cut_date)

                labels = con.sql(
                    f"""
                    WITH timestamp_df AS (
                        SELECT TIMESTAMP '{cut_date}' AS timestamp
                    )
                    SELECT
                        search_ads.{ad_id} AS AdID,
                        t.timestamp,
                        COALESCE(SUM(search_ads.{stream_is_click}), 0) / COALESCE(COUNT(search_ads.{stream_search_id}), 1) AS num_click
                    FROM
                        timestamp_df t
                    LEFT JOIN (
                        ads_src
                        LEFT JOIN search_stream_src
                            ON ads_src.{ad_id} = search_stream_src.{stream_ad_id}
                    ) search_ads
                        ON search_ads.{stream_date} > t.timestamp
                        AND search_ads.{stream_date} <= t.timestamp + INTERVAL '{LABEL_PERIOD_DAYS} days'
                    GROUP BY
                        t.timestamp,
                        search_ads.{ad_id}
                    HAVING
                        SUM(search_ads.{stream_is_click}) > 0
                    """
                ).to_df()

                labels = labels.dropna(subset=["AdID"]).copy()
                frame = features.merge(
                    labels[["timestamp", "AdID", "num_click"]],
                    left_on=["timestamp", f"ad_{ad_id}"],
                    right_on=["timestamp", "AdID"],
                    how="inner",
                ).drop(columns=["AdID"])
                frame["num_click"] = frame["num_click"].fillna(0).astype("float64")
                frames_for_split.append(frame)

            split_frames[split_name] = pd.concat(frames_for_split, ignore_index=True)
    finally:
        con.close()

    df_train = split_frames["train"]
    df_val = split_frames["val"]
    df_test = split_frames["test"]
    target = "num_click"

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
        iterations=500,
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

    val_true = df_val[target].fillna(0).astype("float64").to_numpy()
    test_true = df_test[target].fillna(0).astype("float64").to_numpy()
    val_pred = np.asarray(val_predictions, dtype="float64")
    test_pred = np.asarray(test_predictions, dtype="float64")

    val_metrics = {
        "r2": float(r2(val_true, val_pred)),
        "mae": float(mae(val_true, val_pred)),
        "rmse": float(rmse(val_true, val_pred)),
    }
    test_metrics = {
        "r2": float(r2(test_true, test_pred)),
        "mae": float(mae(test_true, test_pred)),
        "rmse": float(rmse(test_true, test_pred)),
    }

    return df_train, df_val, df_test, val_metrics, test_metrics, len(feature_columns), materialized, target


def main() -> None:
    df_train, df_val, df_test, val_metrics, test_metrics, n_features, materialized, target = run_rel_avito_ad_ctr()
    print("materialized_files:", materialized, flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
    print("lookback_days:", LOOKBACK_DAYS, flush=True)
    print("train_cut_dates:", [cut_date.date() for cut_date in TRAIN_CUT_DATES], flush=True)
    print("train_timestamps:", df_train["timestamp"].nunique(), flush=True)
    print("validation_cut_date:", VALIDATION_CUT_DATE.date(), flush=True)
    print("test_cut_date:", TEST_CUT_DATE.date(), flush=True)
    print("cut_date:", CUT_DATE.date(), flush=True)
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
