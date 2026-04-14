#!/usr/bin/env python
"""RelBench rel-avito: user clicks example aligned to the official task definition."""

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


def run_rel_avito_user_clicks(
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
        con.sql(f"CREATE OR REPLACE VIEW search_info_src AS SELECT * FROM read_parquet('{use_dir / 'SearchInfo.parquet'}')")
        con.sql(f"CREATE OR REPLACE VIEW user_src AS SELECT * FROM read_parquet('{use_dir / 'UserInfo.parquet'}')")
        con.sql(f"CREATE OR REPLACE VIEW visits_src AS SELECT * FROM read_parquet('{use_dir / 'VisitsStream.parquet'}')")
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
        search_info_columns = con.sql("SELECT * FROM search_info_src LIMIT 0").to_df().columns.tolist()
        search_stream_columns = con.sql("SELECT * FROM search_stream_src LIMIT 0").to_df().columns.tolist()
        user_columns = con.sql("SELECT * FROM user_src LIMIT 0").to_df().columns.tolist()
        visits_columns = con.sql("SELECT * FROM visits_src LIMIT 0").to_df().columns.tolist()

        ad_id = {column.lower(): column for column in ads_columns}["adid"]
        category_id = {column.lower(): column for column in category_columns}["categoryid"]
        location_id = {column.lower(): column for column in location_columns}["locationid"]
        search_id = {column.lower(): column for column in search_info_columns}["searchid"]
        search_user_id = {column.lower(): column for column in search_info_columns}["userid"]
        search_info_date = {column.lower(): column for column in search_info_columns}["searchdate"]
        stream_id = {column.lower(): column for column in search_stream_columns}["search_stream_id"]
        stream_search_id = {column.lower(): column for column in search_stream_columns}["searchid"]
        stream_ad_id = {column.lower(): column for column in search_stream_columns}["adid"]
        stream_date = {column.lower(): column for column in search_stream_columns}["searchdate"]
        stream_is_click = {column.lower(): column for column in search_stream_columns}["isclick"]
        user_id = {column.lower(): column for column in user_columns}["userid"]
        visit_user_id = {column.lower(): column for column in visits_columns}["userid"]
        visit_ad_id = {column.lower(): column for column in visits_columns}["adid"]
        visit_date = {column.lower(): column for column in visits_columns}["viewdate"]

        for split_name, cut_dates in {
            "train": TRAIN_CUT_DATES,
            "val": [VALIDATION_CUT_DATE],
            "test": [TEST_CUT_DATE],
        }.items():
            frames_for_split: list[pd.DataFrame] = []

            for cut_date in cut_dates:
                feature_cut_date = cut_date + datetime.timedelta(days=1)

                user_node = DuckdbNode(
                    fpath="user_src",
                    prefix="usr",
                    pk=user_id,
                    date_key=None,
                    columns=user_columns,
                )
                visits_node = DuckdbNode(
                    fpath="visits_src",
                    prefix="vis",
                    pk=visit_ad_id,
                    date_key=visit_date,
                    columns=visits_columns,
                )
                ads_node = DuckdbNode(
                    fpath="ads_src",
                    prefix="ad",
                    pk=ad_id,
                    date_key=None,
                    columns=ads_columns,
                )
                search_info_node = DuckdbNode(
                    fpath="search_info_src",
                    prefix="si",
                    pk=search_id,
                    date_key=search_info_date,
                    columns=search_info_columns,
                )
                search_stream_node = DuckdbNode(
                    fpath="search_stream_src",
                    prefix="ss",
                    pk=stream_id,
                    date_key=stream_date,
                    columns=search_stream_columns,
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

                graph = GraphReduce(
                    name=f"rel_avito_user_clicks_{cut_date.date()}",
                    parent_node=user_node,
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

                for node in [
                    user_node,
                    visits_node,
                    ads_node,
                    search_info_node,
                    search_stream_node,
                    category_node,
                    location_node,
                ]:
                    graph.add_node(node)

                graph.add_entity_edge(user_node, visits_node, parent_key=user_id, relation_key=visit_user_id, reduce=True)
                graph.add_entity_edge(user_node, search_info_node, parent_key=user_id, relation_key=search_user_id, reduce=True)
                graph.add_entity_edge(visits_node, ads_node, parent_key=visit_ad_id, relation_key=ad_id, reduce=True)
                graph.add_entity_edge(search_info_node, search_stream_node, parent_key=search_id, relation_key=stream_search_id, reduce=True)
                graph.add_entity_edge(search_stream_node, ads_node, parent_key=stream_ad_id, relation_key=ad_id, reduce=True)
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
                        search_ads.{search_user_id} AS UserID,
                        t.timestamp,
                        COALESCE(COUNT(search_ads.{stream_ad_id}), 0) > 1 AS num_click
                    FROM
                        timestamp_df t
                    LEFT JOIN
                    (
                        (
                            user_src
                            LEFT JOIN search_info_src
                                ON user_src.{user_id} = search_info_src.{search_user_id}
                        ) user_search_info
                        LEFT JOIN search_stream_src
                            ON user_search_info.{search_id} = search_stream_src.{stream_search_id}
                            AND search_stream_src.{stream_is_click} = 1.0
                    ) search_ads
                        ON search_ads.{search_info_date} > t.timestamp
                        AND search_ads.{search_info_date} <= t.timestamp + INTERVAL '{LABEL_PERIOD_DAYS} days'
                    GROUP BY
                        t.timestamp,
                        search_ads.{search_user_id}
                    """
                ).to_df()

                labels = labels.dropna(subset=["UserID"]).copy()
                frame = features.merge(
                    labels[["timestamp", "UserID", "num_click"]],
                    left_on=["timestamp", f"usr_{user_id}"],
                    right_on=["timestamp", "UserID"],
                    how="inner",
                ).drop(columns=["UserID"])
                frame["num_click"] = frame["num_click"].astype("int8")
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
        and "userid" not in column.lower()
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
    df_train, df_val, df_test, val_metrics, test_metrics, n_features, materialized, target = run_rel_avito_user_clicks()
    print("materialized_files:", materialized, flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
    print("train_cut_dates:", [cut_date.date() for cut_date in TRAIN_CUT_DATES], flush=True)
    print("train_timestamps:", df_train["timestamp"].nunique(), flush=True)
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
