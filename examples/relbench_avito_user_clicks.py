#!/usr/bin/env python
"""RelBench rel-avito: user clicks example aligned to the official task definition."""

from __future__ import annotations

import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

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


def run_rel_avito_user_clicks(
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float] | None, dict[str, float] | None, int, list[str], str]:
    _, db = get_relbench_dataset_db("rel-avito", download=True, upto_test_timestamp=False)
    materialized: list[str] = []

    con = duckdb.connect()
    split_frames: dict[str, pd.DataFrame] = {}

    try:
        register_relbench_db_views(
            con,
            db,
            {
                "AdsInfo": "ads_src",
                "Category": "category_src",
                "Location": "location_src",
                "SearchInfo": "search_info_src",
                "SearchStream": "search_stream_src",
                "UserInfo": "user_src",
                "VisitStream": "visits_src",
            },
            {"SearchStream": "search_stream_id"},
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

        split_tasks = {}
        for split_name in ("train", "val", "test"):
            task, task_table, cut_timestamps = get_relbench_split_task_table(
                "rel-avito", "user-clicks", split_name, download=True, db=db
            )
            split_tasks[split_name] = task
            cut_dates = [timestamp.to_pydatetime() for timestamp in cut_timestamps]
            frame_store = RelBenchFrameStore(f"rel-avito-user-clicks-{split_name}")

            def build_frame(frame_con, cut_date):
                con = frame_con
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
                graph._clean_refs()
                features["timestamp"] = pd.Timestamp(cut_date)

                labels = task_table.df.copy()
                labels[task.time_col] = pd.to_datetime(labels[task.time_col])
                labels = labels[labels[task.time_col] == pd.Timestamp(cut_date)].copy()
                labels = labels.dropna(subset=[task.entity_col]).copy()
                frame = features.merge(
                    labels[[task.time_col, task.entity_col, task.target_col]],
                    left_on=["timestamp", f"usr_{user_id}"],
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
    df_train, df_val, df_test, val_metrics, test_metrics, n_features, materialized, target = run_rel_avito_user_clicks()
    print("materialized_files:", materialized, flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
    print("train_cut_dates:", sorted(df_train["timestamp"].drop_duplicates().dt.date.astype(str).tolist()), flush=True)
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
