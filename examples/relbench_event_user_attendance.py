#!/usr/bin/env python
"""RelBench rel-event: user attendance example aligned to the official task definition."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from relbench_dataset_utils import (
    RelBenchFrameStore,
    get_relbench_dataset_db,
    get_relbench_task,
    get_relbench_split_task_table,
    iter_training_frames,
    register_relbench_db_views,
    target_table_from_frame,
)
from relbench_regression_metrics import add_nmae
from relbench_catboost_utils import TEMPORAL_FEATURE_FAMILIES, fit_tuned_regressor_incremental, set_feature_families

from graphreduce.enum import ComputeLayerEnum, PeriodUnit
from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import DuckdbNode

DATASET_NAME = "rel-event"
TABLES = ["users", "events", "event_attendees", "event_interest", "user_friends"]
TABLE_TO_VIEW = {
    "users": "users_src",
    "events": "events_src",
    "event_attendees": "event_attendees_src",
    "event_interest": "event_interest_src",
    "user_friends": "user_friends_src",
}
ROW_NUMBER_IDS = {
    "event_attendees": "attendee_id",
    "event_interest": "interest_id",
    "user_friends": "friendship_id",
}
DROP_COLUMNS = {
    "event_attendees": ["Unnamed: 0"],
    "user_friends": ["Unnamed: 0"],
}
VALIDATION_CUT_DATE = pd.Timestamp("2012-11-21")
TEST_CUT_DATE = pd.Timestamp("2012-11-29")
LABEL_TIMEDELTA = pd.Timedelta(days=7)
TARGET_COLUMN = "target"


def _task_table_for_split(split_name: str):
    return get_relbench_split_task_table(
        DATASET_NAME, "user-attendance", split_name, download=True
    )


def run_rel_event_user_attendance(
    data_dir: Path | None = None,
) -> tuple[RelBenchFrameStore, pd.DataFrame, pd.DataFrame, dict[str, float] | None, dict[str, float] | None, int, list[str], str]:
    _, db = get_relbench_dataset_db(
        DATASET_NAME, download=True, upto_test_timestamp=False
    )
    materialized: list[str] = []

    con = duckdb.connect()
    split_frames: dict[str, RelBenchFrameStore] = {}

    try:
        register_relbench_db_views(con, db, TABLE_TO_VIEW, ROW_NUMBER_IDS, DROP_COLUMNS)

        bounds = con.sql(
            """
            SELECT
                MIN(ts) AS min_timestamp,
                MAX(ts) AS max_timestamp
            FROM (
                SELECT joinedAt AS ts FROM users_src WHERE joinedAt IS NOT NULL
                UNION ALL
                SELECT start_time AS ts FROM event_attendees_src WHERE start_time IS NOT NULL
                UNION ALL
                SELECT timestamp AS ts FROM event_interest_src WHERE timestamp IS NOT NULL
            )
            """
        ).to_df()
        lookback_start = pd.Timestamp(bounds.loc[0, "min_timestamp"])

        user_columns = con.sql("SELECT * FROM users_src LIMIT 0").to_df().columns.tolist()
        event_columns = con.sql("SELECT * FROM events_src LIMIT 0").to_df().columns.tolist()
        attendee_columns = con.sql("SELECT * FROM event_attendees_src LIMIT 0").to_df().columns.tolist()
        interest_columns = con.sql("SELECT * FROM event_interest_src LIMIT 0").to_df().columns.tolist()
        friend_columns = con.sql("SELECT * FROM user_friends_src LIMIT 0").to_df().columns.tolist()

        user_id_col = {column.lower(): column for column in user_columns}["user_id"]
        user_date_col = {column.lower(): column for column in user_columns}["joinedat"]
        event_id_col = {column.lower(): column for column in event_columns}["event_id"]
        event_user_col = {column.lower(): column for column in event_columns}["user_id"]
        event_date_col = {column.lower(): column for column in event_columns}["start_time"]
        attendee_id_col = {column.lower(): column for column in attendee_columns}["attendee_id"]
        attendee_event_col = {column.lower(): column for column in attendee_columns}["event"]
        attendee_user_col = {column.lower(): column for column in attendee_columns}["user_id"]
        attendee_status_col = {column.lower(): column for column in attendee_columns}["status"]
        attendee_date_col = {column.lower(): column for column in attendee_columns}["start_time"]
        interest_id_col = {column.lower(): column for column in interest_columns}["interest_id"]
        interest_event_col = {column.lower(): column for column in interest_columns}["event"]
        interest_user_col = {column.lower(): column for column in interest_columns}["user"]
        interest_date_col = {column.lower(): column for column in interest_columns}["timestamp"]
        friend_id_col = {column.lower(): column for column in friend_columns}["friendship_id"]
        friend_user_col = {column.lower(): column for column in friend_columns}["user"]

        split_tasks = {}
        for split_name in ["train", "val", "test"]:
            task, task_table, cut_timestamps = _task_table_for_split(split_name)
            split_tasks[split_name] = task
            cut_dates = [timestamp.to_pydatetime() for timestamp in cut_timestamps]
            frame_store = RelBenchFrameStore(
                f"rel-event-user-attendance-{split_name}", persist_each_frame=True
            )

            def build_frame(frame_con, cut_date):
                con = frame_con
                feature_cut_date = pd.Timestamp(cut_date) + pd.Timedelta(seconds=1)

                users_node = DuckdbNode(
                    fpath="users_src",
                    prefix="usr",
                    pk=user_id_col,
                    date_key=user_date_col,
                    columns=user_columns,
                )
                events_node = DuckdbNode(
                    fpath="events_src",
                    prefix="evt",
                    pk=event_id_col,
                    date_key=event_date_col,
                    columns=event_columns,
                )
                attendees_node = DuckdbNode(
                    fpath="event_attendees_src",
                    prefix="att",
                    pk=attendee_id_col,
                    date_key=attendee_date_col,
                    columns=attendee_columns,
                    feature_family_max_columns=4,
                    categorical_top_k=5,
                    context_keys=(attendee_event_col,),
                    annotation_expressions={
                        "is_attending": f"{{{attendee_status_col}}} IN ('yes', 'maybe')",
                        "is_declined": f"{{{attendee_status_col}}} = 'no'",
                    },
                )
                attendee_users_node = DuckdbNode(
                    fpath="users_src",
                    prefix="attusr",
                    pk=user_id_col,
                    date_key=user_date_col,
                    columns=user_columns,
                )
                interest_node = DuckdbNode(
                    fpath="event_interest_src",
                    prefix="int",
                    pk=interest_id_col,
                    date_key=interest_date_col,
                    columns=interest_columns,
                )
                friends_node = DuckdbNode(
                    fpath="user_friends_src",
                    prefix="frd",
                    pk=friend_id_col,
                    date_key=None,
                    columns=friend_columns,
                )

                graph = GraphReduce(
                    name=f"rel_event_user_attendance_{pd.Timestamp(cut_date).date()}",
                    parent_node=users_node,
                    compute_layer=ComputeLayerEnum.duckdb,
                    sql_client=con,
                    cut_date=feature_cut_date.to_pydatetime(),
                    compute_period_val=max(1, int((feature_cut_date - lookback_start).days + 1)),
                    compute_period_unit=PeriodUnit.day,
                    auto_features=True,
                    auto_labels=False,
                    date_filters_on_agg=True,
                    auto_feature_hops_back=3,
                    auto_feature_hops_front=0,
                    use_temp_tables=True,
                )

                nodes = [
                    users_node,
                    events_node,
                    attendees_node,
                    attendee_users_node,
                    interest_node,
                    friends_node,
                ]
                set_feature_families([attendees_node], TEMPORAL_FEATURE_FAMILIES)
                for node in nodes:
                    graph.add_node(node)

                graph.add_entity_edge(users_node, attendees_node, parent_key=user_id_col, relation_key=attendee_user_col, reduce=True)
                graph.add_entity_edge(users_node, interest_node, parent_key=user_id_col, relation_key=interest_user_col, reduce=True)
                graph.add_entity_edge(users_node, friends_node, parent_key=user_id_col, relation_key=friend_user_col, reduce=True)
                graph.add_entity_edge(users_node, events_node, parent_key=user_id_col, relation_key=event_user_col, reduce=True)
                graph.add_entity_edge(attendees_node, events_node, parent_key=attendee_event_col, relation_key=event_id_col, reduce=False)
                graph.add_entity_edge(attendees_node, attendee_users_node, parent_key=attendee_user_col, relation_key=user_id_col, reduce=False)
                graph.add_entity_edge(interest_node, events_node, parent_key=interest_event_col, relation_key=event_id_col, reduce=False)

                graph.do_transformations_sql()
                features = con.sql(f"SELECT * FROM {graph.parent_node._cur_data_ref}").to_df().copy()
                graph._clean_refs()
                features["timestamp"] = pd.Timestamp(cut_date)

                labels = task_table.df.copy()
                labels = labels[
                    pd.to_datetime(labels[task.time_col]) == pd.Timestamp(cut_date)
                ]

                frame = features.merge(
                    labels[[task.time_col, task.entity_col, task.target_col]],
                    left_on=["timestamp", f"usr_{user_id_col}"],
                    right_on=[task.time_col, task.entity_col],
                    how="right",
                    validate="one_to_one",
                )
                frame[task.target_col] = frame[task.target_col].astype("float64")
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

    train_sample = train_store.sample_frame()
    common_columns = set(train_sample.columns) & set(df_val.columns) & set(df_test.columns)
    feature_columns = [
        column
        for column in train_sample.select_dtypes(include=[np.number, "bool"]).columns
        if column != TARGET_COLUMN
        and "label" not in column.lower()
        and "user_id" not in column.lower()
        and column in common_columns
    ]
    if not feature_columns:
        return train_store, df_val, df_test, None, None, 0, materialized, TARGET_COLUMN

    model, best_config, best_val_mae = fit_tuned_regressor_incremental(
        lambda: train_store.iter_batches(),
        feature_columns,
        TARGET_COLUMN,
        df_val[feature_columns].fillna(0),
        df_val[TARGET_COLUMN].astype("float64"),
        batch_count=len(train_store.part_paths),
    )
    print("catboost_config:", best_config, flush=True)
    print("catboost_validation_mae:", best_val_mae, flush=True)
    print("catboost_best_iteration:", model.get_best_iteration(), flush=True)

    val_predictions = np.asarray(model.predict(df_val[feature_columns].fillna(0)), dtype="float64")
    test_predictions = np.asarray(model.predict(df_test[feature_columns].fillna(0)), dtype="float64")
    val_metrics = add_nmae(
        split_tasks["val"].evaluate(
            val_predictions,
            target_table=target_table_from_frame(split_tasks["val"], df_val),
        ),
        df_val[TARGET_COLUMN],
        val_predictions,
        train_store.target_std(TARGET_COLUMN),
    )
    test_metrics = add_nmae(
        split_tasks["test"].evaluate(
            test_predictions,
            target_table=target_table_from_frame(split_tasks["test"], df_test),
        ),
        df_test[TARGET_COLUMN],
        test_predictions,
        train_store.target_std(TARGET_COLUMN),
    )
    return train_store, df_val, df_test, val_metrics, test_metrics, len(feature_columns), materialized, TARGET_COLUMN


def main() -> None:
    df_train, df_val, df_test, val_metrics, test_metrics, n_features, materialized, target = run_rel_event_user_attendance()
    print("materialized_files:", materialized, flush=True)
    print("validation_timestamp:", VALIDATION_CUT_DATE.date(), flush=True)
    print("test_timestamp:", TEST_CUT_DATE.date(), flush=True)
    print("target:", target, flush=True)
    print("train_rows:", df_train.row_count, flush=True)
    print("validation_rows:", len(df_val), flush=True)
    print("test_rows:", len(df_test), flush=True)
    print("feature_count:", n_features, flush=True)
    print("validation_nmae:", val_metrics["nmae"] if val_metrics is not None else "skipped", flush=True)
    print("test_nmae:", test_metrics["nmae"] if test_metrics is not None else "skipped", flush=True)
    print("validation_metrics:", val_metrics if val_metrics is not None else "skipped", flush=True)
    print("test_metrics:", test_metrics if test_metrics is not None else "skipped", flush=True)
    df_train.close()


if __name__ == "__main__":
    main()
