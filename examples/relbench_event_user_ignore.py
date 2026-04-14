#!/usr/bin/env python
"""RelBench rel-event: user ignore example aligned to the official task definition."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.metrics import accuracy, average_precision, f1, roc_auc

from graphreduce.enum import ComputeLayerEnum, PeriodUnit
from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import DuckdbNode

DATASET_NAME = "rel-event"
REL_CACHE_DIR = Path("tests/data/relbench_cache").resolve()
RAW_DATA_DIR = Path("tests/data/relbench/rel-event")
RELBENCH_EVENT_SOURCE_DIR = Path("data/rel-event")
DATA_DIR = Path("tests/data/relbench/rel-event")
TABLES = ["users", "events", "event_attendees", "event_interest", "user_friends"]
VALIDATION_CUT_DATE = pd.Timestamp("2012-11-21")
TEST_CUT_DATE = pd.Timestamp("2012-11-29")
LABEL_TIMEDELTA = pd.Timedelta(days=7)
TARGET_COLUMN = "target"


def _find_raw_data_dir() -> Path | None:
    required_names = {
        "event-recommendation-engine-challenge.zip",
        "users.csv",
        "events.csv",
        "train.csv",
        "event_attendees.csv",
        "user_friends.csv",
    }
    if RAW_DATA_DIR.exists() and any((RAW_DATA_DIR / name).exists() for name in required_names):
        return RAW_DATA_DIR
    return None


def _prepare_relbench_event_source(raw_data_dir: Path) -> None:
    RELBENCH_EVENT_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    for name in [
        "event-recommendation-engine-challenge.zip",
        "users.csv",
        "events.csv",
        "train.csv",
        "event_attendees.csv",
        "user_friends.csv",
        "user_friends.csv.gz",
        "events.csv.gz",
        "event_attendees.csv.gz",
    ]:
        source = raw_data_dir / name
        target = RELBENCH_EVENT_SOURCE_DIR / name
        if source.exists() and not target.exists():
            try:
                target.symlink_to(source.resolve())
            except OSError:
                shutil.copy2(source, target)


def run_rel_event_user_ignore(
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float] | None, dict[str, float] | None, int, list[str], str]:
    raw_data_dir = _find_raw_data_dir()
    if raw_data_dir is None:
        raise RuntimeError(
            "RelBench rel-event requires the Kaggle Event Recommendation Engine Challenge files. "
            "Place them under 'tests/data/relbench/rel-event'."
        )

    os.environ.setdefault("RELBENCH_CACHE_DIR", str(REL_CACHE_DIR))
    _prepare_relbench_event_source(raw_data_dir)
    use_dir = (data_dir or DATA_DIR).resolve()
    use_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_dataset(DATASET_NAME, download=False)
    db = dataset.get_db(upto_test_timestamp=False)
    official_task = get_task(DATASET_NAME, "user-ignore", download=False)
    materialized: list[str] = []
    for table_name in TABLES:
        out_path = use_dir / f"{table_name}.parquet"
        if not out_path.exists():
            db.table_dict[table_name].df.to_parquet(out_path, index=False)
            materialized.append(out_path.name)

    con = duckdb.connect()
    split_frames: dict[str, pd.DataFrame] = {}

    try:
        con.sql(f"CREATE OR REPLACE VIEW users_src AS SELECT * FROM read_parquet('{use_dir / 'users.parquet'}')")
        con.sql(f"CREATE OR REPLACE VIEW events_src AS SELECT * FROM read_parquet('{use_dir / 'events.parquet'}')")
        con.sql(
            f"""
            CREATE OR REPLACE VIEW event_attendees_src AS
            SELECT row_number() OVER () AS attendee_id, * FROM read_parquet('{use_dir / 'event_attendees.parquet'}')
            """
        )
        con.sql(
            f"""
            CREATE OR REPLACE VIEW event_interest_src AS
            SELECT row_number() OVER () AS interest_id, * FROM read_parquet('{use_dir / 'event_interest.parquet'}')
            """
        )
        con.sql(
            f"""
            CREATE OR REPLACE VIEW user_friends_src AS
            SELECT row_number() OVER () AS friendship_id, * FROM read_parquet('{use_dir / 'user_friends.parquet'}')
            """
        )

        bounds = con.sql(
            """
            SELECT
                MIN(ts) AS min_timestamp
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

        official_tables = {
            "train": official_task.get_table("train", mask_input_cols=False).df.copy(),
            "val": official_task.get_table("val", mask_input_cols=False).df.copy(),
            "test": official_task.get_table("test", mask_input_cols=False).df.copy(),
        }

        split_cut_dates = {
            split_name: sorted(pd.to_datetime(table["timestamp"]).drop_duplicates().tolist())
            for split_name, table in official_tables.items()
        }

        for split_name, cut_dates in split_cut_dates.items():
            split_list: list[pd.DataFrame] = []

            for cut_date in cut_dates:
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
                    name=f"rel_event_user_ignore_{pd.Timestamp(cut_date).date()}",
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

                for node in [users_node, events_node, attendees_node, interest_node, friends_node]:
                    graph.add_node(node)

                graph.add_entity_edge(users_node, attendees_node, user_id_col, attendee_user_col, reduce=True)
                graph.add_entity_edge(users_node, interest_node, user_id_col, interest_user_col, reduce=True)
                graph.add_entity_edge(users_node, friends_node, user_id_col, friend_user_col, reduce=True)
                graph.add_entity_edge(users_node, events_node, user_id_col, event_user_col, reduce=True)
                graph.add_entity_edge(attendees_node, events_node, attendee_event_col, event_id_col, reduce=False)
                graph.add_entity_edge(interest_node, events_node, interest_event_col, event_id_col, reduce=False)

                graph.do_transformations_sql()
                features = con.sql(f"SELECT * FROM {graph.parent_node._cur_data_ref}").to_df().copy()
                features["timestamp"] = pd.Timestamp(cut_date)

                labels = official_tables[split_name].copy()
                labels["timestamp"] = pd.to_datetime(labels["timestamp"])
                labels = labels[labels["timestamp"] == pd.Timestamp(cut_date)].copy()
                labels["user"] = labels["user"].astype("int64")

                frame = features.merge(
                    labels,
                    left_on=["timestamp", f"usr_{user_id_col}"],
                    right_on=["timestamp", "user"],
                    how="inner",
                ).drop(columns=["user"])
                frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype("int8")
                split_list.append(frame)

            split_frames[split_name] = pd.concat(split_list, ignore_index=True)
    finally:
        con.close()

    df_train = split_frames["train"]
    df_val = split_frames["val"]
    df_test = split_frames["test"]
    common_columns = set(df_train.columns) & set(df_val.columns) & set(df_test.columns)
    feature_columns = [
        column
        for column in df_train.select_dtypes(include=[np.number, "bool"]).columns
        if column != TARGET_COLUMN and "label" not in column.lower() and "user_id" not in column.lower() and column in common_columns
    ]
    if not feature_columns or df_train[TARGET_COLUMN].nunique() < 2:
        return df_train, df_val, df_test, None, None, len(feature_columns), materialized, TARGET_COLUMN

    model = CatBoostClassifier(
        iterations=5000,
        depth=8,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=50,
        allow_writing_files=False,
    )
    model.fit(df_train[feature_columns].fillna(0), df_train[TARGET_COLUMN])

    val_predictions = np.asarray(model.predict_proba(df_val[feature_columns].fillna(0))[:, 1], dtype="float64")
    test_predictions = np.asarray(model.predict_proba(df_test[feature_columns].fillna(0))[:, 1], dtype="float64")
    val_target = df_val[TARGET_COLUMN].to_numpy(dtype="float64")
    test_target = df_test[TARGET_COLUMN].to_numpy(dtype="float64")
    val_metrics = {
        "accuracy": float(accuracy(val_target, val_predictions)),
        "average_precision": float(average_precision(val_target, val_predictions)),
        "f1": float(f1(val_target, val_predictions)),
        "roc_auc": float(roc_auc(val_target, val_predictions)),
    }
    test_metrics = {
        "accuracy": float(accuracy(test_target, test_predictions)),
        "average_precision": float(average_precision(test_target, test_predictions)),
        "f1": float(f1(test_target, test_predictions)),
        "roc_auc": float(roc_auc(test_target, test_predictions)),
    }
    return df_train, df_val, df_test, val_metrics, test_metrics, len(feature_columns), materialized, TARGET_COLUMN


def main() -> None:
    df_train, df_val, df_test, val_metrics, test_metrics, n_features, materialized, target = run_rel_event_user_ignore()
    print("materialized_files:", materialized, flush=True)
    print("validation_timestamp:", VALIDATION_CUT_DATE.date(), flush=True)
    print("test_timestamp:", TEST_CUT_DATE.date(), flush=True)
    print("target:", target, flush=True)
    print("train_rows:", len(df_train), flush=True)
    print("validation_rows:", len(df_val), flush=True)
    print("test_rows:", len(df_test), flush=True)
    print("feature_count:", n_features, flush=True)
    print("validation_metrics:", val_metrics if val_metrics is not None else "skipped", flush=True)
    print("test_metrics:", test_metrics if test_metrics is not None else "skipped", flush=True)


if __name__ == "__main__":
    main()
