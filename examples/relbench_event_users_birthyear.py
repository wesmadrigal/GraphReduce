#!/usr/bin/env python
"""RelBench rel-event: users birthyear autocomplete example."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from relbench.datasets import get_dataset
from relbench.metrics import mae, r2, rmse

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
TARGET_COLUMN = "usr_birthyear"


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


def run_rel_event_users_birthyear(
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
    materialized: list[str] = []
    for table_name in TABLES:
        out_path = use_dir / f"{table_name}.parquet"
        if not out_path.exists():
            db.table_dict[table_name].df.to_parquet(out_path, index=False)
            materialized.append(out_path.name)

    con = duckdb.connect()

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
                MIN(joinedAt) AS min_timestamp,
                MAX(joinedAt) AS max_timestamp
            FROM users_src
            WHERE joinedAt IS NOT NULL
            """
        ).to_df()
        lookback_start = pd.Timestamp(bounds.loc[0, "min_timestamp"])
        dataset_max_timestamp = pd.Timestamp(bounds.loc[0, "max_timestamp"])

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
        attendee_date_col = {column.lower(): column for column in attendee_columns}["start_time"]
        interest_id_col = {column.lower(): column for column in interest_columns}["interest_id"]
        interest_event_col = {column.lower(): column for column in interest_columns}["event"]
        interest_user_col = {column.lower(): column for column in interest_columns}["user"]
        interest_date_col = {column.lower(): column for column in interest_columns}["timestamp"]
        friend_id_col = {column.lower(): column for column in friend_columns}["friendship_id"]
        friend_user_col = {column.lower(): column for column in friend_columns}["user"]

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
            name="rel_event_users_birthyear",
            parent_node=users_node,
            compute_layer=ComputeLayerEnum.duckdb,
            sql_client=con,
            cut_date=(dataset_max_timestamp + pd.Timedelta(seconds=1)).to_pydatetime(),
            compute_period_val=max(1, int((dataset_max_timestamp - lookback_start).days + 1)),
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
        all_rows = con.sql(f"SELECT * FROM {graph.parent_node._cur_data_ref}").to_df().copy()
    finally:
        con.close()

    train_end = VALIDATION_CUT_DATE - pd.Timedelta(seconds=1)
    val_end = TEST_CUT_DATE - pd.Timedelta(seconds=1)
    all_rows = all_rows.dropna(subset=[TARGET_COLUMN]).copy()
    df_train = all_rows[(all_rows["usr_joinedAt"] > lookback_start) & (all_rows["usr_joinedAt"] <= train_end)].copy()
    df_val = all_rows[(all_rows["usr_joinedAt"] > VALIDATION_CUT_DATE) & (all_rows["usr_joinedAt"] <= val_end)].copy()
    df_test = all_rows[all_rows["usr_joinedAt"] > TEST_CUT_DATE].copy()

    common_columns = set(df_train.columns) & set(df_val.columns) & set(df_test.columns)
    feature_columns = [
        column
        for column in df_train.select_dtypes(include=[np.number, "bool"]).columns
        if column != TARGET_COLUMN
        and "label" not in column.lower()
        and "user_id" not in column.lower()
        and column in common_columns
    ]
    if not feature_columns:
        return df_train, df_val, df_test, None, None, 0, materialized, TARGET_COLUMN

    model = CatBoostRegressor(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        loss_function="MAE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=50,
        allow_writing_files=False,
    )
    model.fit(df_train[feature_columns].fillna(0), df_train[TARGET_COLUMN].astype("float64"))

    val_predictions = np.asarray(model.predict(df_val[feature_columns].fillna(0)), dtype="float64")
    test_predictions = np.asarray(model.predict(df_test[feature_columns].fillna(0)), dtype="float64")
    val_target = df_val[TARGET_COLUMN].to_numpy(dtype="float64")
    test_target = df_test[TARGET_COLUMN].to_numpy(dtype="float64")
    val_metrics = {
        "r2": float(r2(val_target, val_predictions)),
        "mae": float(mae(val_target, val_predictions)),
        "rmse": float(rmse(val_target, val_predictions)),
    }
    test_metrics = {
        "r2": float(r2(test_target, test_predictions)),
        "mae": float(mae(test_target, test_predictions)),
        "rmse": float(rmse(test_target, test_predictions)),
    }
    return df_train, df_val, df_test, val_metrics, test_metrics, len(feature_columns), materialized, TARGET_COLUMN


def main() -> None:
    df_train, df_val, df_test, val_metrics, test_metrics, n_features, materialized, target = run_rel_event_users_birthyear()
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
