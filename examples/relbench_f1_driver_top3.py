#!/usr/bin/env python
"""RelBench rel-f1: driver top-3 qualifying example aligned to the official task definition."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from relbench_dataset_utils import (
    RelBenchFrameStore,
    get_relbench_dataset_db,
    get_relbench_split_task_table,
    get_relbench_task,
    iter_training_frames,
    register_relbench_db_views,
    target_table_from_frame,
)
from relbench_catboost_utils import TEMPORAL_FEATURE_FAMILIES, fit_tuned_classifier_incremental, set_feature_families

from graphreduce.enum import ComputeLayerEnum, PeriodUnit
from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import DuckdbNode

DATASET_NAME = "rel-f1"
TABLES = [
    "circuits",
    "constructors",
    "constructor_results",
    "constructor_standings",
    "drivers",
    "qualifying",
    "races",
    "results",
    "standings",
]

VALIDATION_CUT_DATE = pd.Timestamp("2005-01-01")
TEST_CUT_DATE = pd.Timestamp("2010-01-01")
LABEL_TIMEDELTA = pd.Timedelta(days=30)
NUM_EVAL_TIMESTAMPS = 40
TARGET_COLUMN = "qualifying"
FRAME_STRIDE = 10


def run_rel_f1_driver_top3(
    data_dir: Path | None = None,
) -> tuple[RelBenchFrameStore, pd.DataFrame, pd.DataFrame, dict[str, float] | None, dict[str, float] | None, int, list[str], str]:
    _, db = get_relbench_dataset_db(DATASET_NAME, download=True, upto_test_timestamp=False)
    official_task = get_relbench_task(DATASET_NAME, "driver-top3", download=True)
    materialized: list[str] = []

    con = duckdb.connect()
    split_frames: dict[str, RelBenchFrameStore] = {}

    try:
        register_relbench_db_views(
            con,
            db,
            {table_name: f"{table_name}_src" for table_name in TABLES},
        )

        dataset_bounds = con.sql(
            """
            SELECT
                MIN(date) AS min_timestamp,
                MAX(date) AS max_timestamp
            FROM (
                SELECT date FROM races_src
                UNION ALL
                SELECT date FROM results_src
                UNION ALL
                SELECT date FROM standings_src
                UNION ALL
                SELECT date FROM constructor_results_src
                UNION ALL
                SELECT date FROM constructor_standings_src
                UNION ALL
                SELECT date FROM qualifying_src
            )
            """
        ).to_df()
        lookback_start = pd.Timestamp(dataset_bounds.loc[0, "min_timestamp"])
        split_tasks = {}
        split_specs = {}
        for split_name in ("train", "val", "test"):
            task, task_table, cut_timestamps = get_relbench_split_task_table(
                DATASET_NAME,
                "driver-top3",
                split_name,
                download=True,
                task=official_task,
                db=db,
            )
            split_tasks[split_name] = task
            selected_cut_timestamps = (
                cut_timestamps[::FRAME_STRIDE]
                if split_name == "train"
                else cut_timestamps
            )
            split_specs[split_name] = (
                task,
                task_table,
                [
                    timestamp.to_pydatetime()
                    for timestamp in selected_cut_timestamps
                ],
            )

        driver_columns = con.sql("SELECT * FROM drivers_src LIMIT 0").to_df().columns.tolist()
        result_columns = con.sql("SELECT * FROM results_src LIMIT 0").to_df().columns.tolist()
        standing_columns = con.sql("SELECT * FROM standings_src LIMIT 0").to_df().columns.tolist()
        qualifying_columns = con.sql("SELECT * FROM qualifying_src LIMIT 0").to_df().columns.tolist()
        race_columns = con.sql("SELECT * FROM races_src LIMIT 0").to_df().columns.tolist()
        circuit_columns = con.sql("SELECT * FROM circuits_src LIMIT 0").to_df().columns.tolist()
        constructor_columns = con.sql("SELECT * FROM constructors_src LIMIT 0").to_df().columns.tolist()

        driver_id_col = {column.lower(): column for column in driver_columns}["driverid"]
        result_id_col = {column.lower(): column for column in result_columns}["resultid"]
        result_driver_col = {column.lower(): column for column in result_columns}["driverid"]
        result_race_col = {column.lower(): column for column in result_columns}["raceid"]
        result_constructor_col = {column.lower(): column for column in result_columns}["constructorid"]
        result_date_col = {column.lower(): column for column in result_columns}["date"]
        standing_id_col = {column.lower(): column for column in standing_columns}["driverstandingsid"]
        standing_driver_col = {column.lower(): column for column in standing_columns}["driverid"]
        standing_date_col = {column.lower(): column for column in standing_columns}["date"]
        qualifying_id_col = {column.lower(): column for column in qualifying_columns}["qualifyid"]
        qualifying_driver_col = {column.lower(): column for column in qualifying_columns}["driverid"]
        qualifying_race_col = {column.lower(): column for column in qualifying_columns}["raceid"]
        qualifying_constructor_col = {column.lower(): column for column in qualifying_columns}["constructorid"]
        qualifying_position_col = {column.lower(): column for column in qualifying_columns}["position"]
        qualifying_date_col = {column.lower(): column for column in qualifying_columns}["date"]
        race_id_col = {column.lower(): column for column in race_columns}["raceid"]
        race_circuit_col = {column.lower(): column for column in race_columns}["circuitid"]
        race_date_col = {column.lower(): column for column in race_columns}["date"]
        circuit_id_col = {column.lower(): column for column in circuit_columns}["circuitid"]
        constructor_id_col = {column.lower(): column for column in constructor_columns}["constructorid"]

        for split_name, (task, task_table, cut_dates) in split_specs.items():
            frame_store = RelBenchFrameStore(
                f"rel-f1-driver-top3-{split_name}", persist_each_frame=True
            )

            def build_frame(frame_con, cut_date):
                con = frame_con
                feature_cut_date = pd.Timestamp(cut_date) + pd.Timedelta(seconds=1)

                driver_node = DuckdbNode(
                    fpath="drivers_src",
                    prefix="drv",
                    pk=driver_id_col,
                    date_key=None,
                    columns=driver_columns,
                )
                qualifying_node = DuckdbNode(
                    fpath="qualifying_src",
                    prefix="qua",
                    pk=qualifying_id_col,
                    date_key=qualifying_date_col,
                    columns=qualifying_columns,
                    feature_family_max_columns=4,
                    categorical_top_k=5,
                    context_keys=(qualifying_race_col, qualifying_constructor_col),
                    annotation_expressions={
                        "is_top3": f"{{{qualifying_position_col}}} <= 3",
                    },
                )
                result_node = DuckdbNode(
                    fpath="results_src",
                    prefix="res",
                    pk=result_id_col,
                    date_key=result_date_col,
                    columns=result_columns,
                )
                standing_node = DuckdbNode(
                    fpath="standings_src",
                    prefix="std",
                    pk=standing_id_col,
                    date_key=standing_date_col,
                    columns=standing_columns,
                )
                race_node = DuckdbNode(
                    fpath="races_src",
                    prefix="rac",
                    pk=race_id_col,
                    date_key=race_date_col,
                    columns=race_columns,
                )
                circuit_node = DuckdbNode(
                    fpath="circuits_src",
                    prefix="cir",
                    pk=circuit_id_col,
                    date_key=None,
                    columns=circuit_columns,
                )
                constructor_node = DuckdbNode(
                    fpath="constructors_src",
                    prefix="con",
                    pk=constructor_id_col,
                    date_key=None,
                    columns=constructor_columns,
                )

                graph = GraphReduce(
                    name=f"rel_f1_driver_top3_{pd.Timestamp(cut_date).date()}",
                    parent_node=driver_node,
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

                nodes = [driver_node, qualifying_node, result_node, standing_node, race_node, circuit_node, constructor_node]
                set_feature_families([qualifying_node], TEMPORAL_FEATURE_FAMILIES)
                for node in nodes:
                    graph.add_node(node)

                graph.add_entity_edge(driver_node, qualifying_node, parent_key=driver_id_col, relation_key=qualifying_driver_col, reduce=True)
                graph.add_entity_edge(driver_node, result_node, parent_key=driver_id_col, relation_key=result_driver_col, reduce=True)
                graph.add_entity_edge(driver_node, standing_node, parent_key=driver_id_col, relation_key=standing_driver_col, reduce=True)
                graph.add_entity_edge(qualifying_node, race_node, parent_key=qualifying_race_col, relation_key=race_id_col, reduce=False)
                graph.add_entity_edge(race_node, circuit_node, parent_key=race_circuit_col, relation_key=circuit_id_col, reduce=False)
                graph.add_entity_edge(qualifying_node, constructor_node, parent_key=qualifying_constructor_col, relation_key=constructor_id_col, reduce=False)
                graph.add_entity_edge(result_node, race_node, parent_key=result_race_col, relation_key=race_id_col, reduce=False)
                graph.add_entity_edge(result_node, constructor_node, parent_key=result_constructor_col, relation_key=constructor_id_col, reduce=False)

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
                    left_on=["timestamp", f"drv_{driver_id_col}"],
                    right_on=[task.time_col, task.entity_col],
                    how="right",
                    validate="one_to_one",
                )
                frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype("int8")
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
        and not column.lower().endswith("_id")
        and "driverid" not in column.lower()
        and column in common_columns
    ]
    if not feature_columns or train_store.target_nunique(TARGET_COLUMN) < 2:
        return train_store, df_val, df_test, None, None, len(feature_columns), materialized, TARGET_COLUMN

    model, best_config, best_val_auc = fit_tuned_classifier_incremental(
        lambda: train_store.iter_batches(),
        feature_columns,
        TARGET_COLUMN,
        df_val[feature_columns].fillna(0),
        df_val[TARGET_COLUMN],
        batch_count=len(train_store.part_paths),
    )
    print("catboost_config:", best_config, flush=True)
    print("catboost_validation_auc:", best_val_auc, flush=True)
    print("catboost_best_iteration:", model.get_best_iteration(), flush=True)

    val_predictions = np.asarray(model.predict_proba(df_val[feature_columns].fillna(0))[:, 1], dtype="float64")
    test_predictions = np.asarray(model.predict_proba(df_test[feature_columns].fillna(0))[:, 1], dtype="float64")

    val_metrics = split_tasks["val"].evaluate(
        val_predictions,
        target_table=target_table_from_frame(split_tasks["val"], df_val),
    )
    test_metrics = split_tasks["test"].evaluate(
        test_predictions,
        target_table=target_table_from_frame(split_tasks["test"], df_test),
    )

    return train_store, df_val, df_test, val_metrics, test_metrics, len(feature_columns), materialized, TARGET_COLUMN


def main() -> None:
    df_train, df_val, df_test, val_metrics, test_metrics, n_features, materialized, target = run_rel_f1_driver_top3()
    print("materialized_files:", materialized, flush=True)
    print("validation_timestamp:", VALIDATION_CUT_DATE.date(), flush=True)
    print("test_timestamp:", TEST_CUT_DATE.date(), flush=True)
    print("label_timedelta_days:", int(LABEL_TIMEDELTA / pd.Timedelta(days=1)), flush=True)
    print("num_eval_timestamps:", NUM_EVAL_TIMESTAMPS, flush=True)
    print("training_frame_stride:", FRAME_STRIDE, flush=True)
    print("target:", target, flush=True)
    print("train_rows:", df_train.row_count, flush=True)
    print("validation_rows:", len(df_val), flush=True)
    print("test_rows:", len(df_test), flush=True)
    print("train_timestamps:", df_train.column_nunique("timestamp"), flush=True)
    print("validation_timestamps:", df_val["timestamp"].nunique(), flush=True)
    print("test_timestamps:", df_test["timestamp"].nunique(), flush=True)
    print("columns:", len(df_train.columns), flush=True)
    print("feature_count:", n_features, flush=True)
    print("validation_metrics:", val_metrics if val_metrics is not None else "skipped", flush=True)
    print("test_metrics:", test_metrics if test_metrics is not None else "skipped", flush=True)
    df_train.close()


if __name__ == "__main__":
    main()
