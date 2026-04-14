#!/usr/bin/env python
"""RelBench rel-f1: driver DNF example aligned to the official task definition."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from relbench.datasets import get_dataset
from relbench.metrics import accuracy, average_precision, f1, roc_auc

from graphreduce.enum import ComputeLayerEnum, PeriodUnit
from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import DuckdbNode

DATASET_NAME = "rel-f1"
REL_CACHE_DIR = Path("tests/data/relbench_cache").resolve()
DATA_DIR = Path("tests/data/relbench/rel-f1")
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
TARGET_COLUMN = "did_not_finish"


def run_rel_f1_driver_dnf(
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float] | None, dict[str, float] | None, int, list[str], str]:
    os.environ.setdefault("RELBENCH_CACHE_DIR", str(REL_CACHE_DIR))
    use_dir = (data_dir or DATA_DIR).resolve()
    use_dir.mkdir(parents=True, exist_ok=True)

    materialized: list[str] = []
    if not all((use_dir / f"{table_name}.parquet").exists() for table_name in TABLES):
        dataset = get_dataset(DATASET_NAME, download=True)
        db = dataset.get_db(upto_test_timestamp=False)
        for table_name in TABLES:
            out_path = use_dir / f"{table_name}.parquet"
            db.table_dict[table_name].df.to_parquet(out_path, index=False)
            materialized.append(out_path.name)

    con = duckdb.connect()
    split_frames: dict[str, pd.DataFrame] = {}

    try:
        for table_name in TABLES:
            con.sql(
                f"CREATE OR REPLACE VIEW {table_name}_src AS "
                f"SELECT * FROM read_parquet('{use_dir / f'{table_name}.parquet'}')"
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
        dataset_max_timestamp = pd.Timestamp(dataset_bounds.loc[0, "max_timestamp"])

        train_cut_dates = pd.date_range(
            start=VALIDATION_CUT_DATE - LABEL_TIMEDELTA,
            end=lookback_start,
            freq=-LABEL_TIMEDELTA,
        ).to_pydatetime().tolist()
        val_cut_dates = pd.date_range(
            start=VALIDATION_CUT_DATE,
            end=min(
                VALIDATION_CUT_DATE + LABEL_TIMEDELTA * (NUM_EVAL_TIMESTAMPS - 1),
                TEST_CUT_DATE - LABEL_TIMEDELTA,
            ),
            freq=LABEL_TIMEDELTA,
        ).to_pydatetime().tolist()
        test_cut_dates = pd.date_range(
            start=TEST_CUT_DATE,
            end=min(
                TEST_CUT_DATE + LABEL_TIMEDELTA * (NUM_EVAL_TIMESTAMPS - 1),
                dataset_max_timestamp - LABEL_TIMEDELTA,
            ),
            freq=LABEL_TIMEDELTA,
        ).to_pydatetime().tolist()

        driver_columns = con.sql("SELECT * FROM drivers_src LIMIT 0").to_df().columns.tolist()
        result_columns = con.sql("SELECT * FROM results_src LIMIT 0").to_df().columns.tolist()
        standing_columns = con.sql("SELECT * FROM standings_src LIMIT 0").to_df().columns.tolist()
        race_columns = con.sql("SELECT * FROM races_src LIMIT 0").to_df().columns.tolist()
        circuit_columns = con.sql("SELECT * FROM circuits_src LIMIT 0").to_df().columns.tolist()
        constructor_columns = con.sql("SELECT * FROM constructors_src LIMIT 0").to_df().columns.tolist()

        driver_id_col = {column.lower(): column for column in driver_columns}["driverid"]
        result_id_col = {column.lower(): column for column in result_columns}["resultid"]
        result_driver_col = {column.lower(): column for column in result_columns}["driverid"]
        result_race_col = {column.lower(): column for column in result_columns}["raceid"]
        result_constructor_col = {column.lower(): column for column in result_columns}["constructorid"]
        result_status_col = {column.lower(): column for column in result_columns}["statusid"]
        result_date_col = {column.lower(): column for column in result_columns}["date"]
        standing_id_col = {column.lower(): column for column in standing_columns}["driverstandingsid"]
        standing_driver_col = {column.lower(): column for column in standing_columns}["driverid"]
        standing_date_col = {column.lower(): column for column in standing_columns}["date"]
        race_id_col = {column.lower(): column for column in race_columns}["raceid"]
        race_circuit_col = {column.lower(): column for column in race_columns}["circuitid"]
        race_date_col = {column.lower(): column for column in race_columns}["date"]
        circuit_id_col = {column.lower(): column for column in circuit_columns}["circuitid"]
        constructor_id_col = {column.lower(): column for column in constructor_columns}["constructorid"]

        for split_name, cut_dates in {
            "train": train_cut_dates,
            "val": val_cut_dates,
            "test": test_cut_dates,
        }.items():
            frames_for_split: list[pd.DataFrame] = []

            for cut_date in cut_dates:
                feature_cut_date = pd.Timestamp(cut_date) + pd.Timedelta(seconds=1)

                driver_node = DuckdbNode(
                    fpath="drivers_src",
                    prefix="drv",
                    pk=driver_id_col,
                    date_key=None,
                    columns=driver_columns,
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
                    name=f"rel_f1_driver_dnf_{pd.Timestamp(cut_date).date()}",
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

                for node in [driver_node, result_node, standing_node, race_node, circuit_node, constructor_node]:
                    graph.add_node(node)

                graph.add_entity_edge(driver_node, result_node, parent_key=driver_id_col, relation_key=result_driver_col, reduce=True)
                graph.add_entity_edge(driver_node, standing_node, parent_key=driver_id_col, relation_key=standing_driver_col, reduce=True)
                graph.add_entity_edge(result_node, race_node, parent_key=result_race_col, relation_key=race_id_col, reduce=False)
                graph.add_entity_edge(race_node, circuit_node, parent_key=race_circuit_col, relation_key=circuit_id_col, reduce=False)
                graph.add_entity_edge(result_node, constructor_node, parent_key=result_constructor_col, relation_key=constructor_id_col, reduce=False)

                graph.do_transformations_sql()
                features = con.sql(f"SELECT * FROM {graph.parent_node._cur_data_ref}").to_df().copy()
                features["timestamp"] = pd.Timestamp(cut_date)

                labels = con.sql(
                    f"""
                    WITH timestamp_df AS (
                        SELECT TIMESTAMP '{pd.Timestamp(cut_date)}' AS timestamp
                    )
                    SELECT
                        t.timestamp,
                        re.{result_driver_col} AS driverId,
                        MAX(CASE WHEN re.{result_status_col} != 1 THEN 1 ELSE 0 END) AS did_not_finish
                    FROM timestamp_df t
                    LEFT JOIN results_src re
                        ON re.{result_date_col} > t.timestamp
                        AND re.{result_date_col} <= t.timestamp + INTERVAL '{int(LABEL_TIMEDELTA.total_seconds())} seconds'
                    WHERE re.{result_driver_col} IN (
                        SELECT DISTINCT historical.{result_driver_col}
                        FROM results_src historical
                        WHERE historical.{result_date_col} > t.timestamp - INTERVAL '1 year'
                    )
                    GROUP BY t.timestamp, re.{result_driver_col}
                    """
                ).to_df()

                frame = features.merge(
                    labels,
                    left_on=["timestamp", f"drv_{driver_id_col}"],
                    right_on=["timestamp", "driverId"],
                    how="inner",
                ).drop(columns=["driverId"])
                frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype("int8")
                frames_for_split.append(frame)

            split_frames[split_name] = pd.concat(frames_for_split, ignore_index=True)
    finally:
        con.close()

    df_train = split_frames["train"]
    df_val = split_frames["val"]
    df_test = split_frames["test"]

    common_columns = set(df_train.columns) & set(df_val.columns) & set(df_test.columns)
    feature_columns = [
        column
        for column in df_train.select_dtypes(include=[np.number, "bool"]).columns
        if column != TARGET_COLUMN
        and "label" not in column.lower()
        and not column.lower().endswith("_id")
        and "driverid" not in column.lower()
        and column in common_columns
    ]
    if not feature_columns or df_train[TARGET_COLUMN].nunique() < 2:
        return df_train, df_val, df_test, None, None, len(feature_columns), materialized, TARGET_COLUMN

    model = CatBoostClassifier(
        iterations=500,
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
        "average_precision": float(average_precision(val_target, val_predictions)),
        "accuracy": float(accuracy(val_target, val_predictions)),
        "f1": float(f1(val_target, val_predictions)),
        "roc_auc": float(roc_auc(val_target, val_predictions)),
    }
    test_metrics = {
        "average_precision": float(average_precision(test_target, test_predictions)),
        "accuracy": float(accuracy(test_target, test_predictions)),
        "f1": float(f1(test_target, test_predictions)),
        "roc_auc": float(roc_auc(test_target, test_predictions)),
    }

    return df_train, df_val, df_test, val_metrics, test_metrics, len(feature_columns), materialized, TARGET_COLUMN


def main() -> None:
    df_train, df_val, df_test, val_metrics, test_metrics, n_features, materialized, target = run_rel_f1_driver_dnf()
    print("materialized_files:", materialized, flush=True)
    print("validation_timestamp:", VALIDATION_CUT_DATE.date(), flush=True)
    print("test_timestamp:", TEST_CUT_DATE.date(), flush=True)
    print("label_timedelta_days:", int(LABEL_TIMEDELTA / pd.Timedelta(days=1)), flush=True)
    print("num_eval_timestamps:", NUM_EVAL_TIMESTAMPS, flush=True)
    print("target:", target, flush=True)
    print("train_rows:", len(df_train), flush=True)
    print("validation_rows:", len(df_val), flush=True)
    print("test_rows:", len(df_test), flush=True)
    print("train_timestamps:", df_train["timestamp"].nunique(), flush=True)
    print("validation_timestamps:", df_val["timestamp"].nunique(), flush=True)
    print("test_timestamps:", df_test["timestamp"].nunique(), flush=True)
    print("columns:", len(df_train.columns), flush=True)
    print("feature_count:", n_features, flush=True)
    print("validation_metrics:", val_metrics if val_metrics is not None else "skipped", flush=True)
    print("test_metrics:", test_metrics if test_metrics is not None else "skipped", flush=True)


if __name__ == "__main__":
    main()
