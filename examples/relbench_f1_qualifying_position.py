#!/usr/bin/env python
"""RelBench rel-f1: qualifying position autocomplete example aligned to the official task definition."""

from __future__ import annotations

import os
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

DATASET_NAME = "rel-f1"
REL_CACHE_DIR = Path("tests/data/relbench_cache").resolve()
DATA_DIR = Path("tests/data/relbench/rel-f1")
TABLES = [
    "circuits",
    "constructors",
    "drivers",
    "qualifying",
    "races",
]

VALIDATION_CUT_DATE = pd.Timestamp("2005-01-01")
TEST_CUT_DATE = pd.Timestamp("2010-01-01")
TARGET_COLUMN = "qua_position"


def run_rel_f1_qualifying_position(
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
                SELECT date FROM qualifying_src
            )
            """
        ).to_df()
        lookback_start = pd.Timestamp(dataset_bounds.loc[0, "min_timestamp"])
        dataset_max_timestamp = pd.Timestamp(dataset_bounds.loc[0, "max_timestamp"])

        qualifying_columns = con.sql("SELECT * FROM qualifying_src LIMIT 0").to_df().columns.tolist()
        driver_columns = con.sql("SELECT * FROM drivers_src LIMIT 0").to_df().columns.tolist()
        constructor_columns = con.sql("SELECT * FROM constructors_src LIMIT 0").to_df().columns.tolist()
        race_columns = con.sql("SELECT * FROM races_src LIMIT 0").to_df().columns.tolist()
        circuit_columns = con.sql("SELECT * FROM circuits_src LIMIT 0").to_df().columns.tolist()

        qualifying_id_col = {column.lower(): column for column in qualifying_columns}["qualifyid"]
        qualifying_driver_col = {column.lower(): column for column in qualifying_columns}["driverid"]
        qualifying_constructor_col = {column.lower(): column for column in qualifying_columns}["constructorid"]
        qualifying_race_col = {column.lower(): column for column in qualifying_columns}["raceid"]
        qualifying_date_col = {column.lower(): column for column in qualifying_columns}["date"]
        driver_id_col = {column.lower(): column for column in driver_columns}["driverid"]
        constructor_id_col = {column.lower(): column for column in constructor_columns}["constructorid"]
        race_id_col = {column.lower(): column for column in race_columns}["raceid"]
        race_circuit_col = {column.lower(): column for column in race_columns}["circuitid"]
        race_date_col = {column.lower(): column for column in race_columns}["date"]
        circuit_id_col = {column.lower(): column for column in circuit_columns}["circuitid"]

        qualifying_node = DuckdbNode(
            fpath="qualifying_src",
            prefix="qua",
            pk=qualifying_id_col,
            date_key=qualifying_date_col,
            columns=qualifying_columns,
        )
        driver_node = DuckdbNode(
            fpath="drivers_src",
            prefix="drv",
            pk=driver_id_col,
            date_key=None,
            columns=driver_columns,
        )
        constructor_node = DuckdbNode(
            fpath="constructors_src",
            prefix="con",
            pk=constructor_id_col,
            date_key=None,
            columns=constructor_columns,
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

        graph = GraphReduce(
            name="rel_f1_qualifying_position",
            parent_node=qualifying_node,
            compute_layer=ComputeLayerEnum.duckdb,
            sql_client=con,
            cut_date=(dataset_max_timestamp + pd.Timedelta(seconds=1)).to_pydatetime(),
            compute_period_val=max(1, int((dataset_max_timestamp - lookback_start).days + 1)),
            compute_period_unit=PeriodUnit.day,
            auto_features=True,
            auto_labels=False,
            date_filters_on_agg=True,
            auto_feature_hops_back=2,
            auto_feature_hops_front=0,
            use_temp_tables=True,
        )

        for node in [qualifying_node, driver_node, constructor_node, race_node, circuit_node]:
            graph.add_node(node)

        graph.add_entity_edge(qualifying_node, driver_node, parent_key=qualifying_driver_col, relation_key=driver_id_col, reduce=False)
        graph.add_entity_edge(qualifying_node, constructor_node, parent_key=qualifying_constructor_col, relation_key=constructor_id_col, reduce=False)
        graph.add_entity_edge(qualifying_node, race_node, parent_key=qualifying_race_col, relation_key=race_id_col, reduce=False)
        graph.add_entity_edge(race_node, circuit_node, parent_key=race_circuit_col, relation_key=circuit_id_col, reduce=False)

        graph.do_transformations_sql()
        all_rows = con.sql(f"SELECT * FROM {graph.parent_node._cur_data_ref}").to_df().copy()
    finally:
        con.close()

    all_rows = all_rows.dropna(subset=[TARGET_COLUMN]).copy()
    train_end = VALIDATION_CUT_DATE - pd.Timedelta(seconds=1)
    val_end = TEST_CUT_DATE - pd.Timedelta(seconds=1)

    df_train = all_rows[all_rows["qua_date"] <= train_end].copy()
    df_val = all_rows[(all_rows["qua_date"] > VALIDATION_CUT_DATE) & (all_rows["qua_date"] <= val_end)].copy()
    df_test = all_rows[all_rows["qua_date"] > TEST_CUT_DATE].copy()

    common_columns = set(df_train.columns) & set(df_val.columns) & set(df_test.columns)
    feature_columns = [
        column
        for column in df_train.select_dtypes(include=[np.number, "bool"]).columns
        if column != TARGET_COLUMN
        and "label" not in column.lower()
        and not column.lower().endswith("_id")
        and column not in {"qua_qualifyId", "qua_raceId", "qua_driverId", "qua_constructorId"}
        and column in common_columns
    ]
    if not feature_columns:
        return df_train, df_val, df_test, None, None, 0, materialized, TARGET_COLUMN

    model = CatBoostRegressor(
        iterations=600,
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

    val_target = df_val[TARGET_COLUMN].astype("float64").to_numpy()
    test_target = df_test[TARGET_COLUMN].astype("float64").to_numpy()
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
    df_train, df_val, df_test, val_metrics, test_metrics, n_features, materialized, target = run_rel_f1_qualifying_position()
    print("materialized_files:", materialized, flush=True)
    print("validation_timestamp:", VALIDATION_CUT_DATE.date(), flush=True)
    print("test_timestamp:", TEST_CUT_DATE.date(), flush=True)
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
