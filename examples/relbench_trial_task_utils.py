#!/usr/bin/env python
"""Shared RelBench rel-trial feature builders for task-table examples."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import duckdb
import numpy as np
import pandas as pd

from graphreduce.enum import ComputeLayerEnum, PeriodUnit, SQLOpType
from graphreduce.graph_reduce import GraphReduce
from graphreduce.models import sqlop
from graphreduce.node import DuckdbNode
from relbench_dataset_utils import (
    RelBenchFrameStore,
    get_relbench_dataset_db,
    get_relbench_split_task_table,
    iter_training_frames,
    register_relbench_db_views,
    target_table_from_frame,
)
from relbench_regression_metrics import add_nmae
from relbench_catboost_utils import TEMPORAL_FEATURE_FAMILIES, fit_tuned_regressor_incremental, set_feature_families

LOOKBACK_START = datetime.datetime(2000, 1, 1)
SITE_SUCCESS_FEATURE_FAMILIES = ("base", "semantic", "context")

TABLE_NAME_TO_FILENAME = {
    "studies": "studies.parquet",
    "outcomes": "outcomes.parquet",
    "outcome_analyses": "outcome_analyses.parquet",
    "drop_withdrawals": "drop_withdrawals.parquet",
    "reported_event_totals": "reported_event_totals.parquet",
    "designs": "designs.parquet",
    "eligibilities": "eligibilities.parquet",
    "interventions": "interventions.parquet",
    "conditions": "conditions.parquet",
    "facilities": "facilities.parquet",
    "sponsors": "sponsors.parquet",
    "interventions_studies": "interventions_studies.parquet",
    "conditions_studies": "conditions_studies.parquet",
    "facilities_studies": "facilities_studies.parquet",
    "sponsors_studies": "sponsors_studies.parquet",
}


def prepare_trial_views(con: duckdb.DuckDBPyConnection, data_dir: Path | None = None) -> dict[str, list[str]]:
    _, db = get_relbench_dataset_db("rel-trial", download=True, upto_test_timestamp=False)
    register_relbench_db_views(
        con,
        db,
        {table_name: f"{table_name}_src" for table_name in TABLE_NAME_TO_FILENAME},
    )
    table_columns: dict[str, list[str]] = {}
    for table_name in TABLE_NAME_TO_FILENAME:
        table_columns[table_name] = con.sql(f"SELECT * FROM {table_name}_src LIMIT 0").to_df().columns.tolist()
    return table_columns


def _by_lower(columns: list[str]) -> dict[str, str]:
    return {column.lower(): column for column in columns}


def _select_columns(columns: list[str], required: list[str], optional: list[str] | None = None) -> list[str]:
    by_lower = _by_lower(columns)
    selected = [by_lower[column.lower()] for column in required]
    for column in optional or []:
        resolved = by_lower.get(column.lower())
        if resolved is not None:
            selected.append(resolved)
    return list(dict.fromkeys(selected))


def _feature_cut_date(task_timestamp: pd.Timestamp) -> datetime.datetime:
    return task_timestamp.to_pydatetime() + datetime.timedelta(days=1)


def _select_evenly_spaced_timestamps(
    timestamps: list[pd.Timestamp], max_count: int | None
) -> list[pd.Timestamp]:
    """Keep a bounded, evenly spaced sample while retaining both boundaries."""

    if max_count is None or len(timestamps) <= max_count:
        return list(timestamps)
    if max_count < 2:
        raise ValueError("max_count must be at least 2 when sampling timestamps")

    indices = np.linspace(0, len(timestamps) - 1, num=max_count)
    return [timestamps[int(round(index))] for index in indices]


def _graph(
    con: duckdb.DuckDBPyConnection,
    name: str,
    parent_node: DuckdbNode,
    nodes: list[DuckdbNode],
    cut_date: datetime.datetime,
) -> GraphReduce:
    gr = GraphReduce(
        name=name,
        parent_node=parent_node,
        compute_layer=ComputeLayerEnum.duckdb,
        sql_client=con,
        cut_date=cut_date,
        compute_period_val=max(1, (cut_date - LOOKBACK_START).days + 1),
        compute_period_unit=PeriodUnit.day,
        auto_features=True,
        auto_labels=False,
        date_filters_on_agg=True,
        auto_feature_hops_back=2,
        auto_feature_hops_front=0,
        use_temp_tables=True,
    )
    for node in nodes:
        gr.add_node(node)
    return gr


def build_study_features(
    con: duckdb.DuckDBPyConnection,
    table_columns: dict[str, list[str]],
    task_timestamp: pd.Timestamp,
) -> pd.DataFrame:
    cut_date = _feature_cut_date(task_timestamp)

    studies_cols = _select_columns(
        table_columns["studies"],
        ["nct_id", "start_date"],
        ["enrollment", "number_of_arms", "number_of_groups"],
    )
    outcomes_cols = _select_columns(table_columns["outcomes"], ["id", "nct_id", "date"])
    outcome_analyses_cols = _select_columns(
        table_columns["outcome_analyses"],
        ["id", "nct_id", "outcome_id", "date"],
        [
            "p_value_modifier",
            "param_value",
            "dispersion_value",
            "p_value",
            "ci_percent",
            "ci_lower_limit",
            "ci_upper_limit",
            "ci_upper_limit_raw",
            "ci_lower_limit_raw",
            "p_value_raw",
        ],
    )
    drop_withdrawals_cols = _select_columns(
        table_columns["drop_withdrawals"],
        ["id", "nct_id", "date"],
        ["count"],
    )
    reported_event_totals_cols = _select_columns(
        table_columns["reported_event_totals"],
        ["id", "nct_id", "date"],
        [
            "event_type",
            "classification",
            "subjects_affected",
            "subjects_at_risk",
        ],
    )
    designs_cols = _select_columns(table_columns["designs"], ["id", "nct_id", "date"])
    eligibilities_cols = _select_columns(table_columns["eligibilities"], ["id", "nct_id", "date"])
    interventions_cols = _select_columns(table_columns["interventions"], ["intervention_id"])
    conditions_cols = _select_columns(table_columns["conditions"], ["condition_id"])
    facilities_cols = _select_columns(table_columns["facilities"], ["facility_id"])
    sponsors_cols = _select_columns(table_columns["sponsors"], ["sponsor_id"])
    interventions_studies_cols = _select_columns(
        table_columns["interventions_studies"],
        ["id", "nct_id", "intervention_id", "date"],
    )
    conditions_studies_cols = _select_columns(
        table_columns["conditions_studies"],
        ["id", "nct_id", "condition_id", "date"],
    )
    facilities_studies_cols = _select_columns(
        table_columns["facilities_studies"],
        ["id", "nct_id", "facility_id", "date"],
    )
    sponsors_studies_cols = _select_columns(
        table_columns["sponsors_studies"],
        ["id", "nct_id", "sponsor_id", "date"],
    )

    studies_l = _by_lower(studies_cols)
    outcomes_l = _by_lower(outcomes_cols)
    outcome_analyses_l = _by_lower(outcome_analyses_cols)
    drop_withdrawals_l = _by_lower(drop_withdrawals_cols)
    reported_event_totals_l = _by_lower(reported_event_totals_cols)
    designs_l = _by_lower(designs_cols)
    eligibilities_l = _by_lower(eligibilities_cols)
    interventions_l = _by_lower(interventions_cols)
    conditions_l = _by_lower(conditions_cols)
    facilities_l = _by_lower(facilities_cols)
    sponsors_l = _by_lower(sponsors_cols)
    interventions_studies_l = _by_lower(interventions_studies_cols)
    conditions_studies_l = _by_lower(conditions_studies_cols)
    facilities_studies_l = _by_lower(facilities_studies_cols)
    sponsors_studies_l = _by_lower(sponsors_studies_cols)

    outcome_analysis_annotations = {}
    if "p_value" in outcome_analyses_l:
        outcome_analysis_annotations["is_significant"] = (
            f"{{{outcome_analyses_l['p_value']}}} >= 0 "
            f"AND {{{outcome_analyses_l['p_value']}}} <= 0.05"
        )
    reported_event_annotations = {}
    if "event_type" in reported_event_totals_l:
        reported_event_annotations["is_serious_or_death"] = (
            f"{{{reported_event_totals_l['event_type']}}} IN ('serious', 'deaths')"
        )

    studies = DuckdbNode(
        fpath="studies_src",
        prefix="std",
        pk=studies_l["nct_id"],
        date_key=studies_l["start_date"],
        columns=studies_cols,
        do_filters_ops=[
            sqlop(optype=SQLOpType.where, opval=f"std_{studies_l['nct_id']} is not null"),
            sqlop(
                optype=SQLOpType.where,
                opval=f"std_{studies_l['start_date']} <= '{task_timestamp.date()}'",
            ),
        ],
    )
    outcomes = DuckdbNode(
        fpath="outcomes_src",
        prefix="out",
        pk=outcomes_l["id"],
        date_key="date",
        columns=outcomes_cols,
    )
    outcome_analyses = DuckdbNode(
        fpath="outcome_analyses_src",
        prefix="oa",
        pk=outcome_analyses_l["id"],
        date_key=outcome_analyses_l["date"],
        columns=outcome_analyses_cols,
        feature_family_max_columns=4,
        categorical_top_k=5,
        context_keys=(outcome_analyses_l["nct_id"], outcome_analyses_l["outcome_id"]),
        annotation_expressions=outcome_analysis_annotations,
    )
    drop_withdrawals = DuckdbNode(
        fpath="drop_withdrawals_src",
        prefix="drw",
        pk=drop_withdrawals_l["id"],
        date_key=drop_withdrawals_l["date"],
        columns=drop_withdrawals_cols,
    )
    reported_event_totals = DuckdbNode(
        fpath="reported_event_totals_src",
        prefix="evt",
        pk=reported_event_totals_l["id"],
        date_key=reported_event_totals_l["date"],
        columns=reported_event_totals_cols,
        feature_family_max_columns=4,
        categorical_top_k=5,
        context_keys=(reported_event_totals_l["nct_id"],),
        annotation_expressions=reported_event_annotations,
    )
    designs = DuckdbNode(
        fpath="designs_src",
        prefix="dsg",
        pk=designs_l["id"],
        date_key=designs_l["date"],
        columns=designs_cols,
    )
    eligibilities = DuckdbNode(
        fpath="eligibilities_src",
        prefix="eli",
        pk=eligibilities_l["id"],
        date_key=eligibilities_l["date"],
        columns=eligibilities_cols,
    )
    interventions_studies = DuckdbNode(
        fpath="interventions_studies_src",
        prefix="ist",
        pk=interventions_studies_l["id"],
        date_key=interventions_studies_l["date"],
        columns=interventions_studies_cols,
    )
    conditions_studies = DuckdbNode(
        fpath="conditions_studies_src",
        prefix="cst",
        pk=conditions_studies_l["id"],
        date_key=conditions_studies_l["date"],
        columns=conditions_studies_cols,
    )
    facilities_studies = DuckdbNode(
        fpath="facilities_studies_src",
        prefix="fst",
        pk=facilities_studies_l["id"],
        date_key=facilities_studies_l["date"],
        columns=facilities_studies_cols,
    )
    sponsors_studies = DuckdbNode(
        fpath="sponsors_studies_src",
        prefix="sst",
        pk=sponsors_studies_l["id"],
        date_key=sponsors_studies_l["date"],
        columns=sponsors_studies_cols,
    )
    interventions = DuckdbNode(
        fpath="interventions_src",
        prefix="intv",
        pk=interventions_l["intervention_id"],
        date_key=None,
        columns=interventions_cols,
    )
    conditions = DuckdbNode(
        fpath="conditions_src",
        prefix="cond",
        pk=conditions_l["condition_id"],
        date_key=None,
        columns=conditions_cols,
    )
    facilities = DuckdbNode(
        fpath="facilities_src",
        prefix="fac",
        pk=facilities_l["facility_id"],
        date_key=None,
        columns=facilities_cols,
    )
    sponsors = DuckdbNode(
        fpath="sponsors_src",
        prefix="spn",
        pk=sponsors_l["sponsor_id"],
        date_key=None,
        columns=sponsors_cols,
    )

    nodes = [
        studies,
        outcomes,
        outcome_analyses,
        drop_withdrawals,
        reported_event_totals,
        designs,
        eligibilities,
        interventions_studies,
        conditions_studies,
        facilities_studies,
        sponsors_studies,
        interventions,
        conditions,
        facilities,
        sponsors,
    ]
    set_feature_families(
        [outcome_analyses, reported_event_totals], TEMPORAL_FEATURE_FAMILIES
    )
    gr = _graph(
        con,
        f"rel_trial_study_features_{task_timestamp.date()}",
        studies,
        nodes,
        cut_date,
    )

    study_id = studies_l["nct_id"]
    gr.add_entity_edge(studies, outcomes, study_id, outcomes_l["nct_id"], reduce=True)
    gr.add_entity_edge(studies, outcome_analyses, study_id, outcome_analyses_l["nct_id"], reduce=True)
    gr.add_entity_edge(studies, drop_withdrawals, study_id, drop_withdrawals_l["nct_id"], reduce=True)
    gr.add_entity_edge(studies, reported_event_totals, study_id, reported_event_totals_l["nct_id"], reduce=True)
    gr.add_entity_edge(studies, designs, study_id, designs_l["nct_id"], reduce=True)
    gr.add_entity_edge(studies, eligibilities, study_id, eligibilities_l["nct_id"], reduce=True)
    gr.add_entity_edge(studies, interventions_studies, study_id, interventions_studies_l["nct_id"], reduce=True)
    gr.add_entity_edge(studies, conditions_studies, study_id, conditions_studies_l["nct_id"], reduce=True)
    gr.add_entity_edge(studies, facilities_studies, study_id, facilities_studies_l["nct_id"], reduce=True)
    gr.add_entity_edge(studies, sponsors_studies, study_id, sponsors_studies_l["nct_id"], reduce=True)
    gr.add_entity_edge(
        interventions_studies,
        interventions,
        interventions_studies_l["intervention_id"],
        interventions_l["intervention_id"],
        reduce=True,
    )
    gr.add_entity_edge(
        conditions_studies,
        conditions,
        conditions_studies_l["condition_id"],
        conditions_l["condition_id"],
        reduce=True,
    )
    gr.add_entity_edge(
        facilities_studies,
        facilities,
        facilities_studies_l["facility_id"],
        facilities_l["facility_id"],
        reduce=True,
    )
    gr.add_entity_edge(
        sponsors_studies,
        sponsors,
        sponsors_studies_l["sponsor_id"],
        sponsors_l["sponsor_id"],
        reduce=True,
    )

    gr.do_transformations_sql()
    frame = con.sql(f"SELECT * FROM {gr.parent_node._cur_data_ref}").to_df().copy()
    gr._clean_refs()
    frame["timestamp"] = task_timestamp
    return frame


def build_site_features(
    con: duckdb.DuckDBPyConnection,
    table_columns: dict[str, list[str]],
    task_timestamp: pd.Timestamp,
) -> pd.DataFrame:
    cut_date = _feature_cut_date(task_timestamp)

    facilities_cols = _select_columns(table_columns["facilities"], ["facility_id"])
    facilities_studies_cols = _select_columns(
        table_columns["facilities_studies"],
        ["id", "nct_id", "facility_id", "date"],
    )
    studies_cols = _select_columns(
        table_columns["studies"],
        ["nct_id", "start_date"],
        ["enrollment", "number_of_arms", "number_of_groups"],
    )
    outcomes_cols = _select_columns(table_columns["outcomes"], ["id", "nct_id", "date"])
    outcome_analyses_cols = _select_columns(
        table_columns["outcome_analyses"],
        ["id", "nct_id", "outcome_id", "date"],
        [
            "p_value_modifier",
            "param_value",
            "dispersion_value",
            "p_value",
            "ci_percent",
            "ci_lower_limit",
            "ci_upper_limit",
            "ci_upper_limit_raw",
            "ci_lower_limit_raw",
            "p_value_raw",
        ],
    )
    reported_event_totals_cols = _select_columns(
        table_columns["reported_event_totals"],
        ["id", "nct_id", "date"],
        [
            "event_type",
            "classification",
            "subjects_affected",
            "subjects_at_risk",
        ],
    )

    facilities_l = _by_lower(facilities_cols)
    facilities_studies_l = _by_lower(facilities_studies_cols)
    studies_l = _by_lower(studies_cols)
    outcomes_l = _by_lower(outcomes_cols)
    outcome_analyses_l = _by_lower(outcome_analyses_cols)
    reported_event_totals_l = _by_lower(reported_event_totals_cols)

    outcome_analysis_annotations = {}
    if "p_value" in outcome_analyses_l:
        outcome_analysis_annotations["is_significant"] = (
            f"{{{outcome_analyses_l['p_value']}}} >= 0 "
            f"AND {{{outcome_analyses_l['p_value']}}} <= 0.05"
        )
    reported_event_annotations = {}
    if "event_type" in reported_event_totals_l:
        reported_event_annotations["is_serious_or_death"] = (
            f"{{{reported_event_totals_l['event_type']}}} IN ('serious', 'deaths')"
        )

    facilities = DuckdbNode(
        fpath="facilities_src",
        prefix="fac",
        pk=facilities_l["facility_id"],
        date_key=None,
        columns=facilities_cols,
    )
    facilities_studies = DuckdbNode(
        fpath="facilities_studies_src",
        prefix="fst",
        pk=facilities_studies_l["id"],
        date_key=facilities_studies_l["date"],
        columns=facilities_studies_cols,
    )
    studies = DuckdbNode(
        fpath="studies_src",
        prefix="std",
        pk=studies_l["nct_id"],
        date_key=studies_l["start_date"],
        columns=studies_cols,
    )
    outcomes = DuckdbNode(
        fpath="outcomes_src",
        prefix="out",
        pk=outcomes_l["id"],
        date_key="date",
        columns=outcomes_cols,
    )
    outcome_analyses = DuckdbNode(
        fpath="outcome_analyses_src",
        prefix="oa",
        pk=outcome_analyses_l["id"],
        date_key=outcome_analyses_l["date"],
        columns=outcome_analyses_cols,
        feature_family_max_columns=4,
        categorical_top_k=5,
        context_keys=(outcome_analyses_l["nct_id"], outcome_analyses_l["outcome_id"]),
        annotation_expressions=outcome_analysis_annotations,
    )
    reported_event_totals = DuckdbNode(
        fpath="reported_event_totals_src",
        prefix="evt",
        pk=reported_event_totals_l["id"],
        date_key=reported_event_totals_l["date"],
        columns=reported_event_totals_cols,
        feature_family_max_columns=4,
        categorical_top_k=5,
        context_keys=(reported_event_totals_l["nct_id"],),
        annotation_expressions=reported_event_annotations,
    )

    nodes = [
        facilities,
        facilities_studies,
        studies,
        outcomes,
        outcome_analyses,
        reported_event_totals,
    ]
    set_feature_families(
        [outcome_analyses, reported_event_totals], SITE_SUCCESS_FEATURE_FAMILIES
    )
    gr = _graph(
        con,
        f"rel_trial_site_features_{task_timestamp.date()}",
        facilities,
        nodes,
        cut_date,
    )

    gr.add_entity_edge(
        facilities,
        facilities_studies,
        facilities_l["facility_id"],
        facilities_studies_l["facility_id"],
        reduce=True,
    )
    gr.add_entity_edge(
        facilities_studies,
        studies,
        facilities_studies_l["nct_id"],
        studies_l["nct_id"],
        reduce=False,
    )
    gr.add_entity_edge(
        facilities_studies,
        outcomes,
        facilities_studies_l["nct_id"],
        outcomes_l["nct_id"],
        reduce=True,
    )
    gr.add_entity_edge(
        facilities_studies,
        outcome_analyses,
        facilities_studies_l["nct_id"],
        outcome_analyses_l["nct_id"],
        reduce=True,
    )
    gr.add_entity_edge(
        facilities_studies,
        reported_event_totals,
        facilities_studies_l["nct_id"],
        reported_event_totals_l["nct_id"],
        reduce=True,
    )

    gr.do_transformations_sql()
    frame = con.sql(f"SELECT * FROM {gr.parent_node._cur_data_ref}").to_df().copy()
    gr._clean_refs()
    frame["timestamp"] = task_timestamp
    return frame


def build_task_split_frame(
    con: duckdb.DuckDBPyConnection,
    table_columns: dict[str, list[str]],
    task_name: str,
    split: str,
    feature_builder: Callable[[duckdb.DuckDBPyConnection, dict[str, list[str]], pd.Timestamp], pd.DataFrame],
    feature_entity_col: str,
    max_train_frames: int | None = None,
) -> tuple[object, pd.DataFrame, pd.Timestamp]:
    task, task_table, cut_timestamps = get_relbench_split_task_table(
        "rel-trial", task_name, split, download=True
    )
    if split == "train":
        cut_timestamps = _select_evenly_spaced_timestamps(
            cut_timestamps, max_train_frames
        )
    frame_store = RelBenchFrameStore(
        f"rel-trial-{task_name}-{split}", persist_each_frame=True
    )
    labels = task_table.df.copy()
    labels["_relbench_entity_key"] = labels[task.entity_col].astype(str)
    def build_frame(frame_con: duckdb.DuckDBPyConnection, cut_timestamp: pd.Timestamp) -> pd.DataFrame:
        features = feature_builder(frame_con, table_columns, cut_timestamp)
        features["_relbench_entity_key"] = features[feature_entity_col].astype(str)
        timestamp_labels = labels[labels[task.time_col] == cut_timestamp]
        return features.merge(
            timestamp_labels[["_relbench_entity_key", task.time_col, task.entity_col, task.target_col]],
            left_on=["timestamp", "_relbench_entity_key"],
            right_on=[task.time_col, "_relbench_entity_key"],
            how="right",
            validate="one_to_one",
        ).drop(columns=["_relbench_entity_key"])

    frame_workers = None if split == "train" else 1
    for frame in iter_training_frames(con, cut_timestamps, build_frame, workers=frame_workers):
        frame_store.append(frame)
    return task, frame_store, cut_timestamps[-1]


def select_shared_numeric_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    excluded_cols: set[str],
) -> list[str]:
    common_columns = set(train_df.columns) & set(val_df.columns) & set(test_df.columns)
    return [
        column
        for column in train_df.select_dtypes(include=[np.number, "bool"]).columns
        if column != target_col
        and column not in excluded_cols
        and "label" not in column.lower()
        and not column.lower().endswith("_id")
        and column in common_columns
    ]


def run_rel_trial_regression_task(
    task_name: str,
    feature_builder: Callable[[duckdb.DuckDBPyConnection, dict[str, list[str]], pd.Timestamp], pd.DataFrame],
    feature_entity_col: str,
    data_dir: Path | None = None,
    max_train_frames: int | None = None,
) -> tuple[RelBenchFrameStore, pd.DataFrame, pd.DataFrame, dict[str, float] | None, dict[str, float] | None, int, list[str], str]:
    materialized: list[str] = []

    con = duckdb.connect()
    split_frames: dict[str, RelBenchFrameStore] = {}
    split_tasks: dict[str, object] = {}

    try:
        table_columns = prepare_trial_views(con)
        for split_name in ["train", "val", "test"]:
            task, frame_store, _ = build_task_split_frame(
                con,
                table_columns,
                task_name,
                split_name,
                feature_builder,
                feature_entity_col,
                max_train_frames=max_train_frames,
            )
            split_tasks[split_name] = task
            split_frames[split_name] = frame_store
    finally:
        con.close()

    train_store = split_frames["train"]
    df_val = split_frames["val"].to_dataframe()
    df_test = split_frames["test"].to_dataframe()
    split_frames["val"].close()
    split_frames["test"].close()
    target = split_tasks["train"].target_col

    train_sample = train_store.sample_frame()
    feature_columns = select_shared_numeric_features(
        train_sample,
        df_val,
        df_test,
        target,
        excluded_cols={split_tasks["train"].entity_col, feature_entity_col},
    )
    if not feature_columns:
        return train_store, df_val, df_test, None, None, 0, materialized, target

    def train_batches():
        for batch in train_store.iter_batches():
            batch = batch.copy()
            batch[target] = batch[target].fillna(0).astype("float64")
            yield batch

    model, best_config, best_val_mae = fit_tuned_regressor_incremental(
        train_batches,
        feature_columns,
        target,
        df_val[feature_columns].fillna(0),
        df_val[target].astype("float64"),
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
        df_val[target],
        val_predictions,
        train_store.target_std(target),
    )
    test_metrics = add_nmae(
        split_tasks["test"].evaluate(
            test_predictions,
            target_table=target_table_from_frame(split_tasks["test"], df_test),
        ),
        df_test[target],
        test_predictions,
        train_store.target_std(target),
    )

    return train_store, df_val, df_test, val_metrics, test_metrics, len(feature_columns), materialized, target
