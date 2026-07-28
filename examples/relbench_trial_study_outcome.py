#!/usr/bin/env python
"""RelBench rel-trial study-outcome example with DuckDB + GraphReduce."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import duckdb
import numpy as np
import pandas as pd
from relbench_dataset_utils import (
    get_relbench_dataset_db,
    get_relbench_split_task_table,
    iter_training_frames,
    register_relbench_db_views,
    target_table_from_frame,
)

from graphreduce.enum import ComputeLayerEnum, PeriodUnit, SQLOpType
from graphreduce.graph_reduce import GraphReduce
from graphreduce.models import sqlop
from graphreduce.node import DuckdbNode
from relbench_catboost_utils import TEMPORAL_FEATURE_FAMILIES, fit_tuned_classifier

VAL_TIMESTAMP = datetime.datetime(2020, 1, 1)
TEST_TIMESTAMP = datetime.datetime(2021, 1, 1)
LOOKBACK_START = datetime.datetime(2000, 1, 1)
LABEL_DAYS = 365

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


def _select_columns(
    columns: list[str],
    required: list[str],
    optional: list[str] | None = None,
) -> list[str]:
    by_lower = {column.lower(): column for column in columns}
    selected = [by_lower[name.lower()] for name in required]
    for name in optional or []:
        column = by_lower.get(name.lower())
        if column is not None:
            selected.append(column)
    return list(dict.fromkeys(selected))


def run_rel_trial_study_outcome(
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float | None, float | None, int, list[str], str]:
    _, db = get_relbench_dataset_db("rel-trial", download=True, upto_test_timestamp=False)
    split_tasks = {}
    frame_jobs = []
    for split_name in ("train", "val", "test"):
        task, task_table, cut_timestamps = get_relbench_split_task_table(
            "rel-trial",
            "study-outcome",
            split_name,
            download=True,
            db=db,
        )
        split_tasks[split_name] = task
        frame_jobs.extend(
            (
                split_name,
                f"{split_name}_{timestamp.isoformat()}",
                timestamp,
                task,
                task_table,
            )
            for timestamp in cut_timestamps
        )
    materialized: list[str] = []

    con = duckdb.connect()
    frames_by_name: dict[str, pd.DataFrame] = {}

    try:
        register_relbench_db_views(
            con,
            db,
            {table_name: f"{table_name}_src" for table_name in TABLE_NAME_TO_FILENAME},
        )

        table_columns: dict[str, list[str]] = {}
        for table_name in TABLE_NAME_TO_FILENAME:
            table_columns[table_name] = con.sql(f"SELECT * FROM {table_name}_src LIMIT 0").to_df().columns.tolist()

        def build_frame(frame_con, frame_info):
            con = frame_con
            _, frame_name, cut_date, task, task_table = frame_info
            feature_cut_date = cut_date + datetime.timedelta(days=1)
            studies_cols = _select_columns(
                table_columns["studies"],
                ["nct_id", "start_date"],
                ["enrollment", "number_of_arms", "number_of_groups"],
            )
            outcomes_cols = _select_columns(
                table_columns["outcomes"], ["id", "nct_id", "date"]
            )
            outcome_analyses_cols = _select_columns(
                table_columns["outcome_analyses"],
                ["id", "nct_id", "outcome_id", "p_value", "date"],
                ["p_value_modifier"],
            )
            drop_withdrawals_cols = _select_columns(
                table_columns["drop_withdrawals"], ["id", "nct_id", "date"]
            )
            reported_event_totals_cols = _select_columns(
                table_columns["reported_event_totals"],
                ["id", "nct_id", "date"],
                ["event_type"],
            )
            designs_cols = _select_columns(
                table_columns["designs"], ["id", "nct_id", "date"]
            )
            eligibilities_cols = _select_columns(
                table_columns["eligibilities"], ["id", "nct_id", "date"]
            )
            interventions_cols = _select_columns(
                table_columns["interventions"], ["intervention_id"]
            )
            conditions_cols = _select_columns(
                table_columns["conditions"], ["condition_id"]
            )
            facilities_cols = _select_columns(
                table_columns["facilities"], ["facility_id"]
            )
            sponsors_cols = _select_columns(
                table_columns["sponsors"], ["sponsor_id"]
            )
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

            studies_cols_by_lower = {column.lower(): column for column in studies_cols}
            outcomes_cols_by_lower = {column.lower(): column for column in outcomes_cols}
            outcome_analyses_cols_by_lower = {column.lower(): column for column in outcome_analyses_cols}
            drop_withdrawals_cols_by_lower = {column.lower(): column for column in drop_withdrawals_cols}
            reported_event_totals_cols_by_lower = {column.lower(): column for column in reported_event_totals_cols}
            designs_cols_by_lower = {column.lower(): column for column in designs_cols}
            eligibilities_cols_by_lower = {column.lower(): column for column in eligibilities_cols}
            interventions_cols_by_lower = {column.lower(): column for column in interventions_cols}
            conditions_cols_by_lower = {column.lower(): column for column in conditions_cols}
            facilities_cols_by_lower = {column.lower(): column for column in facilities_cols}
            sponsors_cols_by_lower = {column.lower(): column for column in sponsors_cols}
            interventions_studies_cols_by_lower = {
                column.lower(): column for column in interventions_studies_cols
            }
            conditions_studies_cols_by_lower = {column.lower(): column for column in conditions_studies_cols}
            facilities_studies_cols_by_lower = {column.lower(): column for column in facilities_studies_cols}
            sponsors_studies_cols_by_lower = {column.lower(): column for column in sponsors_studies_cols}

            studies_nct_id = studies_cols_by_lower["nct_id"]
            studies_start_date = studies_cols_by_lower["start_date"]
            outcomes_id = outcomes_cols_by_lower["id"]
            outcomes_nct_id = outcomes_cols_by_lower["nct_id"]
            oa_id = outcome_analyses_cols_by_lower["id"]
            oa_nct_id = outcome_analyses_cols_by_lower["nct_id"]
            oa_outcome_id = outcome_analyses_cols_by_lower["outcome_id"]
            oa_p_value = outcome_analyses_cols_by_lower["p_value"]
            oa_p_value_modifier = outcome_analyses_cols_by_lower.get("p_value_modifier")
            oa_date = outcome_analyses_cols_by_lower["date"]
            drw_id = drop_withdrawals_cols_by_lower["id"]
            drw_nct_id = drop_withdrawals_cols_by_lower["nct_id"]
            drw_date = drop_withdrawals_cols_by_lower["date"]
            evt_id = reported_event_totals_cols_by_lower["id"]
            evt_nct_id = reported_event_totals_cols_by_lower["nct_id"]
            evt_event_type = reported_event_totals_cols_by_lower.get("event_type")
            evt_date = reported_event_totals_cols_by_lower["date"]
            dsg_id = designs_cols_by_lower["id"]
            dsg_nct_id = designs_cols_by_lower["nct_id"]
            dsg_date = designs_cols_by_lower["date"]
            eli_id = eligibilities_cols_by_lower["id"]
            eli_nct_id = eligibilities_cols_by_lower["nct_id"]
            eli_date = eligibilities_cols_by_lower["date"]
            intv_studies_id = interventions_studies_cols_by_lower["id"]
            intv_studies_nct_id = interventions_studies_cols_by_lower["nct_id"]
            intv_studies_intv_id = interventions_studies_cols_by_lower["intervention_id"]
            intv_studies_date = interventions_studies_cols_by_lower["date"]
            cond_studies_id = conditions_studies_cols_by_lower["id"]
            cond_studies_nct_id = conditions_studies_cols_by_lower["nct_id"]
            cond_studies_cond_id = conditions_studies_cols_by_lower["condition_id"]
            cond_studies_date = conditions_studies_cols_by_lower["date"]
            fac_studies_id = facilities_studies_cols_by_lower["id"]
            fac_studies_nct_id = facilities_studies_cols_by_lower["nct_id"]
            fac_studies_fac_id = facilities_studies_cols_by_lower["facility_id"]
            fac_studies_date = facilities_studies_cols_by_lower["date"]
            spn_studies_id = sponsors_studies_cols_by_lower["id"]
            spn_studies_nct_id = sponsors_studies_cols_by_lower["nct_id"]
            spn_studies_spn_id = sponsors_studies_cols_by_lower["sponsor_id"]
            spn_studies_date = sponsors_studies_cols_by_lower["date"]
            interventions_id = interventions_cols_by_lower["intervention_id"]
            conditions_id = conditions_cols_by_lower["condition_id"]
            facilities_id = facilities_cols_by_lower["facility_id"]
            sponsors_id = sponsors_cols_by_lower["sponsor_id"]

            outcome_analysis_annotations = {
                "is_significant": f"{{{oa_p_value}}} >= 0 AND {{{oa_p_value}}} <= 0.05"
            }
            reported_event_annotations = {}
            if evt_event_type is not None:
                reported_event_annotations["is_serious_or_death"] = (
                    f"{{{evt_event_type}}} IN ('serious', 'deaths')"
                )

            studies = DuckdbNode(
                fpath="studies_src",
                prefix="std",
                pk=studies_nct_id,
                date_key=studies_start_date,
                columns=studies_cols,
                do_filters_ops=[
                    sqlop(optype=SQLOpType.where, opval=f"std_{studies_nct_id} is not null"),
                    sqlop(optype=SQLOpType.where, opval=f"std_{studies_start_date} <= '{cut_date.date()}'"),
                ],
                feature_families=TEMPORAL_FEATURE_FAMILIES,
                ts_periods=[30, 90, 180, 365],
                categorical_top_k=5,
                auto_text_features=False,
                feature_family_max_columns=8,
            )
            outcomes = DuckdbNode(
                fpath="outcomes_src",
                prefix="out",
                pk=outcomes_id,
                date_key="date",
                columns=outcomes_cols,
                feature_families=TEMPORAL_FEATURE_FAMILIES,
                ts_periods=[30, 90, 180, 365],
                categorical_top_k=5,
                auto_text_features=False,
                feature_family_max_columns=8,
            )
            outcome_analyses = DuckdbNode(
                fpath="outcome_analyses_src",
                prefix="oa",
                pk=oa_id,
                date_key=oa_date,
                columns=outcome_analyses_cols,
                feature_families=TEMPORAL_FEATURE_FAMILIES,
                ts_periods=[30, 90, 180, 365],
                categorical_top_k=5,
                auto_text_features=False,
                feature_family_max_columns=8,
                context_keys=(oa_nct_id, oa_outcome_id),
                annotation_expressions=outcome_analysis_annotations,
            )
            drop_withdrawals = DuckdbNode(
                fpath="drop_withdrawals_src",
                prefix="drw",
                pk=drw_id,
                date_key=drw_date,
                columns=drop_withdrawals_cols,
                feature_families=TEMPORAL_FEATURE_FAMILIES,
                ts_periods=[30, 90, 180, 365],
                categorical_top_k=5,
                auto_text_features=False,
                feature_family_max_columns=8,
            )
            reported_event_totals = DuckdbNode(
                fpath="reported_event_totals_src",
                prefix="evt",
                pk=evt_id,
                date_key=evt_date,
                columns=reported_event_totals_cols,
                feature_families=TEMPORAL_FEATURE_FAMILIES,
                ts_periods=[30, 90, 180, 365],
                categorical_top_k=5,
                auto_text_features=False,
                feature_family_max_columns=8,
                context_keys=(evt_nct_id,),
                annotation_expressions=reported_event_annotations,
            )
            designs = DuckdbNode(
                fpath="designs_src",
                prefix="dsg",
                pk=dsg_id,
                date_key=dsg_date,
                columns=designs_cols,
                feature_families=TEMPORAL_FEATURE_FAMILIES,
                ts_periods=[30, 90, 180, 365],
                categorical_top_k=5,
                auto_text_features=False,
                feature_family_max_columns=8,
            )
            eligibilities = DuckdbNode(
                fpath="eligibilities_src",
                prefix="eli",
                pk=eli_id,
                date_key=eli_date,
                columns=eligibilities_cols,
                feature_families=TEMPORAL_FEATURE_FAMILIES,
                ts_periods=[30, 90, 180, 365],
                categorical_top_k=5,
                auto_text_features=False,
                feature_family_max_columns=8,
            )
            interventions_studies = DuckdbNode(
                fpath="interventions_studies_src",
                prefix="ist",
                pk=intv_studies_id,
                date_key=intv_studies_date,
                columns=interventions_studies_cols,
                feature_families=TEMPORAL_FEATURE_FAMILIES,
                ts_periods=[30, 90, 180, 365],
                categorical_top_k=5,
                auto_text_features=False,
                feature_family_max_columns=8,
            )
            conditions_studies = DuckdbNode(
                fpath="conditions_studies_src",
                prefix="cst",
                pk=cond_studies_id,
                date_key=cond_studies_date,
                columns=conditions_studies_cols,
                feature_families=TEMPORAL_FEATURE_FAMILIES,
                ts_periods=[30, 90, 180, 365],
                categorical_top_k=5,
                auto_text_features=False,
                feature_family_max_columns=8,
            )
            facilities_studies = DuckdbNode(
                fpath="facilities_studies_src",
                prefix="fst",
                pk=fac_studies_id,
                date_key=fac_studies_date,
                columns=facilities_studies_cols,
                feature_families=TEMPORAL_FEATURE_FAMILIES,
                ts_periods=[30, 90, 180, 365],
                categorical_top_k=5,
                auto_text_features=False,
                feature_family_max_columns=8,
            )
            sponsors_studies = DuckdbNode(
                fpath="sponsors_studies_src",
                prefix="sst",
                pk=spn_studies_id,
                date_key=spn_studies_date,
                columns=sponsors_studies_cols,
                feature_families=TEMPORAL_FEATURE_FAMILIES,
                ts_periods=[30, 90, 180, 365],
                categorical_top_k=5,
                auto_text_features=False,
                feature_family_max_columns=8,
            )
            interventions = DuckdbNode(
                fpath="interventions_src",
                prefix="intv",
                pk=interventions_id,
                date_key=None,
                columns=interventions_cols,
                feature_families=("base",),
                categorical_top_k=5,
                auto_text_features=False,
            )
            conditions = DuckdbNode(
                fpath="conditions_src",
                prefix="cond",
                pk=conditions_id,
                date_key=None,
                columns=conditions_cols,
                feature_families=("base",),
                categorical_top_k=5,
                auto_text_features=False,
            )
            facilities = DuckdbNode(
                fpath="facilities_src",
                prefix="fac",
                pk=facilities_id,
                date_key=None,
                columns=facilities_cols,
                feature_families=("base",),
                categorical_top_k=5,
                auto_text_features=False,
            )
            sponsors = DuckdbNode(
                fpath="sponsors_src",
                prefix="spn",
                pk=sponsors_id,
                date_key=None,
                columns=sponsors_cols,
                feature_families=("base",),
                categorical_top_k=5,
                auto_text_features=False,
            )

            graph = GraphReduce(
                name=f"rel_trial_study_outcome_{cut_date.date()}",
                parent_node=studies,
                compute_layer=ComputeLayerEnum.duckdb,
                sql_client=con,
                cut_date=feature_cut_date,
                compute_period_val=(feature_cut_date - LOOKBACK_START).days + 1,
                compute_period_unit=PeriodUnit.day,
                auto_features=True,
                date_filters_on_agg=True,
                # Study outcome signal is available through the study's
                # direct event/bridge tables. Deeper walks repeatedly expand
                # the 1.8M-row facilities_studies bridge.
                auto_feature_hops_back=2,
                auto_feature_hops_front=0,
                use_temp_tables=True,
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
            for node in nodes:
                graph.add_node(node)

            graph.add_entity_edge(
                studies,
                outcomes,
                parent_key=studies_nct_id,
                relation_key=outcomes_nct_id,
                reduce=True,
            )
            graph.add_entity_edge(
                studies,
                outcome_analyses,
                parent_key=studies_nct_id,
                relation_key=oa_nct_id,
                reduce=True,
            )
            graph.add_entity_edge(
                studies,
                drop_withdrawals,
                parent_key=studies_nct_id,
                relation_key=drw_nct_id,
                reduce=True,
            )
            graph.add_entity_edge(
                studies,
                reported_event_totals,
                parent_key=studies_nct_id,
                relation_key=evt_nct_id,
                reduce=True,
            )
            graph.add_entity_edge(
                studies,
                designs,
                parent_key=studies_nct_id,
                relation_key=dsg_nct_id,
                reduce=True,
            )
            graph.add_entity_edge(
                studies,
                eligibilities,
                parent_key=studies_nct_id,
                relation_key=eli_nct_id,
                reduce=True,
            )
            graph.add_entity_edge(
                studies,
                interventions_studies,
                parent_key=studies_nct_id,
                relation_key=intv_studies_nct_id,
                reduce=True,
            )
            graph.add_entity_edge(
                studies,
                conditions_studies,
                parent_key=studies_nct_id,
                relation_key=cond_studies_nct_id,
                reduce=True,
            )
            graph.add_entity_edge(
                studies,
                facilities_studies,
                parent_key=studies_nct_id,
                relation_key=fac_studies_nct_id,
                reduce=True,
            )
            graph.add_entity_edge(
                studies,
                sponsors_studies,
                parent_key=studies_nct_id,
                relation_key=spn_studies_nct_id,
                reduce=True,
            )
            graph.add_entity_edge(
                interventions_studies,
                interventions,
                parent_key=intv_studies_intv_id,
                relation_key=interventions_id,
                reduce=True,
            )
            graph.add_entity_edge(
                conditions_studies,
                conditions,
                parent_key=cond_studies_cond_id,
                relation_key=conditions_id,
                reduce=True,
            )
            graph.add_entity_edge(
                facilities_studies,
                facilities,
                parent_key=fac_studies_fac_id,
                relation_key=facilities_id,
                reduce=True,
            )
            graph.add_entity_edge(
                sponsors_studies,
                sponsors,
                parent_key=spn_studies_spn_id,
                relation_key=sponsors_id,
                reduce=True,
            )

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
                left_on=["timestamp", f"std_{studies_nct_id}"],
                right_on=[task.time_col, task.entity_col],
                how="right",
                validate="one_to_one",
            )
            target = task.target_col
            frame[target] = frame[target].astype("int8")
            return frame_name, frame, target

        for frame_name, frame, target in iter_training_frames(con, frame_jobs, build_frame):
            frames_by_name[frame_name] = frame
    finally:
        con.close()

    target = split_tasks["train"].target_col
    train_frame_names = [
        frame_name for split_name, frame_name, *_ in frame_jobs if split_name == "train"
    ]
    val_frame_names = [
        frame_name for split_name, frame_name, *_ in frame_jobs if split_name == "val"
    ]
    test_frame_names = [
        frame_name for split_name, frame_name, *_ in frame_jobs if split_name == "test"
    ]
    df_train = pd.concat([frames_by_name[name] for name in train_frame_names], ignore_index=True)
    df_val = pd.concat([frames_by_name[name] for name in val_frame_names], ignore_index=True)
    df_test = pd.concat([frames_by_name[name] for name in test_frame_names], ignore_index=True)

    numeric_columns = [column for column in df_train.select_dtypes(include=[np.number]).columns if column != target]
    feature_columns = [
        column
        for column in numeric_columns
        if "label" not in column.lower()
        and not column.lower().endswith("_id")
        and column != "std_nct_id"
        and column in df_train.columns
        and column in df_val.columns
        and column in df_test.columns
    ]

    if not feature_columns:
        return df_train, df_val, df_test, None, None, 0, materialized, target

    X_train = df_train[feature_columns].fillna(0)
    y_train = df_train[target]
    X_val = df_val[feature_columns].fillna(0)
    y_val = df_val[target]
    if y_train.nunique() < 2 or y_val.nunique() < 2:
        return df_train, df_val, df_test, None, None, len(feature_columns), materialized, target

    model, best_config, best_val_auc = fit_tuned_classifier(
        X_train,
        y_train,
        X_val,
        y_val,
    )
    print("catboost_config:", best_config, flush=True)
    print("catboost_validation_auc:", best_val_auc, flush=True)
    print("catboost_best_iteration:", model.get_best_iteration(), flush=True)

    val_predictions = np.asarray(model.predict_proba(X_val)[:, 1], dtype="float64")
    val_metrics = split_tasks["val"].evaluate(
        val_predictions,
        target_table=target_table_from_frame(split_tasks["val"], df_val),
    )
    catboost_in_time_auc = float(val_metrics["roc_auc"])
    catboost_holdout_auc = None
    if df_test[target].nunique() >= 2:
        test_predictions = np.asarray(
            model.predict_proba(df_test[feature_columns].fillna(0))[:, 1],
            dtype="float64",
        )
        test_metrics = split_tasks["test"].evaluate(
            test_predictions,
            target_table=target_table_from_frame(split_tasks["test"], df_test),
        )
        catboost_holdout_auc = float(test_metrics["roc_auc"])

    return (
        df_train,
        df_val,
        df_test,
        catboost_in_time_auc,
        catboost_holdout_auc,
        len(feature_columns),
        materialized,
        target,
    )


def main() -> None:
    (
        df_train,
        df_val,
        df_test,
        catboost_in_time_auc,
        catboost_holdout_auc,
        n_features,
        materialized,
        target,
    ) = run_rel_trial_study_outcome()
    print("materialized_files:", materialized, flush=True)
    print("val_cut_date:", VAL_TIMESTAMP.date(), flush=True)
    print("test_cut_date:", TEST_TIMESTAMP.date(), flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
    print("label_period_days:", LABEL_DAYS, flush=True)
    print("target:", target, flush=True)
    print("train_rows:", len(df_train), flush=True)
    print("train_timestamps:", df_train["timestamp"].nunique(), flush=True)
    print("val_rows:", len(df_val), flush=True)
    print("val_timestamps:", df_val["timestamp"].nunique(), flush=True)
    print("val_columns:", len(df_val.columns), flush=True)
    print("test_rows:", len(df_test), flush=True)
    print("test_timestamps:", df_test["timestamp"].nunique(), flush=True)
    print("test_columns:", len(df_test.columns), flush=True)
    print("feature_count:", n_features, flush=True)
    print("catboost_in_time_auc:", catboost_in_time_auc if catboost_in_time_auc is not None else "skipped", flush=True)
    print("catboost_holdout_auc:", catboost_holdout_auc if catboost_holdout_auc is not None else "skipped", flush=True)


if __name__ == "__main__":
    main()
