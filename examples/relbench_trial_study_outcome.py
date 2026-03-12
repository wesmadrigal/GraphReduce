#!/usr/bin/env python
"""RelBench rel-trial study-outcome example with DuckDB + GraphReduce."""

from __future__ import annotations

import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from graphreduce.enum import ComputeLayerEnum, PeriodUnit, SQLOpType
from graphreduce.graph_reduce import GraphReduce
from graphreduce.models import sqlop
from graphreduce.node import DuckdbNode

TABLES = [
    "studies.parquet",
    "outcomes.parquet",
    "outcome_analyses.parquet",
    "drop_withdrawals.parquet",
    "reported_event_totals.parquet",
    "designs.parquet",
    "eligibilities.parquet",
    "interventions.parquet",
    "conditions.parquet",
    "facilities.parquet",
    "sponsors.parquet",
    "interventions_studies.parquet",
    "conditions_studies.parquet",
    "facilities_studies.parquet",
    "sponsors_studies.parquet",
]

VAL_TIMESTAMP = datetime.datetime(2020, 1, 1)
TEST_TIMESTAMP = datetime.datetime(2021, 1, 1)
LOOKBACK_START = datetime.datetime(2000, 1, 1)
LABEL_DAYS = 365


def _is_valid_parquet(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    con = duckdb.connect()
    try:
        con.sql(f"select 1 from read_parquet('{path}') limit 1").fetchall()
        return True
    except Exception:
        return False
    finally:
        con.close()


def ensure_local_rel_trial_data(data_dir: Path) -> list[str]:
    missing_or_invalid: list[str] = []
    for table in TABLES:
        out_path = data_dir / table
        if not _is_valid_parquet(out_path):
            missing_or_invalid.append(str(out_path))
    if missing_or_invalid:
        raise FileNotFoundError(
            "Missing/invalid local rel-trial parquet files. "
            "This example does not download data. "
            f"Expected local files under {data_dir}: {missing_or_invalid}"
        )
    return TABLES.copy()


def _prepare_view(con: duckdb.DuckDBPyConnection, view_name: str, parquet_path: Path) -> None:
    con.sql(f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet('{parquet_path}')")


def _infer_columns(con: duckdb.DuckDBPyConnection, view_name: str) -> list[str]:
    return con.sql(f"select * from {view_name} limit 0").to_df().columns.tolist()


def _pick(columns: list[str], candidates: list[str], required: bool = True) -> str | None:
    by_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in by_lower:
            return by_lower[cand.lower()]
    if required:
        raise ValueError(f"Could not find any of {candidates} in columns: {columns}")
    return None


def build_study_outcome_frame(
    con: duckdb.DuckDBPyConnection,
    data_dir: Path,
    cut_date: datetime.datetime,
) -> tuple[pd.DataFrame, str]:
    views = {
        "studies_src": "studies.parquet",
        "outcomes_src": "outcomes.parquet",
        "outcome_analyses_src": "outcome_analyses.parquet",
        "drop_withdrawals_src": "drop_withdrawals.parquet",
        "reported_event_totals_src": "reported_event_totals.parquet",
        "designs_src": "designs.parquet",
        "eligibilities_src": "eligibilities.parquet",
        "interventions_src": "interventions.parquet",
        "conditions_src": "conditions.parquet",
        "facilities_src": "facilities.parquet",
        "sponsors_src": "sponsors.parquet",
        "interventions_studies_src": "interventions_studies.parquet",
        "conditions_studies_src": "conditions_studies.parquet",
        "facilities_studies_src": "facilities_studies.parquet",
        "sponsors_studies_src": "sponsors_studies.parquet",
    }
    for view_name, table_name in views.items():
        _prepare_view(con, view_name, data_dir / table_name)

    studies_cols = _infer_columns(con, "studies_src")
    outcomes_cols = _infer_columns(con, "outcomes_src")
    outcome_analyses_cols = _infer_columns(con, "outcome_analyses_src")
    drop_withdrawals_cols = _infer_columns(con, "drop_withdrawals_src")
    reported_event_totals_cols = _infer_columns(con, "reported_event_totals_src")
    designs_cols = _infer_columns(con, "designs_src")
    eligibilities_cols = _infer_columns(con, "eligibilities_src")
    interventions_cols = _infer_columns(con, "interventions_src")
    conditions_cols = _infer_columns(con, "conditions_src")
    facilities_cols = _infer_columns(con, "facilities_src")
    sponsors_cols = _infer_columns(con, "sponsors_src")
    interventions_studies_cols = _infer_columns(con, "interventions_studies_src")
    conditions_studies_cols = _infer_columns(con, "conditions_studies_src")
    facilities_studies_cols = _infer_columns(con, "facilities_studies_src")
    sponsors_studies_cols = _infer_columns(con, "sponsors_studies_src")

    studies_nct_id = _pick(studies_cols, ["nct_id"])
    studies_start_date = _pick(studies_cols, ["start_date"])
    outcomes_id = _pick(outcomes_cols, ["id"])
    outcomes_nct_id = _pick(outcomes_cols, ["nct_id"])

    oa_id = _pick(outcome_analyses_cols, ["id"])
    oa_nct_id = _pick(outcome_analyses_cols, ["nct_id"])
    oa_outcome_id = _pick(outcome_analyses_cols, ["outcome_id"])
    oa_p_value = _pick(outcome_analyses_cols, ["p_value"])
    oa_p_value_modifier = _pick(outcome_analyses_cols, ["p_value_modifier"], required=False)
    oa_date = _pick(outcome_analyses_cols, ["date"])

    drw_id = _pick(drop_withdrawals_cols, ["id"])
    drw_nct_id = _pick(drop_withdrawals_cols, ["nct_id"])
    drw_date = _pick(drop_withdrawals_cols, ["date"])

    evt_id = _pick(reported_event_totals_cols, ["id"])
    evt_nct_id = _pick(reported_event_totals_cols, ["nct_id"])
    evt_date = _pick(reported_event_totals_cols, ["date"])

    dsg_id = _pick(designs_cols, ["id"])
    dsg_nct_id = _pick(designs_cols, ["nct_id"])
    dsg_date = _pick(designs_cols, ["date"])

    eli_id = _pick(eligibilities_cols, ["id"])
    eli_nct_id = _pick(eligibilities_cols, ["nct_id"])
    eli_date = _pick(eligibilities_cols, ["date"])

    intv_studies_id = _pick(interventions_studies_cols, ["id"])
    intv_studies_nct_id = _pick(interventions_studies_cols, ["nct_id"])
    intv_studies_intv_id = _pick(interventions_studies_cols, ["intervention_id"])
    intv_studies_date = _pick(interventions_studies_cols, ["date"])

    cond_studies_id = _pick(conditions_studies_cols, ["id"])
    cond_studies_nct_id = _pick(conditions_studies_cols, ["nct_id"])
    cond_studies_cond_id = _pick(conditions_studies_cols, ["condition_id"])
    cond_studies_date = _pick(conditions_studies_cols, ["date"])

    fac_studies_id = _pick(facilities_studies_cols, ["id"])
    fac_studies_nct_id = _pick(facilities_studies_cols, ["nct_id"])
    fac_studies_fac_id = _pick(facilities_studies_cols, ["facility_id"])
    fac_studies_date = _pick(facilities_studies_cols, ["date"])

    spn_studies_id = _pick(sponsors_studies_cols, ["id"])
    spn_studies_nct_id = _pick(sponsors_studies_cols, ["nct_id"])
    spn_studies_spn_id = _pick(sponsors_studies_cols, ["sponsor_id"])
    spn_studies_date = _pick(sponsors_studies_cols, ["date"])

    interventions_id = _pick(interventions_cols, ["intervention_id"])
    conditions_id = _pick(conditions_cols, ["condition_id"])
    facilities_id = _pick(facilities_cols, ["facility_id"])
    sponsors_id = _pick(sponsors_cols, ["sponsor_id"])

    pval_modifier_expr = "true"
    if oa_p_value_modifier:
        pval_modifier_expr = "(oa_" + oa_p_value_modifier + " is null or oa_" + oa_p_value_modifier + " != '>')"

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
    )

    outcomes = DuckdbNode(
        fpath="outcomes_src",
        prefix="out",
        pk=outcomes_id,
        date_key="date",
        columns=outcomes_cols,
    )

    outcome_analyses = DuckdbNode(
        fpath="outcome_analyses_src",
        prefix="oa",
        pk=oa_id,
        date_key=oa_date,
        columns=outcome_analyses_cols,
        do_annotate_ops=[
            sqlop(
                optype=SQLOpType.select,
                opval=(
                    "*, "
                    "case when exists ("
                    "select 1 from outcomes_src o "
                    f"where o.{outcomes_id} = oa_{oa_outcome_id} "
                    "and lower(cast(o.outcome_type as varchar)) = 'primary'"
                    ") then 1 else 0 end as oa_is_primary, "
                    f"case when {pval_modifier_expr} "
                    f"and try_cast(oa_{oa_p_value} as double) >= 0 "
                    f"and try_cast(oa_{oa_p_value} as double) <= 1 "
                    "then 1 else 0 end as oa_valid_p_value"
                ),
            )
        ],
        do_labels_ops=[
            sqlop(
                optype=SQLOpType.aggfunc,
                opval=(
                    f"max(case when oa_is_primary = 1 and oa_valid_p_value = 1 and "
                    f"try_cast(oa_{oa_p_value} as double) <= 0.05 then 1 else 0 end) "
                    "as oa_primary_outcome_success_label"
                ),
            ),
            sqlop(optype=SQLOpType.agg, opval=f"oa_{oa_nct_id}"),
        ],
    )

    drop_withdrawals = DuckdbNode(
        fpath="drop_withdrawals_src",
        prefix="drw",
        pk=drw_id,
        date_key=drw_date,
        columns=drop_withdrawals_cols,
    )

    reported_event_totals = DuckdbNode(
        fpath="reported_event_totals_src",
        prefix="evt",
        pk=evt_id,
        date_key=evt_date,
        columns=reported_event_totals_cols,
    )

    designs = DuckdbNode(
        fpath="designs_src",
        prefix="dsg",
        pk=dsg_id,
        date_key=dsg_date,
        columns=designs_cols,
    )

    eligibilities = DuckdbNode(
        fpath="eligibilities_src",
        prefix="eli",
        pk=eli_id,
        date_key=eli_date,
        columns=eligibilities_cols,
    )

    interventions_studies = DuckdbNode(
        fpath="interventions_studies_src",
        prefix="ist",
        pk=intv_studies_id,
        date_key=intv_studies_date,
        columns=interventions_studies_cols,
    )

    conditions_studies = DuckdbNode(
        fpath="conditions_studies_src",
        prefix="cst",
        pk=cond_studies_id,
        date_key=cond_studies_date,
        columns=conditions_studies_cols,
    )

    facilities_studies = DuckdbNode(
        fpath="facilities_studies_src",
        prefix="fst",
        pk=fac_studies_id,
        date_key=fac_studies_date,
        columns=facilities_studies_cols,
    )

    sponsors_studies = DuckdbNode(
        fpath="sponsors_studies_src",
        prefix="sst",
        pk=spn_studies_id,
        date_key=spn_studies_date,
        columns=sponsors_studies_cols,
    )

    interventions = DuckdbNode(
        fpath="interventions_src",
        prefix="intv",
        pk=interventions_id,
        date_key=None,
        columns=interventions_cols,
    )

    conditions = DuckdbNode(
        fpath="conditions_src",
        prefix="cond",
        pk=conditions_id,
        date_key=None,
        columns=conditions_cols,
    )

    facilities = DuckdbNode(
        fpath="facilities_src",
        prefix="fac",
        pk=facilities_id,
        date_key=None,
        columns=facilities_cols,
    )

    sponsors = DuckdbNode(
        fpath="sponsors_src",
        prefix="spn",
        pk=sponsors_id,
        date_key=None,
        columns=sponsors_cols,
    )

    lookback_days = (cut_date - LOOKBACK_START).days + 1
    gr = GraphReduce(
        name=f"rel_trial_study_outcome_{cut_date.date()}",
        parent_node=studies,
        compute_layer=ComputeLayerEnum.duckdb,
        sql_client=con,
        cut_date=cut_date,
        compute_period_val=lookback_days,
        compute_period_unit=PeriodUnit.day,
        auto_features=True,
        date_filters_on_agg=True,
        label_node=outcome_analyses,
        label_period_val=LABEL_DAYS,
        label_period_unit=PeriodUnit.day,
        auto_feature_hops_back=4,
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
        gr.add_node(node)

    gr.add_entity_edge(studies, outcomes, parent_key=studies_nct_id, relation_key=outcomes_nct_id, reduce=True)
    gr.add_entity_edge(studies, outcome_analyses, parent_key=studies_nct_id, relation_key=oa_nct_id, reduce=True)
    gr.add_entity_edge(studies, drop_withdrawals, parent_key=studies_nct_id, relation_key=drw_nct_id, reduce=True)
    gr.add_entity_edge(studies, reported_event_totals, parent_key=studies_nct_id, relation_key=evt_nct_id, reduce=True)
    gr.add_entity_edge(studies, designs, parent_key=studies_nct_id, relation_key=dsg_nct_id, reduce=True)
    gr.add_entity_edge(studies, eligibilities, parent_key=studies_nct_id, relation_key=eli_nct_id, reduce=True)
    gr.add_entity_edge(studies, interventions_studies, parent_key=studies_nct_id, relation_key=intv_studies_nct_id, reduce=True)
    gr.add_entity_edge(studies, conditions_studies, parent_key=studies_nct_id, relation_key=cond_studies_nct_id, reduce=True)
    gr.add_entity_edge(studies, facilities_studies, parent_key=studies_nct_id, relation_key=fac_studies_nct_id, reduce=True)
    gr.add_entity_edge(studies, sponsors_studies, parent_key=studies_nct_id, relation_key=spn_studies_nct_id, reduce=True)

    gr.add_entity_edge(interventions_studies, interventions, parent_key=intv_studies_intv_id, relation_key=interventions_id, reduce=True)
    gr.add_entity_edge(conditions_studies, conditions, parent_key=cond_studies_cond_id, relation_key=conditions_id, reduce=True)
    gr.add_entity_edge(facilities_studies, facilities, parent_key=fac_studies_fac_id, relation_key=facilities_id, reduce=True)
    gr.add_entity_edge(sponsors_studies, sponsors, parent_key=spn_studies_spn_id, relation_key=sponsors_id, reduce=True)

    gr.do_transformations_sql()
    df = con.sql(f"select * from {gr.parent_node._cur_data_ref}").to_df().copy()

    label_cols = [c for c in df.columns if c.startswith("oa_") and "label" in c.lower()]
    if not label_cols:
        raise ValueError("No study outcome label columns found.")
    target = label_cols[0]
    df[target] = df[target].fillna(0).astype("int8")

    return df, target


def train_study_outcome_model(
    df_train: pd.DataFrame,
    target: str,
    df_holdout: pd.DataFrame,
) -> tuple[float | None, float | None, int]:
    numeric_cols = [c for c in df_train.select_dtypes(include=[np.number]).columns if c != target]
    feature_cols = [
        c
        for c in numeric_cols
        if "label" not in c.lower() and not c.lower().endswith("_id") and c not in {"std_nct_id"}
    ]
    feature_cols = [c for c in feature_cols if c in df_holdout.columns]
    if not feature_cols:
        return None, None, 0

    X = df_train[feature_cols].fillna(0)
    y = df_train[target]
    if y.nunique() < 2:
        return None, None, len(feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

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
    model.fit(X_train, y_train)

    in_time_auc = float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    holdout_auc = None
    if df_holdout[target].nunique() >= 2:
        holdout_auc = float(roc_auc_score(df_holdout[target], model.predict_proba(df_holdout[feature_cols].fillna(0))[:, 1]))

    return in_time_auc, holdout_auc, len(feature_cols)


def run_rel_trial_study_outcome(
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float | None, float | None, int, list[str], str]:
    use_dir = data_dir or Path("tests/data/relbench/rel-trial")
    local_tables = ensure_local_rel_trial_data(use_dir)

    con = duckdb.connect()
    try:
        df_val, target = build_study_outcome_frame(con, use_dir, VAL_TIMESTAMP)
        df_test, target_test = build_study_outcome_frame(con, use_dir, TEST_TIMESTAMP)
    finally:
        con.close()

    if target != target_test:
        raise ValueError(f"Target mismatch between val ({target}) and test ({target_test})")

    in_time_auc, holdout_auc, n_features = train_study_outcome_model(df_val, target, df_test)
    return df_val, df_test, in_time_auc, holdout_auc, n_features, local_tables, target


def main() -> None:
    df_val, df_test, in_time_auc, holdout_auc, n_features, local_tables, target = run_rel_trial_study_outcome()
    print("local_tables_verified:", local_tables, flush=True)
    print("val_cut_date:", VAL_TIMESTAMP.date(), flush=True)
    print("test_cut_date:", TEST_TIMESTAMP.date(), flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
    print("label_period_days:", LABEL_DAYS, flush=True)
    print("target:", target, flush=True)
    print("val_rows:", len(df_val), flush=True)
    print("val_columns:", len(df_val.columns), flush=True)
    print("test_rows:", len(df_test), flush=True)
    print("test_columns:", len(df_test.columns), flush=True)
    print("feature_count:", n_features, flush=True)
    print("in_time_auc:", in_time_auc if in_time_auc is not None else "skipped", flush=True)
    print("holdout_auc:", holdout_auc if holdout_auc is not None else "skipped", flush=True)


if __name__ == "__main__":
    main()
