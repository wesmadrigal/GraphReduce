#!/usr/bin/env python
"""RelBench rel-trial study-outcome example with DuckDB + GraphReduce."""

from __future__ import annotations

import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from relbench_dataset_utils import materialize_relbench_dataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from graphreduce.enum import ComputeLayerEnum, PeriodUnit, SQLOpType
from graphreduce.graph_reduce import GraphReduce
from graphreduce.models import sqlop
from graphreduce.node import DuckdbNode

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


def run_rel_trial_study_outcome(
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float | None, float | None, int, list[str], str]:
    use_dir = data_dir or Path("tests/data/relbench/rel-trial")
    materialized = materialize_relbench_dataset("rel-trial", use_dir, TABLE_NAME_TO_FILENAME)

    con = duckdb.connect()
    frames_by_name: dict[str, pd.DataFrame] = {}
    target_by_name: dict[str, str] = {}

    try:
        for table_name, filename in TABLE_NAME_TO_FILENAME.items():
            con.sql(f"CREATE OR REPLACE VIEW {table_name}_src AS SELECT * FROM read_parquet('{use_dir / filename}')")

        table_columns: dict[str, list[str]] = {}
        for table_name in TABLE_NAME_TO_FILENAME:
            table_columns[table_name] = con.sql(f"SELECT * FROM {table_name}_src LIMIT 0").to_df().columns.tolist()

        for frame_name, cut_date in {"val": VAL_TIMESTAMP, "test": TEST_TIMESTAMP}.items():
            feature_cut_date = cut_date + datetime.timedelta(days=1)
            studies_cols = table_columns["studies"]
            outcomes_cols = table_columns["outcomes"]
            outcome_analyses_cols = table_columns["outcome_analyses"]
            drop_withdrawals_cols = table_columns["drop_withdrawals"]
            reported_event_totals_cols = table_columns["reported_event_totals"]
            designs_cols = table_columns["designs"]
            eligibilities_cols = table_columns["eligibilities"]
            interventions_cols = table_columns["interventions"]
            conditions_cols = table_columns["conditions"]
            facilities_cols = table_columns["facilities"]
            sponsors_cols = table_columns["sponsors"]
            interventions_studies_cols = table_columns["interventions_studies"]
            conditions_studies_cols = table_columns["conditions_studies"]
            facilities_studies_cols = table_columns["facilities_studies"]
            sponsors_studies_cols = table_columns["sponsors_studies"]

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

            p_value_modifier_expr = "true"
            if oa_p_value_modifier:
                p_value_modifier_expr = (
                    f"(oa_{oa_p_value_modifier} is null or oa_{oa_p_value_modifier} != '>')"
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
                            f"case when {p_value_modifier_expr} "
                            f"and try_cast(oa_{oa_p_value} as double) >= 0 "
                            f"and try_cast(oa_{oa_p_value} as double) <= 1 "
                            "then 1 else 0 end as oa_valid_p_value"
                        ),
                    )
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
                auto_feature_hops_back=4,
                auto_feature_hops_front=0,
                use_temp_tables=True,
            )

            for node in [
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
            ]:
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
            frame = con.sql(f"SELECT * FROM {graph.parent_node._cur_data_ref}").to_df().copy()
            labels = con.sql(
                f"""
                WITH trial_info AS (
                    SELECT
                        oa.{oa_nct_id} AS nct_id,
                        oa.{oa_p_value} AS p_value,
                        s.{studies_start_date} AS start_date,
                        oa.{oa_date} AS date
                    FROM outcome_analyses_src oa
                    LEFT JOIN outcomes_src o
                        ON oa.{oa_outcome_id} = o.{outcomes_id}
                    LEFT JOIN studies_src s
                        ON s.{studies_nct_id} = o.{outcomes_nct_id}
                    WHERE ({'oa.' + oa_p_value_modifier + ' is null or oa.' + oa_p_value_modifier + " != '>'" if oa_p_value_modifier else 'true'})
                        AND oa.{oa_p_value} >= 0
                        AND oa.{oa_p_value} <= 1
                        AND o.outcome_type = 'Primary'
                )
                SELECT
                    TIMESTAMP '{cut_date}' AS timestamp,
                    tr.nct_id,
                    CASE WHEN MIN(tr.p_value) <= 0.05 THEN 1 ELSE 0 END AS outcome
                FROM trial_info tr
                WHERE tr.start_date <= TIMESTAMP '{cut_date}'
                    AND tr.date > TIMESTAMP '{cut_date}'
                    AND tr.date <= TIMESTAMP '{cut_date}' + INTERVAL '{LABEL_DAYS} days'
                    AND tr.nct_id IS NOT NULL
                GROUP BY tr.nct_id
                """
            ).to_df()
            frame = frame.merge(
                labels[["nct_id", "outcome"]],
                left_on=f"std_{studies_nct_id}",
                right_on="nct_id",
                how="inner",
            ).drop(columns=["nct_id"])
            target = "outcome"
            frame[target] = frame[target].astype("int8")
            frames_by_name[frame_name] = frame
            target_by_name[frame_name] = target
    finally:
        con.close()

    if target_by_name["val"] != target_by_name["test"]:
        raise ValueError(
            f"Target mismatch between val ({target_by_name['val']}) and test ({target_by_name['test']})"
        )

    target = target_by_name["val"]
    df_val = frames_by_name["val"]
    df_test = frames_by_name["test"]

    numeric_columns = [column for column in df_val.select_dtypes(include=[np.number]).columns if column != target]
    feature_columns = [
        column
        for column in numeric_columns
        if "label" not in column.lower()
        and not column.lower().endswith("_id")
        and column != "std_nct_id"
        and column in df_test.columns
    ]

    if not feature_columns:
        return df_val, df_test, None, None, 0, materialized, target

    X = df_val[feature_columns].fillna(0)
    y = df_val[target]
    if y.nunique() < 2:
        return df_val, df_test, None, None, len(feature_columns), materialized, target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

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

    catboost_in_time_auc = float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    catboost_holdout_auc = None
    if df_test[target].nunique() >= 2:
        catboost_holdout_auc = float(
            roc_auc_score(df_test[target], model.predict_proba(df_test[feature_columns].fillna(0))[:, 1])
        )

    return (
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
    print("val_rows:", len(df_val), flush=True)
    print("val_columns:", len(df_val.columns), flush=True)
    print("test_rows:", len(df_test), flush=True)
    print("test_columns:", len(df_test.columns), flush=True)
    print("feature_count:", n_features, flush=True)
    print("catboost_in_time_auc:", catboost_in_time_auc if catboost_in_time_auc is not None else "skipped", flush=True)
    print("catboost_holdout_auc:", catboost_holdout_auc if catboost_holdout_auc is not None else "skipped", flush=True)


if __name__ == "__main__":
    main()
