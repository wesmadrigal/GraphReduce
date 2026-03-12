#!/usr/bin/env python
"""Shared utilities for RelBench rel-avito user-level tasks."""

from __future__ import annotations

import datetime
from pathlib import Path
from urllib.request import urlretrieve

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

BASE_URL = "https://open-relbench.s3.us-east-1.amazonaws.com/rel-avito"
TABLES = [
    "AdsInfo.parquet",
    "Category.parquet",
    "Location.parquet",
    "PhoneRequestsStream.parquet",
    "SearchInfo.parquet",
    "UserInfo.parquet",
    "VisitsStream.parquet",
]

CUT_DATE = datetime.datetime(2015, 5, 14)
LOOKBACK_START = datetime.datetime(2015, 4, 25)
LOOKBACK_DAYS = (CUT_DATE - LOOKBACK_START).days + 1
LABEL_PERIOD_DAYS = 5  # 4-day task window with GraphReduce's strict-less-than boundary.


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


def download_rel_avito_data(data_dir: Path) -> list[str]:
    data_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    for table in TABLES:
        out_path = data_dir / table
        needs_download = not _is_valid_parquet(out_path)
        if needs_download:
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            if tmp_path.exists():
                tmp_path.unlink()
            urlretrieve(f"{BASE_URL}/{table}", tmp_path)
            tmp_path.replace(out_path)
            if not _is_valid_parquet(out_path):
                raise RuntimeError(f"Downloaded parquet is still invalid: {out_path}")
            downloaded.append(table)
    return downloaded


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


def _build_visits_click_annotate(
    vis_click_col: str | None,
    vis_user_col: str,
    vis_ad_col: str,
    vis_date_col: str | None,
    vis_search_col: str | None,
    sea_user_col: str | None,
    sea_ad_col: str | None,
    sea_date_col: str | None,
    sea_search_col: str | None,
    sea_click_col: str | None,
) -> str:
    if vis_click_col:
        return (
            "*, "
            "case when lower(cast(vis_"
            + vis_click_col
            + " as varchar)) in ('1','true','t','yes','y','clicked') "
            "then 1 else 0 end as vis_is_clicked"
        )

    join_terms: list[str] = []
    if vis_search_col and sea_search_col:
        join_terms.append(f"s.{sea_search_col} = vis_{vis_search_col}")
    if sea_user_col:
        join_terms.append(f"s.{sea_user_col} = vis_{vis_user_col}")
    if sea_ad_col:
        join_terms.append(f"s.{sea_ad_col} = vis_{vis_ad_col}")
    if not join_terms or not sea_click_col:
        # Fallback for schemas without explicit click signal.
        return "*, 1 as vis_is_clicked"

    if sea_date_col and vis_date_col:
        join_terms.append(f"s.{sea_date_col} <= vis_{vis_date_col}")

    return (
        "*, "
        + "case when exists ("
        + "select 1 from search_src s "
        + "where "
        + " and ".join(join_terms)
        + f" and coalesce(s.{sea_click_col}, 0) = 1"
        + ") then 1 else 0 end as vis_is_clicked"
    )


def _build_frame(data_dir: Path, mode: str) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        _prepare_view(con, "ads_src", data_dir / "AdsInfo.parquet")
        _prepare_view(con, "search_src", data_dir / "SearchInfo.parquet")
        _prepare_view(con, "user_src", data_dir / "UserInfo.parquet")
        _prepare_view(con, "visits_src", data_dir / "VisitsStream.parquet")

        ads_cols = _infer_columns(con, "ads_src")
        user_cols = _infer_columns(con, "user_src")
        visits_cols = _infer_columns(con, "visits_src")
        search_cols = _infer_columns(con, "search_src")

        user_id_col = _pick(user_cols, ["UserID", "UserId", "user_id", "userid"])
        vis_user_col = _pick(visits_cols, ["UserID", "UserId", "user_id", "userid"])
        vis_ad_col = _pick(visits_cols, ["AdID", "AdId", "ad_id", "adid"])
        vis_date_col = _pick(visits_cols, ["ViewDate", "VisitDate", "EventDate", "Date", "Timestamp", "t_dat"], required=False)
        vis_search_col = _pick(visits_cols, ["SearchID", "SearchId", "search_id", "searchid"], required=False)

        sea_user_col = _pick(search_cols, ["UserID", "UserId", "user_id", "userid"], required=False)
        sea_ad_col = _pick(search_cols, ["AdID", "AdId", "ad_id", "adid"], required=False)
        sea_date_col = _pick(search_cols, ["SearchDate", "EventDate", "Date", "Timestamp"], required=False)
        sea_search_col = _pick(search_cols, ["SearchID", "SearchId", "search_id", "searchid"], required=False)
        vis_click_col = _pick(visits_cols, ["IsClick", "is_click", "isclick", "Clicked", "clicked"], required=False)
        sea_click_col = _pick(search_cols, ["IsClick", "is_click", "isclick", "Clicked", "clicked"], required=False)

        ad_id_col = _pick(ads_cols, ["AdID", "AdId", "ad_id", "adid"])

        user_node = DuckdbNode(
            fpath="user_src",
            prefix="usr",
            pk=user_id_col,
            date_key=None,
            columns=user_cols,
            do_filters_ops=[
                sqlop(
                    optype=SQLOpType.where,
                    opval=(
                        "exists (select 1 from visits_src v "
                        f"where v.{vis_user_col} = usr_{user_id_col} "
                        f"and v.{vis_date_col} < '{CUT_DATE.date()}')"
                        if vis_date_col
                        else f"exists (select 1 from visits_src v where v.{vis_user_col} = usr_{user_id_col})"
                    ),
                )
            ],
        )

        visits_annotate = None
        if mode == "user_clicks":
            visits_annotate = [
                sqlop(
                    optype=SQLOpType.select,
                    opval=_build_visits_click_annotate(
                        vis_click_col=vis_click_col,
                        vis_user_col=vis_user_col,
                        vis_ad_col=vis_ad_col,
                        vis_date_col=vis_date_col,
                        vis_search_col=vis_search_col,
                        sea_user_col=sea_user_col,
                        sea_ad_col=sea_ad_col,
                        sea_date_col=sea_date_col,
                        sea_search_col=sea_search_col,
                        sea_click_col=sea_click_col,
                    ),
                )
            ]

        visits_labels = None
        if mode == "user_clicks":
            visits_labels = [
                sqlop(optype=SQLOpType.aggfunc, opval="sum(vis_is_clicked) as vis_clicked_visits_label"),
                sqlop(
                    optype=SQLOpType.aggfunc,
                    opval=f"count(distinct case when vis_is_clicked = 1 then vis_{vis_ad_col} end) as vis_clicked_ads_label",
                ),
                sqlop(optype=SQLOpType.agg, opval=f"vis_{vis_user_col}"),
            ]
        elif mode == "user_visits":
            visits_labels = [
                sqlop(optype=SQLOpType.aggfunc, opval=f"count(distinct vis_{vis_ad_col}) as vis_distinct_ads_label"),
                sqlop(optype=SQLOpType.agg, opval=f"vis_{vis_user_col}"),
            ]

        visits_node = DuckdbNode(
            fpath="visits_src",
            prefix="vis",
            pk=vis_ad_col,
            date_key=vis_date_col,
            columns=visits_cols,
            do_annotate_ops=visits_annotate,
            do_labels_ops=visits_labels,
        )

        ad_node = DuckdbNode(
            fpath="ads_src",
            prefix="ad",
            pk=ad_id_col,
            date_key=None,
            columns=ads_cols,
        )

        gr = GraphReduce(
            name=f"rel_avito_{mode}",
            parent_node=user_node,
            compute_layer=ComputeLayerEnum.duckdb,
            sql_client=con,
            cut_date=CUT_DATE,
            compute_period_val=LOOKBACK_DAYS,
            compute_period_unit=PeriodUnit.day,
            auto_features=True,
            auto_labels=True,
            date_filters_on_agg=True,
            label_node=visits_node,
            label_field=vis_ad_col,
            label_operation="count",
            label_period_val=LABEL_PERIOD_DAYS,
            label_period_unit=PeriodUnit.day,
            auto_feature_hops_back=3,
            auto_feature_hops_front=0,
            use_temp_tables=True,
        )

        for node in [user_node, visits_node, ad_node]:
            gr.add_node(node)

        gr.add_entity_edge(user_node, visits_node, parent_key=user_id_col, relation_key=vis_user_col, reduce=True)
        gr.add_entity_edge(visits_node, ad_node, parent_key=vis_ad_col, relation_key=ad_id_col, reduce=True)

        if sea_user_col:
            search_node = DuckdbNode(
                fpath="search_src",
                prefix="sea",
                pk=sea_search_col or sea_user_col,
                date_key=sea_date_col,
                columns=search_cols,
            )
            gr.add_node(search_node)
            gr.add_entity_edge(user_node, search_node, parent_key=user_id_col, relation_key=sea_user_col, reduce=True)

        gr.do_transformations_sql()
        out_df = con.sql(f"select * from {gr.parent_node._cur_data_ref}").to_df()
    finally:
        con.close()

    if mode == "user_clicks":
        label_cols = [c for c in out_df.columns if "clicked" in c.lower() and "label" in c.lower()]
        if not label_cols:
            raise ValueError("No clicked label columns found for user_clicks")
        for c in label_cols:
            out_df[c] = out_df[c].fillna(0)
        out_df["user_clicked_next_4d"] = (out_df[label_cols].sum(axis=1) > 0).astype("int8")
    else:
        label_col = "vis_distinct_ads_label"
        if label_col not in out_df.columns:
            candidates = [c for c in out_df.columns if "distinct" in c.lower() and "label" in c.lower()]
            if not candidates:
                raise ValueError("No distinct-visit label column found for user_visits")
            label_col = candidates[0]
        out_df[label_col] = out_df[label_col].fillna(0)
        out_df["user_multi_visit_next_4d"] = (out_df[label_col] > 1).astype("int8")

    return out_df


def _train_binary_model(df: pd.DataFrame, target: str) -> tuple[float | None, int]:
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    feature_cols = [
        c
        for c in numeric_cols
        if "label" not in c.lower() and not c.lower().endswith("_id") and "userid" not in c.lower()
    ]
    if not feature_cols:
        return None, 0

    X = df[feature_cols].fillna(0)
    y = df[target]
    if y.nunique() < 2:
        return None, len(feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = CatBoostClassifier(
        iterations=400,
        depth=8,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=50,
        allow_writing_files=False,
    )
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, preds))
    return auc, len(feature_cols)


def run_avito_task(mode: str, data_dir: Path | None = None) -> tuple[pd.DataFrame, float | None, int, list[str], str]:
    if mode not in {"user_clicks", "user_visits"}:
        raise ValueError("mode must be 'user_clicks' or 'user_visits'")

    use_dir = data_dir or Path("tests/data/relbench/rel-avito")
    downloaded = download_rel_avito_data(use_dir)
    df = _build_frame(use_dir, mode=mode)

    if mode == "user_clicks":
        target = "user_clicked_next_4d"
    else:
        target = "user_multi_visit_next_4d"

    auc, n_features = _train_binary_model(df, target=target)
    return df, auc, n_features, downloaded, target
