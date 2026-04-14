#!/usr/bin/env python
"""Notebook-friendly rel-avito user-visits GraphReduce example.

Copy/paste this file into a Jupyter notebook cell, then run:
    df = run_rel_avito_user_visits_graphreduce()
    df.head()
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from urllib.request import urlretrieve

import duckdb
import pandas as pd

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
    "SearchStream.parquet",
    "UserInfo.parquet",
    "VisitsStream.parquet",
]

CUT_DATE = datetime.datetime(2015, 5, 14)
LOOKBACK_START = datetime.datetime(2015, 4, 25)
LOOKBACK_DAYS = (CUT_DATE - LOOKBACK_START).days + 1
LABEL_PERIOD_DAYS = 5  # 4-day task horizon with GraphReduce strict-less-than boundary.


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
        if not _is_valid_parquet(out_path):
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            if tmp_path.exists():
                tmp_path.unlink()
            urlretrieve(f"{BASE_URL}/{table}", tmp_path)
            tmp_path.replace(out_path)
            if not _is_valid_parquet(out_path):
                raise RuntimeError(f"Downloaded parquet is invalid: {out_path}")
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


def _configure_duckdb_for_large_rel_avito(con: duckdb.DuckDBPyConnection) -> None:
    temp_dir = os.getenv("GRAPHREDUCE_DUCKDB_TEMP_DIR", "/tmp/duckdb_tmp")
    max_temp = os.getenv("GRAPHREDUCE_DUCKDB_MAX_TEMP_SIZE", "200GiB")
    memory_limit = os.getenv("GRAPHREDUCE_DUCKDB_MEMORY_LIMIT", "16GiB")
    threads = os.getenv("GRAPHREDUCE_DUCKDB_THREADS", "4")

    con.sql(f"SET temp_directory='{temp_dir}'")
    con.sql(f"SET max_temp_directory_size='{max_temp}'")
    con.sql(f"SET memory_limit='{memory_limit}'")
    con.sql(f"SET threads={threads}")
    con.sql("SET preserve_insertion_order=false")


def run_rel_avito_user_visits_graphreduce(data_dir: Path | None = None) -> pd.DataFrame:
    """Build rel-avito user-visits labels/features with GraphReduce and return a pandas DataFrame."""
    use_dir = data_dir or Path("tests/data/relbench/rel-avito")
    _ = download_rel_avito_data(use_dir)

    con = duckdb.connect()
    try:
        _configure_duckdb_for_large_rel_avito(con)
        _prepare_view(con, "ads_src", use_dir / "AdsInfo.parquet")
        _prepare_view(con, "search_info_src", use_dir / "SearchInfo.parquet")
        _prepare_view(con, "search_stream_src", use_dir / "SearchStream.parquet")
        _prepare_view(con, "user_src", use_dir / "UserInfo.parquet")
        _prepare_view(con, "visits_src", use_dir / "VisitsStream.parquet")
        _prepare_view(con, "phone_src", use_dir / "PhoneRequestsStream.parquet")

        ads_cols = _infer_columns(con, "ads_src")
        user_cols = _infer_columns(con, "user_src")
        visits_cols = _infer_columns(con, "visits_src")
        search_info_cols = _infer_columns(con, "search_info_src")
        search_stream_cols = _infer_columns(con, "search_stream_src")
        phone_cols = _infer_columns(con, "phone_src")

        user_id_col = _pick(user_cols, ["UserID", "UserId", "user_id", "userid"])
        vis_user_col = _pick(visits_cols, ["UserID", "UserId", "user_id", "userid"])
        vis_ad_col = _pick(visits_cols, ["AdID", "AdId", "ad_id", "adid"])
        vis_date_col = _pick(
            visits_cols,
            ["ViewDate", "VisitDate", "EventDate", "Date", "Timestamp", "t_dat"],
            required=False,
        )
        sinfo_search_id_col = _pick(search_info_cols, ["SearchID", "SearchId", "search_id", "searchid"])
        sinfo_user_col = _pick(search_info_cols, ["UserID", "UserId", "user_id", "userid"], required=False)
        sinfo_ad_col = _pick(search_info_cols, ["AdID", "AdId", "ad_id", "adid"], required=False)
        sinfo_date_col = _pick(search_info_cols, ["SearchDate", "EventDate", "Date", "Timestamp"], required=False)
        sstream_search_id_col = _pick(search_stream_cols, ["SearchID", "SearchId", "search_id", "searchid"])
        sstream_ad_col = _pick(search_stream_cols, ["AdID", "AdId", "ad_id", "adid"], required=False)
        sstream_date_col = _pick(search_stream_cols, ["SearchDate", "EventDate", "Date", "Timestamp"], required=False)
        phone_user_col = _pick(phone_cols, ["UserID", "UserId", "user_id", "userid"], required=False)
        phone_pk_col = _pick(
            phone_cols,
            ["PhoneRequestID", "RequestID", "Id", "ID", "AdID", "AdId", "ad_id"],
            required=False,
        ) or user_id_col
        phone_date_col = _pick(phone_cols, ["RequestDate", "Date", "EventDate", "Timestamp", "SearchDate"], required=False)
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

        visits_node = DuckdbNode(
            fpath="visits_src",
            prefix="vis",
            pk=vis_ad_col,
            date_key=vis_date_col,
            columns=visits_cols,
            do_labels_ops=[
                sqlop(optype=SQLOpType.aggfunc, opval=f"count(distinct vis_{vis_ad_col}) as vis_distinct_ads_label"),
                sqlop(optype=SQLOpType.agg, opval=f"vis_{vis_user_col}"),
            ],
        )

        search_info_node = DuckdbNode(
            fpath="search_info_src",
            prefix="sinf",
            pk=sinfo_search_id_col,
            date_key=sinfo_date_col,
            columns=search_info_cols,
        )
        search_stream_node = DuckdbNode(
            fpath="search_stream_src",
            prefix="sstr",
            pk=sstream_search_id_col,
            date_key=sstream_date_col,
            columns=search_stream_cols,
        )
        phone_node = DuckdbNode(
            fpath="phone_src",
            prefix="ph",
            pk=phone_pk_col,
            date_key=phone_date_col,
            columns=phone_cols,
        )
        ads_from_visits_node = DuckdbNode(
            fpath="ads_src",
            prefix="adv",
            pk=ad_id_col,
            date_key=None,
            columns=ads_cols,
        )
        ads_from_search_node = DuckdbNode(
            fpath="ads_src",
            prefix="ads",
            pk=ad_id_col,
            date_key=None,
            columns=ads_cols,
        )

        gr = GraphReduce(
            name="rel_avito_user_visits_jupyter",
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
            auto_feature_hops_back=4,
            auto_feature_hops_front=0,
            use_temp_tables=True,
        )

        for node in [
            user_node,
            visits_node,
            search_info_node,
            search_stream_node,
            phone_node,
            ads_from_visits_node,
            ads_from_search_node,
        ]:
            gr.add_node(node)

        gr.add_entity_edge(user_node, visits_node, parent_key=user_id_col, relation_key=vis_user_col, reduce=True)
        if sinfo_user_col:
            gr.add_entity_edge(user_node, search_info_node, parent_key=user_id_col, relation_key=sinfo_user_col, reduce=True)
        else:
            raise ValueError("SearchInfo must include UserID/UserId to connect to UserInfo for this task")
        gr.add_entity_edge(search_info_node, search_stream_node, parent_key=sinfo_search_id_col, relation_key=sstream_search_id_col, reduce=True)
        if phone_user_col:
            gr.add_entity_edge(user_node, phone_node, parent_key=user_id_col, relation_key=phone_user_col, reduce=True)
        gr.add_entity_edge(visits_node, ads_from_visits_node, parent_key=vis_ad_col, relation_key=ad_id_col, reduce=True)
        if sinfo_ad_col:
            gr.add_entity_edge(search_info_node, ads_from_search_node, parent_key=sinfo_ad_col, relation_key=ad_id_col, reduce=True)
        elif sstream_ad_col:
            gr.add_entity_edge(search_stream_node, ads_from_search_node, parent_key=sstream_ad_col, relation_key=ad_id_col, reduce=True)

        gr.do_transformations_sql()
        df = con.sql(f"select * from {gr.parent_node._cur_data_ref}").to_df()
    finally:
        con.close()

    label_col = "vis_distinct_ads_label"
    if label_col not in df.columns:
        candidates = [c for c in df.columns if "distinct" in c.lower() and "label" in c.lower()]
        if not candidates:
            raise ValueError("No distinct-visit label column found")
        label_col = candidates[0]
    df[label_col] = df[label_col].fillna(0)
    df["user_multi_visit_next_4d"] = (df[label_col] > 1).astype("int8")
    return df


if __name__ == "__main__":
    out = run_rel_avito_user_visits_graphreduce()
    print("cut_date:", CUT_DATE.date())
    print("lookback_start:", LOOKBACK_START.date())
    print("lookback_days:", LOOKBACK_DAYS)
    print("label_period_days:", LABEL_PERIOD_DAYS)
    print("rows:", len(out))
    print("columns:", len(out.columns))
    print("target_col:", "user_multi_visit_next_4d")
