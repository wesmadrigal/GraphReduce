#!/usr/bin/env python
"""RelBench rel-avito: ad CTR regression example."""

from __future__ import annotations

import datetime
from pathlib import Path
from urllib.request import urlretrieve

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
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
    "SearchStream.parquet",
    "UserInfo.parquet",
    "VisitsStream.parquet",
]

CUT_DATE = datetime.datetime(2015, 5, 14)
LOOKBACK_START = datetime.datetime(2015, 4, 25)
LOOKBACK_DAYS = (CUT_DATE - LOOKBACK_START).days + 1
LABEL_PERIOD_DAYS = 5


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
        if _is_valid_parquet(out_path):
            continue
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


def build_ad_ctr_frame(con: duckdb.DuckDBPyConnection, data_dir: Path) -> tuple[pd.DataFrame, str]:
    _prepare_view(con, "ads_src", data_dir / "AdsInfo.parquet")
    _prepare_view(con, "search_stream_src", data_dir / "SearchStream.parquet")
    _prepare_view(con, "search_info_src", data_dir / "SearchInfo.parquet")
    _prepare_view(con, "visits_src", data_dir / "VisitsStream.parquet")
    _prepare_view(con, "category_src", data_dir / "Category.parquet")
    _prepare_view(con, "location_src", data_dir / "Location.parquet")

    ads_cols = _infer_columns(con, "ads_src")
    search_stream_cols = _infer_columns(con, "search_stream_src")
    search_info_cols = _infer_columns(con, "search_info_src")
    visits_cols = _infer_columns(con, "visits_src")
    category_cols = _infer_columns(con, "category_src")
    location_cols = _infer_columns(con, "location_src")

    ads_id_col = _pick(ads_cols, ["AdID", "AdId", "ad_id", "adid"])

    sstr_search_id_col = _pick(search_stream_cols, ["SearchID", "SearchId", "search_id", "searchid"])
    sstr_ad_col = _pick(search_stream_cols, ["AdID", "AdId", "ad_id", "adid"])
    sstr_click_col = _pick(search_stream_cols, ["IsClick", "is_click", "isclick", "Clicked", "clicked"])
    sstr_date_col = _pick(search_stream_cols, ["SearchDate", "EventDate", "Date", "Timestamp"], required=False)

    sinf_search_id_col = _pick(search_info_cols, ["SearchID", "SearchId", "search_id", "searchid"], required=False)
    sinf_ad_col = _pick(search_info_cols, ["AdID", "AdId", "ad_id", "adid"], required=False)
    sinf_date_col = _pick(search_info_cols, ["SearchDate", "EventDate", "Date", "Timestamp"], required=False)

    vis_ad_col = _pick(visits_cols, ["AdID", "AdId", "ad_id", "adid"], required=False)
    vis_date_col = _pick(visits_cols, ["ViewDate", "VisitDate", "EventDate", "Date", "Timestamp", "t_dat"], required=False)

    cat_id_col = _pick(category_cols, ["CategoryID", "CategoryId", "category_id", "cat_id", "id"], required=False)
    loc_id_col = _pick(location_cols, ["LocationID", "LocationId", "location_id", "loc_id", "id"], required=False)
    ads_cat_col = _pick(ads_cols, ["CategoryID", "CategoryId", "category_id"], required=False)
    ads_loc_col = _pick(ads_cols, ["LocationID", "LocationId", "location_id"], required=False)

    ads_node = DuckdbNode(
        fpath="ads_src",
        prefix="ad",
        pk=ads_id_col,
        date_key=None,
        columns=ads_cols,
        do_filters_ops=[sqlop(optype=SQLOpType.where, opval=f"ad_{ads_id_col} is not null")],
    )

    search_stream_node = DuckdbNode(
        fpath="search_stream_src",
        prefix="sstr",
        pk=sstr_search_id_col,
        date_key=sstr_date_col,
        columns=search_stream_cols,
        do_labels_ops=[
            sqlop(
                optype=SQLOpType.aggfunc,
                opval=(
                    f"sum(case when coalesce(try_cast(sstr_{sstr_click_col} as double), 0) > 0 then 1 else 0 end) "
                    "/ nullif(count(*), 0)::double as sstr_ctr_label"
                ),
            ),
            sqlop(optype=SQLOpType.aggfunc, opval="count(*) as sstr_impressions_label"),
            sqlop(optype=SQLOpType.agg, opval=f"sstr_{sstr_ad_col}"),
        ],
    )

    search_info_node = DuckdbNode(
        fpath="search_info_src",
        prefix="sinf",
        pk=sinf_search_id_col or ads_id_col,
        date_key=sinf_date_col,
        columns=search_info_cols,
    )

    visits_node = DuckdbNode(
        fpath="visits_src",
        prefix="vis",
        pk=vis_ad_col or ads_id_col,
        date_key=vis_date_col,
        columns=visits_cols,
    )

    category_node = DuckdbNode(
        fpath="category_src",
        prefix="cat",
        pk=cat_id_col or ads_id_col,
        date_key=None,
        columns=category_cols,
    )

    location_node = DuckdbNode(
        fpath="location_src",
        prefix="loc",
        pk=loc_id_col or ads_id_col,
        date_key=None,
        columns=location_cols,
    )

    gr = GraphReduce(
        name="rel_avito_ad_ctr",
        parent_node=ads_node,
        compute_layer=ComputeLayerEnum.duckdb,
        sql_client=con,
        cut_date=CUT_DATE,
        compute_period_val=LOOKBACK_DAYS,
        compute_period_unit=PeriodUnit.day,
        auto_features=True,
        date_filters_on_agg=True,
        label_node=search_stream_node,
        label_period_val=LABEL_PERIOD_DAYS,
        label_period_unit=PeriodUnit.day,
        auto_feature_hops_back=4,
        auto_feature_hops_front=0,
        use_temp_tables=True,
    )

    for node in [ads_node, search_stream_node, search_info_node, visits_node, category_node, location_node]:
        gr.add_node(node)

    gr.add_entity_edge(ads_node, search_stream_node, parent_key=ads_id_col, relation_key=sstr_ad_col, reduce=True)

    if sinf_ad_col:
        gr.add_entity_edge(ads_node, search_info_node, parent_key=ads_id_col, relation_key=sinf_ad_col, reduce=True)
    if sinf_search_id_col:
        gr.add_entity_edge(search_info_node, search_stream_node, parent_key=sinf_search_id_col, relation_key=sstr_search_id_col, reduce=True)
    if vis_ad_col:
        gr.add_entity_edge(ads_node, visits_node, parent_key=ads_id_col, relation_key=vis_ad_col, reduce=True)
    if ads_cat_col and cat_id_col:
        gr.add_entity_edge(ads_node, category_node, parent_key=ads_cat_col, relation_key=cat_id_col, reduce=True)
    if ads_loc_col and loc_id_col:
        gr.add_entity_edge(ads_node, location_node, parent_key=ads_loc_col, relation_key=loc_id_col, reduce=True)

    gr.do_transformations_sql()
    out_df = con.sql(f"select * from {gr.parent_node._cur_data_ref}").to_df().copy()

    target = "sstr_ctr_label"
    if target not in out_df.columns:
        candidates = [c for c in out_df.columns if "ctr" in c.lower() and "label" in c.lower()]
        if not candidates:
            raise ValueError("No CTR label column found")
        target = candidates[0]

    out_df[target] = out_df[target].fillna(0.0).astype("float64")
    return out_df, target


def train_ad_ctr_model(df: pd.DataFrame, target: str) -> tuple[float | None, int]:
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    feature_cols = [
        c
        for c in numeric_cols
        if "label" not in c.lower() and not c.lower().endswith("_id") and c not in {"ad_AdID", "ad_AdId", "ad_ad_id"}
    ]
    if not feature_cols:
        return None, 0

    X = df[feature_cols].fillna(0)
    y = df[target]
    if y.nunique() <= 1:
        return None, len(feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = CatBoostRegressor(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    return mae, len(feature_cols)


def run_rel_avito_ad_ctr(data_dir: Path | None = None) -> tuple[pd.DataFrame, float | None, int, list[str], str]:
    use_dir = data_dir or Path("tests/data/relbench/rel-avito")
    downloaded = download_rel_avito_data(use_dir)
    con = duckdb.connect()
    try:
        df, target = build_ad_ctr_frame(con, use_dir)
    finally:
        con.close()

    mae, n_features = train_ad_ctr_model(df, target)
    return df, mae, n_features, downloaded, target


def main() -> None:
    df, mae, n_features, downloaded, target = run_rel_avito_ad_ctr()
    print("downloaded_files:", downloaded, flush=True)
    print("cut_date:", CUT_DATE.date(), flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
    print("lookback_days:", LOOKBACK_DAYS, flush=True)
    print("label_period_days:", LABEL_PERIOD_DAYS, flush=True)
    print("target:", target, flush=True)
    print("rows:", len(df), flush=True)
    print("columns:", len(df.columns), flush=True)
    print("feature_count:", n_features, flush=True)
    print("model_mae:", mae if mae is not None else "skipped", flush=True)


if __name__ == "__main__":
    main()
