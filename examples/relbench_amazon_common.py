#!/usr/bin/env python
"""Shared utilities for RelBench rel-amazon churn tasks."""

from __future__ import annotations

import datetime
from pathlib import Path
from urllib.request import urlretrieve

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from graphreduce.enum import ComputeLayerEnum, PeriodUnit, SQLOpType
from graphreduce.graph_reduce import GraphReduce
from graphreduce.models import sqlop
from graphreduce.node import DuckdbNode

BASE_URL = "https://open-relbench.s3.us-east-1.amazonaws.com/rel-amazon"
TABLES = ["customer.parquet", "product.parquet", "review.parquet"]

CUT_DATE = datetime.datetime(2016, 1, 1)
LOOKBACK_START = datetime.datetime(1996, 6, 25)
LOOKBACK_DAYS = (CUT_DATE - LOOKBACK_START).days + 1
LABEL_PERIOD_DAYS = 90


def download_rel_amazon_data(data_dir: Path) -> list[str]:
    data_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    for table in TABLES:
        out_path = data_dir / table
        if not out_path.exists():
            urlretrieve(f"{BASE_URL}/{table}", out_path)
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


def _train_binary(df: pd.DataFrame, target: str) -> tuple[float | None, int]:
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    feature_cols = [
        c
        for c in numeric_cols
        if "label" not in c.lower() and not c.lower().endswith("_id") and c.lower() not in {"customerid", "productid"}
    ]
    if not feature_cols:
        return None, 0

    X = df[feature_cols].fillna(0)
    y = df[target]
    if y.nunique() < 2:
        return None, len(feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=400, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, preds))
    return auc, len(feature_cols)


def _build_frame(data_dir: Path, mode: str) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        _prepare_view(con, "customer_src", data_dir / "customer.parquet")
        _prepare_view(con, "product_src", data_dir / "product.parquet")
        _prepare_view(con, "review_src", data_dir / "review.parquet")

        customer_cols = _infer_columns(con, "customer_src")
        product_cols = _infer_columns(con, "product_src")
        review_cols = _infer_columns(con, "review_src")

        cust_pk = _pick(customer_cols, ["customer_id", "CustomerID", "reviewerID", "user_id", "id"])
        prod_pk = _pick(product_cols, ["product_id", "ProductID", "asin", "item_id", "id"])
        rev_customer = _pick(review_cols, ["customer_id", "CustomerID", "reviewerID", "user_id", "UserID"])
        rev_product = _pick(review_cols, ["product_id", "ProductID", "asin", "item_id", "AdID"])
        rev_date = _pick(
            review_cols,
            ["review_time", "review_date", "ReviewTime", "timestamp", "date", "t_dat", "unixReviewTime"],
        )
        rev_pk = _pick(review_cols, ["review_id", "ReviewID", "id"], required=False) or rev_product

        customer_node = DuckdbNode(
            fpath="customer_src",
            prefix="cust",
            pk=cust_pk,
            date_key=None,
            columns=customer_cols,
            do_filters_ops=[
                sqlop(
                    optype=SQLOpType.where,
                    opval=(
                        "exists (select 1 from review_src r "
                        f"where r.{rev_customer} = cust_{cust_pk} "
                        f"and r.{rev_date} < '{CUT_DATE.date()}')"
                    ),
                )
            ],
        )

        product_node = DuckdbNode(
            fpath="product_src",
            prefix="prod",
            pk=prod_pk,
            date_key=None,
            columns=product_cols,
            do_filters_ops=[
                sqlop(
                    optype=SQLOpType.where,
                    opval=(
                        "exists (select 1 from review_src r "
                        f"where r.{rev_product} = prod_{prod_pk} "
                        f"and r.{rev_date} < '{CUT_DATE.date()}')"
                    ),
                )
            ],
        )

        review_node = DuckdbNode(
            fpath="review_src",
            prefix="rev",
            pk=rev_pk,
            date_key=rev_date,
            columns=review_cols,
        )

        if mode == "user_churn":
            parent_node = customer_node
            label_field = rev_pk
        elif mode == "item_churn":
            parent_node = product_node
            label_field = rev_pk
        else:
            raise ValueError("mode must be user_churn or item_churn")

        gr = GraphReduce(
            name=f"rel_amazon_{mode}",
            parent_node=parent_node,
            compute_layer=ComputeLayerEnum.duckdb,
            sql_client=con,
            cut_date=CUT_DATE,
            compute_period_val=LOOKBACK_DAYS,
            compute_period_unit=PeriodUnit.day,
            auto_features=True,
            auto_labels=True,
            date_filters_on_agg=True,
            label_node=review_node,
            label_field=label_field,
            label_operation="count",
            label_period_val=LABEL_PERIOD_DAYS,
            label_period_unit=PeriodUnit.day,
            auto_feature_hops_back=3,
            auto_feature_hops_front=0,
            use_temp_tables=True,
        )

        for node in [customer_node, product_node, review_node]:
            gr.add_node(node)

        gr.add_entity_edge(customer_node, review_node, parent_key=cust_pk, relation_key=rev_customer, reduce=True)
        gr.add_entity_edge(product_node, review_node, parent_key=prod_pk, relation_key=rev_product, reduce=True)

        gr.do_transformations_sql()
        out_df = con.sql(f"select * from {gr.parent_node._cur_data_ref}").to_df()

        label_cols = [c for c in out_df.columns if c.startswith("rev_") and "label" in c.lower()]
        if not label_cols:
            raise ValueError("No review label columns found in output.")
        for c in label_cols:
            out_df[c] = out_df[c].fillna(0)

        if mode == "user_churn":
            out_df["user_churn_90d"] = (out_df[label_cols].sum(axis=1) == 0).astype("int8")
        else:
            out_df["item_has_review_next_90d"] = (out_df[label_cols].sum(axis=1) > 0).astype("int8")

        return out_df
    finally:
        con.close()


def run_amazon_task(mode: str, data_dir: Path | None = None) -> tuple[pd.DataFrame, float | None, int, list[str], str]:
    use_dir = data_dir or Path("tests/data/relbench/rel-amazon")
    downloaded = download_rel_amazon_data(use_dir)
    df = _build_frame(use_dir, mode=mode)

    if mode == "user_churn":
        target = "user_churn_90d"
    else:
        target = "item_has_review_next_90d"

    auc, n_features = _train_binary(df, target=target)
    return df, auc, n_features, downloaded, target
