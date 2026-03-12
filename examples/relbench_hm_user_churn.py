#!/usr/bin/env python
"""RelBench rel-hm user churn example with DuckDB + GraphReduce."""

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

BASE_URL = "https://open-relbench.s3.us-east-1.amazonaws.com/rel-hm"
TABLES = ["article.parquet", "customer.parquet", "transactions.parquet"]
CUT_DATE = datetime.datetime(2020, 9, 14)
LOOKBACK_START = datetime.datetime(2018, 9, 20)
LOOKBACK_DAYS = (CUT_DATE - LOOKBACK_START).days
LABEL_DAYS = 7


def download_rel_hm_data(data_dir: Path) -> list[str]:
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


def build_user_churn_frame(con: duckdb.DuckDBPyConnection, data_dir: Path) -> pd.DataFrame:
    _prepare_view(con, "article_src", data_dir / "article.parquet")
    _prepare_view(con, "customer_src", data_dir / "customer.parquet")
    _prepare_view(con, "transactions_src", data_dir / "transactions.parquet")

    article_columns = _infer_columns(con, "article_src")
    customer_columns = _infer_columns(con, "customer_src")
    transaction_columns = _infer_columns(con, "transactions_src")

    article_id_col = "article_id"
    customer_id_col = "customer_id"
    tx_date_col = "t_dat"

    customer = DuckdbNode(
        fpath="customer_src",
        prefix="cust",
        pk=customer_id_col,
        date_key=None,
        columns=customer_columns,
        do_filters_ops=[
            sqlop(
                optype=SQLOpType.where,
                opval=(
                    f"exists ("
                    f"select 1 from transactions_src tx "
                    f"where tx.{customer_id_col} = cust_{customer_id_col} "
                    f"and tx.{tx_date_col} < '{CUT_DATE.date()}'"
                    f")"
                ),
            )
        ],
    )

    article = DuckdbNode(
        fpath="article_src",
        prefix="art",
        pk=article_id_col,
        date_key=None,
        columns=article_columns,
    )

    transactions = DuckdbNode(
        fpath="transactions_src",
        prefix="txn",
        pk=article_id_col,
        date_key=tx_date_col,
        columns=transaction_columns,
    )

    gr = GraphReduce(
        name="rel_hm_user_churn",
        parent_node=customer,
        compute_layer=ComputeLayerEnum.duckdb,
        sql_client=con,
        cut_date=CUT_DATE,
        compute_period_val=LOOKBACK_DAYS,
        compute_period_unit=PeriodUnit.day,
        auto_features=True,
        auto_labels=True,
        date_filters_on_agg=True,
        label_node=transactions,
        label_field=article_id_col,
        label_operation="count",
        label_period_val=LABEL_DAYS,
        label_period_unit=PeriodUnit.day,
        auto_feature_hops_back=3,
        auto_feature_hops_front=0,
    )

    for node in [customer, article, transactions]:
        gr.add_node(node)

    gr.add_entity_edge(customer, transactions, parent_key=customer_id_col, relation_key=customer_id_col, reduce=True)
    gr.add_entity_edge(transactions, article, parent_key=article_id_col, relation_key=article_id_col, reduce=True)

    gr.do_transformations_sql()
    out_df = con.sql(f"select * from {gr.parent_node._cur_data_ref}").to_df()

    label_cols = [c for c in out_df.columns if c.startswith("txn_") and "label" in c.lower()]
    if not label_cols:
        raise ValueError("No transaction label columns found in output dataframe.")

    for c in label_cols:
        out_df[c] = out_df[c].fillna(0)

    out_df["user_churn_7d"] = (out_df[label_cols].sum(axis=1) == 0).astype("int8")
    return out_df


def train_user_churn_model(df: pd.DataFrame) -> tuple[float | None, int]:
    target = "user_churn_7d"
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    feature_cols = [
        c
        for c in numeric_cols
        if "label" not in c.lower() and not c.lower().endswith("_id") and c not in {"cust_customer_id", "txn_article_id"}
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


def run_rel_hm_user_churn(data_dir: Path | None = None) -> tuple[pd.DataFrame, float | None, int, list[str]]:
    use_dir = data_dir or Path("tests/data/relbench/rel-hm")
    downloaded = download_rel_hm_data(use_dir)
    con = duckdb.connect()
    try:
        df = build_user_churn_frame(con, use_dir)
    finally:
        con.close()
    auc, n_features = train_user_churn_model(df)
    return df, auc, n_features, downloaded


def main() -> None:
    df, auc, n_features, downloaded = run_rel_hm_user_churn()
    print("downloaded_files:", downloaded, flush=True)
    print("cut_date:", CUT_DATE.date(), flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
    print("lookback_days:", LOOKBACK_DAYS, flush=True)
    print("label_period_days:", LABEL_DAYS, flush=True)
    print("rows:", len(df), flush=True)
    print("columns:", len(df.columns), flush=True)
    print("feature_count:", n_features, flush=True)
    print("model_auc:", auc if auc is not None else "skipped", flush=True)


if __name__ == "__main__":
    main()
