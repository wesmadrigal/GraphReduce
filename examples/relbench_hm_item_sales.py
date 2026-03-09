#!/usr/bin/env python
"""RelBench rel-hm: item sales regression example with DuckDB + GraphReduce."""

from __future__ import annotations

import datetime
from pathlib import Path
from urllib.request import urlretrieve

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

from graphreduce.enum import ComputeLayerEnum, PeriodUnit, SQLOpType
from graphreduce.graph_reduce import GraphReduce
from graphreduce.models import sqlop
from graphreduce.node import DuckdbNode

BASE_URL = "https://open-relbench.s3.us-east-1.amazonaws.com/rel-hm"
TABLES = ["article.parquet", "customer.parquet", "transactions.parquet"]
LOOKBACK_START = datetime.datetime(2018, 9, 20)
EVAL_DATE = datetime.datetime(2020, 9, 7)
HOLDOUT_DATE = datetime.datetime(2020, 9, 14)
# GraphReduce label horizon is [cut_date, cut_date + period), so use 8 to include 7 full days.
LABEL_DAYS = 8


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


def _pick(columns: list[str], candidates: list[str], required: bool = True) -> str | None:
    by_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in by_lower:
            return by_lower[cand.lower()]
    if required:
        raise ValueError(f"Could not find any of {candidates} in columns: {columns}")
    return None


def build_item_sales_frame(con: duckdb.DuckDBPyConnection, data_dir: Path, cut_date: datetime.datetime) -> pd.DataFrame:
    _prepare_view(con, "article_src", data_dir / "article.parquet")
    _prepare_view(con, "customer_src", data_dir / "customer.parquet")
    _prepare_view(con, "transactions_src", data_dir / "transactions.parquet")

    article_columns = _infer_columns(con, "article_src")
    customer_columns = _infer_columns(con, "customer_src")
    transaction_columns = _infer_columns(con, "transactions_src")

    article_id_col = _pick(article_columns, ["article_id", "articleid", "id"])
    customer_id_col = _pick(customer_columns, ["customer_id", "customerid", "id"])
    tx_customer_col = _pick(transaction_columns, ["customer_id", "customerid"])
    tx_article_col = _pick(transaction_columns, ["article_id", "articleid"])
    tx_date_col = _pick(transaction_columns, ["t_dat", "date", "transaction_date", "timestamp"])
    tx_price_col = _pick(transaction_columns, ["price", "amount", "sales", "purchase_amount"])

    article = DuckdbNode(
        fpath="article_src",
        prefix="art",
        pk=article_id_col,
        date_key=None,
        columns=article_columns,
        do_filters_ops=[
            sqlop(
                optype=SQLOpType.where,
                opval=(
                    f"exists ("
                    f"select 1 from transactions_src tx "
                    f"where tx.{tx_article_col} = art_{article_id_col} "
                    f"and tx.{tx_date_col} < '{cut_date.date()}'"
                    f")"
                ),
            )
        ],
    )

    customer = DuckdbNode(
        fpath="customer_src",
        prefix="cust",
        pk=customer_id_col,
        date_key=None,
        columns=customer_columns,
    )

    transactions = DuckdbNode(
        fpath="transactions_src",
        prefix="txn",
        pk=tx_article_col,
        date_key=tx_date_col,
        columns=transaction_columns,
    )

    lookback_days = (cut_date - LOOKBACK_START).days

    gr = GraphReduce(
        name=f"rel_hm_item_sales_{cut_date.date()}",
        parent_node=article,
        compute_layer=ComputeLayerEnum.duckdb,
        sql_client=con,
        cut_date=cut_date,
        compute_period_val=lookback_days,
        compute_period_unit=PeriodUnit.day,
        auto_features=True,
        auto_labels=True,
        date_filters_on_agg=True,
        label_node=transactions,
        label_field=tx_price_col,
        label_operation="sum",
        label_period_val=LABEL_DAYS,
        label_period_unit=PeriodUnit.day,
        auto_feature_hops_back=3,
        auto_feature_hops_front=0,
    )

    for node in [article, customer, transactions]:
        gr.add_node(node)

    gr.add_entity_edge(article, transactions, parent_key=article_id_col, relation_key=tx_article_col, reduce=True)
    gr.add_entity_edge(customer, transactions, parent_key=customer_id_col, relation_key=tx_customer_col, reduce=True)

    gr.do_transformations_sql()
    out_df = con.sql(f"select * from {gr.parent_node._cur_data_ref}").to_df()

    label_cols = [c for c in out_df.columns if c.startswith("txn_") and "label" in c.lower()]
    if not label_cols:
        raise ValueError("No transaction label columns found in output dataframe.")

    for c in label_cols:
        out_df[c] = out_df[c].fillna(0)

    out_df["item_sales_7d_usd"] = out_df[label_cols].sum(axis=1).astype("float64")
    return out_df


def run_rel_hm_item_sales(
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float | None, int, list[str], str]:
    use_dir = data_dir or Path("tests/data/relbench/rel-hm")
    downloaded = download_rel_hm_data(use_dir)
    con = duckdb.connect()
    try:
        df_eval = build_item_sales_frame(con, use_dir, cut_date=EVAL_DATE)
        df_holdout = build_item_sales_frame(con, use_dir, cut_date=HOLDOUT_DATE)
    finally:
        con.close()

    target = "item_sales_7d_usd"
    numeric_cols = [c for c in df_eval.select_dtypes(include=[np.number]).columns if c != target]
    feature_cols = [
        c
        for c in numeric_cols
        if "label" not in c.lower() and not c.lower().endswith("_id") and c not in {"art_article_id", "txn_article_id"}
    ]
    feature_cols = [c for c in feature_cols if c in df_holdout.columns]

    if not feature_cols:
        return df_eval, df_holdout, None, 0, downloaded, target

    model = CatBoostRegressor(
        iterations=700,
        depth=8,
        learning_rate=0.05,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    X_eval = df_eval[feature_cols].fillna(0)
    y_eval = df_eval[target].fillna(0).astype("float64")
    X_holdout = df_holdout[feature_cols].fillna(0)
    y_holdout = df_holdout[target].fillna(0).astype("float64")
    model.fit(X_eval, y_eval)
    preds = model.predict(X_holdout)
    holdout_mae = float(mean_absolute_error(y_holdout, preds))
    return df_eval, df_holdout, holdout_mae, len(feature_cols), downloaded, target


def main() -> None:
    df_eval, df_holdout, holdout_mae, n_features, downloaded, target = run_rel_hm_item_sales()
    print("downloaded_files:", downloaded, flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
    print("eval_timestamp:", EVAL_DATE.date(), flush=True)
    print("holdout_timestamp:", HOLDOUT_DATE.date(), flush=True)
    print("eval_lookback_days:", (EVAL_DATE - LOOKBACK_START).days, flush=True)
    print("holdout_lookback_days:", (HOLDOUT_DATE - LOOKBACK_START).days, flush=True)
    print("label_period_days:", LABEL_DAYS, flush=True)
    print("target:", target, flush=True)
    print("eval_rows:", len(df_eval), flush=True)
    print("holdout_rows:", len(df_holdout), flush=True)
    print("columns:", len(df_eval.columns), flush=True)
    print("feature_count:", n_features, flush=True)
    print("holdout_mae:", holdout_mae if holdout_mae is not None else "skipped", flush=True)


if __name__ == "__main__":
    main()
