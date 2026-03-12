#!/usr/bin/env python
"""RelBench rel-amazon: item churn end-to-end example."""

from __future__ import annotations

import datetime
from pathlib import Path
from urllib.request import urlretrieve

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split

from graphreduce.enum import ComputeLayerEnum, PeriodUnit, SQLOpType
from graphreduce.graph_reduce import GraphReduce
from graphreduce.models import sqlop
from graphreduce.node import DuckdbNode

BASE_URL = "https://open-relbench.s3.us-east-1.amazonaws.com/rel-amazon"
TABLES = ["customer.parquet", "product.parquet", "review.parquet"]

VALIDATION_CUT_DATE = datetime.datetime(2015, 1, 1)
HOLDOUT_CUT_DATE = datetime.datetime(2016, 1, 1)
CUT_DATE = HOLDOUT_CUT_DATE
LOOKBACK_START = datetime.datetime(1996, 6, 25)
LOOKBACK_DAYS = (HOLDOUT_CUT_DATE - LOOKBACK_START).days + 1
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
    preds = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, preds))
    return auc, len(feature_cols)


def _train_regression(df: pd.DataFrame, target: str) -> tuple[float | None, int]:
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    feature_cols = [
        c
        for c in numeric_cols
        if "label" not in c.lower() and not c.lower().endswith("_id") and c.lower() not in {"customerid", "productid"}
    ]
    if not feature_cols:
        return None, 0

    X = df[feature_cols].fillna(0)
    y = df[target].fillna(0).astype("float64")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = CatBoostRegressor(
        iterations=700,
        depth=8,
        learning_rate=0.05,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=42,
        verbose=50,
        allow_writing_files=False,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    return mae, len(feature_cols)


def _build_frame(data_dir: Path, mode: str, cut_date: datetime.datetime) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        _prepare_view(con, "customer_src_raw", data_dir / "customer.parquet")
        _prepare_view(con, "product_src_raw", data_dir / "product.parquet")
        _prepare_view(con, "review_src_raw", data_dir / "review.parquet")

        customer_cols = _infer_columns(con, "customer_src_raw")
        product_cols = _infer_columns(con, "product_src_raw")
        review_cols = _infer_columns(con, "review_src_raw")

        cust_pk = _pick(customer_cols, ["customer_id", "CustomerID", "reviewerID", "user_id", "id"])
        prod_pk = _pick(product_cols, ["product_id", "ProductID", "asin", "item_id", "id"])
        rev_customer = _pick(review_cols, ["customer_id", "CustomerID", "reviewerID", "user_id", "UserID"])
        rev_product = _pick(review_cols, ["product_id", "ProductID", "asin", "item_id", "AdID"])
        rev_date = _pick(
            review_cols,
            ["review_time", "review_date", "ReviewTime", "timestamp", "date", "t_dat", "unixReviewTime"],
        )
        rev_pk = _pick(review_cols, ["review_id", "ReviewID", "id"], required=False) or rev_product
        review_amount_col = _pick(
            review_cols,
            ["price", "Price", "purchase_amount", "PurchaseAmount", "amount", "Amount", "total", "Total"],
            required=False,
        )
        product_amount_col = _pick(
            product_cols,
            ["price", "Price", "purchase_amount", "PurchaseAmount", "amount", "Amount", "total", "Total"],
            required=False,
        )

        con.sql("CREATE OR REPLACE VIEW customer_src AS SELECT * FROM customer_src_raw")
        if product_amount_col:
            con.sql(
                f"""
                CREATE OR REPLACE VIEW product_src AS
                SELECT *
                FROM product_src_raw
                WHERE TRY_CAST({product_amount_col} AS DOUBLE) IS NOT NULL
                """
            )
        else:
            con.sql("CREATE OR REPLACE VIEW product_src AS SELECT * FROM product_src_raw")
        amount_expr = "0.0"
        if review_amount_col and product_amount_col:
            amount_expr = (
                f"COALESCE(TRY_CAST(r.{review_amount_col} AS DOUBLE), TRY_CAST(p.{product_amount_col} AS DOUBLE), 0.0)"
            )
        elif review_amount_col:
            amount_expr = f"COALESCE(TRY_CAST(r.{review_amount_col} AS DOUBLE), 0.0)"
        elif product_amount_col:
            amount_expr = f"COALESCE(TRY_CAST(p.{product_amount_col} AS DOUBLE), 0.0)"
        con.sql(
            f"""
            CREATE OR REPLACE VIEW review_src AS
            SELECT
                r.*,
                {amount_expr} AS _gr_ltv_amount
            FROM review_src_raw r
            LEFT JOIN product_src_raw p
              ON r.{rev_product} = p.{prod_pk}
            """
        )
        review_cols = _infer_columns(con, "review_src")

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
                        f"and r.{rev_date} < '{cut_date.date()}')"
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
                        f"and r.{rev_date} < '{cut_date.date()}')"
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
            label_operation = "count"
        elif mode == "item_churn":
            parent_node = product_node
            label_field = rev_pk
            label_operation = "count"
        elif mode == "user_ltv":
            parent_node = customer_node
            label_field = "_gr_ltv_amount"
            label_operation = "sum"
        elif mode == "item_ltv":
            parent_node = product_node
            label_field = "_gr_ltv_amount"
            label_operation = "sum"
        else:
            raise ValueError("mode must be user_churn, item_churn, user_ltv, or item_ltv")

        lookback_days = (cut_date - LOOKBACK_START).days + 1

        gr = GraphReduce(
            name=f"rel_amazon_{mode}",
            parent_node=parent_node,
            compute_layer=ComputeLayerEnum.duckdb,
            sql_client=con,
            cut_date=cut_date,
            compute_period_val=lookback_days,
            compute_period_unit=PeriodUnit.day,
            auto_features=True,
            auto_labels=True,
            date_filters_on_agg=True,
            label_node=review_node,
            label_field=label_field,
            label_operation=label_operation,
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
        elif mode == "item_churn":
            out_df["item_has_review_next_90d"] = (out_df[label_cols].sum(axis=1) > 0).astype("int8")
        elif mode == "user_ltv":
            out_df["user_ltv_90d_usd"] = out_df[label_cols].sum(axis=1).astype("float64")
        else:
            out_df["item_ltv_90d_usd"] = out_df[label_cols].sum(axis=1).astype("float64")

        return out_df
    finally:
        con.close()


def run_amazon_task(
    mode: str,
    data_dir: Path | None = None,
    cut_date: datetime.datetime | None = None,
) -> tuple[pd.DataFrame, float | None, int, list[str], str]:
    use_dir = data_dir or Path("tests/data/relbench/rel-amazon")
    use_cut_date = cut_date or CUT_DATE
    downloaded = download_rel_amazon_data(use_dir)
    df = _build_frame(use_dir, mode=mode, cut_date=use_cut_date)

    if mode == "user_churn":
        target = "user_churn_90d"
        auc, n_features = _train_binary(df, target=target)
        return df, auc, n_features, downloaded, target
    if mode == "item_churn":
        target = "item_has_review_next_90d"
        auc, n_features = _train_binary(df, target=target)
        return df, auc, n_features, downloaded, target
    if mode == "user_ltv":
        target = "user_ltv_90d_usd"
        mae, n_features = _train_regression(df, target=target)
        return df, mae, n_features, downloaded, target
    if mode == "item_ltv":
        target = "item_ltv_90d_usd"
        mae, n_features = _train_regression(df, target=target)
        return df, mae, n_features, downloaded, target
    raise ValueError("mode must be user_churn, item_churn, user_ltv, or item_ltv")


def run_amazon_temporal_regression_task(
    mode: str,
    data_dir: Path | None = None,
    validation_cut_date: datetime.datetime = VALIDATION_CUT_DATE,
    holdout_cut_date: datetime.datetime = HOLDOUT_CUT_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, float | None, int, list[str], str]:
    if mode not in {"user_ltv", "item_ltv"}:
        raise ValueError("mode must be user_ltv or item_ltv")

    use_dir = data_dir or Path("tests/data/relbench/rel-amazon")
    downloaded = download_rel_amazon_data(use_dir)
    df_validation = _build_frame(use_dir, mode=mode, cut_date=validation_cut_date)
    df_holdout = _build_frame(use_dir, mode=mode, cut_date=holdout_cut_date)

    if mode == "user_ltv":
        target = "user_ltv_90d_usd"
    else:
        target = "item_ltv_90d_usd"

    numeric_cols = [c for c in df_validation.select_dtypes(include=[np.number]).columns if c != target]
    feature_cols = [
        c
        for c in numeric_cols
        if "label" not in c.lower() and not c.lower().endswith("_id") and c.lower() not in {"customerid", "productid"}
    ]
    feature_cols = [c for c in feature_cols if c in df_holdout.columns]
    if not feature_cols:
        return df_validation, df_holdout, None, 0, downloaded, target

    model = CatBoostRegressor(
        iterations=700,
        depth=8,
        learning_rate=0.05,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=42,
        verbose=50,
        allow_writing_files=False,
    )
    X_validation = df_validation[feature_cols].fillna(0)
    y_validation = df_validation[target].fillna(0).astype("float64")
    X_holdout = df_holdout[feature_cols].fillna(0)
    y_holdout = df_holdout[target].fillna(0).astype("float64")
    model.fit(X_validation, y_validation)
    holdout_preds = model.predict(X_holdout)
    holdout_mae = float(mean_absolute_error(y_holdout, holdout_preds))
    return df_validation, df_holdout, holdout_mae, len(feature_cols), downloaded, target


def main() -> None:
    df, auc, n_features, downloaded, target = run_amazon_task("item_churn")
    print("downloaded_files:", downloaded, flush=True)
    print("cut_date:", CUT_DATE.date(), flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
    print("lookback_days:", LOOKBACK_DAYS, flush=True)
    print("label_period_days:", LABEL_PERIOD_DAYS, flush=True)
    print("target:", target, flush=True)
    print("rows:", len(df), flush=True)
    print("columns:", len(df.columns), flush=True)
    print("feature_count:", n_features, flush=True)
    print("model_auc:", auc if auc is not None else "skipped", flush=True)


if __name__ == "__main__":
    main()
