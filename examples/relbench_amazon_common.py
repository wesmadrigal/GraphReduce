#!/usr/bin/env python
"""Shared utilities for RelBench rel-amazon churn and LTV tasks."""

from __future__ import annotations

import datetime
from pathlib import Path

import duckdb
import pandas as pd

from graphreduce.enum import ComputeLayerEnum, PeriodUnit, SQLOpType
from graphreduce.graph_reduce import GraphReduce
from graphreduce.models import sqlop
from graphreduce.node import DuckdbNode
from relbench_dataset_utils import get_relbench_dataset_db, register_relbench_db_views

TABLE_NAME_TO_FILENAME = {
    "customer": "customer.parquet",
    "product": "product.parquet",
    "review": "review.parquet",
}

VALIDATION_CUT_DATE = datetime.datetime(2015, 10, 1)
HOLDOUT_CUT_DATE = datetime.datetime(2016, 1, 1)
CUT_DATE = HOLDOUT_CUT_DATE
LOOKBACK_START = datetime.datetime(2008, 1, 1)
LOOKBACK_DAYS = (HOLDOUT_CUT_DATE - LOOKBACK_START).days + 1
LABEL_PERIOD_DAYS = 90


def materialize_rel_amazon_data(data_dir: Path) -> list[str]:
    return []


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


def _build_frame(mode: str, cut_date: datetime.datetime) -> pd.DataFrame:
    _, db = get_relbench_dataset_db("rel-amazon", download=True, upto_test_timestamp=False)
    con = duckdb.connect()
    try:
        register_relbench_db_views(
            con,
            db,
            {
                "customer": "customer_src_raw",
                "product": "product_src_raw",
                "review": "review_src_raw",
            },
            {"review": "review_id"},
        )

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
    if cut_date not in (None, CUT_DATE):
        raise ValueError(
            "Custom scoring cutoffs are disabled; RelBench examples evaluate only "
            "the official test task table"
        )
    if mode == "user_churn":
        from relbench_amazon_user_churn import run_rel_amazon_user_churn

        runner = run_rel_amazon_user_churn
        metric_name = "roc_auc"
    elif mode == "item_churn":
        from relbench_amazon_item_churn import run_rel_amazon_item_churn

        runner = run_rel_amazon_item_churn
        metric_name = "roc_auc"
    elif mode == "user_ltv":
        from relbench_amazon_user_ltv import run_rel_amazon_user_ltv

        runner = run_rel_amazon_user_ltv
        metric_name = "mae"
    elif mode == "item_ltv":
        from relbench_amazon_item_ltv import run_rel_amazon_item_ltv

        runner = run_rel_amazon_item_ltv
        metric_name = "mae"
    else:
        raise ValueError("mode must be user_churn, item_churn, user_ltv, or item_ltv")

    train_store, _, df_test, _, test_metrics, n_features, materialized, target = (
        runner(data_dir=data_dir)
    )
    train_store.close()
    test_score = None if test_metrics is None else float(test_metrics[metric_name])
    return df_test, test_score, n_features, materialized, target


def run_amazon_temporal_regression_task(
    mode: str,
    data_dir: Path | None = None,
    validation_cut_date: datetime.datetime = VALIDATION_CUT_DATE,
    holdout_cut_date: datetime.datetime = HOLDOUT_CUT_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, float | None, int, list[str], str]:
    if mode not in {"user_ltv", "item_ltv"}:
        raise ValueError("mode must be user_ltv or item_ltv")
    if (
        validation_cut_date != VALIDATION_CUT_DATE
        or holdout_cut_date != HOLDOUT_CUT_DATE
    ):
        raise ValueError(
            "Custom scoring cutoffs are disabled; RelBench examples evaluate only "
            "the official validation and test task tables"
        )
    if mode == "user_ltv":
        from relbench_amazon_user_ltv import run_rel_amazon_user_ltv

        runner = run_rel_amazon_user_ltv
    else:
        from relbench_amazon_item_ltv import run_rel_amazon_item_ltv

        runner = run_rel_amazon_item_ltv

    train_store, df_validation, df_holdout, _, test_metrics, n_features, materialized, target = (
        runner(data_dir=data_dir)
    )
    train_store.close()
    test_mae = None if test_metrics is None else float(test_metrics["mae"])
    return df_validation, df_holdout, test_mae, n_features, materialized, target
