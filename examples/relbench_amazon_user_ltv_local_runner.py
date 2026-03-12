#!/usr/bin/env python
"""Run rel-amazon user-ltv example for docs interactive mode."""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from urllib.request import urlretrieve

import duckdb
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

from graphreduce.enum import ComputeLayerEnum, PeriodUnit, SQLOpType
from graphreduce.graph_reduce import GraphReduce
from graphreduce.models import sqlop
from graphreduce.node import DuckdbNode

BASE_URL = "https://open-relbench.s3.us-east-1.amazonaws.com/rel-amazon"
TABLES = ["customer.parquet", "product.parquet", "review.parquet"]
LOOKBACK_START = datetime.datetime(1996, 6, 25)
VALIDATION_CUT_DATE = datetime.datetime(2015, 1, 1)
HOLDOUT_CUT_DATE = datetime.datetime(2016, 1, 1)
LABEL_PERIOD_DAYS = 90


def _is_interactive_mode() -> bool:
    return os.getenv("GRAPHREDUCE_INTERACTIVE", "0").strip().lower() in {"1", "true", "yes"}


def _pick_col(columns: list[str], candidates: list[str], required: bool = True) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(f"Could not find any of {candidates} in columns: {columns}")
    return None


def main() -> None:
    print("Running rel-amazon user-ltv example...", flush=True)

    data_dir = Path("tests/data/relbench/rel-amazon")
    data_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    for table in TABLES:
        out_path = data_dir / table
        if not out_path.exists():
            urlretrieve(f"{BASE_URL}/{table}", out_path)
            downloaded.append(table)

    # Build eval and holdout frames sequentially.
    frames: dict[str, object] = {}
    for split_name, cut_date in [("eval", VALIDATION_CUT_DATE), ("holdout", HOLDOUT_CUT_DATE)]:
        con = duckdb.connect()
        try:
            con.sql(f"CREATE OR REPLACE VIEW customer_src_raw AS SELECT * FROM read_parquet('{data_dir / 'customer.parquet'}')")
            con.sql(f"CREATE OR REPLACE VIEW product_src_raw AS SELECT * FROM read_parquet('{data_dir / 'product.parquet'}')")
            con.sql(f"CREATE OR REPLACE VIEW review_src_raw AS SELECT * FROM read_parquet('{data_dir / 'review.parquet'}')")

            customer_cols = con.sql("select * from customer_src_raw limit 0").to_df().columns.tolist()
            product_cols = con.sql("select * from product_src_raw limit 0").to_df().columns.tolist()
            review_cols = con.sql("select * from review_src_raw limit 0").to_df().columns.tolist()

            cust_pk = _pick_col(customer_cols, ["customer_id", "CustomerID", "reviewerID", "user_id", "id"])
            prod_pk = _pick_col(product_cols, ["product_id", "ProductID", "asin", "item_id", "id"])
            rev_customer = _pick_col(review_cols, ["customer_id", "CustomerID", "reviewerID", "user_id", "UserID"])
            rev_product = _pick_col(review_cols, ["product_id", "ProductID", "asin", "item_id", "AdID"])
            rev_date = _pick_col(
                review_cols,
                ["review_time", "review_date", "ReviewTime", "timestamp", "date", "t_dat", "unixReviewTime"],
            )
            rev_pk = _pick_col(review_cols, ["review_id", "ReviewID", "id"], required=False) or rev_product

            review_amount_col = _pick_col(
                review_cols,
                ["price", "Price", "purchase_amount", "PurchaseAmount", "amount", "Amount", "total", "Total"],
                required=False,
            )
            product_amount_col = _pick_col(
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
                SELECT r.*, {amount_expr} AS _gr_ltv_amount
                FROM review_src_raw r
                LEFT JOIN product_src_raw p
                  ON r.{rev_product} = p.{prod_pk}
                """
            )
            review_cols = con.sql("select * from review_src limit 0").to_df().columns.tolist()

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
            )
            review_node = DuckdbNode(
                fpath="review_src",
                prefix="rev",
                pk=rev_pk,
                date_key=rev_date,
                columns=review_cols,
            )

            lookback_days = (cut_date - LOOKBACK_START).days + 1
            gr = GraphReduce(
                name=f"rel_amazon_user_ltv_{split_name}",
                parent_node=customer_node,
                compute_layer=ComputeLayerEnum.duckdb,
                sql_client=con,
                cut_date=cut_date,
                compute_period_val=lookback_days,
                compute_period_unit=PeriodUnit.day,
                auto_features=True,
                auto_labels=True,
                date_filters_on_agg=True,
                label_node=review_node,
                label_field="_gr_ltv_amount",
                label_operation="sum",
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
            df = con.sql(f"select * from {gr.parent_node._cur_data_ref}").to_df()
        finally:
            con.close()

        label_cols = [c for c in df.columns if c.startswith("rev_") and "label" in c.lower()]
        for c in label_cols:
            df[c] = df[c].fillna(0)
        df["user_ltv_90d_usd"] = df[label_cols].sum(axis=1).astype("float64")
        frames[split_name] = df

    df_eval = frames["eval"]
    df_holdout = frames["holdout"]

    target = "user_ltv_90d_usd"
    numeric_cols = [c for c in df_eval.select_dtypes(include=[np.number]).columns if c != target]
    feature_cols = [
        c
        for c in numeric_cols
        if "label" not in c.lower() and not c.lower().endswith("_id") and c.lower() not in {"customerid", "productid"}
    ]
    feature_cols = [c for c in feature_cols if c in df_holdout.columns]

    holdout_mae = None
    if feature_cols:
        X_eval = df_eval[feature_cols].fillna(0)
        y_eval = df_eval[target].fillna(0).astype("float64")
        X_holdout = df_holdout[feature_cols].fillna(0)
        y_holdout = df_holdout[target].fillna(0).astype("float64")

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
        model.fit(X_eval, y_eval)
        holdout_mae = float(mean_absolute_error(y_holdout, model.predict(X_holdout)))

    print("downloaded_files:", downloaded, flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
    print("validation_timestamp:", VALIDATION_CUT_DATE.date(), flush=True)
    print("holdout_timestamp:", HOLDOUT_CUT_DATE.date(), flush=True)
    print("validation_lookback_days:", (VALIDATION_CUT_DATE - LOOKBACK_START).days + 1, flush=True)
    print("holdout_lookback_days:", (HOLDOUT_CUT_DATE - LOOKBACK_START).days + 1, flush=True)
    print("label_period_days:", LABEL_PERIOD_DAYS, flush=True)
    print("target:", target, flush=True)
    print("eval_rows:", len(df_eval), flush=True)
    print("holdout_rows:", len(df_holdout), flush=True)
    print("columns:", len(df_eval.columns), flush=True)
    print("feature_count:", len(feature_cols), flush=True)
    print("holdout_mae:", holdout_mae if holdout_mae is not None else "skipped", flush=True)
    if _is_interactive_mode():
        print("eval.columns:", df_eval.columns, flush=True)


if __name__ == "__main__":
    main()
