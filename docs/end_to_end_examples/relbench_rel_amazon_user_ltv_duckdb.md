# rel-amazon: user LTV

This example implements the RelBench rel-amazon user LTV setup:

* parent node: `customer.parquet`
* label node: `review.parquet`
* context node: `product.parquet`
* compute window (eval): `1996-06-25` to `2015-01-01`
* holdout timestamp: `2016-01-01`
* label period: `90` days (3 months)
* target: total dollar value of products the user buys/reviews in the next 90 days

Data source:

* `https://open-relbench.s3.us-east-1.amazonaws.com/rel-amazon`

## Complete Example (Full Code)

```python
import datetime
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


def download_rel_amazon_data(data_dir: Path) -> list[str]:
    data_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
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


def build_user_ltv_frame(data_dir: Path, cut_date: datetime.datetime):
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
            name=f"rel_amazon_user_ltv_{cut_date.date()}",
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
        out_df = con.sql(f"select * from {gr.parent_node._cur_data_ref}").to_df()

        label_cols = [c for c in out_df.columns if c.startswith("rev_") and "label" in c.lower()]
        if not label_cols:
            raise ValueError("No review label columns found in output.")

        for c in label_cols:
            out_df[c] = out_df[c].fillna(0)

        out_df["user_ltv_90d_usd"] = out_df[label_cols].sum(axis=1).astype("float64")
        return out_df
    finally:
        con.close()


data_dir = Path("tests/data/relbench/rel-amazon")
downloaded = download_rel_amazon_data(data_dir)
df_eval = build_user_ltv_frame(data_dir, VALIDATION_CUT_DATE)
df_holdout = build_user_ltv_frame(data_dir, HOLDOUT_CUT_DATE)

numeric_cols = [c for c in df_eval.select_dtypes(include=[np.number]).columns if c != "user_ltv_90d_usd"]
feature_cols = [
    c
    for c in numeric_cols
    if "label" not in c.lower() and not c.lower().endswith("_id") and c.lower() not in {"customerid", "productid"}
]
feature_cols = [c for c in feature_cols if c in df_holdout.columns]

X_eval = df_eval[feature_cols].fillna(0)
y_eval = df_eval["user_ltv_90d_usd"].fillna(0).astype("float64")
X_holdout = df_holdout[feature_cols].fillna(0)
y_holdout = df_holdout["user_ltv_90d_usd"].fillna(0).astype("float64")

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
holdout_mae = mean_absolute_error(y_holdout, model.predict(X_holdout))

print("downloaded_files:", downloaded)
print("lookback_start:", LOOKBACK_START.date())
print("validation_timestamp:", VALIDATION_CUT_DATE.date())
print("holdout_timestamp:", HOLDOUT_CUT_DATE.date())
print("label_period_days:", LABEL_PERIOD_DAYS)
print("feature_count:", len(feature_cols))
print("holdout_mae:", round(float(holdout_mae), 4))
```

Full runnable scripts:

* `examples/relbench_amazon_user_ltv.py`
* `examples/relbench_amazon_user_ltv_local_runner.py`

## Run Interactive

<div class="modal-runner" data-modal-runner data-api-base="https://runner.13.218.155.128.sslip.io" data-example="relbench_amazon_user_ltv">
  <div class="modal-runner-controls">
    <input class="modal-runner-input" data-api-input value="https://runner.13.218.155.128.sslip.io" />
    <button data-save-api-btn>Save API URL</button>
    <button data-run-btn>Run rel-amazon User LTV</button>
  </div>
  <div class="modal-runner-status" data-status>Idle</div>
  <pre class="modal-runner-log" data-log></pre>
</div>
