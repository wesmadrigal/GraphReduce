# rel-hm: item sales

This example implements the RelBench rel-hm item-sales setup:

* parent node: `article.parquet`
* label node: `transactions.parquet`
* context node: `customer.parquet`
* compute window (eval): `2018-09-20` to `2020-09-07`
* holdout timestamp: `2020-09-14`
* label period: `8` days (to include 7 full days with GraphReduce's exclusive upper bound)
* target: total sales amount for each item in the next week

Data source:

* `https://open-relbench.s3.us-east-1.amazonaws.com/rel-hm`

## Complete Example

### Data Preparation + GraphReduce

```python
from relbench_hm_item_sales import EVAL_DATE, HOLDOUT_DATE, LABEL_DAYS, LOOKBACK_START, run_rel_hm_item_sales

df_eval, df_holdout, holdout_mae, n_features, downloaded, target = run_rel_hm_item_sales()

print("lookback_start:", LOOKBACK_START.date())
print("eval_timestamp:", EVAL_DATE.date())
print("holdout_timestamp:", HOLDOUT_DATE.date())
print("label_period_days:", LABEL_DAYS)
print("target:", target)
print("eval_rows:", len(df_eval), "holdout_rows:", len(df_holdout))
```

### Model Training

```python
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

numeric_cols = [c for c in df_eval.select_dtypes(include=[np.number]).columns if c != "item_sales_7d_usd"]
feature_cols = [
    c
    for c in numeric_cols
    if "label" not in c.lower() and not c.lower().endswith("_id") and c not in {"art_article_id", "txn_article_id"}
]
feature_cols = [c for c in feature_cols if c in df_holdout.columns]

X_eval = df_eval[feature_cols].fillna(0)
y_eval = df_eval["item_sales_7d_usd"].fillna(0).astype("float64")
X_holdout = df_holdout[feature_cols].fillna(0)
y_holdout = df_holdout["item_sales_7d_usd"].fillna(0).astype("float64")

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
print("holdout_mae:", round(float(holdout_mae), 4))
```

Full runnable scripts:

* `examples/relbench_hm_item_sales.py`
* `examples/relbench_hm_item_sales_local_runner.py`

## Run Interactive

<div class="modal-runner" data-modal-runner data-api-base="https://runner.13.218.155.128.sslip.io" data-example="relbench_hm_item_sales">
  <div class="modal-runner-controls">
    <input class="modal-runner-input" data-api-input value="https://runner.13.218.155.128.sslip.io" />
    <button data-save-api-btn>Save API URL</button>
    <button data-run-btn>Run rel-hm Item Sales</button>
  </div>
  <div class="modal-runner-status" data-status>Idle</div>
  <pre class="modal-runner-log" data-log></pre>
</div>
