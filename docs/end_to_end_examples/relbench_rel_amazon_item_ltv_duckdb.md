# rel-amazon: item LTV

This example implements the RelBench rel-amazon item LTV setup:

* parent node: `product.parquet`
* label node: `review.parquet`
* context node: `customer.parquet`
* compute window (eval): `1996-06-25` to `2015-01-01`
* holdout timestamp: `2016-01-01`
* label period: `90` days (3 months)
* target: total dollar value of purchases/reviews the item receives in the next 90 days

Data source:

* `https://open-relbench.s3.us-east-1.amazonaws.com/rel-amazon`

## Complete Example

### Data Preparation + GraphReduce

```python
from pathlib import Path
from urllib.request import urlretrieve

from relbench_amazon_common import (
    BASE_URL,
    TABLES,
    LABEL_PERIOD_DAYS,
    LOOKBACK_START,
    VALIDATION_CUT_DATE,
    HOLDOUT_CUT_DATE,
    run_amazon_temporal_regression_task,
)

out_dir = Path("tests/data/relbench/rel-amazon")
out_dir.mkdir(parents=True, exist_ok=True)
for table in TABLES:
    path = out_dir / table
    if not path.exists():
        urlretrieve(f"{BASE_URL}/{table}", path)

df_eval, df_holdout, holdout_mae, n_features, downloaded, target = run_amazon_temporal_regression_task(
    "item_ltv",
    data_dir=out_dir,
)

print("lookback_start:", LOOKBACK_START.date())
print("validation_timestamp:", VALIDATION_CUT_DATE.date())
print("holdout_timestamp:", HOLDOUT_CUT_DATE.date())
print("label_period_days:", LABEL_PERIOD_DAYS)
print("target:", target)
print("eval_rows:", len(df_eval), "holdout_rows:", len(df_holdout))
```

### Model Training

```python
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

numeric_cols = [c for c in df_eval.select_dtypes(include=[np.number]).columns if c != "item_ltv_90d_usd"]
feature_cols = [
    c
    for c in numeric_cols
    if "label" not in c.lower() and not c.lower().endswith("_id") and c.lower() not in {"customerid", "productid"}
]
feature_cols = [c for c in feature_cols if c in df_holdout.columns]

X_eval = df_eval[feature_cols].fillna(0)
y_eval = df_eval["item_ltv_90d_usd"].fillna(0).astype("float64")
X_holdout = df_holdout[feature_cols].fillna(0)
y_holdout = df_holdout["item_ltv_90d_usd"].fillna(0).astype("float64")

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

* `examples/relbench_amazon_common.py`
* `examples/relbench_amazon_item_ltv.py`
* `examples/relbench_amazon_item_ltv_local_runner.py`

## Run Interactive

<div class="modal-runner" data-modal-runner data-api-base="https://runner.13.218.155.128.sslip.io" data-example="relbench_amazon_item_ltv">
  <div class="modal-runner-controls">
    <input class="modal-runner-input" data-api-input value="https://runner.13.218.155.128.sslip.io" />
    <button data-save-api-btn>Save API URL</button>
    <button data-run-btn>Run rel-amazon Item LTV</button>
  </div>
  <div class="modal-runner-status" data-status>Idle</div>
  <pre class="modal-runner-log" data-log></pre>
</div>
