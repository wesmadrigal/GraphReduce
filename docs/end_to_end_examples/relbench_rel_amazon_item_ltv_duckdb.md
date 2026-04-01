# rel-amazon: item LTV (Regression)

This example now pulls `rel-amazon` directly from `relbench`, materializes the official tables locally, and then runs the existing GraphReduce pipeline on those files.

```python
from relbench_amazon_item_ltv import run_rel_amazon_item_ltv

df_val, df_holdout, holdout_mae, n_features, materialized, target = run_rel_amazon_item_ltv()
print(materialized)
print(target, holdout_mae, n_features, len(df_val), len(df_holdout))
```

Current implementation:

* [`examples/relbench_amazon_item_ltv.py`](../../examples/relbench_amazon_item_ltv.py)
* [`examples/relbench_amazon_common.py`](../../examples/relbench_amazon_common.py)

Current RelBench-aligned dates:

* validation cut date: `2015-10-01`
* holdout cut date: `2016-01-01`
* lookback start: `2008-01-01`

## Interactive Runner

<div class="modal-runner" data-modal-runner data-api-base="https://runner.13.218.155.128.sslip.io" data-example="relbench_amazon_item_ltv">
  <div class="modal-runner-controls">
    <input class="modal-runner-input" data-api-input value="https://runner.13.218.155.128.sslip.io" />
    <button data-save-api-btn>Save API URL</button>
    <button data-run-btn>Run rel-amazon item LTV</button>
  </div>
  <div class="modal-runner-status" data-status>Idle</div>
  <pre class="modal-runner-log" data-log></pre>
</div>
