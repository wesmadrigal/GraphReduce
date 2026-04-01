# rel-amazon: user LTV (Regression)

This runner now materializes the official `rel-amazon` tables through the `relbench` Python library and trains on the library-aligned temporal split instead of the old S3 snapshot flow.

```python
from relbench_amazon_user_ltv import run_rel_amazon_user_ltv

df_val, df_holdout, holdout_mae, n_features, materialized, target = run_rel_amazon_user_ltv()
print(materialized)
print(target, holdout_mae, n_features, len(df_val), len(df_holdout))
```

Current implementation:

* [`examples/relbench_amazon_user_ltv.py`](../../examples/relbench_amazon_user_ltv.py)
* [`examples/relbench_amazon_common.py`](../../examples/relbench_amazon_common.py)

Current RelBench-aligned dates:

* validation cut date: `2015-10-01`
* holdout cut date: `2016-01-01`
* lookback start: `2008-01-01`

## Interactive Runner

<div class="modal-runner" data-modal-runner data-api-base="https://runner.13.218.155.128.sslip.io" data-example="relbench_amazon_user_ltv">
  <div class="modal-runner-controls">
    <input class="modal-runner-input" data-api-input value="https://runner.13.218.155.128.sslip.io" />
    <button data-save-api-btn>Save API URL</button>
    <button data-run-btn>Run rel-amazon user LTV</button>
  </div>
  <div class="modal-runner-status" data-status>Idle</div>
  <pre class="modal-runner-log" data-log></pre>
</div>
