# rel-amazon: user churn (Classification)

This example now sources data from the `relbench` Python library, not the old S3 parquet mirror. The runner materializes the official `rel-amazon` tables locally through `get_dataset("rel-amazon", download=True)` before building the GraphReduce frame.

```python
from relbench_amazon_user_churn import run_rel_amazon_user_churn

df, auc, n_features, materialized, target = run_rel_amazon_user_churn()
print(materialized)
print(target, auc, n_features, len(df))
```

Current implementation:

* [`examples/relbench_amazon_user_churn.py`](../../examples/relbench_amazon_user_churn.py)
* [`examples/relbench_amazon_common.py`](../../examples/relbench_amazon_common.py)

Current RelBench-aligned dates:

* validation cut date: `2015-10-01`
* holdout cut date: `2016-01-01`
* lookback start: `2008-01-01`

## Interactive Runner

<div class="modal-runner" data-modal-runner data-api-base="https://runner.13.218.155.128.sslip.io" data-example="relbench_amazon_user_churn">
  <div class="modal-runner-controls">
    <input class="modal-runner-input" data-api-input value="https://runner.13.218.155.128.sslip.io" />
    <button data-save-api-btn>Save API URL</button>
    <button data-run-btn>Run rel-amazon user churn</button>
  </div>
  <div class="modal-runner-status" data-status>Idle</div>
  <pre class="modal-runner-log" data-log></pre>
</div>
