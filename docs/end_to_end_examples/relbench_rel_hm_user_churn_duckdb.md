# rel-hm: user churn (Classification)

This example now pulls `rel-hm` from the `relbench` Python library and materializes the official tables locally before GraphReduce runs. The old S3 parquet download path has been removed.

```python
from relbench_hm_user_churn import run_rel_hm_user_churn

df, auc, n_features, materialized = run_rel_hm_user_churn()
print(materialized)
print(auc, n_features, len(df))
```

Current implementation:

* [`examples/relbench_hm_user_churn.py`](../../examples/relbench_hm_user_churn.py)

Current RelBench-aligned dates:

* cut date: `2020-09-14`
* lookback start: `2019-09-07`

## Interactive Runner

<div class="modal-runner" data-modal-runner data-api-base="https://runner.13.218.155.128.sslip.io" data-example="relbench_hm_user_churn">
  <div class="modal-runner-controls">
    <input class="modal-runner-input" data-api-input value="https://runner.13.218.155.128.sslip.io" />
    <button data-save-api-btn>Save API URL</button>
    <button data-run-btn>Run rel-hm user churn</button>
  </div>
  <div class="modal-runner-status" data-status>Idle</div>
  <pre class="modal-runner-log" data-log></pre>
</div>
