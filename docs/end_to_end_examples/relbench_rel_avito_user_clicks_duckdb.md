# rel-avito: user clicks (Classification)

This runner now uses the official `relbench` dataset loader and materializes the Avito tables locally before GraphReduce runs. The example no longer depends on the legacy S3 parquet mirror.

```python
from relbench_avito_user_clicks import run_rel_avito_user_clicks

df, auc, n_features, materialized, target = run_rel_avito_user_clicks()
print(materialized)
print(target, auc, n_features, len(df))
```

Current implementation:

* [`examples/relbench_avito_user_clicks.py`](../../examples/relbench_avito_user_clicks.py)
* [`examples/relbench_avito_common.py`](../../examples/relbench_avito_common.py)

## Interactive Runner

<div class="modal-runner" data-modal-runner data-api-base="https://runner.13.218.155.128.sslip.io" data-example="relbench_avito_user_clicks">
  <div class="modal-runner-controls">
    <input class="modal-runner-input" data-api-input value="https://runner.13.218.155.128.sslip.io" />
    <button data-save-api-btn>Save API URL</button>
    <button data-run-btn>Run rel-avito user clicks</button>
  </div>
  <div class="modal-runner-status" data-status>Idle</div>
  <pre class="modal-runner-log" data-log></pre>
</div>
