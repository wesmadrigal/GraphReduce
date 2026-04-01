# rel-avito: user visits (Classification)

This example now materializes the official `rel-avito` tables through the `relbench` Python library and then runs the user-visits GraphReduce task against those local files.

```python
from relbench_avito_user_visits import run_rel_avito_user_visits

df, auc, n_features, materialized, target = run_rel_avito_user_visits()
print(materialized)
print(target, auc, n_features, len(df))
```

Current implementation:

* [`examples/relbench_avito_user_visits.py`](../../examples/relbench_avito_user_visits.py)
* [`examples/relbench_avito_common.py`](../../examples/relbench_avito_common.py)

## Interactive Runner

<div class="modal-runner" data-modal-runner data-api-base="https://runner.13.218.155.128.sslip.io" data-example="relbench_avito_user_visits">
  <div class="modal-runner-controls">
    <input class="modal-runner-input" data-api-input value="https://runner.13.218.155.128.sslip.io" />
    <button data-save-api-btn>Save API URL</button>
    <button data-run-btn>Run rel-avito user visits</button>
  </div>
  <div class="modal-runner-status" data-status>Idle</div>
  <pre class="modal-runner-log" data-log></pre>
</div>
