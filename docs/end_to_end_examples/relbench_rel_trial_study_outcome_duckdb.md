# rel-trial: study outcome (Classification)

This example now materializes the official `rel-trial` tables from the `relbench` Python library and runs the GraphReduce pipeline on those local parquet files.

```python
from relbench_trial_study_outcome import run_rel_trial_study_outcome

df_val, df_test, in_time_auc, holdout_auc, n_features, materialized, target = run_rel_trial_study_outcome()
print(materialized)
print(target, in_time_auc, holdout_auc, n_features, len(df_val), len(df_test))
```

Current implementation:

* [`examples/relbench_trial_study_outcome.py`](../../examples/relbench_trial_study_outcome.py)

## Interactive Runner

<div class="modal-runner" data-modal-runner data-api-base="https://runner.13.218.155.128.sslip.io" data-example="relbench_trial_study_outcome">
  <div class="modal-runner-controls">
    <input class="modal-runner-input" data-api-input value="https://runner.13.218.155.128.sslip.io" />
    <button data-save-api-btn>Save API URL</button>
    <button data-run-btn>Run rel-trial study outcome</button>
  </div>
  <div class="modal-runner-status" data-status>Idle</div>
  <pre class="modal-runner-log" data-log></pre>
</div>
