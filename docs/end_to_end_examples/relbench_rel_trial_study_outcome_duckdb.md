# rel-trial: study outcome (Classification)

This example now materializes the official `rel-trial` tables from the `relbench` Python library and runs one GraphReduce instance for each official yearly training cut date, plus the validation and test cut dates.

```python
from relbench_trial_study_outcome import run_rel_trial_study_outcome

df_train, df_val, df_test, in_time_auc, holdout_auc, n_features, materialized, target = run_rel_trial_study_outcome()
print(materialized)
print(target, in_time_auc, holdout_auc, n_features, len(df_train), len(df_val), len(df_test))
```

Current implementation:

* [`examples/relbench_trial_study_outcome.py`](https://github.com/wesmadrigal/graphreduce/blob/master/examples/relbench_trial_study_outcome.py)

## Latest Tested Result

Latest local run from `tests/data/relbench/run_reports/relbench_results.json`:

* in-time AUC: `0.6208`
* holdout AUC: `0.6434`
* feature count: `282`
* temporal instances: `21` (`19` train, `1` validation, `1` test)
* rows: train `11,994`, validation `960`, test `825`
* runtime: approximately `45s` on the local development machine

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
