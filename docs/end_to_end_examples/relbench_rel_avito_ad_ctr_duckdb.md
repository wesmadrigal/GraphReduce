# rel-avito: ad CTR (Regression)

This example now materializes `rel-avito` from the `relbench` Python library. The local files keep the historical example filenames, but the source of truth is `get_dataset("rel-avito", download=True)`.

```python
from relbench_avito_ad_ctr import run_rel_avito_ad_ctr

df, mae, n_features, materialized, target = run_rel_avito_ad_ctr()
print(materialized)
print(target, mae, n_features, len(df))
```

Current implementation:

* [`examples/relbench_avito_ad_ctr.py`](../../examples/relbench_avito_ad_ctr.py)
* [`examples/relbench_avito_common.py`](../../examples/relbench_avito_common.py)

## Latest Tested Result

Latest local run from `tests/data/relbench/run_reports/relbench_results.json`:

* test MAE: `0.0313`
* test RMSE: `0.0943`
* feature count: `50`
* rows: train `5100`, validation `1766`, test `1816`
* runtime: `32.4s` (`2026-04-09T22:24:09Z`)

## Interactive Runner

<div class="modal-runner" data-modal-runner data-api-base="https://runner.13.218.155.128.sslip.io" data-example="relbench_avito_ad_ctr">
  <div class="modal-runner-controls">
    <input class="modal-runner-input" data-api-input value="https://runner.13.218.155.128.sslip.io" />
    <button data-save-api-btn>Save API URL</button>
    <button data-run-btn>Run rel-avito ad CTR</button>
  </div>
  <div class="modal-runner-status" data-status>Idle</div>
  <pre class="modal-runner-log" data-log></pre>
</div>
