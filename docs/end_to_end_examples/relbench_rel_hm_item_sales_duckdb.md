# rel-hm: item sales (Regression)

This runner now materializes `rel-hm` through `relbench` before building the GraphReduce frames. It no longer downloads parquet files from the example S3 bucket.

```python
from relbench_hm_item_sales import run_rel_hm_item_sales

df_eval, df_holdout, holdout_mae, n_features, materialized, target = run_rel_hm_item_sales()
print(materialized)
print(target, holdout_mae, n_features, len(df_eval), len(df_holdout))
```

Current implementation:

* [`examples/relbench_hm_item_sales.py`](../../examples/relbench_hm_item_sales.py)

Current RelBench-aligned dates:

* eval cut date: `2020-09-07`
* holdout cut date: `2020-09-14`
* lookback start: `2019-09-07`

## Interactive Runner

<div class="modal-runner" data-modal-runner data-api-base="https://runner.13.218.155.128.sslip.io" data-example="relbench_hm_item_sales">
  <div class="modal-runner-controls">
    <input class="modal-runner-input" data-api-input value="https://runner.13.218.155.128.sslip.io" />
    <button data-save-api-btn>Save API URL</button>
    <button data-run-btn>Run rel-hm item sales</button>
  </div>
  <div class="modal-runner-status" data-status>Idle</div>
  <pre class="modal-runner-log" data-log></pre>
</div>
