# rel-avito: ad CTR regression

This example uses `AdsInfo.parquet` as the parent node and predicts ad-level
CTR in the next 4-day horizon (implemented as `label_period_days=5` because the
label window end is exclusive).

* parent node: `AdsInfo.parquet`
* label node: `SearchStream.parquet`
* task type: regression
* label SQL: `sum(clicks) / count(*)`

Label definition in `do_labels_ops`:

```sql
sum(case when IsClick > 0 then 1 else 0 end) / count(*) as ctr
```

In DuckDB form used here:

```sql
sum(case when coalesce(try_cast(IsClick as double), 0) > 0 then 1 else 0 end)
/ nullif(count(*), 0)::double as sstr_ctr_label
```

## Graph

* `AdsInfo -> SearchStream` (label branch)
* `AdsInfo -> SearchInfo`
* `SearchInfo -> SearchStream`
* `AdsInfo -> VisitsStream`
* optional dimensions when keys exist:
  * `AdsInfo -> Category`
  * `AdsInfo -> Location`

## Run Example

```bash
python examples/relbench_avito_ad_ctr.py
```

## Run Interactive

<div class="modal-runner" data-modal-runner data-api-base="https://runner.13.218.155.128.sslip.io" data-example="relbench_avito_ad_ctr">
  <div class="modal-runner-controls">
    <input class="modal-runner-input" data-api-input value="https://runner.13.218.155.128.sslip.io" />
    <button data-save-api-btn>Save API URL</button>
    <button data-run-btn>Run rel-avito Ad CTR</button>
  </div>
  <div class="modal-runner-status" data-status>Idle</div>
  <pre class="modal-runner-log" data-log></pre>
</div>
