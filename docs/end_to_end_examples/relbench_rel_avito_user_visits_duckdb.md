# rel-avito: user visits

[![RelBench rel-avito user-visits graphreduce flow](relbench_rel_avito_user_visits_overview.svg)](relbench_rel_avito_user_visits_overview.svg)

Open full-size: [SVG](relbench_rel_avito_user_visits_overview.svg)

This task uses `VisitsStream.parquet` as the label node and predicts whether a
user will visit **more than one unique `AdID`** in the next 4-day horizon.

* parent node: `UserInfo.parquet`
* label node: `VisitsStream.parquet`
* label aggregation: `count(distinct AdID)`
* target: `count(distinct AdID) > 1`
* GraphReduce label period: `5` days (exclusive end boundary)
* cut date: `2015-05-14`
* lookback start: `2015-04-25`

Data source:

* `https://open-relbench.s3.us-east-1.amazonaws.com/rel-avito`

## Label Logic

`VisitsStream` label ops include:

```sql
count(distinct vis_AdID) as vis_distinct_ads_label
```

and the final binary target is:

```python
user_multi_visit_next_4d = (vis_distinct_ads_label > 1).astype("int8")
```

## Run Example

```bash
python examples/relbench_avito_user_visits.py
```

## Run Interactive

<div class="modal-runner" data-modal-runner data-api-base="https://runner.13.218.155.128.sslip.io" data-example="relbench_avito_user_visits">
  <div class="modal-runner-controls">
    <input class="modal-runner-input" data-api-input value="https://runner.13.218.155.128.sslip.io" />
    <button data-save-api-btn>Save API URL</button>
    <button data-run-btn>Run rel-avito User Visits</button>
  </div>
  <div class="modal-runner-status" data-status>Idle</div>
  <pre class="modal-runner-log" data-log></pre>
</div>
