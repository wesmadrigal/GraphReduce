# RelBench GraphReduce + CatBoost Performance Comparison

Current local report: `tests/data/relbench/run_reports/relbench_results.json`

* finished at: `2026-04-09T22:55:14Z`
* tasks passed: `25/25`

## Published Baseline Comparison

This table keeps the published RelBench comparison baselines already referenced in the docs and updates only the `GraphReduce + CatBoost` column to match the latest local run.

## Best-Per-Task Summary

* GraphReduce + CatBoost: **1 win**
* Kumo (fine tuned): **5 wins**
* RDL: **1 win** (tie)
* Data Scientist: **3 wins**
* Kumo (in context): **0 wins**

Notes:

* ties are counted for each tied solution
* `AUCROC` values are shown as percentages
* `MAE` values are shown in task-native units from the local report

| Problem | Data Scientist | RDL | Kumo (in context) | Kumo (fine tuned) | GraphReduce + CatBoost | Metric |
|---|---:|---:|---:|---:|---:|---|
| rel-stack-user-engagement | 90.3 | 90.2 | 87.09 | **90.7** | 89.12 | AUCROC |
| rel-stack-user-badges | 86.2 | **89.86** | 80 | **89.86** | 84.42 | AUCROC |
| rel-trial-study-outcome | **72.00** | 68.60 | 70.79 | 71.16 | 61.57 | AUCROC |
| rel-amazon-user-churn | 67.6 | 70.42 | 67.29 | **70.46** | 69.83 | AUCROC |
| rel-amazon-item-churn | 81.8 | 82.81 | 79.93 | **82.83** | 82.49 | AUCROC |
| rel-hm-user-churn | 69 | 69.88 | 67.71 | **71.23** | 69.52 | AUCROC |
| rel-stack-post-votes | 0.068 | 0.065 | 0.065 | 0.065 | **0.0625** | MAE |
| rel-hm-item-sales | 0.036 | 0.056 | 0.04 | **0.034** | 0.0362 | MAE |
| rel-amazon-user-ltv | **13.92** | 14.31 | 16.16 | 14.22 | 14.02 | MAE |
| rel-amazon-item-ltv | **41.12** | 50.05 | 55.25 | 48.67 | 42.71 | MAE |

## Additional Tasks In The Current Run

These tasks were also exercised by the same local run report but are not part of the baseline comparison table above.

| Problem | GraphReduce + CatBoost | Metric |
|---|---:|---|
| rel-avito-user-clicks | 78.95 | AUCROC |
| rel-avito-user-visits | 82.99 | AUCROC |
| rel-avito-ad-ctr | 0.0313 | MAE |
| rel-event-interest-interested | 62.52 | AUCROC |
| rel-event-interest-not-interested | 32.15 | AUCROC |
| rel-event-user-attendance | 0.2930 | MAE |
| rel-event-user-ignore | 60.47 | AUCROC |
| rel-event-user-repeat | 64.96 | AUCROC |
| rel-event-users-birthyear | 5.3363 | MAE |
| rel-f1-driver-circuit-compete | 0.0000 | MAP |
| rel-f1-driver-dnf | 71.51 | AUCROC |
| rel-f1-driver-position | 3.5905 | MAE |
| rel-f1-driver-top3 | 82.46 | AUCROC |
| rel-f1-qualifying-position | 5.3486 | MAE |
| rel-f1-results-position | 3.4612 | MAE |
