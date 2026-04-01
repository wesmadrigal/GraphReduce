# RelBench GraphReduce + CatBoost Performance Comparison

This table compares performance across RelBench tasks currently listed in these docs.

## Best-Per-Task Summary

- GraphReduce + CatBoost: **6 wins**
- Kumo (fine tuned): **4 wins**
- RDL: **1 win** (tie)
- Data Scientist: **0 wins**
- Kumo (in context): **0 wins**

Notes:
- Ties are counted for each tied solution.

| Problem | Data Scientist | RDL | Kumo (in context) | Kumo (fine tuned) | GraphReduce + CatBoost | Metric |
|---|---:|---:|---:|---:|---:|---|
| rel-stack-user-engagement | 90.3 | 90.2 | 87.09 | **90.7** | 89.21 | AUCROC |
| rel-stack-user-badges | 86.2 | **89.86** | 80 | **89.86** | 84.30 | AUCROC |
| rel-trial-study-outcome | 72.00 | 68.60 | 70.79 | 71.16 | **93.20** | AUCROC |
| rel-amazon-user-churn | 67.6 | 70.42 | 67.29 | 70.46 | **72.00** | AUCROC |
| rel-amazon-item-churn | 81.8 | 82.81 | 79.93 | **82.83** | 81.00 | AUCROC |
| rel-hm-user-churn | 69 | 69.88 | 67.71 | 71.23 | **76.50** | AUCROC |
| rel-stack-post-votes | 0.068 | 0.065 | 0.065 | 0.065 | **0.0626** | MAE |
| rel-hm-item-sales | 0.036 | 0.056 | 0.04 | **0.034** | 0.0429 | MAE |
| rel-amazon-user-ltv | 13.92 | 14.31 | 16.16 | 14.22 | **6.593** | MAE |
| rel-amazon-item-ltv | 41.12 | 50.05 | 55.25 | 48.67 | **18.58** | MAE |
