# rel-stack: user engagement (Classification)

This docs example now matches the current local runner: it materializes the official `rel-stack` tables through the `relbench` library and then builds the GraphReduce user-engagement frame from those CSVs.

```python
from pathlib import Path
from relbench_dataset_utils import materialize_relbench_dataset

materialized = materialize_relbench_dataset(
    "rel-stack",
    Path("tests/data/relbench/rel-stack"),
    {
        "users": "Users.csv",
        "posts": "Posts.csv",
        "badges": "Badges.csv",
        "postHistory": "PostHistory.csv",
        "postLinks": "PostLinks.csv",
        "votes": "Votes.csv",
        "comments": "Comments.csv",
    },
)
print(materialized)
```

Current implementation:

* [`examples/relbench_user_engagement_local_runner.py`](../../examples/relbench_user_engagement_local_runner.py)

## Latest Tested Result

Latest local run from `tests/data/relbench/run_reports/relbench_results.json`:

* test ROC AUC: `0.8912`
* test average precision: `0.3767`
* feature count: `362`
* rows: train `83388`, validation `85702`, test `88014`
* runtime: `34.0s` (`2026-04-09T22:55:14Z`)

## Interactive Runner

<div class="modal-runner" data-modal-runner data-api-base="https://runner.13.218.155.128.sslip.io" data-example="relbench_user_engagement">
  <div class="modal-runner-controls">
    <input class="modal-runner-input" data-api-input value="https://runner.13.218.155.128.sslip.io" />
    <button data-save-api-btn>Save API URL</button>
    <button data-run-btn>Run rel-stack user engagement</button>
  </div>
  <div class="modal-runner-status" data-status>Idle</div>
  <pre class="modal-runner-log" data-log></pre>
</div>
