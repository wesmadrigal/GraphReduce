# rel-stack: user badges (Classification)

This docs example now reflects the current runner path: the local script materializes the official `rel-stack` tables from the `relbench` library and then runs the GraphReduce badge pipeline. The old S3 CSV download and separate `Tags.csv` dependency are no longer part of the example.

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

* [`examples/relbench_user_badges_local_runner.py`](../../examples/relbench_user_badges_local_runner.py)

## Latest Tested Result

Latest local run from `tests/data/relbench/run_reports/relbench_results.json`:

* test ROC AUC: `0.8442`
* test average precision: `0.2749`
* feature count: `490`
* rows: train `239926`, validation `247379`, test `255341`
* runtime: `89.8s` (`2026-04-09T22:54:40Z`)

## Interactive Runner

<div class="modal-runner" data-modal-runner data-api-base="https://runner.13.218.155.128.sslip.io" data-example="relbench_user_badges">
  <div class="modal-runner-controls">
    <input class="modal-runner-input" data-api-input value="https://runner.13.218.155.128.sslip.io" />
    <button data-save-api-btn>Save API URL</button>
    <button data-run-btn>Run rel-stack user badges</button>
  </div>
  <div class="modal-runner-status" data-status>Idle</div>
  <pre class="modal-runner-log" data-log></pre>
</div>
