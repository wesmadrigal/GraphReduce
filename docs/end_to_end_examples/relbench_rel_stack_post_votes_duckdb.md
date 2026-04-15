# rel-stack: post votes (Regression)

[![RelBench rel-stack post-votes graphreduce flow](relbench_rel_stack_post_votes_duckdb_overview.svg)](relbench_rel_stack_post_votes_duckdb_overview.svg)

Open full-size: [SVG](relbench_rel_stack_post_votes_duckdb_overview.svg)

This docs page now reflects the current runner path: the example materializes the official `rel-stack` tables from the `relbench` Python library and then builds the GraphReduce post-votes regression frame locally.

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

* [`examples/relbench_post_votes_local_runner.py`](../../examples/relbench_post_votes_local_runner.py)

## Latest Tested Result

Latest local run from `tests/data/relbench/run_reports/relbench_results.json`:

* in-time holdout MAE (`2020`): `0.0613`
* out-of-time MAE (`2021`): `0.0625`
* feature count: `217`
* rows: train `139575`, future eval `160172`
* runtime: `60.3s` (`2026-04-09T22:52:48Z`)

## Interactive Runner

<div class="modal-runner" data-modal-runner data-api-base="https://runner.13.218.155.128.sslip.io" data-example="relbench_post_votes">
  <div class="modal-runner-controls">
    <input class="modal-runner-input" data-api-input value="https://runner.13.218.155.128.sslip.io" />
    <button data-save-api-btn>Save API URL</button>
    <button data-run-btn>Run rel-stack post votes</button>
  </div>
  <div class="modal-runner-status" data-status>Idle</div>
  <pre class="modal-runner-log" data-log></pre>
</div>
