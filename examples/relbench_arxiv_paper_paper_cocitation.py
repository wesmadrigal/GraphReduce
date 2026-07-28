#!/usr/bin/env python
"""Official rel-arxiv/paper-paper-cocitation baseline."""

from relbench_v2_task_utils import run_relbench_v2_link_task


if __name__ == "__main__":
    run_relbench_v2_link_task("rel-arxiv", "paper-paper-cocitation")
