#!/usr/bin/env python
"""GraphReduce example for official rel-arxiv/author-category."""

from relbench_v2_task_utils import run_relbench_v2_entity_task


if __name__ == "__main__":
    run_relbench_v2_entity_task("rel-arxiv", "author-category")
