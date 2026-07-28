#!/usr/bin/env python
"""Compatibility entry point for the rel-stack user-badge task-table runner.

The previous version of this example manually built labels with custom SQL and
used a date-node variant. The task-table path now delegates supervision to the
RelBench `user-badge` task and runs GraphReduce once per split timestamp.
"""

from __future__ import annotations

from relbench_user_badges_local_runner import main


if __name__ == "__main__":
    main()
