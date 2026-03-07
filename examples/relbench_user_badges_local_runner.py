#!/usr/bin/env python
"""Run rel-stack user-badges benchmark via pytest entrypoint."""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    env = os.environ.copy()
    env["RUN_RELBENCH_BENCHMARKS"] = "1"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-s",
        "tests/test_benchmarks.py::test_relbench_user_badges_duckdb",
    ]
    print("Running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
