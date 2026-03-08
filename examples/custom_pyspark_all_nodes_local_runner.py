#!/usr/bin/env python
"""Run custom PySpark all-nodes cust_data example and print a summary."""

from __future__ import annotations

import os

from custom_pyspark_all_nodes import run_custom_pyspark_all_nodes


def _is_interactive_mode() -> bool:
    return os.getenv("GRAPHREDUCE_INTERACTIVE", "0").strip().lower() in {"1", "true", "yes"}


def main() -> None:
    print("Running custom pyspark all-nodes backend...", flush=True)
    df = run_custom_pyspark_all_nodes()

    rows = df.count()
    cols = len(df.columns)
    print("rows:", rows, flush=True)
    print("columns:", cols, flush=True)
    print("column_names:", df.columns, flush=True)
    print("shape:", (rows, cols), flush=True)
    if _is_interactive_mode():
        print("df.columns:", df.columns, flush=True)


if __name__ == "__main__":
    main()
