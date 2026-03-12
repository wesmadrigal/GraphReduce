#!/usr/bin/env python
"""Run rel-trial study-outcome example for docs interactive mode."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.request import urlretrieve

from relbench_trial_study_outcome import (
    LABEL_DAYS,
    LOOKBACK_START,
    TABLES,
    TEST_TIMESTAMP,
    VAL_TIMESTAMP,
    run_rel_trial_study_outcome,
)

BASE_URL = "https://open-relbench.s3.us-east-1.amazonaws.com/rel-trial"


def _is_interactive_mode() -> bool:
    return os.getenv("GRAPHREDUCE_INTERACTIVE", "0").strip().lower() in {"1", "true", "yes"}


def main() -> None:
    print("Running rel-trial study-outcome example...", flush=True)
    data_dir = Path("tests/data/relbench/rel-trial")
    data_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    for table in TABLES:
        out_path = data_dir / table
        if not out_path.exists():
            urlretrieve(f"{BASE_URL}/{table}", out_path)
            downloaded.append(table)

    df_val, df_test, in_time_auc, holdout_auc, n_features, local_tables, target = run_rel_trial_study_outcome()

    print("downloaded_files:", downloaded, flush=True)
    print("local_data_path:", "tests/data/relbench/rel-trial", flush=True)
    print("local_tables_verified:", local_tables, flush=True)
    print("val_cut_date:", VAL_TIMESTAMP.date(), flush=True)
    print("test_cut_date:", TEST_TIMESTAMP.date(), flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
    print("label_period_days:", LABEL_DAYS, flush=True)
    print("target:", target, flush=True)
    print("val_rows:", len(df_val), flush=True)
    print("val_columns:", len(df_val.columns), flush=True)
    print("test_rows:", len(df_test), flush=True)
    print("test_columns:", len(df_test.columns), flush=True)
    print("feature_count:", n_features, flush=True)
    print("in_time_auc:", in_time_auc if in_time_auc is not None else "skipped", flush=True)
    print("holdout_auc:", holdout_auc if holdout_auc is not None else "skipped", flush=True)
    if _is_interactive_mode():
        print("val.columns:", df_val.columns, flush=True)


if __name__ == "__main__":
    main()
