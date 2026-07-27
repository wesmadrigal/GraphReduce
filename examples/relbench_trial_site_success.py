#!/usr/bin/env python
"""RelBench rel-trial: site success example using official task tables."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from relbench_trial_task_utils import build_site_features, run_rel_trial_regression_task


def run_rel_trial_site_success(
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float] | None, dict[str, float] | None, int, list[str], str]:
    return run_rel_trial_regression_task(
        task_name="site-success",
        feature_builder=build_site_features,
        feature_entity_col="fac_facility_id",
        data_dir=data_dir,
        max_train_frames=5,
    )


def main() -> None:
    df_train, df_val, df_test, val_metrics, test_metrics, n_features, materialized, target = (
        run_rel_trial_site_success()
    )
    print("materialized_files:", materialized, flush=True)
    print("task:", "site-success", flush=True)
    print("target:", target, flush=True)
    print("train_rows:", len(df_train), flush=True)
    print("validation_rows:", len(df_val), flush=True)
    print("test_rows:", len(df_test), flush=True)
    print("train_timestamps:", df_train.column_nunique("timestamp"), flush=True)
    print("validation_timestamps:", df_val["timestamp"].nunique(), flush=True)
    print("test_timestamps:", df_test["timestamp"].nunique(), flush=True)
    print("columns:", len(df_train.columns), flush=True)
    print("feature_count:", n_features, flush=True)
    print("validation_nmae:", val_metrics["nmae"] if val_metrics is not None else "skipped", flush=True)
    print("test_nmae:", test_metrics["nmae"] if test_metrics is not None else "skipped", flush=True)
    print("validation_metrics:", val_metrics if val_metrics is not None else "skipped", flush=True)
    print("test_metrics:", test_metrics if test_metrics is not None else "skipped", flush=True)


if __name__ == "__main__":
    main()
