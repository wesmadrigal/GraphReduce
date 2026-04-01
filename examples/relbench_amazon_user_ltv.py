#!/usr/bin/env python
"""RelBench rel-amazon: user LTV end-to-end example."""

from __future__ import annotations

from pathlib import Path

from relbench_amazon_common import (
    HOLDOUT_CUT_DATE,
    LABEL_PERIOD_DAYS,
    LOOKBACK_START,
    VALIDATION_CUT_DATE,
    run_amazon_temporal_regression_task,
)


def run_rel_amazon_user_ltv(data_dir: Path | None = None):
    return run_amazon_temporal_regression_task("user_ltv", data_dir=data_dir)


def main() -> None:
    df_val, df_holdout, holdout_mae, n_features, materialized, target = run_rel_amazon_user_ltv()
    print("materialized_files:", materialized, flush=True)
    print("validation_cut_date:", VALIDATION_CUT_DATE.date(), flush=True)
    print("holdout_cut_date:", HOLDOUT_CUT_DATE.date(), flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
    print("validation_lookback_days:", (VALIDATION_CUT_DATE - LOOKBACK_START).days + 1, flush=True)
    print("holdout_lookback_days:", (HOLDOUT_CUT_DATE - LOOKBACK_START).days + 1, flush=True)
    print("label_period_days:", LABEL_PERIOD_DAYS, flush=True)
    print("target:", target, flush=True)
    print("validation_rows:", len(df_val), flush=True)
    print("holdout_rows:", len(df_holdout), flush=True)
    print("columns:", len(df_val.columns), flush=True)
    print("feature_count:", n_features, flush=True)
    print("holdout_mae:", holdout_mae if holdout_mae is not None else "skipped", flush=True)


if __name__ == "__main__":
    main()
