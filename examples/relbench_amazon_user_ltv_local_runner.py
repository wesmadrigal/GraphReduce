#!/usr/bin/env python
"""Run rel-amazon user-LTV example for docs interactive mode."""

from __future__ import annotations

import os

from relbench_amazon_user_ltv import (
    LABEL_PERIOD_DAYS,
    LOOKBACK_START,
    TEST_CUT_DATE,
    VALIDATION_CUT_DATE,
    run_rel_amazon_user_ltv,
)


def _is_interactive_mode() -> bool:
    return os.getenv("GRAPHREDUCE_INTERACTIVE", "0").strip().lower() in {"1", "true", "yes"}


def main() -> None:
    print("Running rel-amazon user-LTV example...", flush=True)
    df_train, df_val, df_test, val_metrics, test_metrics, n_features, materialized, target = run_rel_amazon_user_ltv()
    print("materialized_files:", materialized, flush=True)
    print("validation_cut_date:", VALIDATION_CUT_DATE.date(), flush=True)
    print("test_cut_date:", TEST_CUT_DATE.date(), flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
    print("train_timestamps:", df_train["timestamp"].nunique(), flush=True)
    print("label_period_days:", LABEL_PERIOD_DAYS, flush=True)
    print("target:", target, flush=True)
    print("train_rows:", len(df_train), flush=True)
    print("validation_rows:", len(df_val), flush=True)
    print("test_rows:", len(df_test), flush=True)
    print("columns:", len(df_train.columns), flush=True)
    print("feature_count:", n_features, flush=True)
    print("validation_nmae:", val_metrics["nmae"] if val_metrics is not None else "skipped", flush=True)
    print("test_nmae:", test_metrics["nmae"] if test_metrics is not None else "skipped", flush=True)
    print("validation_metrics:", val_metrics if val_metrics is not None else "skipped", flush=True)
    print("test_metrics:", test_metrics if test_metrics is not None else "skipped", flush=True)
    if _is_interactive_mode():
        print("train.columns:", df_train.columns, flush=True)


if __name__ == "__main__":
    main()
