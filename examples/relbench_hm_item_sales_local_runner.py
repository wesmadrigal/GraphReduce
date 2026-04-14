#!/usr/bin/env python
"""Run rel-hm item-sales example and print summary for docs interactive mode."""

from __future__ import annotations

import os

from relbench_hm_item_sales import (
    HOLDOUT_DATE,
    LABEL_DAYS,
    LOOKBACK_START,
    TEST_CUT_DATE,
    TRAIN_CUT_DATES,
    VALIDATION_CUT_DATE,
    run_rel_hm_item_sales,
)


def _is_interactive_mode() -> bool:
    return os.getenv("GRAPHREDUCE_INTERACTIVE", "0").strip().lower() in {"1", "true", "yes"}


def main() -> None:
    print("Running rel-hm item-sales example...", flush=True)
    df_train, df_val, df_test, val_metrics, test_metrics, n_features, materialized, target = run_rel_hm_item_sales()

    print("materialized_files:", materialized, flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
    print("train_cut_dates:", [cut_date.date() for cut_date in TRAIN_CUT_DATES], flush=True)
    print("train_timestamps:", df_train["timestamp"].nunique(), flush=True)
    print("validation_timestamp:", VALIDATION_CUT_DATE.date(), flush=True)
    print("test_timestamp:", TEST_CUT_DATE.date(), flush=True)
    print("holdout_timestamp:", HOLDOUT_DATE.date(), flush=True)
    print("label_period_days:", LABEL_DAYS, flush=True)
    print("target:", target, flush=True)
    print("train_rows:", len(df_train), flush=True)
    print("validation_rows:", len(df_val), flush=True)
    print("test_rows:", len(df_test), flush=True)
    print("columns:", len(df_train.columns), flush=True)
    print("feature_count:", n_features, flush=True)
    print("validation_metrics:", val_metrics if val_metrics is not None else "skipped", flush=True)
    print("test_metrics:", test_metrics if test_metrics is not None else "skipped", flush=True)
    if _is_interactive_mode():
        print("train.columns:", df_train.columns, flush=True)


if __name__ == "__main__":
    main()
