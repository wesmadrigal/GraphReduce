#!/usr/bin/env python
"""Run rel-avito ad-ctr example for docs interactive mode."""

from __future__ import annotations

import os

from relbench_avito_ad_ctr import (
    TRAIN_CUT_DATES,
    CUT_DATE,
    LABEL_PERIOD_DAYS,
    LOOKBACK_DAYS,
    LOOKBACK_START,
    TEST_CUT_DATE,
    VALIDATION_CUT_DATE,
    run_rel_avito_ad_ctr,
)


def _is_interactive_mode() -> bool:
    return os.getenv("GRAPHREDUCE_INTERACTIVE", "0").strip().lower() in {"1", "true", "yes"}


def main() -> None:
    print("Running rel-avito ad-ctr example...", flush=True)
    df_train, df_val, df_test, val_metrics, test_metrics, n_features, materialized, target = run_rel_avito_ad_ctr()
    print("materialized_files:", materialized, flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
    print("lookback_days:", LOOKBACK_DAYS, flush=True)
    print("train_cut_dates:", [cut_date.date() for cut_date in TRAIN_CUT_DATES], flush=True)
    print("train_timestamps:", df_train["timestamp"].nunique(), flush=True)
    print("validation_cut_date:", VALIDATION_CUT_DATE.date(), flush=True)
    print("test_cut_date:", TEST_CUT_DATE.date(), flush=True)
    print("cut_date:", CUT_DATE.date(), flush=True)
    print("label_period_days:", LABEL_PERIOD_DAYS, flush=True)
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
