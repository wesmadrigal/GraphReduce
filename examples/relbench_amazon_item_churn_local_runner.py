#!/usr/bin/env python
"""Run rel-amazon item-churn example for docs interactive mode."""

from __future__ import annotations

import os

from relbench_amazon_common import (
    CUT_DATE,
    LABEL_PERIOD_DAYS,
    LOOKBACK_DAYS,
    LOOKBACK_START,
    VALIDATION_CUT_DATE,
)
from relbench_amazon_item_churn import run_rel_amazon_item_churn


def _is_interactive_mode() -> bool:
    return os.getenv("GRAPHREDUCE_INTERACTIVE", "0").strip().lower() in {"1", "true", "yes"}


def main() -> None:
    print("Running rel-amazon item-churn example...", flush=True)
    (
        df_train,
        df_val,
        df_test,
        val_metrics,
        test_metrics,
        n_features,
        materialized,
        target,
    ) = run_rel_amazon_item_churn()
    print("materialized_files:", materialized, flush=True)
    print("validation_cut_date:", VALIDATION_CUT_DATE.date(), flush=True)
    print("test_cut_date:", CUT_DATE.date(), flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
    print("lookback_days:", LOOKBACK_DAYS, flush=True)
    print("label_period_days:", LABEL_PERIOD_DAYS, flush=True)
    print("target:", target, flush=True)
    print("train_rows:", df_train.row_count, flush=True)
    print("validation_rows:", len(df_val), flush=True)
    print("test_rows:", len(df_test), flush=True)
    print("feature_count:", n_features, flush=True)
    print("validation_metrics:", val_metrics if val_metrics is not None else "skipped", flush=True)
    print("test_metrics:", test_metrics if test_metrics is not None else "skipped", flush=True)
    if _is_interactive_mode():
        print("df_test.columns:", df_test.columns, flush=True)
    df_train.close()


if __name__ == "__main__":
    main()
