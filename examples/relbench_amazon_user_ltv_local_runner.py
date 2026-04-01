#!/usr/bin/env python
"""Run rel-amazon user-LTV example for docs interactive mode."""

from __future__ import annotations

import os

from relbench_amazon_common import HOLDOUT_CUT_DATE, LABEL_PERIOD_DAYS, LOOKBACK_START, VALIDATION_CUT_DATE
from relbench_amazon_user_ltv import run_rel_amazon_user_ltv


def _is_interactive_mode() -> bool:
    return os.getenv("GRAPHREDUCE_INTERACTIVE", "0").strip().lower() in {"1", "true", "yes"}


def main() -> None:
    print("Running rel-amazon user-LTV example...", flush=True)
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
    if _is_interactive_mode():
        print("validation.columns:", df_val.columns, flush=True)


if __name__ == "__main__":
    main()
