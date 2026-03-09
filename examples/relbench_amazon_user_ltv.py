#!/usr/bin/env python
"""RelBench rel-amazon: user LTV end-to-end example."""

from __future__ import annotations

from relbench_amazon_common import (
    HOLDOUT_CUT_DATE,
    LABEL_PERIOD_DAYS,
    LOOKBACK_START,
    VALIDATION_CUT_DATE,
    run_amazon_temporal_regression_task,
)


def main() -> None:
    df_eval, df_holdout, holdout_mae, n_features, downloaded, target = run_amazon_temporal_regression_task("user_ltv")
    eval_lookback_days = (VALIDATION_CUT_DATE - LOOKBACK_START).days + 1
    holdout_lookback_days = (HOLDOUT_CUT_DATE - LOOKBACK_START).days + 1
    print("downloaded_files:", downloaded, flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
    print("validation_timestamp:", VALIDATION_CUT_DATE.date(), flush=True)
    print("holdout_timestamp:", HOLDOUT_CUT_DATE.date(), flush=True)
    print("validation_lookback_days:", eval_lookback_days, flush=True)
    print("holdout_lookback_days:", holdout_lookback_days, flush=True)
    print("label_period_days:", LABEL_PERIOD_DAYS, flush=True)
    print("target:", target, flush=True)
    print("eval_rows:", len(df_eval), flush=True)
    print("holdout_rows:", len(df_holdout), flush=True)
    print("columns:", len(df_eval.columns), flush=True)
    print("feature_count:", n_features, flush=True)
    print("holdout_mae:", holdout_mae if holdout_mae is not None else "skipped", flush=True)


if __name__ == "__main__":
    main()
