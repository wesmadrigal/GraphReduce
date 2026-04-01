#!/usr/bin/env python
"""RelBench rel-amazon: user churn end-to-end example."""

from __future__ import annotations

from pathlib import Path

from relbench_amazon_common import CUT_DATE, LABEL_PERIOD_DAYS, LOOKBACK_DAYS, LOOKBACK_START, run_amazon_task


def run_rel_amazon_user_churn(data_dir: Path | None = None):
    return run_amazon_task("user_churn", data_dir=data_dir)


def main() -> None:
    df, auc, n_features, materialized, target = run_rel_amazon_user_churn()
    print("materialized_files:", materialized, flush=True)
    print("cut_date:", CUT_DATE.date(), flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
    print("lookback_days:", LOOKBACK_DAYS, flush=True)
    print("label_period_days:", LABEL_PERIOD_DAYS, flush=True)
    print("target:", target, flush=True)
    print("rows:", len(df), flush=True)
    print("columns:", len(df.columns), flush=True)
    print("feature_count:", n_features, flush=True)
    print("model_auc:", auc if auc is not None else "skipped", flush=True)


if __name__ == "__main__":
    main()
