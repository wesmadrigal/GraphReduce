#!/usr/bin/env python
"""Run rel-hm item-sales example and print summary for docs interactive mode."""

from __future__ import annotations

import os

from relbench_hm_item_sales import EVAL_DATE, HOLDOUT_DATE, LABEL_DAYS, LOOKBACK_START, run_rel_hm_item_sales


def _is_interactive_mode() -> bool:
    return os.getenv("GRAPHREDUCE_INTERACTIVE", "0").strip().lower() in {"1", "true", "yes"}


def main() -> None:
    print("Running rel-hm item-sales example...", flush=True)
    df_eval, df_holdout, holdout_mae, n_features, downloaded, target = run_rel_hm_item_sales()

    print("downloaded_files:", downloaded, flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
    print("eval_timestamp:", EVAL_DATE.date(), flush=True)
    print("holdout_timestamp:", HOLDOUT_DATE.date(), flush=True)
    print("eval_lookback_days:", (EVAL_DATE - LOOKBACK_START).days, flush=True)
    print("holdout_lookback_days:", (HOLDOUT_DATE - LOOKBACK_START).days, flush=True)
    print("label_period_days:", LABEL_DAYS, flush=True)
    print("target:", target, flush=True)
    print("eval_rows:", len(df_eval), flush=True)
    print("holdout_rows:", len(df_holdout), flush=True)
    print("columns:", len(df_eval.columns), flush=True)
    print("feature_count:", n_features, flush=True)
    print("holdout_mae:", holdout_mae if holdout_mae is not None else "skipped", flush=True)
    if _is_interactive_mode():
        print("eval.columns:", df_eval.columns, flush=True)


if __name__ == "__main__":
    main()
