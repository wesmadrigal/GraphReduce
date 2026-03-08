#!/usr/bin/env python
"""Run rel-hm user churn example and print summary for docs interactive mode."""

from __future__ import annotations

import os

from relbench_hm_user_churn import (
    CUT_DATE,
    LABEL_DAYS,
    LOOKBACK_DAYS,
    LOOKBACK_START,
    run_rel_hm_user_churn,
)


def _is_interactive_mode() -> bool:
    return os.getenv("GRAPHREDUCE_INTERACTIVE", "0").strip().lower() in {"1", "true", "yes"}


def main() -> None:
    print("Running rel-hm user churn example...", flush=True)
    df, auc, n_features, downloaded = run_rel_hm_user_churn()

    print("downloaded_files:", downloaded, flush=True)
    print("cut_date:", CUT_DATE.date(), flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
    print("lookback_days:", LOOKBACK_DAYS, flush=True)
    print("label_period_days:", LABEL_DAYS, flush=True)
    print("rows:", len(df), flush=True)
    print("columns:", len(df.columns), flush=True)
    print("feature_count:", n_features, flush=True)
    print("model_auc:", auc if auc is not None else "skipped", flush=True)
    if _is_interactive_mode():
        print("df.columns:", df.columns, flush=True)


if __name__ == "__main__":
    main()
