#!/usr/bin/env python
"""Run rel-avito ad-ctr example for docs interactive mode."""

from __future__ import annotations

import os

from relbench_avito_ad_ctr import (
    CUT_DATE,
    LABEL_PERIOD_DAYS,
    LOOKBACK_DAYS,
    LOOKBACK_START,
    run_rel_avito_ad_ctr,
)


def _is_interactive_mode() -> bool:
    return os.getenv("GRAPHREDUCE_INTERACTIVE", "0").strip().lower() in {"1", "true", "yes"}


def main() -> None:
    print("Running rel-avito ad-ctr example...", flush=True)
    df, mae, n_features, downloaded, target = run_rel_avito_ad_ctr()
    print("downloaded_files:", downloaded, flush=True)
    print("cut_date:", CUT_DATE.date(), flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
    print("lookback_days:", LOOKBACK_DAYS, flush=True)
    print("label_period_days:", LABEL_PERIOD_DAYS, flush=True)
    print("target:", target, flush=True)
    print("rows:", len(df), flush=True)
    print("columns:", len(df.columns), flush=True)
    print("feature_count:", n_features, flush=True)
    print("model_mae:", mae if mae is not None else "skipped", flush=True)
    if _is_interactive_mode():
        print("df.columns:", df.columns, flush=True)


if __name__ == "__main__":
    main()
