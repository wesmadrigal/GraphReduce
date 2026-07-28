#!/usr/bin/env python
"""Run rel-trial study-outcome example for docs interactive mode."""

from __future__ import annotations

import os

from relbench_trial_study_outcome import (
    LABEL_DAYS,
    LOOKBACK_START,
    TEST_TIMESTAMP,
    VAL_TIMESTAMP,
    run_rel_trial_study_outcome,
)


def _is_interactive_mode() -> bool:
    return os.getenv("GRAPHREDUCE_INTERACTIVE", "0").strip().lower() in {"1", "true", "yes"}


def main() -> None:
    print("Running rel-trial study-outcome example...", flush=True)
    (
        df_train,
        df_val,
        df_test,
        catboost_in_time_auc,
        catboost_holdout_auc,
        n_features,
        materialized,
        target,
    ) = run_rel_trial_study_outcome()

    print("materialized_files:", materialized, flush=True)
    print("data_source:", "official relbench dataset", flush=True)
    print("val_cut_date:", VAL_TIMESTAMP.date(), flush=True)
    print("test_cut_date:", TEST_TIMESTAMP.date(), flush=True)
    print("lookback_start:", LOOKBACK_START.date(), flush=True)
    print("label_period_days:", LABEL_DAYS, flush=True)
    print("target:", target, flush=True)
    print("train_rows:", len(df_train), flush=True)
    print("train_timestamps:", df_train["timestamp"].nunique(), flush=True)
    print("val_rows:", len(df_val), flush=True)
    print("val_timestamps:", df_val["timestamp"].nunique(), flush=True)
    print("val_columns:", len(df_val.columns), flush=True)
    print("test_rows:", len(df_test), flush=True)
    print("test_timestamps:", df_test["timestamp"].nunique(), flush=True)
    print("test_columns:", len(df_test.columns), flush=True)
    print("feature_count:", n_features, flush=True)
    print("catboost_in_time_auc:", catboost_in_time_auc if catboost_in_time_auc is not None else "skipped", flush=True)
    print("catboost_holdout_auc:", catboost_holdout_auc if catboost_holdout_auc is not None else "skipped", flush=True)
    if _is_interactive_mode():
        print("val.columns:", df_val.columns, flush=True)


if __name__ == "__main__":
    main()
