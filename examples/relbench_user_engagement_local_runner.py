#!/usr/bin/env python
"""Run rel-stack user-engagement with RelBench task tables."""

from __future__ import annotations

import duckdb

from relbench_stack_task_utils import (
    build_task_split_frame,
    build_user_engagement_features,
    materialize_rel_stack,
    prepare_stack_views,
    select_shared_numeric_features,
    target_table_from_frame,
)
from relbench_catboost_utils import fit_incremental_classifier


def main() -> None:
    materialized_files = materialize_rel_stack()

    con = duckdb.connect()
    prepare_stack_views(con)
    print("Running rel-stack user-engagement task...", flush=True)

    task, _, train_store, train_cut_date = build_task_split_frame(
        con,
        task_name="user-engagement",
        split="train",
        feature_builder=build_user_engagement_features,
        feature_entity_col="user_Id",
        use_all_timestamps=True,
    )
    _, _, val_store, val_cut_date = build_task_split_frame(
        con,
        task_name="user-engagement",
        split="val",
        feature_builder=build_user_engagement_features,
        feature_entity_col="user_Id",
    )
    _, _, test_store, test_cut_date = build_task_split_frame(
        con,
        task_name="user-engagement",
        split="test",
        feature_builder=build_user_engagement_features,
        feature_entity_col="user_Id",
    )

    df_val = val_store.to_dataframe()
    df_test = test_store.to_dataframe()
    val_store.close()
    test_store.close()
    train_sample = train_store.sample_frame()
    target = task.target_col
    features = select_shared_numeric_features(
        train_sample,
        df_val,
        df_test,
        target_col=target,
        excluded_cols={"user_Id", "user_AccountId", task.entity_col},
    )

    print("materialized_files:", materialized_files, flush=True)
    print("train_cut_date:", train_cut_date.date(), flush=True)
    print("validation_cut_date:", val_cut_date.date(), flush=True)
    print("test_cut_date:", test_cut_date.date(), flush=True)
    print("target:", target, flush=True)
    print("train_rows:", train_store.row_count, flush=True)
    print("validation_rows:", len(df_val), flush=True)
    print("test_rows:", len(df_test), flush=True)
    print("feature_count:", len(features), flush=True)

    if not features or train_store.target_nunique(target) < 2:
        print(
            "insufficient features or single-class training target; skipping model fit",
            flush=True,
        )
        con.close()
        return

    model, _ = fit_incremental_classifier(
        lambda: train_store.iter_batches(),
        features,
        target,
        df_val[features].fillna(0),
        df_val[target],
        batch_count=len(train_store.part_paths),
        config={
            "iterations": 300,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
        },
        auto_class_weights="Balanced",
    )

    val_pred = model.predict_proba(df_val[features].fillna(0))[:, 1]
    test_pred = model.predict_proba(df_test[features].fillna(0))[:, 1]

    val_metrics = task.evaluate(val_pred, target_table_from_frame(task, df_val))
    test_metrics = task.evaluate(test_pred, target_table_from_frame(task, df_test))

    print("validation_metrics:", val_metrics, flush=True)
    print("test_metrics:", test_metrics, flush=True)
    train_store.close()
    con.close()


if __name__ == "__main__":
    main()
