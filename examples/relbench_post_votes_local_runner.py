#!/usr/bin/env python
"""Run rel-stack post-votes with RelBench task tables."""

from __future__ import annotations

import duckdb

from relbench_stack_task_utils import (
    build_post_votes_features,
    build_task_split_frame,
    materialize_rel_stack,
    prepare_stack_views,
    select_shared_numeric_features,
    target_table_from_frame,
)
from relbench_regression_metrics import add_nmae
from relbench_catboost_utils import fit_incremental_regressor


def _print_steps_summary(materialized_files: list[str], result_text: str) -> None:
    print("\nSteps completed:", flush=True)
    print(
        f"1. Loaded official RelBench DB views: {len(materialized_files)} materialized file(s).",
        flush=True,
    )
    print("2. Built train/validation/test GraphReduce frames at task timestamps.", flush=True)
    print("3. Joined RelBench task-table labels to each feature frame.", flush=True)
    print("4. Trained CatBoost and evaluated with the RelBench task metrics.", flush=True)
    print(f"5. Achieved the following result: {result_text}", flush=True)


def main() -> None:
    materialized_files = materialize_rel_stack()

    con = duckdb.connect()
    prepare_stack_views(con)
    print("Starting rel-stack post-votes pipeline...", flush=True)

    task, _, train_store, train_cut_date = build_task_split_frame(
        con,
        task_name="post-votes",
        split="train",
        feature_builder=build_post_votes_features,
        feature_entity_col="post_Id",
        use_all_timestamps=True,
    )
    _, _, val_store, val_cut_date = build_task_split_frame(
        con,
        task_name="post-votes",
        split="val",
        feature_builder=build_post_votes_features,
        feature_entity_col="post_Id",
    )
    _, _, test_store, test_cut_date = build_task_split_frame(
        con,
        task_name="post-votes",
        split="test",
        feature_builder=build_post_votes_features,
        feature_entity_col="post_Id",
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
        excluded_cols={"post_Id", "post_OwnerUserId", task.entity_col},
    )

    print("train_cut_date:", train_cut_date.date(), flush=True)
    print("validation_cut_date:", val_cut_date.date(), flush=True)
    print("test_cut_date:", test_cut_date.date(), flush=True)
    print("train_timestamps:", train_store.column_nunique(task.time_col), flush=True)
    print("validation_timestamps:", df_val[task.time_col].nunique(), flush=True)
    print("test_timestamps:", df_test[task.time_col].nunique(), flush=True)
    print("train_rows:", train_store.row_count, flush=True)
    print("validation_rows:", len(df_val), flush=True)
    print("test_rows:", len(df_test), flush=True)
    print("target:", target, flush=True)
    print("num_features:", len(features), flush=True)

    if not features:
        print("no numerical features; skipping model fit", flush=True)
        _print_steps_summary(
            materialized_files, "model fit skipped due to no numerical features"
        )
        con.close()
        return

    model, _ = fit_incremental_regressor(
        lambda: train_store.iter_batches(),
        features,
        target,
        df_val[features].fillna(0),
        df_val[target],
        batch_count=len(train_store.part_paths),
        config={
            "iterations": 1000,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
        },
    )

    val_pred = model.predict(df_val[features].fillna(0))
    test_pred = model.predict(df_test[features].fillna(0))
    val_metrics = add_nmae(
        task.evaluate(val_pred, target_table_from_frame(task, df_val)),
        df_val[target],
        val_pred,
        train_store.target_std(target),
    )
    test_metrics = add_nmae(
        task.evaluate(test_pred, target_table_from_frame(task, df_test)),
        df_test[target],
        test_pred,
        train_store.target_std(target),
    )

    print("validation_nmae:", val_metrics["nmae"], flush=True)
    print("test_nmae:", test_metrics["nmae"], flush=True)
    print("validation_metrics:", val_metrics, flush=True)
    print("test_metrics:", test_metrics, flush=True)
    _print_steps_summary(
        materialized_files,
        (
            f"CatBoost validation metrics = {val_metrics}; "
            f"CatBoost test metrics = {test_metrics}"
        ),
    )
    train_store.close()
    con.close()


if __name__ == "__main__":
    main()
