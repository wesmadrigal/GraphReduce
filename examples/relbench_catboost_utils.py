"""Validation-tuned CatBoost helpers for the RelBench examples."""

from __future__ import annotations

from math import ceil
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score


CLASSIFIER_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "depth6_regularized",
        "iterations": 2000,
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 10.0,
    },
    {
        "name": "depth7_regularized",
        "iterations": 1500,
        "depth": 7,
        "learning_rate": 0.03,
        "l2_leaf_reg": 10.0,
    },
    {
        "name": "depth5_regularized",
        "iterations": 2500,
        "depth": 5,
        "learning_rate": 0.02,
        "l2_leaf_reg": 12.0,
    },
    {
        "name": "depth6_fast",
        "iterations": 1000,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 12.0,
    },
)

REGRESSOR_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "depth6_regularized",
        "iterations": 2000,
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 10.0,
    },
    {
        "name": "depth7_regularized",
        "iterations": 1500,
        "depth": 7,
        "learning_rate": 0.03,
        "l2_leaf_reg": 10.0,
    },
    {
        "name": "depth5_regularized",
        "iterations": 2500,
        "depth": 5,
        "learning_rate": 0.02,
        "l2_leaf_reg": 12.0,
    },
    {
        "name": "depth6_fast",
        "iterations": 1000,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 12.0,
    },
)

ALL_FEATURE_FAMILIES = (
    "base",
    "semantic",
    "conditional",
    "temporal",
    "sequence",
    "episode",
    "context",
)
TEMPORAL_FEATURE_FAMILIES = (
    "base",
    "semantic",
    "conditional",
    "temporal",
    "sequence",
    "episode",
    "context",
)


def enable_all_feature_families(nodes: Iterable[Any]) -> None:
    """Enable every SQL feature family on a graph's nodes."""

    for node in nodes:
        node.feature_families = ALL_FEATURE_FAMILIES


def set_feature_families(nodes: Iterable[Any], families: Sequence[str]) -> None:
    """Select SQL auto-feature families without adding custom feature logic."""

    selected = tuple(dict.fromkeys(families))
    unknown = set(selected) - set(ALL_FEATURE_FAMILIES)
    if unknown:
        raise ValueError(f"Unknown SQL auto-feature families: {sorted(unknown)}")
    for node in nodes:
        node.feature_families = selected


def fit_tuned_classifier(
    train_inputs: pd.DataFrame,
    train_target: pd.Series,
    val_inputs: pd.DataFrame,
    val_target: pd.Series,
    *,
    cat_features: Sequence[int] | None = None,
    auto_class_weights: str | None = None,
    configs: Sequence[dict[str, Any]] = CLASSIFIER_CONFIGS,
) -> tuple[CatBoostClassifier, dict[str, Any], float]:
    """Select a classifier by validation AUROC without touching the test split."""

    best_model: CatBoostClassifier | None = None
    best_config: dict[str, Any] | None = None
    best_auc = float("-inf")
    for config in configs:
        params = {
            "iterations": int(config["iterations"]),
            "depth": int(config["depth"]),
            "learning_rate": float(config["learning_rate"]),
            "l2_leaf_reg": float(config["l2_leaf_reg"]),
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": 42,
            "random_strength": 1.0,
            "bagging_temperature": 1.0,
            "verbose": False,
            "allow_writing_files": False,
        }
        if auto_class_weights is not None:
            params["auto_class_weights"] = auto_class_weights
        model = CatBoostClassifier(**params)
        model.fit(
            train_inputs,
            train_target,
            cat_features=cat_features,
            eval_set=(val_inputs, val_target),
            use_best_model=True,
            early_stopping_rounds=200,
        )
        predictions = model.predict_proba(val_inputs)[:, 1]
        auc = float(roc_auc_score(val_target.to_numpy(dtype="float64"), predictions))
        if auc > best_auc:
            best_model = model
            best_config = config
            best_auc = auc

    if best_model is None or best_config is None:
        raise RuntimeError("CatBoost classifier tuning produced no model")
    return best_model, best_config, best_auc


def fit_tuned_regressor(
    train_inputs: pd.DataFrame,
    train_target: pd.Series,
    val_inputs: pd.DataFrame,
    val_target: pd.Series,
    *,
    configs: Sequence[dict[str, Any]] = REGRESSOR_CONFIGS,
) -> tuple[CatBoostRegressor, dict[str, Any], float]:
    """Select a regressor by validation MAE without touching the test split."""

    best_model: CatBoostRegressor | None = None
    best_config: dict[str, Any] | None = None
    best_mae = float("inf")
    for config in configs:
        model = CatBoostRegressor(
            iterations=int(config["iterations"]),
            depth=int(config["depth"]),
            learning_rate=float(config["learning_rate"]),
            l2_leaf_reg=float(config["l2_leaf_reg"]),
            loss_function="MAE",
            eval_metric="MAE",
            random_seed=42,
            random_strength=1.0,
            bagging_temperature=1.0,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            train_inputs,
            train_target,
            eval_set=(val_inputs, val_target),
            use_best_model=True,
            early_stopping_rounds=200,
        )
        predictions = np.asarray(model.predict(val_inputs), dtype="float64")
        mae = float(mean_absolute_error(val_target.to_numpy(dtype="float64"), predictions))
        if mae < best_mae:
            best_model = model
            best_config = config
            best_mae = mae

    if best_model is None or best_config is None:
        raise RuntimeError("CatBoost regressor tuning produced no model")
    return best_model, best_config, best_mae


def _incremental_model_params(config: dict[str, Any], loss_function: str, iterations: int) -> dict[str, Any]:
    return {
        "iterations": iterations,
        "depth": int(config["depth"]),
        "learning_rate": float(config["learning_rate"]),
        "l2_leaf_reg": float(config["l2_leaf_reg"]),
        "loss_function": loss_function,
        "random_seed": 42,
        "random_strength": 1.0,
        "bagging_temperature": 1.0,
        "verbose": False,
        "allow_writing_files": False,
    }


def fit_incremental_classifier(
    train_batch_factory: Callable[[], Iterator[pd.DataFrame]],
    feature_columns: Sequence[str],
    target_column: str,
    val_inputs: pd.DataFrame,
    val_target: pd.Series,
    *,
    batch_count: int,
    config: dict[str, Any],
    cat_features: Sequence[int] | None = None,
    auto_class_weights: str | None = None,
) -> tuple[CatBoostClassifier, float]:
    """Fit one CatBoost classifier by continuing across disk-backed batches."""

    iterations = max(1, ceil(int(config["iterations"]) / max(1, batch_count)))
    params = _incremental_model_params(config, "Logloss", iterations)
    if auto_class_weights is not None:
        params["auto_class_weights"] = auto_class_weights

    model: CatBoostClassifier | None = None
    for batch in train_batch_factory():
        target = batch[target_column]
        if target.nunique(dropna=True) < 2:
            continue
        inputs = batch.reindex(columns=list(feature_columns), fill_value=0).fillna(0)
        next_model = CatBoostClassifier(**params)
        next_model.fit(
            inputs,
            target,
            cat_features=cat_features,
            init_model=model,
            use_best_model=False,
        )
        model = next_model

    if model is None:
        raise RuntimeError("Incremental classifier received no batch with two target classes")
    predictions = model.predict_proba(val_inputs)[:, 1]
    return model, float(roc_auc_score(val_target.to_numpy(dtype="float64"), predictions))


def fit_incremental_regressor(
    train_batch_factory: Callable[[], Iterator[pd.DataFrame]],
    feature_columns: Sequence[str],
    target_column: str,
    val_inputs: pd.DataFrame,
    val_target: pd.Series,
    *,
    batch_count: int,
    config: dict[str, Any],
) -> tuple[CatBoostRegressor, float]:
    """Fit one CatBoost regressor by continuing across disk-backed batches."""

    iterations = max(1, ceil(int(config["iterations"]) / max(1, batch_count)))
    params = _incremental_model_params(config, "MAE", iterations)
    model: CatBoostRegressor | None = None
    for batch in train_batch_factory():
        target = batch[target_column]
        # CatBoost cannot fit a regression batch whose target has no
        # variation. Sparse early cutoff frames can legitimately have this
        # shape; skip them and continue the incremental model on the next
        # informative frame.
        if target.nunique(dropna=True) < 2:
            continue
        inputs = batch.reindex(columns=list(feature_columns), fill_value=0).fillna(0)
        next_model = CatBoostRegressor(**params)
        next_model.fit(
            inputs,
            target.astype("float64"),
            init_model=model,
            use_best_model=False,
        )
        model = next_model

    if model is None:
        raise RuntimeError(
            "Incremental regressor received no batch with varying targets"
        )
    predictions = np.asarray(model.predict(val_inputs), dtype="float64")
    return model, float(mean_absolute_error(val_target.to_numpy(dtype="float64"), predictions))


def fit_tuned_classifier_incremental(
    train_batch_factory: Callable[[], Iterator[pd.DataFrame]],
    feature_columns: Sequence[str],
    target_column: str,
    val_inputs: pd.DataFrame,
    val_target: pd.Series,
    *,
    batch_count: int,
    cat_features: Sequence[int] | None = None,
    auto_class_weights: str | None = None,
    configs: Sequence[dict[str, Any]] = CLASSIFIER_CONFIGS,
) -> tuple[CatBoostClassifier, dict[str, Any], float]:
    best_model: CatBoostClassifier | None = None
    best_config: dict[str, Any] | None = None
    best_auc = float("-inf")
    for config in configs:
        model, auc = fit_incremental_classifier(
            train_batch_factory,
            feature_columns,
            target_column,
            val_inputs,
            val_target,
            batch_count=batch_count,
            config=config,
            cat_features=cat_features,
            auto_class_weights=auto_class_weights,
        )
        if auc > best_auc:
            best_model, best_config, best_auc = model, config, auc
    if best_model is None or best_config is None:
        raise RuntimeError("Incremental classifier tuning produced no model")
    return best_model, best_config, best_auc


def fit_tuned_regressor_incremental(
    train_batch_factory: Callable[[], Iterator[pd.DataFrame]],
    feature_columns: Sequence[str],
    target_column: str,
    val_inputs: pd.DataFrame,
    val_target: pd.Series,
    *,
    batch_count: int,
    configs: Sequence[dict[str, Any]] = REGRESSOR_CONFIGS,
) -> tuple[CatBoostRegressor, dict[str, Any], float]:
    best_model: CatBoostRegressor | None = None
    best_config: dict[str, Any] | None = None
    best_mae = float("inf")
    for config in configs:
        model, mae = fit_incremental_regressor(
            train_batch_factory,
            feature_columns,
            target_column,
            val_inputs,
            val_target,
            batch_count=batch_count,
            config=config,
        )
        if mae < best_mae:
            best_model, best_config, best_mae = model, config, mae
    if best_model is None or best_config is None:
        raise RuntimeError("Incremental regressor tuning produced no model")
    return best_model, best_config, best_mae
