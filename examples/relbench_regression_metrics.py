#!/usr/bin/env python
"""Regression metrics shared by RelBench examples."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def _as_float_array(values: Any) -> np.ndarray:
    return pd.Series(values).fillna(0).astype("float64").to_numpy()


def train_split_target_std(train_target: Any) -> float:
    if np.isscalar(train_target):
        std = float(train_target)
        if not np.isfinite(std) or std <= 0:
            raise ValueError("Cannot compute NMAE with a zero or non-finite train target standard deviation.")
        return std
    values = _as_float_array(train_target)
    std = float(np.std(values))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Cannot compute NMAE with a zero or non-finite train target standard deviation.")
    return std


def normalized_mae(y_true: Any, y_pred: Any, train_target: Any) -> float:
    truth = _as_float_array(y_true)
    pred = _as_float_array(y_pred)
    if truth.shape != pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape; got {truth.shape} and {pred.shape}.")
    return float(np.mean(np.abs(truth - pred)) / train_split_target_std(train_target))


def add_nmae(metrics: Mapping[str, float], y_true: Any, y_pred: Any, train_target: Any) -> dict[str, float]:
    metrics_with_nmae = dict(metrics)
    metrics_with_nmae["nmae"] = normalized_mae(y_true, y_pred, train_target)
    return metrics_with_nmae
