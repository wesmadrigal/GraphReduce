from __future__ import annotations

import pandas as pd
import pytest

from examples.relbench_catboost_utils import (
    fit_incremental_regressor,
    set_feature_families,
)


class _Node:
    feature_families = None


def test_set_feature_families_deduplicates_and_preserves_order():
    nodes = [_Node(), _Node()]

    set_feature_families(nodes, ("temporal", "base", "temporal"))

    assert [node.feature_families for node in nodes] == [
        ("temporal", "base"),
        ("temporal", "base"),
    ]


def test_set_feature_families_rejects_unknown_family():
    with pytest.raises(ValueError, match="Unknown SQL auto-feature families"):
        set_feature_families([_Node()], ("base", "unknown"))


def test_incremental_regressor_skips_constant_target_frames():
    batches = [
        pd.DataFrame({"feature": [0, 1], "target": [0.0, 0.0]}),
        pd.DataFrame({"feature": [2, 3, 4], "target": [1.0, 2.0, 3.0]}),
    ]

    model, mae = fit_incremental_regressor(
        lambda: iter(batches),
        ["feature"],
        "target",
        pd.DataFrame({"feature": [1, 2]}),
        pd.Series([1.0, 2.0]),
        batch_count=len(batches),
        config={
            "iterations": 10,
            "depth": 2,
            "learning_rate": 0.1,
            "l2_leaf_reg": 3.0,
        },
    )

    assert model is not None
    assert mae >= 0.0
