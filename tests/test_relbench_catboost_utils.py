from __future__ import annotations

import pytest

from examples.relbench_catboost_utils import set_feature_families


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
