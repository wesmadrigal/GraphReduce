import datetime

import pytest

from graphreduce.enum import ComputeLayerEnum
from graphreduce.node import DuckdbNode


def _node(*, cutoff_boundary: str = "strict") -> DuckdbNode:
    return DuckdbNode(
        fpath="events",
        prefix="evt",
        pk="entity_id",
        date_key="timestamp",
        columns=["entity_id", "timestamp", "value"],
        compute_layer=ComputeLayerEnum.duckdb,
        cut_date=datetime.datetime(2025, 1, 2),
        cutoff_boundary=cutoff_boundary,
    )


def test_feature_cutoff_is_strict_by_default() -> None:
    filters = _node().prep_for_features()

    assert filters[0].opval == "evt_timestamp < '2025-01-02 00:00:00'"


def test_feature_cutoff_can_include_rows_at_the_cutoff() -> None:
    filters = _node(cutoff_boundary="inclusive").prep_for_features()

    assert filters[0].opval == "evt_timestamp <= '2025-01-02 00:00:00'"


def test_dynamic_entity_cutoff_honors_inclusive_boundary() -> None:
    date_node = DuckdbNode(
        fpath="cutoffs",
        prefix="cut",
        pk="entity_id",
        date_key="cutoff",
        columns=["entity_id", "cutoff"],
        compute_layer=ComputeLayerEnum.duckdb,
        is_date_node=True,
    )
    node = _node(cutoff_boundary="inclusive")
    node.date_node = date_node

    filters = node.prep_for_features()

    assert filters[0].opval == "evt_timestamp <= cut_cutoff"


def test_feature_cutoff_rejects_unknown_boundary() -> None:
    with pytest.raises(ValueError, match="cutoff_boundary"):
        _node(cutoff_boundary="unknown")
