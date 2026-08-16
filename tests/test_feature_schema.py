from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from graphreduce.enum import ComputeLayerEnum
from graphreduce.feature_schema import FeatureRole, profile_feature_schema
from graphreduce.node import SQLNode


def _training_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "study_id": [1, 2, 3, 4],
            "sponsor_id": [10, 10, 11, 12],
            "event_time": pd.to_datetime(
                ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]
            ),
            "phase": ["I", "II", "II", "III"],
            "enrollment": [10.0, 25.0, 40.0, 100.0],
            "has_monitor": [True, False, True, False],
            "brief_title": [
                "A sufficiently descriptive title for the first study",
                "A sufficiently descriptive title for the second study",
                "A sufficiently descriptive title for the third study",
                "A sufficiently descriptive title for the fourth study",
            ],
            "outcome": [0, 1, 1, 0],
            "future_duration": [100, 200, 300, 400],
            "constant": [1, 1, 1, 1],
            "empty": [None, None, None, None],
        }
    )


def test_profile_feature_schema_emits_typed_cutoff_safe_manifest() -> None:
    manifest = profile_feature_schema(
        _training_sample(),
        primary_keys="study_id",
        foreign_keys="sponsor_id",
        date_keys="event_time",
        target_columns="outcome",
        unsafe_columns="future_duration",
        source="studies",
    )

    roles = {name: column.role for name, column in manifest.by_name.items()}
    assert roles["study_id"] == FeatureRole.PRIMARY_KEY
    assert roles["sponsor_id"] == FeatureRole.FOREIGN_KEY
    assert roles["event_time"] == FeatureRole.DATE_KEY
    assert roles["phase"] == FeatureRole.CATEGORICAL
    assert roles["enrollment"] == FeatureRole.NUMERICAL
    assert roles["has_monitor"] == FeatureRole.BOOLEAN
    assert roles["brief_title"] == FeatureRole.TEXT
    assert roles["outcome"] == FeatureRole.TARGET
    assert roles["future_duration"] == FeatureRole.UNSAFE

    assert manifest.graph_columns == (
        "study_id",
        "sponsor_id",
        "event_time",
        "phase",
        "enrollment",
        "has_monitor",
        "brief_title",
    )
    assert manifest.feature_columns == (
        "phase",
        "enrollment",
        "has_monitor",
        "brief_title",
    )
    assert manifest.categorical_columns == ("phase",)
    assert manifest.text_columns == ("brief_title",)
    assert manifest.categorical_indices() == (0,)
    assert manifest.select_feature_frame(_training_sample()).columns.tolist() == [
        "phase",
        "enrollment",
        "has_monitor",
        "brief_title",
    ]
    assert manifest.by_name["constant"].selected is False
    assert manifest.by_name["empty"].selected is False
    assert manifest.by_name["future_duration"].cutoff_safe is False


def test_profile_feature_schema_bounds_high_cardinality_categories() -> None:
    frame = pd.DataFrame(
        {
            "entity": list(range(10)),
            "category": [f"category-{index}" for index in range(10)],
        }
    )

    manifest = profile_feature_schema(
        frame,
        primary_keys="entity",
        max_categorical_cardinality=5,
        text_minimum_average_length=100,
    )

    category = manifest.by_name["category"]
    assert category.role == FeatureRole.CATEGORICAL
    assert category.selected is False
    assert "cardinality" in category.reason


def test_node_can_apply_profiled_graph_columns() -> None:
    frame = _training_sample()
    connection = sqlite3.connect(":memory:")
    frame.to_sql("studies", connection, index=False)
    node = SQLNode(
        fpath="studies",
        pk="study_id",
        prefix="std",
        date_key="event_time",
        client=connection,
        compute_layer=ComputeLayerEnum.sqlite,
    )

    manifest = node.infer_feature_manifest(
        frame,
        foreign_keys="sponsor_id",
        target_columns="outcome",
        unsafe_columns="future_duration",
        apply_columns=True,
    )

    assert node.feature_manifest is manifest
    assert node.columns == list(manifest.graph_columns)
    assert "outcome" not in node.columns
    assert "future_duration" not in node.columns
    connection.close()


def test_node_applies_unprefixed_columns_from_transformed_sample() -> None:
    frame = _training_sample().rename(
        columns={column: f"std_{column}" for column in _training_sample().columns}
    )
    connection = sqlite3.connect(":memory:")
    node = SQLNode(
        fpath="studies",
        pk="study_id",
        prefix="std",
        date_key="event_time",
        client=connection,
        compute_layer=ComputeLayerEnum.sqlite,
    )

    node.infer_feature_manifest(
        frame,
        foreign_keys="sponsor_id",
        target_columns="outcome",
        unsafe_columns="future_duration",
        apply_columns=True,
    )

    assert node.columns[0:3] == ["study_id", "sponsor_id", "event_time"]
    assert all(not column.startswith("std_") for column in node.columns)
    connection.close()


def test_profile_rejects_unknown_declared_columns() -> None:
    with pytest.raises(KeyError, match="absent"):
        profile_feature_schema(_training_sample(), unsafe_columns="not_present")
