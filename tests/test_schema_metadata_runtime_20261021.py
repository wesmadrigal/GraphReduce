"""Run explicitly against the isolated clean GraphReduce candidate runtime."""

from copy import deepcopy
from types import SimpleNamespace

import duckdb
import pandas as pd
import pytest

from graphreduce.enum import ComputeLayerEnum
from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import DuckdbNode, SQLNode


@pytest.fixture
def node_factory():
    if not hasattr(DuckdbNode, "get_schema_frame"):
        pytest.skip("Only the isolated clean candidate runtime has this method")

    def make(connection):
        return DuckdbNode(fpath="samples", prefix="sample", pk="id", client=connection,
            compute_layer=ComputeLayerEnum.duckdb)
    return make


def test_clean_schema_api_keeps_cold_inference_and_warm_columns(node_factory):
    with duckdb.connect() as connection:
        connection.sql("CREATE TABLE samples AS SELECT 1 AS id, 3.2 AS value")
        original, candidate = node_factory(connection), node_factory(connection)
        pd.testing.assert_frame_equal(original.get_sample(), candidate.get_schema_frame(), check_exact=True)
        assert original._stypes == candidate._stypes
        types = deepcopy(candidate._stypes)
        schema = candidate.get_schema_frame()
        assert len(schema) == 0 and schema.columns.tolist() == ["id", "value"]
        assert candidate._stypes == types
        connection.sql("CREATE TABLE next_relation AS SELECT 2 AS id, 'x' AS other")
        candidate._cur_data_ref = "next_relation"
        assert candidate.get_schema_frame().columns.tolist() == ["id", "other"]


def test_custom_node_get_sample_is_never_bypassed(node_factory):
    with duckdb.connect() as connection:
        node = node_factory(connection)
        node._stypes = {"original": "categorical"}
        node.get_sample = lambda: pd.DataFrame({"custom": [7]})
        assert node.get_schema_frame().to_dict("list") == {"custom": [7]}


def test_other_engine_and_non_sql_custom_nodes_keep_original_behavior(node_factory):
    # SQLNode's default metadata operation is exactly the existing sample call.
    frame = pd.DataFrame({"custom": [9]})
    dummy = SimpleNamespace(get_sample=lambda: frame)
    assert SQLNode.get_schema_frame(dummy) is frame
    assert GraphReduce._join_schema_frame(dummy) is frame
