import sqlite3

import duckdb
import pandas as pd
import pytest
from dask import dataframe as dd

from graphreduce.enum import ComputeLayerEnum, SQLOpType
from graphreduce.graph_reduce import GraphReduce
from graphreduce.models import sqlop
from graphreduce.node import DuckdbNode, DynamicNode, SQLNode


class FrameNode(DynamicNode):
    def do_data(self):
        self.calls.append((self.prefix, "do_data"))

    def do_annotate(self):
        self.calls.append((self.prefix, "do_annotate"))
        self.df[self.colabbr("annotated")] = 1

    def do_filters(self):
        self.calls.append((self.prefix, "do_filters"))
        if self.drop_rows:
            self.df = self.df.head(0)

    def do_normalize(self):
        self.calls.append((self.prefix, "do_normalize"))

    def do_reduce(self, reduce_key):
        self.calls.append((self.prefix, "do_reduce"))
        return (
            self.df.groupby(list(self.colabbrs(reduce_key)))
            .size()
            .to_frame(name=self.colabbr("nrows"))
            .reset_index()
        )

    def do_post_join_annotate(self):
        self.calls.append((self.prefix, "do_post_join_annotate"))

    def do_post_join_filters(self):
        self.calls.append((self.prefix, "do_post_join_filters"))


class SQLHooks:
    def do_data(self):
        self.calls.append((self.prefix, "do_data"))
        return super().do_data()

    def do_annotate(self):
        self.calls.append((self.prefix, "do_annotate"))
        return [
            sqlop(optype=SQLOpType.select, opval=f"*, 1 AS {self.prefix}_annotated")
        ]

    def do_filters(self):
        self.calls.append((self.prefix, "do_filters"))
        if self.drop_rows:
            return [sqlop(optype=SQLOpType.where, opval="false")]

    def do_normalize(self):
        self.calls.append((self.prefix, "do_normalize"))

    def do_reduce(self, reduce_key):
        self.calls.append((self.prefix, "do_reduce"))
        return [
            sqlop(optype=SQLOpType.agg, opval=self.key_sql(reduce_key)),
            sqlop(optype=SQLOpType.aggfunc, opval=f"count(*) AS {self.prefix}_nrows"),
        ]

    def do_post_join_annotate(self):
        self.calls.append((self.prefix, "do_post_join_annotate"))

    def do_post_join_filters(self):
        self.calls.append((self.prefix, "do_post_join_filters"))


class DuckNode(SQLHooks, DuckdbNode):
    pass


class SQLiteNode(SQLHooks, SQLNode):
    pass


@pytest.fixture(params=["pandas", "dask", "duckdb", "sqlite"])
def backend(request):
    return request.param


def make_graph(backend, frames=None, sample_size=None):
    if frames is None:
        frames = {
            "root": pd.DataFrame({"id": [1, 2, 3]}),
            "child": pd.DataFrame({"id": [10, 11, 20], "parent_id": [1, 1, 2]}),
            "leaf": pd.DataFrame(
                {"id": [100, 101, 102, 200], "parent_id": [10, 10, 11, 20]}
            ),
        }
    layer = ComputeLayerEnum(backend)
    connection = (
        duckdb.connect()
        if backend == "duckdb"
        else sqlite3.connect(":memory:")
        if backend == "sqlite"
        else None
    )
    calls = []
    nodes = {}
    for prefix, frame in frames.items():
        kwargs = dict(prefix=prefix, pk="id", compute_layer=layer)
        if connection:
            if backend == "duckdb":
                connection.register("input_frame", frame)
                connection.sql(
                    f"CREATE TABLE {prefix}_source AS SELECT * FROM input_frame"
                )
            else:
                frame.to_sql(f"{prefix}_source", connection, index=False)
            cls = DuckNode if backend == "duckdb" else SQLiteNode
            node = cls(
                fpath=f"{prefix}_source",
                columns=list(frame.columns),
                client=connection,
                **kwargs,
            )
        else:
            node = FrameNode(**kwargs)
            node.df = frame.add_prefix(f"{prefix}_")
            if backend == "dask":
                node.df = dd.from_pandas(node.df, npartitions=2)
        node.calls = calls
        node.drop_rows = False
        nodes[prefix] = node
    graph = GraphReduce(
        parent_node=nodes["root"],
        compute_layer=layer,
        sql_client=connection,
        sample_size=sample_size,
        auto_feature_hops_back=10,
    )
    graph.add_nodes_from(nodes.values())
    if "child" in nodes:
        graph.add_entity_edge(
            nodes["root"], nodes["child"], parent_key="id", relation_key="parent_id"
        )
    if "leaf" in nodes:
        graph.add_entity_edge(
            nodes["child"], nodes["leaf"], parent_key="id", relation_key="parent_id"
        )
    return graph, nodes, calls


def frame_for(node, data=None):
    if isinstance(node, SQLNode):
        return node.execute_query(f"SELECT * FROM {data or node.get_current_ref()}")
    frame = node.df if data is None else data
    return frame.compute() if isinstance(frame, dd.DataFrame) else frame


def test_stop_after_annotation_before_filters(backend):
    graph, nodes, calls = make_graph(backend)
    result = graph.run_until("child", "do_annotate")
    assert result.node is nodes["child"]
    assert result.operation == "do_annotate"
    assert calls[-1] == ("child", "do_annotate")
    assert "child_annotated" in frame_for(result.node, result.data)
    assert ("child", "do_filters") not in calls
    assert not any(method == "do_reduce" for _, method in calls)


def test_stop_after_load_before_annotation(backend):
    graph, nodes, calls = make_graph(backend)
    result = graph.run_until(nodes["root"], "do_data", sample_size=1)
    assert calls == [("root", "do_data")]
    assert len(frame_for(result.node, result.data)) == 1


@pytest.mark.parametrize("target", ["leaf", "child"])
def test_reduction_finishes_descendants_but_stops_before_parent_join(backend, target):
    graph, nodes, calls = make_graph(backend)
    result = graph.run_until(nodes[target], "do_reduce", sample_size=1)
    assert calls[-1] == (target, "do_reduce")
    reduced = frame_for(result.node, result.data)
    expected = [2, 1] if target == "leaf" else [2]
    assert sorted(reduced[f"{target}_nrows"]) == sorted(expected)
    parent = nodes["child"] if target == "leaf" else nodes["root"]
    assert f"{target}_nrows" not in frame_for(parent).columns
    if target == "child":
        child = nodes["child"]
        before_reduce = (
            child.get_ref_name("join", lookup=True)
            if isinstance(child, SQLNode)
            else None
        )
        assert "leaf_nrows" in frame_for(child, before_reduce).columns


@pytest.mark.parametrize(
    "size, expected", [(0, [0, 0, 0]), (1, [1, 2, 3]), (100, [3, 3, 4])]
)
def test_root_limit_propagates_to_all_descendant_rows(backend, size, expected):
    graph, nodes, calls = make_graph(backend, sample_size=size)
    graph.load_data()
    assert [len(frame_for(node)) for node in nodes.values()] == expected
    assert calls == [(name, "do_data") for name in nodes]
    if backend in {"duckdb", "sqlite"}:
        child_queries = [record["sql"] for record in graph.executed_op_records[1:]]
        assert all("EXISTS (SELECT 1" in query for query in child_queries)
        assert all("LIMIT" not in query for query in child_queries)


def test_sample_follows_load_stage_keys_before_parent_filters(backend):
    graph, nodes, _ = make_graph(backend)
    nodes["root"].drop_rows = True
    result = graph.run_until(nodes["child"], "do_annotate", sample_size=1)
    assert frame_for(result.node, result.data)["child_id"].tolist() == [10, 11]


def test_full_run_and_legacy_runner_use_constructor_sample(backend):
    for legacy in [False, True]:
        graph, nodes, _ = make_graph(backend, sample_size=1)
        if legacy:
            if backend in {"duckdb", "sqlite"}:
                graph.do_transformations_sql()
            else:
                graph.do_transformations()
        else:
            result = graph.run()
            assert result.operation == "complete"
            assert result.node is nodes["root"]
        parent = frame_for(nodes["root"])
        assert parent["root_id"].tolist() == [1]
        assert parent["child_nrows"].tolist() == [2]


def test_shared_descendant_unions_paths_without_duplicating_rows(backend):
    frames = {
        "root": pd.DataFrame({"id": [1, 2]}),
        "left": pd.DataFrame({"id": [10, 20], "root_id": [1, 2]}),
        "right": pd.DataFrame({"id": [30, 40], "root_id": [1, 2]}),
        "shared": pd.DataFrame(
            {
                "id": [100, 101, 102, 102, 200],
                "left_id": [10, 20, 10, 10, 20],
                "right_id": [40, 30, 30, 30, 40],
            }
        ),
        "deep": pd.DataFrame({"id": [1000, 2000], "shared_id": [102, 200]}),
    }
    graph, nodes, _ = make_graph(backend, frames)
    # Insert the shared descendant's edges before its second parent's edge.
    for parent, child, foreign_key in [
        ("root", "left", "root_id"),
        ("left", "shared", "left_id"),
        ("shared", "deep", "shared_id"),
        ("right", "shared", "right_id"),
        ("root", "right", "root_id"),
    ]:
        graph.add_entity_edge(
            nodes[parent], nodes[child], parent_key="id", relation_key=foreign_key
        )
    graph.load_data(sample_size=1)
    assert sorted(frame_for(nodes["shared"])["shared_id"]) == [100, 101, 102, 102]
    assert frame_for(nodes["deep"])["deep_id"].tolist() == [1000]


def test_composite_keys_nulls_duplicate_parents_and_quoted_values(backend):
    frames = {
        "root": pd.DataFrame(
            {"tenant id": ["a'b", "a'b", "x", None], "id": [7, 7, 7, 8]}
        ),
        "events": pd.DataFrame(
            {
                "tenant id": ["a'b", "x", None, "a'b"],
                "parent_id": [7, 7, 8, 999],
                "id": [10, 20, 30, 40],
            }
        ),
    }
    graph, nodes, _ = make_graph(backend, frames)
    graph.add_entity_edge(
        nodes["root"],
        nodes["events"],
        parent_key=("tenant id", "id"),
        relation_key=("tenant id", "parent_id"),
    )
    graph.load_data(sample_size=4)
    assert sorted(frame_for(nodes["events"])["events_id"]) == [10, 20]


def test_custom_do_data_predicate_is_preserved():
    graph, nodes, _ = make_graph("duckdb")
    nodes["child"].do_data_ops = sqlop(
        optype=SQLOpType.custom,
        opval="SELECT id AS child_id, parent_id AS child_parent_id FROM child_source WHERE id <> 11;",
    )
    graph.load_data(sample_size=1)
    assert frame_for(nodes["child"])["child_id"].tolist() == [10]
    assert frame_for(nodes["leaf"])["leaf_id"].tolist() == [100, 101]


@pytest.mark.parametrize(
    "operation",
    ["do_filters", "do_normalize", "do_post_join_annotate", "do_post_join_filters"],
)
def test_other_operation_boundaries_stop_after_the_hook(backend, operation):
    graph, nodes, calls = make_graph(backend)
    result = graph.run_until(nodes["child"], operation)
    assert calls[-1] == ("child", operation)
    assert result.operation == operation
    if operation.startswith("do_post_join"):
        assert "leaf_nrows" in frame_for(result.node, result.data)
        assert ("child", "do_reduce") not in calls


def test_auto_features_are_included_in_reduce_boundary():
    graph, nodes, _ = make_graph("duckdb")
    graph.auto_features = True
    nodes["leaf"].do_reduce = SQLNode.do_reduce.__get__(nodes["leaf"], SQLNode)
    result = graph.run_until(nodes["leaf"], "do_reduce", sample_size=1)
    reduced = frame_for(result.node, result.data)
    assert set(reduced["leaf_parent_id"]) == {10, 11}
    assert any(column != "leaf_parent_id" for column in reduced.columns)
    assert graph.executed_op_records[-1]["auto_generated"] is True
    assert "leaf_nrows" not in frame_for(nodes["child"])


def test_label_stop_precedes_label_join():
    graph, nodes, _ = make_graph("duckdb")
    nodes["leaf"].do_labels_ops = [
        sqlop(optype=SQLOpType.agg, opval="leaf_parent_id"),
        sqlop(optype=SQLOpType.aggfunc, opval="count(*) AS outcome"),
    ]
    result = graph.run_until(nodes["leaf"], "do_labels", sample_size=1)
    assert sorted(frame_for(result.node, result.data)["outcome"]) == [1, 2]
    assert "outcome" not in frame_for(nodes["child"])


def test_frozen_sample_plan_does_not_capture_sample_ids_or_temporary_references():
    graph, _, _ = make_graph("duckdb")
    graph.load_data(sample_size=1)
    plan = graph.freeze_execution_plan()
    replay, nodes, _ = make_graph("duckdb")
    replay.apply_execution_plan(plan)
    replay.load_data()
    assert [len(frame_for(node)) for node in nodes.values()] == [3, 3, 4]


def test_sampling_actual_csv_loads(tmp_path):
    root_path, child_path = tmp_path / "root.csv", tmp_path / "child.csv"
    pd.DataFrame({"id": [1, 2]}).to_csv(root_path, index=False)
    pd.DataFrame({"id": [10, 11, 20], "root_id": [1, 1, 2]}).to_csv(
        child_path, index=False
    )
    root = DynamicNode(fpath=str(root_path), fmt="csv", pk="id", prefix="r")
    child = DynamicNode(fpath=str(child_path), fmt="csv", pk="id", prefix="c")
    graph = GraphReduce(parent_node=root, compute_layer=ComputeLayerEnum.pandas)
    graph.add_entity_edge(root, child, parent_key="id", relation_key="root_id")
    graph.load_data(sample_size=1)
    assert root.df["r_id"].tolist() == [1]
    assert child.df["c_id"].tolist() == [10, 11]


def test_sampled_date_node_load_uses_root_keys():
    graph, nodes, _ = make_graph(
        "duckdb",
        {
            "root": pd.DataFrame({"id": [1, 2]}),
            "dates": pd.DataFrame(
                {"id": [1, 2], "cutoff": ["2024-01-01", "2024-02-01"]}
            ),
        },
    )
    nodes["dates"].is_date_node = True
    graph.date_node = nodes["dates"]
    graph.add_entity_edge(
        nodes["root"], nodes["dates"], parent_key="id", relation_key="id"
    )
    graph.load_data(sample_size=1)
    assert frame_for(nodes["dates"])["dates_id"].tolist() == [1]


@pytest.mark.parametrize("size", [-1, True, 1.5, "100"])
def test_invalid_sample_size_rejected_before_loading(size):
    graph, _, calls = make_graph("pandas")
    with pytest.raises(ValueError, match="sample_size"):
        graph.load_data(sample_size=size)
    assert calls == []


@pytest.mark.parametrize(
    "target, operation",
    [
        ("missing", "do_data"),
        ("child", "do_unknown"),
        ("root", "do_reduce"),
        ("leaf", "do_post_join_annotate"),
    ],
)
def test_invalid_targets_fail_before_loading(target, operation):
    graph, _, calls = make_graph("duckdb")
    with pytest.raises(ValueError):
        graph.run_until(target, operation)
    assert calls == []


def test_cycles_and_disconnected_samples_are_rejected_before_loading():
    graph, nodes, calls = make_graph("pandas")
    graph.add_entity_edge(
        nodes["leaf"], nodes["root"], parent_key="id", relation_key="id"
    )
    with pytest.raises(ValueError, match="acyclic"):
        graph.load_data(sample_size=1)
    graph.remove_edge(nodes["leaf"], nodes["root"])
    graph.remove_edge(nodes["child"], nodes["leaf"])
    with pytest.raises(ValueError, match="reachable"):
        graph.load_data(sample_size=1)
    assert calls == []
