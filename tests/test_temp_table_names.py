import duckdb

from graphreduce.enum import ComputeLayerEnum
from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import DuckdbNode


def _make_graph(connection):
    node = DuckdbNode(
        fpath="events",
        table_name="events",
        prefix="evt",
        pk="id",
        date_key="created_at",
        compute_layer=ComputeLayerEnum.duckdb,
        client=connection,
    )
    graph = GraphReduce(
        name="parallel-temp-name-test",
        parent_node=node,
        compute_layer=ComputeLayerEnum.duckdb,
        sql_client=connection,
    )
    graph.add_node(node)
    graph.hydrate_graph_attrs()
    return graph, node


def test_graphreduce_temp_refs_are_unique_per_graph_instance():
    connection = duckdb.connect()
    try:
        first_graph, first_node = _make_graph(connection)
        second_graph, second_node = _make_graph(connection)

        first_ref = first_node.get_ref_name("do_data")
        second_ref = second_node.get_ref_name("do_data")

        assert first_graph.execution_namespace != second_graph.execution_namespace
        assert first_ref != second_ref
        assert first_graph.execution_namespace in first_ref
        assert second_graph.execution_namespace in second_ref
    finally:
        connection.close()
