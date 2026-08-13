import duckdb

from graphreduce.enum import ComputeLayerEnum, SQLOpType
from graphreduce.models import sqlop
from graphreduce.node import DatabricksNode, DuckdbNode


def _duckdb_node(client=None, columns=None, do_data_ops=None):
    return DuckdbNode(
        fpath="teams",
        prefix="Team",
        compute_layer=ComputeLayerEnum.duckdb,
        client=client,
        columns=columns or [],
        do_data_ops=do_data_ops,
    )


def test_render_identifier_quotes_only_when_required():
    node = _duckdb_node()

    assert node.render_identifier("yearID") == "yearID"
    assert node.render_identifier("2B") == '"2B"'
    assert node.render_identifier("customer name") == '"customer name"'
    assert node.render_identifier("order") == '"order"'
    assert node.render_identifier('column"quote') == '"column""quote"'


def test_render_identifier_uses_databricks_backticks():
    node = DatabricksNode(
        fpath="teams",
        prefix="Team",
        compute_layer=ComputeLayerEnum.databricks,
    )

    assert node.render_identifier("2B") == "`2B`"
    assert node.render_identifier("column`quote") == "`column``quote`"


def test_do_data_quotes_duckdb_numeric_and_reserved_identifiers():
    client = duckdb.connect()
    client.sql(
        """
        CREATE TABLE teams (
            yearID INTEGER,
            "2B" INTEGER,
            "3B" INTEGER,
            "order" VARCHAR
        )
        """
    )
    client.sql("INSERT INTO teams VALUES (2026, 42, 7, 'first')")
    node = _duckdb_node(
        client=client,
        columns=["yearID", "2B", "3B", "order"],
    )

    query = node.build_query(node.do_data())
    result = node.execute_query(query)

    assert '"2B" as Team_2B' in query
    assert '"3B" as Team_3B' in query
    assert '"order" as Team_order' in query
    assert result.to_dict("records") == [
        {
            "Team_yearID": 2026,
            "Team_2B": 42,
            "Team_3B": 7,
            "Team_order": "first",
        }
    ]
    client.close()


def test_custom_do_data_expressions_are_not_rewritten():
    custom_op = sqlop(optype=SQLOpType.select, opval="COUNT(*) AS total")
    node = _duckdb_node(do_data_ops=[custom_op])

    assert node.do_data() == [custom_op]
