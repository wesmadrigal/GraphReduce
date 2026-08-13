import sqlite3

import duckdb
import pandas as pd
import pytest
from dask import dataframe as dd

from graphreduce.enum import ComputeLayerEnum, SQLOpType
from graphreduce.graph_reduce import GraphReduce
from graphreduce.models import sqlop
from graphreduce.node import (
    DuckdbNode,
    DynamicNode,
    SQLNode,
    key_parts,
    normalize_key,
)


def _pandas_nodes():
    parent = DynamicNode(
        prefix="acct",
        pk=("tenant_id", "account_id"),
        compute_layer=ComputeLayerEnum.pandas,
    )
    relation = DynamicNode(
        prefix="evt",
        pk=("tenant_id", "account_id", "event_id"),
        compute_layer=ComputeLayerEnum.pandas,
    )
    parent.df = pd.DataFrame(
        {
            "acct_tenant_id": [1, 2],
            "acct_account_id": [7, 7],
            "acct_name": ["one", "two"],
        }
    )
    relation.df = pd.DataFrame(
        {
            "evt_tenant_id": [1, 2],
            "evt_account_ref": [7, 7],
            "evt_event_id": [100, 200],
            "evt_amount": [10, 50],
        }
    )
    return parent, relation


def test_key_specs_normalize_without_breaking_scalars():
    assert key_parts("id") == ("id",)
    assert normalize_key(["id"]) == "id"
    assert normalize_key(["tenant_id", "id"]) == ("tenant_id", "id")

    node = DynamicNode(prefix="acct", pk=["tenant_id", "id"])
    assert node.pk == ("tenant_id", "id")
    assert node.colabbrs(node.pk) == ("acct_tenant_id", "acct_id")
    assert node.key_sql(node.pk) == "acct_tenant_id, acct_id"


@pytest.mark.parametrize(
    "key, error_type",
    [
        ([], ValueError),
        (["id", "id"], ValueError),
        (["id", ""], ValueError),
        (["id", 1], ValueError),
        ({"id"}, TypeError),
    ],
)
def test_invalid_key_specs_are_rejected(key, error_type):
    with pytest.raises(error_type):
        key_parts(key)


def test_edge_rejects_composite_key_arity_mismatch():
    parent, relation = _pandas_nodes()
    graph = GraphReduce(
        parent_node=parent,
        compute_layer=ComputeLayerEnum.pandas,
    )

    with pytest.raises(ValueError, match="same number of columns"):
        graph.add_entity_edge(
            parent,
            relation,
            parent_key=("tenant_id", "account_id"),
            relation_key="account_ref",
        )


def test_pandas_composite_join_does_not_cross_tenants():
    parent, relation = _pandas_nodes()
    graph = GraphReduce(
        parent_node=parent,
        compute_layer=ComputeLayerEnum.pandas,
    )
    graph.add_entity_edge(
        parent,
        relation,
        parent_key=("tenant_id", "account_id"),
        relation_key=("tenant_id", "account_ref"),
        reduce=False,
    )

    joined = graph.join(parent, relation)

    assert joined["evt_amount"].tolist() == [10, 50]
    assert len(joined) == 2


def test_dask_composite_join_does_not_cross_tenants():
    parent, relation = _pandas_nodes()
    parent.compute_layer = ComputeLayerEnum.dask
    relation.compute_layer = ComputeLayerEnum.dask
    parent.df = dd.from_pandas(parent.df, npartitions=1)
    relation.df = dd.from_pandas(relation.df, npartitions=1)
    graph = GraphReduce(
        parent_node=parent,
        compute_layer=ComputeLayerEnum.dask,
    )
    graph.add_entity_edge(
        parent,
        relation,
        parent_key=("tenant_id", "account_id"),
        relation_key=("tenant_id", "account_ref"),
        reduce=False,
    )

    joined = graph.join(parent, relation).compute()

    assert joined["evt_amount"].tolist() == [10, 50]
    assert len(joined) == 2


def test_pandas_composite_reduction_joins_at_full_entity_grain():
    parent, relation = _pandas_nodes()
    relation.df = pd.concat(
        [
            relation.df,
            pd.DataFrame(
                {
                    "evt_tenant_id": [1],
                    "evt_account_ref": [7],
                    "evt_event_id": [101],
                    "evt_amount": [5],
                }
            ),
        ],
        ignore_index=True,
    )
    graph = GraphReduce(
        parent_node=parent,
        compute_layer=ComputeLayerEnum.pandas,
    )
    graph.add_entity_edge(
        parent,
        relation,
        parent_key=("tenant_id", "account_id"),
        relation_key=("tenant_id", "account_ref"),
        reduce=True,
    )
    reduce_columns = list(
        relation.colabbrs(("tenant_id", "account_ref"))
    )
    reduced = (
        relation.df.groupby(reduce_columns)
        .agg(evt_amount_sum=pd.NamedAgg(column="evt_amount", aggfunc="sum"))
        .reset_index()
    )

    joined = graph.join(parent, relation, relation_df=reduced)

    assert joined["evt_amount_sum"].tolist() == [15, 50]
    assert len(joined) == 2


def test_pandas_auto_features_group_by_every_composite_key_component():
    node = DynamicNode(
        prefix="evt",
        pk=("tenant_id", "account_id", "event_id"),
        compute_layer=ComputeLayerEnum.pandas,
    )
    node.df = pd.DataFrame(
        {
            "evt_tenant_id": [1, 1, 2],
            "evt_account_id": [7, 7, 7],
            "evt_event_id": [100, 101, 200],
            "evt_amount": [10.0, 5.0, 50.0],
        }
    )

    result = node.pandas_auto_features(
        ("tenant_id", "account_id"),
        type_func_map={"numerical": ["sum"]},
    )

    assert result["evt_tenant_id"].tolist() == [1, 2]
    assert result["evt_amount_sum"].tolist() == [15.0, 50.0]


def test_pandas_graph_transformation_supports_composite_reduction(tmp_path):
    parent_path = tmp_path / "accounts.csv"
    relation_path = tmp_path / "events.csv"
    pd.DataFrame(
        {
            "tenant_id": [1, 2],
            "account_id": [7, 7],
        }
    ).to_csv(parent_path, index=False)
    pd.DataFrame(
        {
            "tenant_id": [1, 1, 2],
            "account_ref": [7, 7, 7],
            "event_id": [100, 101, 200],
            "amount": [10.0, 5.0, 50.0],
        }
    ).to_csv(relation_path, index=False)

    parent = DynamicNode(
        fpath=str(parent_path),
        fmt="csv",
        prefix="acct",
        pk=("tenant_id", "account_id"),
        compute_layer=ComputeLayerEnum.pandas,
    )
    relation = DynamicNode(
        fpath=str(relation_path),
        fmt="csv",
        prefix="evt",
        pk=("tenant_id", "account_ref", "event_id"),
        compute_layer=ComputeLayerEnum.pandas,
    )
    graph = GraphReduce(
        parent_node=parent,
        compute_layer=ComputeLayerEnum.pandas,
        auto_features=True,
    )
    graph.add_entity_edge(
        parent,
        relation,
        parent_key=("tenant_id", "account_id"),
        relation_key=("tenant_id", "account_ref"),
        reduce=True,
    )

    graph.do_transformations()

    result = parent.df.sort_values("acct_tenant_id")
    assert result["evt_amount_sum"].tolist() == [15.0, 50.0]
    assert "evt_account_ref_sum" not in result.columns
    assert len(result) == 2


def test_default_labels_group_by_every_composite_key_component():
    node = DynamicNode(
        prefix="evt",
        pk=("tenant_id", "account_id", "event_id"),
        compute_layer=ComputeLayerEnum.pandas,
    )
    node.df = pd.DataFrame(
        {
            "evt_tenant_id": [1, 1, 2],
            "evt_account_ref": [7, 7, 7],
            "evt_event_id": [100, 101, 200],
            "evt_target": [1, 0, 3],
        }
    )

    result = node.default_label(
        op="sum",
        field="target",
        reduce_key=("tenant_id", "account_ref"),
    )

    assert result["evt_tenant_id"].tolist() == [1, 2]
    assert result["evt_target_label"].tolist() == [1, 3]


def test_sql_query_and_join_use_every_composite_key_component():
    connection = sqlite3.connect(":memory:")
    try:
        pd.DataFrame(
            {
                "acct_tenant_id": [1, 2],
                "acct_account_id": [7, 7],
                "acct_name": ["one", "two"],
            }
        ).to_sql("accounts", connection, index=False)
        pd.DataFrame(
            {
                "evt_tenant_id": [1, 1, 2],
                "evt_account_ref": [7, 7, 7],
                "evt_amount": [10, 5, 50],
            }
        ).to_sql("events", connection, index=False)

        parent = SQLNode(
            fpath="accounts",
            prefix="acct",
            pk=("tenant_id", "account_id"),
            compute_layer=ComputeLayerEnum.sqlite,
            client=connection,
        )
        relation = SQLNode(
            fpath="events",
            prefix="evt",
            pk=("tenant_id", "account_ref"),
            compute_layer=ComputeLayerEnum.sqlite,
            client=connection,
        )
        reduce_ops = [
            sqlop(
                optype=SQLOpType.aggfunc,
                opval="sum(evt_amount) as evt_amount_sum",
            ),
            sqlop(
                optype=SQLOpType.agg,
                opval=relation.key_sql(("tenant_id", "account_ref")),
            ),
        ]
        reduce_sql = relation.build_query(reduce_ops)
        relation.create_ref(reduce_sql, "composite_reduce")

        assert "GROUP BY evt_tenant_id, evt_account_ref" in reduce_sql

        graph = GraphReduce(
            parent_node=parent,
            compute_layer=ComputeLayerEnum.sqlite,
            sql_client=connection,
        )
        graph.add_entity_edge(
            parent,
            relation,
            parent_key=("tenant_id", "account_id"),
            relation_key=("tenant_id", "account_ref"),
            reduce=True,
        )
        graph.join_sql(parent, relation)

        result = pd.read_sql_query(
            f"SELECT * FROM {parent._cur_data_ref} ORDER BY acct_tenant_id",
            connection,
        )
        join_sql = graph.sql_ops[-2]

        assert (
            "parent.acct_tenant_id = relation.evt_tenant_id" in join_sql
        )
        assert (
            "parent.acct_account_id = relation.evt_account_ref" in join_sql
        )
        assert result["evt_amount_sum"].tolist() == [15, 50]
        assert len(result) == 2
    finally:
        connection.close()


def test_composite_date_node_propagates_at_full_entity_grain():
    connection = duckdb.connect()
    try:
        connection.sql(
            "CREATE TABLE accounts(tenant_id INTEGER, account_id INTEGER)"
        )
        connection.sql("INSERT INTO accounts VALUES (1, 7), (2, 7)")
        connection.sql(
            """
            CREATE TABLE events(
                tenant_id INTEGER,
                account_ref INTEGER,
                event_id INTEGER,
                ts TIMESTAMP
            )
            """
        )
        connection.sql(
            """
            INSERT INTO events VALUES
                (1, 7, 100, '2026-01-01'),
                (2, 7, 200, '2026-01-02')
            """
        )
        connection.sql(
            """
            CREATE TABLE cutoffs(
                tenant_id INTEGER,
                account_id INTEGER,
                cutoff TIMESTAMP
            )
            """
        )
        connection.sql(
            """
            INSERT INTO cutoffs VALUES
                (1, 7, '2026-06-01'),
                (2, 7, '2026-07-01')
            """
        )

        parent = DuckdbNode(
            fpath="accounts",
            prefix="acct",
            pk=("tenant_id", "account_id"),
            columns=["tenant_id", "account_id"],
            client=connection,
            compute_layer=ComputeLayerEnum.duckdb,
        )
        relation = DuckdbNode(
            fpath="events",
            prefix="evt",
            pk=("tenant_id", "account_ref", "event_id"),
            date_key="ts",
            columns=["tenant_id", "account_ref", "event_id", "ts"],
            client=connection,
            compute_layer=ComputeLayerEnum.duckdb,
        )
        date_node = DuckdbNode(
            fpath="cutoffs",
            prefix="cut",
            pk=("tenant_id", "account_id"),
            date_key="cutoff",
            columns=["tenant_id", "account_id", "cutoff"],
            client=connection,
            compute_layer=ComputeLayerEnum.duckdb,
            is_date_node=True,
        )
        graph = GraphReduce(
            parent_node=parent,
            compute_layer=ComputeLayerEnum.duckdb,
            sql_client=connection,
            date_node=date_node,
        )
        graph.add_entity_edge(
            parent,
            relation,
            parent_key=("tenant_id", "account_id"),
            relation_key=("tenant_id", "account_ref"),
            reduce=False,
        )

        graph.do_transformations_sql()
        result = connection.sql(
            f"""
            SELECT acct_tenant_id, evt_event_id, cut_cutoff
            FROM {parent._cur_data_ref}
            ORDER BY acct_tenant_id
            """
        ).to_df()

        assert result["evt_event_id"].tolist() == [100, 200]
        assert result["cut_cutoff"].dt.month.tolist() == [6, 7]
    finally:
        connection.close()
