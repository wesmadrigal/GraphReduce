import datetime
import os
import sqlite3
from pathlib import Path

import duckdb
import pandas as pd

from graphreduce.node import DynamicNode, SQLNode, DuckdbNode
from graphreduce.graph_reduce import GraphReduce
from graphreduce.enum import ComputeLayerEnum, PeriodUnit, SQLOpType
from graphreduce.models import sqlop

base = "tests/data/cust_data"


def test_automated_fe():
    # Each node gets a unique prefix so merged columns keep table lineage.
    nodes = {
        "cust": DynamicNode(
            fpath=f"{base}/cust.csv",
            fmt="csv",
            pk="id",
            prefix="cu",
            date_key=None,
            compute_layer=ComputeLayerEnum.pandas,
        ),
        "orders": DynamicNode(
            fpath=f"{base}/orders.csv",
            fmt="csv",
            pk="id",
            prefix="ord",
            date_key="ts",
            compute_layer=ComputeLayerEnum.pandas,
        ),
        "order_products": DynamicNode(
            fpath=f"{base}/order_products.csv",
            fmt="csv",
            pk="id",
            prefix="op",
            date_key=None,
            compute_layer=ComputeLayerEnum.pandas,
        ),
        "notifications": DynamicNode(
            fpath=f"{base}/notifications.csv",
            fmt="csv",
            pk="id",
            prefix="notif",
            date_key="ts",
            compute_layer=ComputeLayerEnum.pandas,
        ),
        "notification_interactions": DynamicNode(
            fpath=f"{base}/notification_interactions.csv",
            fmt="csv",
            pk="id",
            prefix="ni",
            date_key="ts",
            compute_layer=ComputeLayerEnum.pandas,
        ),
    }

    gr = GraphReduce(
        name="customers_auto_fe",
        parent_node=nodes["cust"],
        fmt="csv",
        compute_layer=ComputeLayerEnum.pandas,
        cut_date=datetime.datetime(2023, 9, 1),
        compute_period_val=365,
        compute_period_unit=PeriodUnit.day,
        auto_features=True,
        auto_feature_hops_front=1,
        auto_feature_hops_back=2,
        label_node=nodes["orders"],
        label_operation="count",
        label_field="id",
        label_period_val=60,
        label_period_unit=PeriodUnit.day,
    )

    for node in nodes.values():
        gr.add_node(node)

    # Graph topology:
    # cust -> orders -> order_products
    # cust -> notifications -> notification_interactions
    gr.add_entity_edge(
        parent_node=nodes["cust"],
        relation_node=nodes["orders"],
        parent_key="id",
        relation_key="customer_id",
        reduce=True,
    )
    gr.add_entity_edge(
        parent_node=nodes["orders"],
        relation_node=nodes["order_products"],
        parent_key="id",
        relation_key="order_id",
        reduce=True,
    )
    gr.add_entity_edge(
        parent_node=nodes["cust"],
        relation_node=nodes["notifications"],
        parent_key="id",
        relation_key="customer_id",
        reduce=True,
    )
    gr.add_entity_edge(
        parent_node=nodes["notifications"],
        relation_node=nodes["notification_interactions"],
        parent_key="id",
        relation_key="notification_id",
        reduce=True,
    )

    gr.do_transformations()

    # Final training / analytics frame at customer grain.
    df = gr.parent_node.df
    print(df.shape)
    print(df.head())


def test_automated_fe_sql():
    conn = sqlite3.connect(":memory:")
    pd.read_csv(os.path.join(base, "cust.csv")).to_sql(
        "cust", conn, index=False, if_exists="replace"
    )
    pd.read_csv(os.path.join(base, "orders.csv")).to_sql(
        "orders", conn, index=False, if_exists="replace"
    )
    pd.read_csv(os.path.join(base, "order_products.csv")).to_sql(
        "order_products", conn, index=False, if_exists="replace"
    )

    class OrdersSQLNode(SQLNode):
        def do_reduce(self, reduce_key):
            return [
                sqlop(
                    optype=SQLOpType.aggfunc,
                    opval=f"count(*) as {self.colabbr('id_count')}",
                ),
                sqlop(
                    optype=SQLOpType.aggfunc,
                    opval=f"min({self.colabbr('ts')}) as {self.colabbr('ts_min')}",
                ),
                sqlop(
                    optype=SQLOpType.aggfunc,
                    opval=f"max({self.colabbr('ts')}) as {self.colabbr('ts_max')}",
                ),
                sqlop(optype=SQLOpType.agg, opval=f"{self.colabbr(reduce_key)}"),
            ]

        def do_labels(self, reduce_key):
            return [
                sqlop(
                    optype=SQLOpType.aggfunc,
                    opval=f"count(*) as {self.colabbr('id_label')}",
                ),
                sqlop(optype=SQLOpType.agg, opval=f"{self.colabbr(reduce_key)}"),
            ]

    class OrderProductsSQLNode(SQLNode):
        def do_reduce(self, reduce_key):
            return [
                sqlop(
                    optype=SQLOpType.aggfunc,
                    opval=f"count(*) as {self.colabbr('id_count')}",
                ),
                sqlop(
                    optype=SQLOpType.aggfunc,
                    opval=f"count(distinct {self.colabbr('product_id')}) as {self.colabbr('product_id_nunique')}",
                ),
                sqlop(optype=SQLOpType.agg, opval=f"{self.colabbr(reduce_key)}"),
            ]

    cust = SQLNode(
        fpath="cust",
        prefix="cu",
        client=conn,
        columns=["id", "name"],
        compute_layer=ComputeLayerEnum.sqlite,
        date_key=None,
    )
    orders = OrdersSQLNode(
        fpath="orders",
        prefix="ord",
        client=conn,
        columns=["id", "customer_id", "ts", "amount"],
        compute_layer=ComputeLayerEnum.sqlite,
        date_key="ts",
    )
    order_products = OrderProductsSQLNode(
        fpath="order_products",
        prefix="op",
        client=conn,
        columns=["id", "order_id", "product_id"],
        compute_layer=ComputeLayerEnum.sqlite,
        date_key=None,
    )

    gr = GraphReduce(
        name="customers_auto_fe_sql",
        parent_node=cust,
        compute_layer=ComputeLayerEnum.sqlite,
        sql_client=conn,
        cut_date=datetime.datetime(2023, 9, 1),
        compute_period_val=365,
        compute_period_unit=PeriodUnit.day,
        label_node=orders,
        label_period_val=60,
        label_period_unit=PeriodUnit.day,
    )

    for node in [cust, orders, order_products]:
        gr.add_node(node)

    gr.add_entity_edge(
        parent_node=cust,
        relation_node=orders,
        parent_key="id",
        relation_key="customer_id",
        reduce=True,
    )
    gr.add_entity_edge(
        parent_node=orders,
        relation_node=order_products,
        parent_key="id",
        relation_key="order_id",
        reduce=True,
    )

    gr.do_transformations_sql()
    assert len(gr.sql_ops) > 0
    assert any(
        "group by ord_customer_id" in s.lower()
        for s in gr.sql_ops
        if isinstance(s, str)
    )

    out = cust.get_sample()
    assert "ord_id_label" in out.columns
    assert len(out) == 4


def test_date_nodes():
    cut_date = datetime.datetime(2023, 5, 1)
    #data_dir = Path("tests/data/cust_data")
    data_dir = Path("/usr/local/lake/cust_data")
    con = duckdb.connect()
    cust = DuckdbNode(
        fpath=f"'{data_dir / 'cust.csv'}'",
        prefix="cust",
        pk="id",
        date_key=None,
        columns=['id','name'],
        table_name="cust",
        #do_filters_ops=[sqlop(optype=SQLOpType.where, opval=f"user_CreationDate <= '{cut_date.date()}'")],
    )
    order = DuckdbNode(
            fpath=f"'{data_dir / 'orders.csv'}'",
            prefix="ord",
            pk="id",
            date_key="ts",
            columns=['id','cid','ts', 'order_type', 'total'],
            table_name="ord",
        )
    notification = DuckdbNode(
            fpath=f"'{data_dir / 'notifications.csv'}'",
            prefix="notif",
            pk="id",
            date_key="ts",
            columns=['id','ident_cust','ts'],
            table_name="notif"
        )

    ni = DuckdbNode(
            fpath=f"'{data_dir / 'notification_interactions.csv'}'",
            prefix="ni",
            pk="id",
            date_key="ts",
            columns=['id','notification_id','interaction_type_id','ts'],
            table_name="notifation_interaction",
        )

    nit = DuckdbNode(
            fpath=f"'{data_dir / 'notification_interaction_types.csv'}'",
            prefix="nit",
            pk="id",
            date_key='ts',
            columns=['id','name'],
            table_name="nits",
        )

    op = DuckdbNode(
            fpath=f"'{data_dir / 'order_products.csv'}'",
            prefix="op",
            pk="id",
            date_key=None,
            columns=['id','product_id', 'order_id'],
            table_name="order_products",
        )

    date_node = DuckdbNode(
            fpath=f"'{data_dir / 'orders.csv'}'",
            prefix='cd',
            date_key='first_order_date',
            table_name="cut_date",
            do_data_ops=sqlop(
                    optype=SQLOpType.custom,
                    opval="""
                    SELECT cid as cd_cid, min(ts) as cd_first_order_date
                    from '/usr/local/lake/cust_data/orders.csv'
                    where order_type = 'subscription'
                    group by cid
                    """
                ),
            client=con
    )

    gr = GraphReduce(
            name=f"rel-stack-badges-{cut_date.date()}",
            parent_node=cust,
            compute_layer=ComputeLayerEnum.duckdb,
            sql_client=con,
            cut_date=cut_date,
            compute_period_val=730,
            compute_period_unit=PeriodUnit.day,
            auto_features=True,
            auto_labels=True,
            label_node=notification,
            label_field="id",
            label_operation="count",
            label_period_val=180,
            label_period_unit=PeriodUnit.day,
            auto_feature_hops_back=4,
            auto_feature_hops_front=0,
            dry_run=False,
            date_node=date_node
    )

    for node in [cust,order,op,notification,ni,nit]:
        gr.add_node(node)


    gr.add_entity_edge(parent_node=cust,relation_node=order,parent_key='id',relation_key='cid',reduce=True)
    gr.add_entity_edge(parent_node=cust,relation_node=notification,parent_key='id',relation_key='ident_cust',reduce=True)

    gr.add_entity_edge(parent_node=order,relation_node=op, parent_key='id',relation_key='order_id',reduce=True)
    gr.add_entity_edge(parent_node=notification,relation_node=ni,parent_key='id',relation_key='notification_id',reduce=True)
    gr.add_entity_edge(parent_node=ni,relation_node=nit,parent_key='interaction_type_id',relation_key='id',reduce=False)

    # Date node relationships
    gr.add_entity_edge(parent_node=cust, relation_node=date_node, parent_key='id', relation_key='cid')

    gr.do_transformations_sql()

    df = con.execute(f"select * from {gr.parent_node._cur_data_ref}").df()
    print(df.head())

    label = [c for c in df.columns if 'label' in c][0]
    print(df[['cust_id',label]])

    assert len(df) == 6


def test_date_node_not_joined_when_train_false():
    cut_date = datetime.datetime(2023, 5, 1)
    data_dir = Path(base).resolve()
    con = duckdb.connect()

    cust = DuckdbNode(
        fpath=f"'{data_dir / 'cust.csv'}'",
        prefix="cust",
        pk="id",
        date_key=None,
        columns=["id", "name"],
        table_name="cust",
        client=con,
    )
    order = DuckdbNode(
        fpath=f"'{data_dir / 'orders.csv'}'",
        prefix="ord",
        pk="id",
        date_key="ts",
        columns=["id", "customer_id", "ts", "amount"],
        table_name="ord",
        client=con,
    )
    date_node = DuckdbNode(
        fpath=f"'{data_dir / 'orders.csv'}'",
        prefix="cd",
        pk="customer_id",
        date_key="first_order_date",
        table_name="cut_date",
        do_data_ops=sqlop(
            optype=SQLOpType.custom,
            opval=f"""
                SELECT
                    customer_id AS cd_customer_id,
                    min(ts) AS cd_first_order_date
                FROM '{data_dir / 'orders.csv'}'
                GROUP BY customer_id
            """,
        ),
        client=con,
    )

    gr = GraphReduce(
        name="customers_no_date_join_on_score",
        parent_node=cust,
        compute_layer=ComputeLayerEnum.duckdb,
        sql_client=con,
        cut_date=cut_date,
        compute_period_val=365,
        compute_period_unit=PeriodUnit.day,
        auto_features=False,
        label_node=None,
        label_period_val=None,
        label_period_unit=None,
        date_node=date_node,
        train=False,
    )

    for node in [cust, order]:
        gr.add_node(node)

    gr.add_entity_edge(
        parent_node=cust,
        relation_node=order,
        parent_key="id",
        relation_key="customer_id",
        reduce=True,
    )
    gr.add_entity_edge(
        parent_node=cust,
        relation_node=date_node,
        parent_key="id",
        relation_key="customer_id",
    )

    gr.do_transformations_sql()

    df = con.execute(f"select * from {gr.parent_node._cur_data_ref}").df()
    assert "cd_first_order_date" not in df.columns
    assert not any(col.startswith("cd_") for col in df.columns)
    assert len(df) > 0
    con.close()
