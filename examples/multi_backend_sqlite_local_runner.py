#!/usr/bin/env python
"""Run multi-backend sqlite cust+notifications example."""

from __future__ import annotations

import sqlite3

import pandas as pd

from graphreduce.enum import ComputeLayerEnum, SQLOpType
from graphreduce.graph_reduce import GraphReduce
from graphreduce.models import sqlop
from graphreduce.node import SQLNode

DATA_PATH = "tests/data/cust_data"


class CustNode(SQLNode):
    def do_annotate(self):
        return [sqlop(optype=SQLOpType.select, opval=f"*, LENGTH({self.colabbr('name')}) as {self.colabbr('name_length')}")]

    def do_filters(self):
        return [sqlop(optype=SQLOpType.where, opval=f"{self.colabbr('id')} < 3")]

    def do_normalize(self):
        pass

    def do_reduce(self, reduce_key):
        pass

    def do_post_join_annotate(self):
        pass

    def do_post_join_filters(self):
        pass


class NotificationNode(SQLNode):
    def do_annotate(self):
        return [sqlop(optype=SQLOpType.select, opval=f"*, strftime('%m', {self.colabbr('ts')}) as {self.colabbr('ts_month')}")]

    def do_filters(self):
        return [sqlop(optype=SQLOpType.where, opval=f"{self.colabbr('ts')} > '2022-06-01'")]

    def do_normalize(self):
        pass

    def do_reduce(self, reduce_key):
        return [
            sqlop(optype=SQLOpType.aggfunc, opval=f"count(*) as {self.colabbr('num_notifications')}"),
            sqlop(optype=SQLOpType.agg, opval=f"{self.colabbr(reduce_key)}"),
        ]


def main() -> None:
    print("Running sqlite backend...", flush=True)
    conn = sqlite3.connect(":memory:")
    for table in ["cust", "notifications"]:
        pd.read_csv(f"{DATA_PATH}/{table}.csv").to_sql(table, conn, if_exists="replace", index=False)

    cust = CustNode(
        fpath="cust",
        prefix="cust",
        pk="id",
        client=conn,
        compute_layer=ComputeLayerEnum.sqlite,
        columns=["id", "name"],
    )
    notif = NotificationNode(
        fpath="notifications",
        prefix="not",
        pk="id",
        date_key="ts",
        client=conn,
        compute_layer=ComputeLayerEnum.sqlite,
        columns=["id", "customer_id", "ts"],
    )

    gr = GraphReduce(
        name="cust_notif_sqlite",
        parent_node=cust,
        compute_layer=ComputeLayerEnum.sqlite,
        sql_client=conn,
        use_temp_tables=True,
    )
    gr.add_node(cust)
    gr.add_node(notif)
    gr.add_entity_edge(cust, notif, parent_key="id", relation_key="customer_id", reduce=True)
    gr.do_transformations_sql()

    out_df = conn.execute(f"select * from {gr.parent_node._cur_data_ref}").fetchall()
    cols = [x[0] for x in conn.execute(f"select * from {gr.parent_node._cur_data_ref} limit 1").description]
    print("rows:", len(out_df), flush=True)
    print("columns:", len(cols), flush=True)
    print("shape:", (len(out_df), len(cols)), flush=True)


if __name__ == "__main__":
    main()
