#!/usr/bin/env python
"""Run multi-backend duckdb cust+notifications example."""

from __future__ import annotations

import duckdb

from graphreduce.enum import ComputeLayerEnum, SQLOpType
from graphreduce.graph_reduce import GraphReduce
from graphreduce.models import sqlop
from graphreduce.node import SQLNode


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
        return [sqlop(optype=SQLOpType.select, opval=f"*, strftime({self.colabbr('ts')}, '%m') as {self.colabbr('ts_month')}")]

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
    print("Running duckdb backend...", flush=True)
    con = duckdb.connect()
    con.sql("CREATE OR REPLACE VIEW cust AS SELECT * FROM read_csv_auto('tests/data/cust_data/cust.csv', header=true)")
    con.sql("CREATE OR REPLACE VIEW notifications AS SELECT * FROM read_csv_auto('tests/data/cust_data/notifications.csv', header=true)")

    cust = CustNode(
        fpath="cust",
        prefix="cust",
        pk="id",
        client=con,
        compute_layer=ComputeLayerEnum.duckdb,
        columns=["id", "name"],
    )
    notif = NotificationNode(
        fpath="notifications",
        prefix="not",
        pk="id",
        date_key="ts",
        client=con,
        compute_layer=ComputeLayerEnum.duckdb,
        columns=["id", "customer_id", "ts"],
    )

    gr = GraphReduce(
        name="cust_notif_duckdb",
        parent_node=cust,
        compute_layer=ComputeLayerEnum.duckdb,
        sql_client=con,
        use_temp_tables=True,
    )
    gr.add_node(cust)
    gr.add_node(notif)
    gr.add_entity_edge(cust, notif, parent_key="id", relation_key="customer_id", reduce=True)
    gr.do_transformations_sql()

    out_df = con.sql(f"select * from {gr.parent_node._cur_data_ref}").to_df()
    print("rows:", len(out_df), flush=True)
    print("columns:", len(out_df.columns), flush=True)
    print("shape:", out_df.shape, flush=True)


if __name__ == "__main__":
    main()
