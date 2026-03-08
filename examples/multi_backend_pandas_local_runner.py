#!/usr/bin/env python
"""Run multi-backend pandas cust+notifications example."""

from __future__ import annotations

import datetime
import os

import pandas as pd

from graphreduce.enum import ComputeLayerEnum
from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import GraphReduceNode

DATA_PATH = "tests/data/cust_data"


def _is_interactive_mode() -> bool:
    return os.getenv("GRAPHREDUCE_INTERACTIVE", "0").strip().lower() in {"1", "true", "yes"}


class CustNode(GraphReduceNode):
    def do_annotate(self):
        self.df[self.colabbr("name_length")] = self.df[self.colabbr("name")].fillna("").str.len()
        return self.df

    def do_filters(self):
        self.df = self.df[self.df[self.colabbr("id")] < 3]
        return self.df

    def do_normalize(self):
        pass

    def do_reduce(self, reduce_key):
        pass

    def do_labels(self, reduce_key):
        pass

    def do_post_join_annotate(self):
        pass

    def do_post_join_filters(self):
        pass


class NotificationNode(GraphReduceNode):
    def do_annotate(self):
        self.df[self.colabbr("ts_month")] = self.df[self.colabbr("ts")].astype(str).str.slice(5, 7)
        return self.df

    def do_filters(self):
        self.df = self.df[self.df[self.colabbr("ts")] > "2022-06-01"]
        return self.df

    def do_normalize(self):
        pass

    def do_reduce(self, reduce_key):
        return (
            self.prep_for_features()
            .groupby(self.colabbr(reduce_key))
            .agg(**{self.colabbr("num_notifications"): pd.NamedAgg(column=self.colabbr(self.pk), aggfunc="count")})
            .reset_index()
        )

    def do_labels(self, reduce_key):
        pass

    def do_post_join_annotate(self):
        pass

    def do_post_join_filters(self):
        pass


def main() -> None:
    print("Running pandas backend...", flush=True)
    cust = CustNode(
        fpath=os.path.join(DATA_PATH, "cust.csv"),
        fmt="csv",
        prefix="cust",
        pk="id",
        compute_layer=ComputeLayerEnum.pandas,
        columns=["id", "name"],
    )
    notif = NotificationNode(
        fpath=os.path.join(DATA_PATH, "notifications.csv"),
        fmt="csv",
        prefix="not",
        pk="id",
        date_key="ts",
        compute_layer=ComputeLayerEnum.pandas,
        columns=["id", "customer_id", "ts"],
    )

    gr = GraphReduce(
        name="cust_notif_pandas",
        parent_node=cust,
        compute_layer=ComputeLayerEnum.pandas,
        cut_date=datetime.datetime(2023, 6, 30),
    )
    gr.add_node(cust)
    gr.add_node(notif)
    gr.add_entity_edge(cust, notif, parent_key="id", relation_key="customer_id", reduce=True)
    gr.do_transformations()

    print("rows:", len(gr.parent_node.df), flush=True)
    print("columns:", len(gr.parent_node.df.columns), flush=True)
    print("column_names:", gr.parent_node.df.columns.tolist(), flush=True)
    print("shape:", gr.parent_node.df.shape, flush=True)
    if _is_interactive_mode():
        print("df.columns:", gr.parent_node.df.columns, flush=True)


if __name__ == "__main__":
    main()
