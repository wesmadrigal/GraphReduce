#!/usr/bin/env python
"""Run the Preserve Child Grain example locally and print a summary.

Usage:
  python examples/preserve_child_grain_local_runner.py
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd

from graphreduce.enum import ComputeLayerEnum, PeriodUnit
from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import GraphReduceNode


class CustNode(GraphReduceNode):
    def do_filters(self):
        return self.df

    def do_annotate(self):
        return self.df

    def do_post_join_annotate(self):
        return self.df

    def do_normalize(self):
        return self.df

    def do_post_join_filters(self):
        return self.df

    def do_reduce(self, reduce_key):
        return self.df

    def do_labels(self, reduce_key):
        return self.df


class OrderNode(GraphReduceNode):
    def do_filters(self):
        return self.df

    def do_annotate(self):
        return self.df

    def do_post_join_annotate(self):
        return self.df

    def do_normalize(self):
        return self.df

    def do_post_join_filters(self):
        return self.df

    def do_reduce(self, reduce_key):
        return (
            self.prep_for_features()
            .groupby(self.colabbr(reduce_key))
            .agg(
                **{
                    self.colabbr("num_orders"): pd.NamedAgg(
                        column=self.colabbr(self.pk), aggfunc="count"
                    )
                }
            )
            .reset_index()
        )

    def do_labels(self, reduce_key):
        return self.df


class NotificationNode(GraphReduceNode):
    def do_filters(self):
        return self.df

    def do_annotate(self):
        return self.df

    def do_post_join_annotate(self):
        return self.df

    def do_normalize(self):
        return self.df

    def do_post_join_filters(self):
        return self.df

    def do_reduce(self, reduce_key):
        return self.df

    def do_labels(self, reduce_key):
        return self.df


def main() -> None:
    data_path = Path("tests/data/cust_data")

    cust_node = CustNode(
        fpath=str(data_path / "cust.csv"),
        fmt="csv",
        prefix="cust",
        date_key=None,
        pk="id",
        compute_layer=ComputeLayerEnum.pandas,
    )

    orders_node = OrderNode(
        fpath=str(data_path / "orders.csv"),
        fmt="csv",
        prefix="ord",
        date_key="ts",
        pk="id",
        compute_layer=ComputeLayerEnum.pandas,
    )

    notifications_node = NotificationNode(
        fpath=str(data_path / "notifications.csv"),
        fmt="csv",
        prefix="not",
        date_key="ts",
        pk="id",
        compute_layer=ComputeLayerEnum.pandas,
    )

    gr = GraphReduce(
        name="preserve_child_grain_local",
        parent_node=cust_node,
        fmt="csv",
        compute_layer=ComputeLayerEnum.pandas,
        auto_features=False,
        auto_labels=False,
        cut_date=datetime.datetime(2023, 6, 30),
        compute_period_unit=PeriodUnit.day,
        compute_period_val=365,
    )

    gr.add_node(cust_node)
    gr.add_node(orders_node)
    gr.add_node(notifications_node)

    gr.add_entity_edge(
        parent_node=cust_node,
        relation_node=orders_node,
        parent_key="id",
        relation_key="customer_id",
        relation_type="parent_child",
        reduce=True,
    )

    gr.add_entity_edge(
        parent_node=cust_node,
        relation_node=notifications_node,
        parent_key="id",
        relation_key="customer_id",
        relation_type="parent_child",
        reduce=False,
    )

    print("Starting preserve-child-grain transformation...", flush=True)
    gr.do_transformations()

    df = gr.parent_node.df
    print("\nCompleted.", flush=True)
    print(f"rows: {len(df)}", flush=True)
    print(f"columns: {len(df.columns)}", flush=True)
    print("sample columns:", list(df.columns[:10]), flush=True)
    print("shape:", df.shape, flush=True)
    print("expected rows: 9", flush=True)


if __name__ == "__main__":
    main()
