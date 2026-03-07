#!/usr/bin/env python
"""Run the GraphReduce Hello World example locally and print a summary.

Usage:
  python examples/hello_world_local_runner.py
"""

from __future__ import annotations

import datetime
from pathlib import Path

from graphreduce.enum import ComputeLayerEnum, PeriodUnit
from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import DynamicNode


def main() -> None:
    data_path = Path("tests/data/cust_data")

    cust_node = DynamicNode(
        fpath=str(data_path / "cust.csv"),
        fmt="csv",
        prefix="cust",
        date_key=None,
        pk="id",
    )

    orders_node = DynamicNode(
        fpath=str(data_path / "orders.csv"),
        fmt="csv",
        prefix="ord",
        date_key="ts",
        pk="id",
    )

    notifications_node = DynamicNode(
        fpath=str(data_path / "notifications.csv"),
        fmt="csv",
        prefix="not",
        date_key="ts",
        pk="id",
    )

    gr = GraphReduce(
        name="hello_world_local",
        parent_node=cust_node,
        fmt="csv",
        compute_layer=ComputeLayerEnum.pandas,
        auto_features=True,
        auto_labels=True,
        cut_date=datetime.datetime(2023, 6, 30),
        compute_period_unit=PeriodUnit.day,
        compute_period_val=365,
        label_node=orders_node,
        label_field="id",
        label_operation="count",
        label_period_unit=PeriodUnit.day,
        label_period_val=30,
        auto_feature_hops_back=3,
        auto_feature_hops_front=0,
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
        reduce=True,
    )

    print("Starting GraphReduce transformation...", flush=True)
    gr.do_transformations()

    df = gr.parent_node.df
    print("\nCompleted.", flush=True)
    print(f"rows: {len(df)}", flush=True)
    print(f"columns: {len(df.columns)}", flush=True)
    print("sample columns:", list(df.columns[:10]), flush=True)
    print("shape:", df.shape, flush=True)


if __name__ == "__main__":
    main()
