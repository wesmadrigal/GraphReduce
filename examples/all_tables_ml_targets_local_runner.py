#!/usr/bin/env python
"""Run the all-tables reduced example locally and print a summary."""

from __future__ import annotations

import datetime
from pathlib import Path

from graphreduce.enum import ComputeLayerEnum, PeriodUnit
from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import DynamicNode


def main() -> None:
    data_path = Path("tests/data/cust_data")

    cust_node = DynamicNode(fpath=str(data_path / "cust.csv"), fmt="csv", prefix="cust", date_key=None, pk="id")
    orders_node = DynamicNode(fpath=str(data_path / "orders.csv"), fmt="csv", prefix="ord", date_key="ts", pk="id")
    order_products_node = DynamicNode(
        fpath=str(data_path / "order_products.csv"), fmt="csv", prefix="op", date_key=None, pk="id"
    )
    notifications_node = DynamicNode(
        fpath=str(data_path / "notifications.csv"), fmt="csv", prefix="not", date_key="ts", pk="id"
    )
    notification_interactions_node = DynamicNode(
        fpath=str(data_path / "notification_interactions.csv"), fmt="csv", prefix="ni", date_key="ts", pk="id"
    )

    gr = GraphReduce(
        name="all_tables_ml_targets_local",
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
        auto_feature_hops_front=1,
    )

    for node in [cust_node, orders_node, order_products_node, notifications_node, notification_interactions_node]:
        gr.add_node(node)

    gr.add_entity_edge(cust_node, orders_node, parent_key="id", relation_key="customer_id", relation_type="parent_child", reduce=True)
    gr.add_entity_edge(orders_node, order_products_node, parent_key="id", relation_key="order_id", relation_type="parent_child", reduce=True)
    gr.add_entity_edge(cust_node, notifications_node, parent_key="id", relation_key="customer_id", relation_type="parent_child", reduce=True)
    gr.add_entity_edge(
        notifications_node,
        notification_interactions_node,
        parent_key="id",
        relation_key="notification_id",
        relation_type="parent_child",
        reduce=True,
    )

    print("Starting all-tables reduced transformation...", flush=True)
    gr.do_transformations()

    df = gr.parent_node.df
    print("\nCompleted.", flush=True)
    print(f"rows: {len(df)}", flush=True)
    print(f"columns: {len(df.columns)}", flush=True)
    print("shape:", df.shape, flush=True)
    print("sample columns:", list(df.columns[:10]), flush=True)


if __name__ == "__main__":
    main()
