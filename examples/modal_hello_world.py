#!/usr/bin/env python

"""
Run the GraphReduce Hello World example on Modal.

Usage:
  modal run examples/modal_hello_world.py
"""

from __future__ import annotations

import pathlib
import urllib.request

import modal


app = modal.App("graphreduce-hello-world")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "graphreduce",
        "pandas",
        "numpy",
        "dask[dataframe]",
        "networkx",
        "structlog",
    )
)


def _download(url: str, out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, out_path)


@app.function(image=image, timeout=600)
def run_hello_world() -> dict:
    import datetime
    from pathlib import Path

    from graphreduce.enum import ComputeLayerEnum, PeriodUnit
    from graphreduce.graph_reduce import GraphReduce
    from graphreduce.node import DynamicNode

    base = "https://raw.githubusercontent.com/wesmadrigal/GraphReduce/master/tests/data/cust_data"
    data_dir = Path("/tmp/cust_data")

    _download(f"{base}/cust.csv", data_dir / "cust.csv")
    _download(f"{base}/orders.csv", data_dir / "orders.csv")
    _download(f"{base}/notifications.csv", data_dir / "notifications.csv")

    cust_node = DynamicNode(
        fpath=str(data_dir / "cust.csv"),
        fmt="csv",
        prefix="cust",
        date_key=None,
        pk="id",
    )

    orders_node = DynamicNode(
        fpath=str(data_dir / "orders.csv"),
        fmt="csv",
        prefix="ord",
        date_key="ts",
        pk="id",
    )

    notifications_node = DynamicNode(
        fpath=str(data_dir / "notifications.csv"),
        fmt="csv",
        prefix="not",
        date_key="ts",
        pk="id",
    )

    gr = GraphReduce(
        name="hello_world_modal",
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

    gr.do_transformations()
    df = gr.parent_node.df
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "sample_columns": [str(c) for c in df.columns[:10]],
    }


@app.local_entrypoint()
def main():
    result = run_hello_world.remote()
    print("Remote result:", flush=True)
    print(result, flush=True)
