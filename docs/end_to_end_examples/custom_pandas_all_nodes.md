# Custom Pandas Graph: All `cust_data` Nodes

This example runs the same custom all-nodes graph in pandas across:

* `cust.csv`
* `orders.csv`
* `order_products.csv`
* `notifications.csv`
* `notification_interactions.csv`
* `notification_interaction_types.csv`

## Complete Example

```python
#!/usr/bin/env python
"""Custom pandas GraphReduce example using all cust_data nodes."""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd

from graphreduce.enum import ComputeLayerEnum
from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import GraphReduceNode

DATA_PATH = Path("tests/data/cust_data")


class CustNode(GraphReduceNode):
    def do_annotate(self):
        self.df[self.colabbr("name_length")] = self.df[self.colabbr("name")].fillna("").astype(str).str.len()
        return self.df

    def do_filters(self):
        self.df = self.df[self.df[self.colabbr("id")].notna()]
        return self.df

    def do_normalize(self):
        self.df[self.colabbr("name")] = (
            self.df[self.colabbr("name")]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        return self.df

    def do_reduce(self, reduce_key):
        return self.df

    def do_labels(self, reduce_key):
        return self.df

    def do_post_join_annotate(self):
        ord_ct = self.df["ord_num_orders"].fillna(0) if "ord_num_orders" in self.df.columns else 0
        not_ct = self.df["not_num_notifications"].fillna(0) if "not_num_notifications" in self.df.columns else 0
        engaged = (
            self.df["not_num_engaged_interactions"].fillna(0)
            if "not_num_engaged_interactions" in self.df.columns
            else 0
        )
        self.df["cust_total_events"] = ord_ct + not_ct + engaged
        self.df["cust_activity_tier"] = pd.cut(
            self.df["cust_total_events"],
            bins=[-1, 1, 4, float("inf")],
            labels=["low", "medium", "high"],
        ).astype(str)
        return self.df

    def do_post_join_filters(self):
        self.df = self.df[self.df["cust_total_events"] >= 0]
        return self.df


class OrderNode(GraphReduceNode):
    def do_annotate(self):
        self.df[self.colabbr("amount_dbl")] = pd.to_numeric(self.df[self.colabbr("amount")], errors="coerce")
        return self.df

    def do_filters(self):
        self.df = self.df[self.df[self.colabbr("ts")] >= "2022-01-01"]
        return self.df

    def do_normalize(self):
        self.df[self.colabbr("amount_dbl")] = self.df[self.colabbr("amount_dbl")].fillna(0.0)
        return self.df

    def do_reduce(self, reduce_key):
        return (
            self.prep_for_features()
            .groupby(self.colabbr(reduce_key))
            .agg(
                **{
                    self.colabbr("num_orders"): pd.NamedAgg(column=self.colabbr(self.pk), aggfunc="nunique"),
                    self.colabbr("sum_amount"): pd.NamedAgg(column=self.colabbr("amount_dbl"), aggfunc="sum"),
                    self.colabbr("avg_amount"): pd.NamedAgg(column=self.colabbr("amount_dbl"), aggfunc="mean"),
                }
            )
            .reset_index()
        )

    def do_labels(self, reduce_key):
        return self.df

    def do_post_join_annotate(self):
        return self.df

    def do_post_join_filters(self):
        return self.df


class OrderProductsNode(GraphReduceNode):
    def do_annotate(self):
        self.df[self.colabbr("product_id_int")] = pd.to_numeric(self.df[self.colabbr("product_id")], errors="coerce")
        return self.df

    def do_filters(self):
        self.df = self.df[self.df[self.colabbr("product_id_int")].notna()]
        return self.df

    def do_normalize(self):
        return self.df

    def do_reduce(self, reduce_key):
        return (
            self.prep_for_features()
            .groupby(self.colabbr(reduce_key))
            .agg(
                **{
                    self.colabbr("num_order_products"): pd.NamedAgg(column=self.colabbr(self.pk), aggfunc="count"),
                    self.colabbr("num_distinct_products"): pd.NamedAgg(
                        column=self.colabbr("product_id_int"), aggfunc="nunique"
                    ),
                }
            )
            .reset_index()
        )

    def do_labels(self, reduce_key):
        return self.df

    def do_post_join_annotate(self):
        return self.df

    def do_post_join_filters(self):
        return self.df


class NotificationNode(GraphReduceNode):
    def do_annotate(self):
        self.df[self.colabbr("ts_month")] = self.df[self.colabbr("ts")].astype(str).str.slice(5, 7)
        return self.df

    def do_filters(self):
        self.df = self.df[self.df[self.colabbr("ts")] >= "2022-01-01"]
        return self.df

    def do_normalize(self):
        return self.df

    def do_reduce(self, reduce_key):
        return (
            self.prep_for_features()
            .groupby(self.colabbr(reduce_key))
            .agg(
                **{
                    self.colabbr("num_notifications"): pd.NamedAgg(column=self.colabbr(self.pk), aggfunc="nunique"),
                    self.colabbr("max_notification_ts"): pd.NamedAgg(column=self.colabbr("ts"), aggfunc="max"),
                    self.colabbr("num_interactions"): pd.NamedAgg(
                        column="ni_num_interactions", aggfunc="sum"
                    ),
                    self.colabbr("num_engaged_interactions"): pd.NamedAgg(
                        column="ni_num_engaged_interactions", aggfunc="sum"
                    ),
                }
            )
            .reset_index()
        )

    def do_labels(self, reduce_key):
        return self.df

    def do_post_join_annotate(self):
        return self.df

    def do_post_join_filters(self):
        return self.df


class NotificationInteractionsNode(GraphReduceNode):
    def do_annotate(self):
        self.df[self.colabbr("ts_day")] = self.df[self.colabbr("ts")].astype(str).str.slice(0, 10)
        return self.df

    def do_filters(self):
        self.df = self.df[self.df[self.colabbr("ts")] >= "2022-01-01"]
        return self.df

    def do_normalize(self):
        return self.df

    def do_reduce(self, reduce_key):
        return (
            self.prep_for_features()
            .groupby(self.colabbr(reduce_key))
            .agg(
                **{
                    self.colabbr("num_interactions"): pd.NamedAgg(column=self.colabbr(self.pk), aggfunc="count"),
                    self.colabbr("num_interaction_types"): pd.NamedAgg(
                        column=self.colabbr("interaction_type_id"), aggfunc="nunique"
                    ),
                    self.colabbr("num_engaged_interactions"): pd.NamedAgg(
                        column="nit_is_engagement_type", aggfunc="sum"
                    ),
                }
            )
            .reset_index()
        )

    def do_labels(self, reduce_key):
        return self.df

    def do_post_join_annotate(self):
        return self.df

    def do_post_join_filters(self):
        return self.df


class NotificationInteractionTypeNode(GraphReduceNode):
    def do_annotate(self):
        self.df[self.colabbr("name")] = self.df[self.colabbr("name")].astype(str).str.strip().str.lower()
        self.df[self.colabbr("is_engagement_type")] = (
            self.df[self.colabbr("name")].isin(["clicked", "dismissed"]).astype(int)
        )
        self.df[self.colabbr("is_view_event")] = self.df[self.colabbr("name")].str.contains("view", na=False).astype(int)
        return self.df

    def do_filters(self):
        self.df = self.df[self.df[self.colabbr("id")].notna()]
        return self.df

    def do_normalize(self):
        return self.df

    def do_reduce(self, reduce_key):
        return (
            self.prep_for_features()
            .groupby(self.colabbr(reduce_key))
            .agg(
                **{
                    self.colabbr("name"): pd.NamedAgg(column=self.colabbr("name"), aggfunc="first"),
                    self.colabbr("is_engagement_type"): pd.NamedAgg(
                        column=self.colabbr("is_engagement_type"), aggfunc="max"
                    ),
                    self.colabbr("is_view_event"): pd.NamedAgg(column=self.colabbr("is_view_event"), aggfunc="max"),
                }
            )
            .reset_index()
        )

    def do_labels(self, reduce_key):
        return self.df

    def do_post_join_annotate(self):
        return self.df

    def do_post_join_filters(self):
        return self.df


def build_custom_pandas_graph() -> GraphReduce:
    cust = CustNode(
        fpath=str(DATA_PATH / "cust.csv"),
        fmt="csv",
        prefix="cust",
        pk="id",
        compute_layer=ComputeLayerEnum.pandas,
        columns=["id", "name"],
    )
    orders = OrderNode(
        fpath=str(DATA_PATH / "orders.csv"),
        fmt="csv",
        prefix="ord",
        pk="id",
        date_key="ts",
        compute_layer=ComputeLayerEnum.pandas,
        columns=["id", "customer_id", "ts", "amount"],
    )
    order_products = OrderProductsNode(
        fpath=str(DATA_PATH / "order_products.csv"),
        fmt="csv",
        prefix="op",
        pk="id",
        compute_layer=ComputeLayerEnum.pandas,
        columns=["id", "order_id", "product_id"],
    )
    notifications = NotificationNode(
        fpath=str(DATA_PATH / "notifications.csv"),
        fmt="csv",
        prefix="not",
        pk="id",
        date_key="ts",
        compute_layer=ComputeLayerEnum.pandas,
        columns=["id", "customer_id", "ts"],
    )
    notification_interactions = NotificationInteractionsNode(
        fpath=str(DATA_PATH / "notification_interactions.csv"),
        fmt="csv",
        prefix="ni",
        pk="id",
        date_key="ts",
        compute_layer=ComputeLayerEnum.pandas,
        columns=["id", "notification_id", "interaction_type_id", "ts"],
    )
    interaction_types = NotificationInteractionTypeNode(
        fpath=str(DATA_PATH / "notification_interaction_types.csv"),
        fmt="csv",
        prefix="nit",
        pk="id",
        compute_layer=ComputeLayerEnum.pandas,
        columns=["id", "name"],
    )

    gr = GraphReduce(
        name="custom_pandas_all_nodes",
        parent_node=cust,
        compute_layer=ComputeLayerEnum.pandas,
        cut_date=datetime.datetime(2023, 6, 30),
    )

    for node in [cust, orders, order_products, notifications, notification_interactions, interaction_types]:
        gr.add_node(node)

    gr.add_entity_edge(cust, orders, parent_key="id", relation_key="customer_id", reduce=True)
    gr.add_entity_edge(orders, order_products, parent_key="id", relation_key="order_id", reduce=True)
    gr.add_entity_edge(cust, notifications, parent_key="id", relation_key="customer_id", reduce=True)
    gr.add_entity_edge(notifications, notification_interactions, parent_key="id", relation_key="notification_id", reduce=True)
    gr.add_entity_edge(notification_interactions, interaction_types, parent_key="interaction_type_id", relation_key="id", reduce=True)

    return gr


def run_custom_pandas_all_nodes() -> pd.DataFrame:
    gr = build_custom_pandas_graph()
    gr.do_transformations()
    return gr.parent_node.df


def main() -> None:
    df = run_custom_pandas_all_nodes()
    print("rows:", len(df), flush=True)
    print("columns:", len(df.columns), flush=True)
    print("column_names:", list(df.columns), flush=True)


if __name__ == "__main__":
    main()
```

Full runnable script:

* `examples/custom_pandas_all_nodes.py`

## Run Interactive

<div class="modal-runner" data-modal-runner data-api-base="https://runner.13.218.155.128.sslip.io" data-example="custom_pandas_all_nodes">
  <div class="modal-runner-controls">
    <input class="modal-runner-input" data-api-input value="https://runner.13.218.155.128.sslip.io" />
    <button data-save-api-btn>Save API URL</button>
    <button data-run-btn>Run custom pandas all-nodes</button>
  </div>
  <div class="modal-runner-status" data-status>Idle</div>
  <pre class="modal-runner-log" data-log></pre>
</div>
