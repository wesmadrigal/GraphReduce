# End-to-end automated feature engineering

This tutorial shows a full batch feature engineering flow on a multi-table graph
using `GraphReduce.do_transformations()` with `auto_features=True`.

It covers:

1. Building a graph of related tabular entities.
2. Enforcing point-in-time windows with `cut_date` + compute period.
3. Automatically generating aggregate features on child tables.
4. Generating a label and joining everything back to a parent grain.

## End-to-end example

```python
import datetime
from sklearn.model_selection import train_test_split
from graphreduce.node import DynamicNode
from graphreduce.graph_reduce import GraphReduce
from graphreduce.enum import ComputeLayerEnum, PeriodUnit

base = "graphreduce/datasets/customers"

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

# Example supervised training split.
target_col = "ord_id_label"
feature_cols = [
    c
    for c, dtype in dict(df.dtypes).items()
    if c != target_col and (str(dtype).startswith("int") or str(dtype).startswith("float"))
]
model_df = df[feature_cols + [target_col]].dropna()

X = model_df[feature_cols]
y = model_df[target_col]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Requires installing tabpfn first:
#   pip install tabpfn
#
# from tabpfn import TabPFNClassifier
# clf = TabPFNClassifier()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print(y_pred[:10])
```

## End-to-end SQL example with `do_transformations_sql` and `sqlop`

```python
import datetime
import sqlite3
import pandas as pd
from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import SQLNode
from graphreduce.models import sqlop
from graphreduce.enum import ComputeLayerEnum, PeriodUnit, SQLOpType

base = "graphreduce/datasets/customers"
conn = sqlite3.connect(":memory:")

# Load csv files into sqlite tables for this example.
pd.read_csv(f"{base}/cust.csv").to_sql("cust", conn, index=False, if_exists="replace")
pd.read_csv(f"{base}/orders.csv").to_sql("orders", conn, index=False, if_exists="replace")
pd.read_csv(f"{base}/order_products.csv").to_sql("order_products", conn, index=False, if_exists="replace")


class OrdersSQLNode(SQLNode):
    def do_reduce(self, reduce_key):
        return [
            sqlop(optype=SQLOpType.aggfunc, opval=f"count(*) as {self.colabbr('id_count')}"),
            sqlop(optype=SQLOpType.aggfunc, opval=f"min({self.colabbr('ts')}) as {self.colabbr('ts_min')}"),
            sqlop(optype=SQLOpType.aggfunc, opval=f"max({self.colabbr('ts')}) as {self.colabbr('ts_max')}"),
            sqlop(optype=SQLOpType.agg, opval=f"{self.colabbr(reduce_key)}"),
        ]

    def do_labels(self, reduce_key):
        return [
            sqlop(optype=SQLOpType.aggfunc, opval=f"count(*) as {self.colabbr('id_label')}"),
            sqlop(optype=SQLOpType.agg, opval=f"{self.colabbr(reduce_key)}"),
        ]


class OrderProductsSQLNode(SQLNode):
    def do_reduce(self, reduce_key):
        return [
            sqlop(optype=SQLOpType.aggfunc, opval=f"count(*) as {self.colabbr('id_count')}"),
            sqlop(optype=SQLOpType.aggfunc, opval=f"count(distinct {self.colabbr('product_id')}) as {self.colabbr('product_id_nunique')}"),
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
    columns=["id", "customer_id", "ts"],
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
    # Optional: collect SQL without executing statements.
    # dry_run=True,
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

# SQL pipeline entrypoint.
gr.do_transformations_sql()

# Generated SQL operations (in execution order).
for stmt in gr.sql_ops:
    if isinstance(stmt, str):
        print(stmt)

# Materialized output at customer grain.
out = cust.get_sample()
print(out.columns)
print(out.head())
```

## What to expect

After execution, `gr.parent_node.df` is a customer-grain dataset that includes:

* base customer columns (`cu_*`)
* reduced and auto-generated features propagated upward from child tables (`ord_*`, `op_*`, `notif_*`, `ni_*`)
* label columns for supervised learning (for example `ord_id_label`)

This is the typical batch ML/AI output shape: one row per parent entity with
historical multi-table features and optional future-looking labels.
