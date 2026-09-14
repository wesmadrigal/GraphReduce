# Quickstart

This walkthrough builds a two-table customer feature graph with the sample CSV
files included in the repository. Run it from the repository root.

## What you will build

```text
cust.csv                       orders.csv
one row per customer           many rows per customer
        id  1 ───────────── *  customer_id
           output grain          reduce before join
```

The graph uses order history to create customer features and uses future order
rows to create a customer label.

## 1. Install the project

```bash
pip install -e ".[dev]"
```

The sample files live under `tests/data/cust_data/`.

## 2. Define the nodes

```python
import datetime as dt
from pathlib import Path

from graphreduce.enum import ComputeLayerEnum, PeriodUnit
from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import DynamicNode

data_dir = Path("tests/data/cust_data")

customers = DynamicNode(
    fpath=str(data_dir / "cust.csv"),
    fmt="csv",
    pk="id",
    prefix="cust",
    compute_layer=ComputeLayerEnum.pandas,
)

orders = DynamicNode(
    fpath=str(data_dir / "orders.csv"),
    fmt="csv",
    pk="id",
    prefix="ord",
    date_key="ts",
    compute_layer=ComputeLayerEnum.pandas,
)
```

The prefixes are unique, so output columns retain their source: `cust_name`
comes from customers and `ord_amount_sum` comes from orders. The orders node
has a `date_key`; the static customer table does not need one.

## 3. Configure the graph

```python
graph = GraphReduce(
    name="customer_quickstart",
    parent_node=customers,
    compute_layer=ComputeLayerEnum.pandas,
    cut_date=dt.datetime(2023, 6, 30),
    compute_period_val=365,
    compute_period_unit=PeriodUnit.day,
    auto_features=True,
    label_node=orders,
    label_field="id",
    label_operation="count",
    label_period_val=30,
    label_period_unit=PeriodUnit.day,
)
```

This declares:

- one output row per customer;
- a one-year order-history window ending just before June 30, 2023; and
- a 30-day future window beginning just after that cutoff for the order-count
  label.

## 4. Connect the nodes

```python
graph.add_node(customers)
graph.add_node(orders)

graph.add_entity_edge(
    parent_node=customers,
    relation_node=orders,
    parent_key="id",
    relation_key="customer_id",
    reduce=True,
)
```

`reduce=True` is essential here. It tells GraphReduce to aggregate all matching
orders to one row per `customer_id` before joining them to customers.

## 5. Execute and inspect

```python
graph.do_transformations()

features = graph.parent_node.df
print(features.shape)
print(features.head())
```

The sample produces four rows—one for each customer. The exact feature count
may evolve as automatic feature generation improves, so assert the intended
row grain and required columns rather than a fixed total column count.

```python
assert len(features) == 4
assert features["cust_id"].is_unique
assert "ord_id_label" in features.columns
```

Customers with no order in the label window have a null label after the left
join. If “no matching event” means zero for your task, fill that value
explicitly before model training; do not apply that rule blindly to targets
where missing has a different meaning.

## What happened

GraphReduce performed the following work:

1. Loaded and prefixed both CSV files.
2. Filtered order feature rows to the configured historical window.
3. Generated order aggregates grouped by `ord_customer_id`.
4. Left-joined those aggregates to customers.
5. Counted orders in the future label window and joined the label.
6. Stored the customer-grain result on `graph.parent_node.df`.

## Next steps

- Add a third table in the full [Hello World](end_to_end_examples/hello_world.md).
- Learn the exact [time-window boundaries](fundamentals/time_and_leakage.md).
- Replace automatic logic with [custom node operations](tutorial_custom_graph.md).
- Move the same graph shape to another [compute layer](tutorial_swapping_compute.md).
- Build and inspect [SQL-native graphs](tutorial_sql_backends_dynamic_queries.md).
