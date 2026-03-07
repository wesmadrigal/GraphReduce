# SQL Backends and Dynamic Query Construction

This page highlights GraphReduce SQL pluggability and starts with a single-node
example showing how SQL is dynamically constructed at runtime.

SQL can already be used with:

* sqlite
* duckdb
* snowflake SQL
* databricks SQL
* aws Athena
* aws redshift

## Single Node Example (duckdb backend)

This mirrors the `test_duckdb_node` pattern and shows the exact dynamic query
construction call:

```python
from pathlib import Path
import duckdb

from graphreduce.node import DuckdbNode
from graphreduce.enum import ComputeLayerEnum

data_path = Path("tests/data/cust_data")
con = duckdb.connect()

node = DuckdbNode(
    fpath=f"'{data_path / 'cust.csv'}'",
    pk="id",
    prefix="cust",
    compute_layer=ComputeLayerEnum.duckdb,
    client=con,
    columns=["id", "name"],
    table_name="customer",  # required for filesystem-backed DuckDB reads
)

print(node.build_query(node.do_data()))
print(node.build_query(node.do_annotate()))
```

What this demonstrates:

* `node.do_data()` emits the base SQL op sequence
* `node.build_query(...)` compiles SQL text from those ops
* `node.do_annotate()` shows prefixed column-selection SQL generation

## Single Node Example (sqlite backend with custom SQLNode)

This version shows dynamic SQL from user-defined node methods.

```python
import sqlite3
from pathlib import Path
import typing
import pandas as pd

from graphreduce.node import SQLNode
from graphreduce.enum import ComputeLayerEnum, SQLOpType
from graphreduce.models import sqlop

data_path = Path("tests/data/cust_data")
conn = sqlite3.connect(":memory:")

# Load csv into sqlite table.
pd.read_csv(data_path / "orders.csv").to_sql("orders", conn, index=False, if_exists="replace")

class OrdersSQLNode(SQLNode):
    def do_filters(self) -> typing.List[sqlop]:
        return [sqlop(optype=SQLOpType.where, opval=f"{self.colabbr('ts')} >= '2023-01-01'")]

    def do_reduce(self, reduce_key):
        return [
            sqlop(optype=SQLOpType.aggfunc, opval=f"count(*) as {self.colabbr('num_orders')}"),
            sqlop(optype=SQLOpType.agg, opval=f"{self.colabbr(reduce_key)}"),
        ]

orders = OrdersSQLNode(
    fpath="orders",
    prefix="ord",
    pk="id",
    date_key="ts",
    client=conn,
    compute_layer=ComputeLayerEnum.sqlite,
    columns=["id", "customer_id", "ts", "amount"],
)

print(orders.build_query(orders.do_data()))
print(orders.build_query(orders.do_annotate()))
print(orders.build_query(orders.do_filters()))
print(orders.build_query(orders.do_filters() + orders.do_reduce("customer_id")))

conn.close()
```

## Why This Matters

* You define SQL behavior in small composable ops.
* GraphReduce compiles those ops into executable SQL at runtime.
* The same node/graph abstraction can target different SQL engines by changing
  backend configuration and node class.
