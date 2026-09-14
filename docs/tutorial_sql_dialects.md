# SQL Operations and Dialects

GraphReduce represents SQL as small `sqlop` objects. A `SQLNode` combines those
objects with its current source reference, and `build_query()` renders an
executable statement for the selected compute layer.

## Operation types

Common `SQLOpType` values include:

| Type | SQL role |
| --- | --- |
| `select` | Projection or derived column |
| `where` | Filter predicate |
| `aggfunc` | Aggregate expression in the projection |
| `agg` | `GROUP BY` expression |
| `having` | Post-aggregation filter |
| `order` | Sort expression |
| `window` | Window expression |

## Build a query

```python
import sqlite3

from graphreduce.enum import ComputeLayerEnum, SQLOpType
from graphreduce.models import sqlop
from graphreduce.node import SQLNode

connection = sqlite3.connect(":memory:")

orders = SQLNode(
    fpath="orders",
    prefix="ord",
    pk="id",
    date_key="ts",
    compute_layer=ComputeLayerEnum.sqlite,
    client=connection,
    columns=["id", "customer_id", "ts", "amount"],
)

ops = [
    sqlop(optype=SQLOpType.where, opval="ord_ts >= '2023-01-01'"),
    sqlop(
        optype=SQLOpType.aggfunc,
        opval="sum(ord_amount) as ord_amount_sum",
    ),
    sqlop(optype=SQLOpType.agg, opval="ord_customer_id"),
]

print(orders.build_query(ops))
```

The node knows its source relation and prefixes. The operations specify only
the changes needed for this stage.

## Put operations on a node

For reusable behavior, subclass `SQLNode`:

```python
class OrderNode(SQLNode):
    def do_filters(self):
        return [
            sqlop(
                optype=SQLOpType.where,
                opval=f"{self.colabbr('amount')} >= 0",
            )
        ]

    def do_reduce(self, reduce_key):
        return [
            sqlop(
                optype=SQLOpType.aggfunc,
                opval=(
                    f"sum({self.colabbr('amount')}) "
                    f"as {self.colabbr('amount_sum')}"
                ),
            ),
            sqlop(optype=SQLOpType.agg, opval=self.key_sql(reduce_key)),
        ]
```

`do_transformations_sql()` calls these methods while traversing the graph,
compiles the returned operations, and creates intermediate references.

## Dialect selection matters

Choose the enum value for the actual engine—such as `sqlite`, `duckdb`,
`snowflake`, `redshift`, `athena`, `postgres`, `mysql`, `trino`, or
`databricks`. GraphReduce uses it for date arithmetic, quoting, temporary
objects, and other syntax differences.

Portable custom SQL should use simple expressions and GraphReduce helpers such
as `colabbr()` and `key_sql()`. Backend-specific functions may still require a
specialized node implementation.

Continue with:

- [SQL backends and dynamic query construction](tutorial_sql_backends_dynamic_queries.md)
- [SQL auto-feature families](sql_auto_feature_families.md)
- [Frozen SQL plans](tutorial_frozen_sql_plans.md)
