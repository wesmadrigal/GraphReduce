# Core Abstractions

GraphReduce has a deliberately small object model: tables are nodes,
relationships are edges, and one graph coordinates execution.

## `GraphReduce`

`GraphReduce` extends `networkx.DiGraph`. It stores the table graph and owns
configuration that must be consistent across nodes:

- `parent_node`: the intended output entity and row grain;
- `compute_layer`: the execution backend;
- `cut_date`: the point in time at which features are evaluated;
- `compute_period_*`: the historical feature window;
- `label_period_*`: the future target window;
- `label_node`, `label_field`, and `label_operation`: optional automatic label
  configuration; and
- `auto_features`: whether GraphReduce should synthesize relation features.

The normal entry points are:

```python
graph.do_transformations()       # dataframe execution
graph.do_transformations_sql()   # SQL execution
```

For dataframe execution, the final materialized result is
`graph.parent_node.df`.

## Node types

Every node represents one source relation and requires a unique `prefix`.

| Node | Use it when |
| --- | --- |
| `DynamicNode` | Default loading and automatic dataframe features are sufficient |
| `GraphReduceNode` subclass | A dataframe source needs custom operations |
| `SQLNode` | Operations should be represented as composable SQL |
| Backend SQL subclass | The backend needs specialized behavior, such as `DuckdbNode` or `AthenaNode` |

Common node settings include:

- `fpath`: a file path or SQL relation, depending on the node type;
- `fmt`: source format such as `csv`, `parquet`, or `sql`;
- `pk`: a primary key string or composite-key sequence;
- `prefix`: column-lineage prefix, unique within the graph;
- `date_key`: the timestamp used for time filtering; and
- `compute_layer`: a value from `ComputeLayerEnum`.

### `DynamicNode`

Use `DynamicNode` for schema-driven dataframe features:

```python
from graphreduce.enum import ComputeLayerEnum
from graphreduce.node import DynamicNode

orders = DynamicNode(
    fpath="orders.parquet",
    fmt="parquet",
    pk="id",
    prefix="ord",
    date_key="created_at",
    compute_layer=ComputeLayerEnum.pandas,
)
```

### Custom `GraphReduceNode`

A custom subclass owns seven operation hooks. Methods work with `self.df`; they
do not receive a dataframe argument.

```python
import pandas as pd

from graphreduce.node import GraphReduceNode


class OrderNode(GraphReduceNode):
    def do_filters(self):
        self.df = self.df[self.df[self.colabbr("amount")] >= 0]

    def do_annotate(self):
        pass

    def do_normalize(self):
        pass

    def do_reduce(self, reduce_key):
        key = self.colabbr(reduce_key)
        return self.prep_for_features().groupby(key).agg(
            **{
                self.colabbr("amount_sum"): pd.NamedAgg(
                    column=self.colabbr("amount"), aggfunc="sum"
                )
            }
        )

    def do_labels(self, reduce_key):
        pass

    def do_post_join_annotate(self):
        pass

    def do_post_join_filters(self):
        pass
```

The key helpers are:

- `colabbr("amount")` returns the prefixed physical column name;
- `prep_for_features()` applies the node's historical window; and
- `prep_for_labels()` applies the node's future label window.

### `SQLNode`

SQL methods return `sqlop` objects rather than mutating `self.df`. GraphReduce
compiles those operations and manages intermediate backend references.

```python
from graphreduce.enum import ComputeLayerEnum, SQLOpType
from graphreduce.models import sqlop
from graphreduce.node import SQLNode


class OrderSQLNode(SQLNode):
    def do_reduce(self, reduce_key):
        key = self.colabbr(reduce_key)
        return [
            sqlop(
                optype=SQLOpType.aggfunc,
                opval=f"sum({self.colabbr('amount')}) as {self.colabbr('amount_sum')}",
            ),
            sqlop(optype=SQLOpType.agg, opval=key),
        ]
```

See [SQL backends and dynamic queries](tutorial_sql_backends_dynamic_queries.md)
for complete examples.

## Edges

An entity edge records the direction, join keys, and reduction behavior:

```python
graph.add_entity_edge(
    parent_node=customers,
    relation_node=orders,
    parent_key="id",
    relation_key="customer_id",
    reduce=True,
)
```

The direction is meaningful: the relation is processed and joined into its
parent. With `reduce=True`, GraphReduce aggregates the relation to its foreign
key before the join.

Both keys may be ordered tuples for composite relationships. Their lengths
must match.

## Prefixes and physical columns

Nodes prefix columns as data enters the graph. An `orders.amount` source column
on a node with `prefix="ord"` becomes `ord_amount`. Always use `colabbr()` in
custom operations instead of hard-coding the transformed name.

Prefixes must be unique across all graph nodes. They provide lineage and avoid
ambiguous columns after repeated joins.

## Put configuration at the right level

- Put table-specific behavior and schema meaning on the node.
- Put relationship cardinality and join keys on the edge.
- Put shared execution, time-window, label, and graph-wide automatic-feature
  settings on `GraphReduce`.

This separation is the core design constraint behind the library.
