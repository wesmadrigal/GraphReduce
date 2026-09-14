# Define Custom Node Operations

Use a `GraphReduceNode` subclass when automatic features are not sufficient.
The node becomes the single home for logic specific to one source table.

## Required methods

Custom nodes implement seven hooks:

```python
from graphreduce.node import GraphReduceNode


class MyNode(GraphReduceNode):
    def do_filters(self):
        pass

    def do_annotate(self):
        pass

    def do_normalize(self):
        pass

    def do_reduce(self, reduce_key):
        pass

    def do_labels(self, reduce_key):
        pass

    def do_post_join_annotate(self):
        pass

    def do_post_join_filters(self):
        pass
```

Methods use `self.df`; they do not receive a dataframe argument. Leave an
unused hook as `pass`.

## A complete order node

```python
import pandas as pd

from graphreduce.node import GraphReduceNode


class OrderNode(GraphReduceNode):
    def do_filters(self):
        amount = self.colabbr("amount")
        self.df = self.df[self.df[amount].between(0, 100_000)]

    def do_annotate(self):
        amount = self.colabbr("amount")
        self.df[self.colabbr("is_large")] = self.df[amount] >= 1_000

    def do_normalize(self):
        pass

    def do_reduce(self, reduce_key):
        key = self.colabbr(reduce_key)
        history = self.prep_for_features()
        return history.groupby(key).agg(
            **{
                self.colabbr("order_count"): pd.NamedAgg(
                    column=self.colabbr("id"), aggfunc="count"
                ),
                self.colabbr("amount_sum"): pd.NamedAgg(
                    column=self.colabbr("amount"), aggfunc="sum"
                ),
                self.colabbr("amount_max"): pd.NamedAgg(
                    column=self.colabbr("amount"), aggfunc="max"
                ),
                self.colabbr("large_order_count"): pd.NamedAgg(
                    column=self.colabbr("is_large"), aggfunc="sum"
                ),
            }
        )

    def do_labels(self, reduce_key):
        key = self.colabbr(reduce_key)
        future = self.prep_for_labels()
        labels = future.groupby(key).agg(
            **{
                self.colabbr("order_label"): pd.NamedAgg(
                    column=self.colabbr("id"), aggfunc="count"
                )
            }
        )
        return labels

    def do_post_join_annotate(self):
        pass

    def do_post_join_filters(self):
        pass
```

Instantiate it like any other node:

```python
from graphreduce.enum import ComputeLayerEnum

orders = OrderNode(
    fpath="orders.csv",
    fmt="csv",
    pk="id",
    prefix="ord",
    date_key="ts",
    compute_layer=ComputeLayerEnum.pandas,
)
```

When the graph is hydrated, its shared cutoff and time periods are pushed to
the node.

## Two helpers prevent common mistakes

`self.colabbr(name)` returns the physical prefixed column. For example,
`self.colabbr("amount")` returns `ord_amount` when the prefix is `ord`.

`self.prep_for_features()` and `self.prep_for_labels()` return the appropriate
time slices. Call them inside reduction and label logic instead of grouping the
unfiltered dataframe.

## Reduction contract

`do_reduce(reduce_key)` must return data keyed by the prefixed relation key. On
an edge from customers to orders, GraphReduce calls
`orders.do_reduce("customer_id")` and expects one aggregate row per
`ord_customer_id`.

If you return duplicate relation keys, the later join can multiply parent
rows. Add a focused assertion while developing a node:

```python
reduced = orders.do_reduce("customer_id")
assert reduced.index.is_unique
```

## Post-join logic

Use `do_post_join_annotate` or `do_post_join_filters` only for expressions that
need columns from a child already joined into the node. When execution order
depends on particular children, declare a
[post-join dependency](tutorial_method_deps.md).
