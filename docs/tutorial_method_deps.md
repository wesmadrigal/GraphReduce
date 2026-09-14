# Post-Join Dependencies

Some annotations and filters require columns from one or more child nodes. A
post-join dependency prevents that operation from running until those children
have been merged.

## Dataframe nodes

Use `method_requires` with node classes:

```python
from graphreduce.context import method_requires
from graphreduce.node import GraphReduceNode


class OrderNode(GraphReduceNode):
    # Implement the required node methods here.
    ...


class CustomerNode(GraphReduceNode):
    @method_requires(nodes=[OrderNode])
    def do_post_join_annotate(self):
        signup = self.colabbr("signup_date")
        self.df[self.colabbr("days_to_last_order")] = (
            self.df["ord_ts_max"] - self.df[signup]
        ).dt.days

    # Implement the other required node methods here.
    ...
```

GraphReduce records merged relation classes on the parent. The decorated method
returns without executing until every required class is present.

For multiple dependencies:

```python
@method_requires(nodes=[OrderNode, NotificationNode])
def do_post_join_filters(self):
    self.df = self.df[
        self.df["ord_id_count"].notna()
        | self.df["not_id_count"].notna()
    ]
```

## SQL nodes

SQL nodes accept dependency instances as constructor settings:

```python
customers = SQLNode(
    # ...normal node configuration...
    do_post_join_annotate_ops=[...],
    do_post_join_annotate_requires=[orders],
    do_post_join_filters_ops=[...],
    do_post_join_filters_requires=[orders, notifications],
)
```

The SQL path checks whether columns with each required node's prefix are
available before returning the configured operations.

## Guidance

- Use pre-join hooks for logic that depends only on the node's own columns.
- Use post-join hooks only when another relation is genuinely required.
- Depend on the smallest set of nodes needed by the expression.
- Reference prefixed columns, because source names have already changed by the
  time a post-join method runs.
- Cover dependency order with a small graph test; the operation's correctness
  depends on both graph topology and prefixes.
