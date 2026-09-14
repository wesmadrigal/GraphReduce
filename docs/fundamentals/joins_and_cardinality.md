# Joins and Cardinality

The most important edge decision is whether a child relation should be reduced
before it is joined. That decision determines the row grain of the result.

## Define an edge

```python
graph.add_entity_edge(
    parent_node=customers,
    relation_node=orders,
    parent_key="id",
    relation_key="customer_id",
    reduce=True,
)
```

This means:

- `customers` is on the one side of the relationship;
- `orders` is on the many side;
- `customers.id = orders.customer_id` is the join condition; and
- orders must be aggregated by `customer_id` before the join.

GraphReduce uses a left join when it attaches the child result, so parent rows
without matching child rows remain in the output.

## Choose the reduction mode

| Configuration | Behavior | Typical use |
| --- | --- | --- |
| `reduce=True` | Aggregate the child to its edge key, then join | Preserve parent grain across a one-to-many relationship |
| `reduce=False` | Join the child rows directly | Intentionally produce or preserve child grain |
| `reduce_after_join=True` | Join first, then reduce the enriched parent side | Aggregate only after required child context is present |

`reduce` and `reduce_after_join` cannot both be true.

### Why `reduce=True` is the normal choice

If customer 7 has three orders, a raw join produces three customer rows.
Reducing orders first produces one row containing summaries such as order count
and total amount; joining that row back preserves one customer row.

### When `reduce=False` is appropriate

Use it only when row multiplication is intended—for example, when building one
output row per notification while attaching customer attributes. Verify the
grain explicitly after execution. See
[Preserve child grain](../end_to_end_examples/preserve_child_grain.md).

## Composite keys

Keys may be a string or an ordered list/tuple of strings:

```python
graph.add_entity_edge(
    parent_node=accounts,
    relation_node=events,
    parent_key=("tenant_id", "account_id"),
    relation_key=("tenant_id", "account_id"),
    reduce=True,
)
```

Both sides must contain the same number of key columns, in matching order.

## Avoid accidental many-to-many joins

Before executing a graph, answer these questions for every edge:

1. Is the parent key unique at the point of the join?
2. How many child rows can match one parent row?
3. Does the relation need to be reduced once, or after another child has joined?
4. What should one row represent immediately after this edge?
5. Can the row count be asserted on a small sample?

Prefixes prevent column-name collisions; they do not prevent row
multiplication. Cardinality must be modeled deliberately.
