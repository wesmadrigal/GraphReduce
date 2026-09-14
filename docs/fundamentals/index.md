# Fundamentals

GraphReduce treats relational feature engineering as graph execution. The
graph makes three decisions explicit:

1. **What does one output row represent?** The `parent_node` defines this
   grain.
2. **How are tables related?** Each edge records the key pair and whether the
   relation must be reduced.
3. **Which rows are valid at prediction time?** Date keys and graph-level
   windows separate historical features from future labels.

## A small example

```text
customers ──< orders ──< order_products
    │
    └───────< notifications
```

If `customers` is the parent node, a bottom-up execution works like this:

1. Reduce `order_products` to one row per order.
2. Join those features to `orders`.
3. Reduce the enriched orders to one row per customer.
4. Reduce notifications to one row per customer.
5. Join both sets of customer-level features to `customers`.

This ordering preserves information from deep relations while keeping the
final table at customer grain.

## The three objects

| Object | Represents | Owns |
| --- | --- | --- |
| `GraphReduceNode` | A table, file, or queryable relation | Data access, filtering, annotation, normalization, reduction, labels |
| Entity edge | A parent/child relationship | Join keys, `reduce`, and `reduce_after_join` behavior |
| `GraphReduce` | The full compute graph | Parent grain, time windows, traversal, joins, and execution backend |

`DynamicNode` supplies default behavior for automatic feature engineering.
Subclass `GraphReduceNode` when a table needs custom logic. Use `SQLNode` or a
backend-specific subclass when operations should be compiled into SQL.

## Four rules to remember

1. Every node needs a unique `prefix`.
2. Every time-varying node needs a `date_key` for point-in-time filtering.
3. Use `reduce=True` for a normal one-to-many child that must roll up to its
   parent.
4. Read the result from `graph.parent_node.df` for dataframe execution; SQL
   execution instead maintains backend references and generated statements.

Continue with:

- [Operations pipeline](operations.md)
- [Joins and cardinality](joins_and_cardinality.md)
- [Time windows and leakage safety](time_and_leakage.md)
- [Core abstractions](../abstractions.md)
- [Quickstart](../tutorial_index.md)
