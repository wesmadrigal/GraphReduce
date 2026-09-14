# Operations Pipeline

GraphReduce separates operations that use one table from operations that need
joined data. This makes the location and timing of each transformation
predictable.

## Node operations

| Method | When it runs | Purpose |
| --- | --- | --- |
| `do_data` | First | Load a dataframe or establish a SQL source |
| `do_annotate` | Before reduction | Add columns derived from the node alone |
| `do_filters` | Before reduction | Remove rows using node-local conditions |
| `do_normalize` | Before reduction | Clean or normalize node-local values |
| `do_reduce` | During edge traversal | Aggregate a relation to its parent key |
| `do_labels` | During edge traversal | Aggregate target values from the label window |
| `do_post_join_annotate` | After a child join | Add columns that depend on joined data |
| `do_post_join_filters` | After a child join | Filter using joined data |
| `do_post_join_reduce` | After a child join, when configured | Reduce an enriched intermediate relation |

`DynamicNode` implements the required hooks with automatic/default behavior.
A custom `GraphReduceNode` subclass must implement the abstract hooks, even
when a hook only contains `pass`.

## Dataframe execution

`GraphReduce.do_transformations()` performs this high-level sequence:

```text
hydrate graph configuration
        │
load every node
        │
check prefix uniqueness
        │
annotate → filter → normalize each node
        │
traverse edges from leaves toward the parent
        │
for each edge:
  reduce child if requested
  + generate automatic features if enabled
  + join child result to parent
  + compute/join labels when applicable
  + run eligible post-join operations
        │
materialized parent dataframe
```

The traversal is bottom-up. Features created on a deep child can therefore be
carried through an intermediate node and reduced again at the next edge.

## SQL execution

`GraphReduce.do_transformations_sql()` follows the same relational intent, but
node methods return lists of `sqlop` values. GraphReduce compiles those values,
creates intermediate references in the SQL backend, and records generated SQL
in `graph.sql_ops`.

SQL execution is not a drop-in call on a dataframe graph. Use `SQLNode` or a
backend-specific SQL subclass, supply the appropriate client, and select the
matching `ComputeLayerEnum` value.

See:

- [Custom node operations](../tutorial_custom_graph.md)
- [Post-join dependencies](../tutorial_method_deps.md)
- [SQL backends and dynamic queries](../tutorial_sql_backends_dynamic_queries.md)
- [Frozen SQL plans](../tutorial_frozen_sql_plans.md)
