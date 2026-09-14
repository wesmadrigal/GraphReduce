# Configuration Reference

This page covers the settings needed for most graphs. The Python signatures
remain the source of truth for less common backend-specific and experimental
options.

## `GraphReduce`

### Graph identity and execution

| Parameter | Meaning |
| --- | --- |
| `name` | Human-readable graph name |
| `parent_node` | Node that anchors traversal and normally defines output grain |
| `compute_layer` | Backend selected from `ComputeLayerEnum` |
| `auto_features` | Generate schema-aware relation features during reduction |
| `debug` | Enable additional diagnostic logging |
| `train` | Include training-only behavior such as label computation; scoring graphs normally set this to `False` |

### Time windows

| Parameter | Meaning |
| --- | --- |
| `cut_date` | Point-in-time reference for feature and label windows |
| `compute_period_val` | Size of the historical feature window |
| `compute_period_unit` | Unit for the feature-window value |
| `label_period_val` | Size of the future label window |
| `label_period_unit` | Unit for the label-window value |

If `label_node` is set, both label-period settings are required. See
[Time windows and leakage safety](fundamentals/time_and_leakage.md) for exact
boundaries.

### Labels

| Parameter | Meaning |
| --- | --- |
| `label_node` | Relation from which the target is computed |
| `label_field` | Source field passed to automatic label aggregation |
| `label_operation` | Aggregation string or callable, such as `count`, `sum`, or `bool` |

For `DynamicNode`, GraphReduce calls the default label implementation. A custom
`GraphReduceNode` supplies `do_labels(reduce_key)` instead.

Automatic labels are left-joined to the parent. An entity with no matching
label-window row therefore receives null, not an implicit zero.

### Traversal

| Parameter | Meaning |
| --- | --- |
| `auto_feature_hops_back` | Depth limit for descendant traversal from the parent |
| `auto_feature_hops_front` | Limit for optional joins through predecessor nodes |

Keep `auto_feature_hops_back` at least as deep as the child relations that must
contribute to the output. The default is `2`.

### SQL execution

| Parameter | Meaning |
| --- | --- |
| `sql_client` | Backend client shared with SQL nodes |
| `lazy_execution` | Defer backend execution where supported |
| `dry_run` | Compile/log SQL without executing it |
| `checkpoint_schema` | Schema used for SQL intermediate materialization |

Automatic SQL feature settings are documented separately under
[SQL auto-feature families](sql_auto_feature_families.md).

## `GraphReduceNode`

### Source and schema

| Parameter | Meaning |
| --- | --- |
| `fpath` | File path or SQL relation, depending on node type |
| `fmt` | Storage/source format such as `csv`, `parquet`, or `sql` |
| `pk` | Primary key as a string or ordered composite-key sequence |
| `prefix` | Required column-lineage prefix; must be unique in the graph |
| `date_key` | Timestamp used by feature and label filtering |
| `columns` | Known source columns, especially useful for SQL nodes |
| `compute_layer` | Node backend; normally matches the graph backend |

Graph hydration overwrites shared execution and time settings on nodes with the
graph-level values. Put one cutoff and one window policy on the graph unless a
specialized workflow explicitly requires something else.

### SQL-only additions

`SQLNode` accepts a backend `client`, optional `table_name`, and optional lists
of `sqlop` values for each operation hook. A backend subclass may require more
connection, catalog, or storage configuration.

## Entity edges

`GraphReduce.add_entity_edge()` accepts:

| Parameter | Meaning |
| --- | --- |
| `parent_node` | Destination of the relation's features |
| `relation_node` | Child relation being joined upward |
| `parent_key` | Key on the parent |
| `relation_key` | Matching key on the relation |
| `relation_type` | Relationship label; normally `parent_child` |
| `reduce` | Aggregate the relation before joining |
| `reduce_after_join` | Join first and reduce the enriched parent afterward |

`parent_key` and `relation_key` must have the same number of columns.
`reduce` and `reduce_after_join` cannot both be true.

## Results and inspection

| Path | Result |
| --- | --- |
| `graph.parent_node.df` | Materialized dataframe result |
| `graph.sql_ops` | Generated SQL statements in execution order |
| `graph.get_executed_op_records()` | Structured SQL operation records |
| `graph.freeze_execution_plan()` | Serializable record of a completed SQL plan |
| `graph.plot_graph(path)` | Interactive HTML visualization of graph topology |

For SQL plan reuse, see [Frozen SQL plans](tutorial_frozen_sql_plans.md).
