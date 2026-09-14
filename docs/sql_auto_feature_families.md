# SQL Auto-Feature Families

GraphReduce can generate relational features while compiling a graph through
`GraphReduce.do_transformations_sql()`. A feature family is a bounded group of
SQL expressions created when a child relation is reduced to its parent key.

Only `base` is enabled by default. Every other family is opt-in.

## At a glance

| Family | Question it answers | Main requirement |
| --- | --- | --- |
| `base` | How much activity occurred, and what are its basic summaries? | None |
| `temporal` | How did numeric values differ across lookback windows? | `date_key` and `ts_periods` |
| `sequence` | Was activity steady, recent, or bursty? | `date_key` and `ts_periods` |
| `conditional` | What kinds of activity occurred? | Eligible categories or predicate annotations |
| `episode` | How many rows and distinct events occurred? | A primary key for distinct-event counts |
| `semantic` | Which domain concepts should become features? | `annotation_expressions` or auto-annotation |
| `context` | How did a value compare with its peers? | Explicit `context_keys` |

## Configure a node

Feature families can be selected on an individual SQL node:

```python
from graphreduce.enum import ComputeLayerEnum
from graphreduce.node import DuckdbNode

events = DuckdbNode(
    fpath="events",
    fmt="sql",
    pk="event_id",
    prefix="evt",
    date_key="event_time",
    compute_layer=ComputeLayerEnum.duckdb,
    feature_families=("base", "temporal", "sequence", "conditional"),
    ts_periods=(7, 30, 90, 365),
    feature_family_max_columns=4,
    categorical_cardinality_threshold=20,
    categorical_top_k=5,
)
```

Or set schema-independent defaults on the graph:

```python
graph = GraphReduce(
    parent_node=customers,
    compute_layer=ComputeLayerEnum.duckdb,
    auto_features=True,
    feature_families=("base", "temporal", "sequence"),
    ts_periods=(7, 30, 90, 365),
    feature_family_max_columns=4,
    categorical_cardinality_threshold=20,
    categorical_top_k=5,
)

graph.do_transformations_sql()
```

Graph-level feature settings default to `None`. When omitted, the node's value
is preserved; when supplied, the graph value overrides every node during graph
hydration. Node-specific concepts such as `annotation_expressions` and
`context_keys` should remain on the node whose schema defines them.

## Time windows

`ts_periods` contains lookback lengths in days. Each period is evaluated
relative to the current point-in-time reference when one is available, or to
the graph's `cut_date` otherwise.

Three controls work together:

- `date_key` identifies the event timestamp on the node.
- `compute_period_val` and `compute_period_unit` bound all available history.
- `ts_periods` creates summaries within that already-bounded history.

A 365-day `ts_periods` entry cannot recover rows if the graph's compute window
contains only 90 days. Set `ts_periods=()` to disable rolling-window features.

## `base`: schema-aware summaries

The `base` family inspects a sample and emits aggregations appropriate for the
inferred physical and semantic types:

- identifier-like columns contribute a count without duplicating the same
  count for every identifier;
- numeric columns use compatible functions from `type_func_map`;
- booleans produce true counts and shares;
- dates and timestamps produce configured date summaries;
- eligible categorical columns produce distinct counts and bounded
  count/share/presence features; and
- text-like columns can produce inexpensive length, word-count, empty-value,
  URL, number, and punctuation summaries.

For a time-series relation, `base` also creates recency, windowed event counts,
and adjacent-window change ratios.

A categorical column is eligible only when its observed cardinality is at most
`categorical_cardinality_threshold`. At most `categorical_top_k` frequent
values are expanded; an `other` bucket is added when the top-value limit omits
eligible values. Higher-cardinality columns are skipped.

Text summaries are controlled independently by `auto_text_features`. They do
not tokenize, embed, or call a language model.

## `temporal`: windowed numeric values

For each selected numeric column and lookback period, `temporal` generates
windowed `sum`, `avg`, `min`, and `max` features. The reduce key, date column,
and identifier-like values are excluded.

Approximate feature count for `N` selected numeric columns and `P` periods:

```text
4 × N × P
```

Use `feature_family_max_columns` to keep this family bounded on wide tables.

## `sequence`: activity shape

`sequence` describes cadence without exposing an ordered event list. It can
generate:

- events per day within each period;
- each period's share of lifetime activity;
- burst ratios between adjacent periods;
- the lifetime active span; and
- lifetime activity per active day.

These features help distinguish steady activity, recent bursts, and dormant
entities with similar lifetime counts.

## `conditional`: activity composition

`conditional` selects conditions from two sources:

- predicate columns created by semantic annotation; and
- frequent values from categorical columns within the cardinality threshold.

For each selected condition and period, it generates count, share, and
presence features, plus count-change ratios between adjacent periods. Semantic
predicates are selected before generic categories.

This family can grow quickly. Bound both `feature_family_max_columns` and
`categorical_top_k`.

## `episode`: rows versus distinct events

`episode` emits a total row count and, when a primary key is available, a
distinct primary-key count. Time-series nodes also receive those counts by
lookback period.

This distinction is useful after many-to-many joins: the row count measures
join volume, while the distinct count better approximates underlying events.

## `semantic`: explicit domain meaning

`semantic` compiles caller-supplied SQL expressions before reduction:

```python
events = DuckdbNode(
    # ...normal node configuration...
    feature_families=("base", "semantic", "conditional", "temporal"),
    annotation_expressions={
        "is_verified": "{status} = 'verified'",
        "is_high_value": "{amount} >= 100",
        "line_value": ("value", "{amount} * {quantity}"),
    },
)
```

A plain expression is treated as a predicate and becomes a `0/1` indicator.
The `("value", expression)` form preserves a numeric result for normal
aggregation. Braced placeholders are resolved against the node's columns.

The family does not guess business meaning from names. With
`auto_annotate_features=True`, it can additionally infer a bounded set of
categorical indicators, text indicators, and category-gated numeric values.

## `context`: peer-relative values

`context` compares numeric values within an explicit peer group before the
child is reduced:

```python
items = DuckdbNode(
    # ...normal node configuration...
    feature_families=("base", "context"),
    context_keys=("category_id",),
)
```

For each context key, GraphReduce can create peer-group size and the difference
between a row's numeric value and its peer-group average. The resulting values
are then aggregated to the parent.

GraphReduce deliberately does not infer peer groups. A plausible but incorrect
group creates misleading features, so `context_keys` is required.

## Keep the feature budget deliberate

The most important controls are:

| Setting | Controls |
| --- | --- |
| `feature_families` | Families enabled on a node |
| `ts_periods` | Count and duration of time windows |
| `feature_family_max_columns` | Source columns or conditions used by expanding families |
| `feature_family_max_features_per_column` | Derived features emitted per selected column |
| `categorical_cardinality_threshold` | Maximum observed cardinality for automatic categories |
| `categorical_top_k` | Values expanded per eligible category |
| `auto_text_features` | Automatic text summaries |
| `annotation_expressions` | Deterministic domain predicates and values |
| `context_keys` | Explicit peer groups |

A practical rollout is:

1. Establish a baseline with `("base",)`.
2. Add `temporal` for meaningful numeric measurements.
3. Add `sequence` when cadence or burstiness may matter.
4. Add `conditional` for important statuses or categories.
5. Add `episode` when rows and logical events differ.
6. Add `semantic` for known domain concepts.
7. Add `context` only when the peer group is unambiguous.

Feature counts multiply across columns, periods, relations, and graph hops.
Enable a family where its signal is plausible instead of applying a maximal
configuration to every node.
