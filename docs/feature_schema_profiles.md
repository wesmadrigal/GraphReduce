# Feature Schema Profiles

A feature schema profile records which columns are structural, predictive, or
unsafe. Fit it on training data before automatic feature generation so schema
decisions do not learn from validation or test distributions.

## Create a manifest

```python
from graphreduce.feature_schema import profile_feature_schema

manifest = profile_feature_schema(
    training_sample,
    primary_keys="study_id",
    foreign_keys=("sponsor_id",),
    date_keys="event_time",
    target_columns="outcome",
    unsafe_columns=("future_duration",),
    source="studies",
)
```

The manifest separates:

- primary and foreign keys used for graph structure;
- date keys used for temporal behavior;
- target columns used for supervision;
- unsafe columns that must not become features; and
- eligible model features with inferred types.

Typical consumers are:

```python
node.columns = list(manifest.graph_columns)
categorical_columns = manifest.categorical_columns
```

## Infer through a node

Nodes expose the same workflow:

```python
manifest = node.infer_feature_manifest(
    training_sample,
    foreign_keys=("sponsor_id",),
    target_columns="outcome",
    unsafe_columns=("future_duration",),
    apply_columns=True,
)
```

With `apply_columns=True`, the inferred graph columns are assigned back to the
node.

## Prevent schema leakage

Use only the training sample to decide:

- semantic and physical types;
- missing-value coverage;
- categorical cardinality;
- feature inclusion; and
- any value-frequency threshold.

Validation and test frames may be aligned to the completed manifest, but must
not change it.

GraphReduce cannot determine availability time from a dtype or column name. A
field such as `resolved_at`, `final_status`, or `lifetime_value` may contain
future information even when it looks harmless. Put mutable, post-outcome, or
future-derived fields in `unsafe_columns` based on dataset knowledge.

## Relationship to automatic features

The manifest decides which columns are safe and how they should be understood.
Automatic feature families decide which relational operations to generate from
those columns. For example, a safe categorical `event_type` can produce bounded
indicators, while a safe numeric `amount` can produce reductions or windowed
summaries.

See [SQL auto-feature families](sql_auto_feature_families.md) for those
generation controls.
