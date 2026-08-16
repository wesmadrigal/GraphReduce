# Feature schema profiles

GraphReduce can fit a typed, model-agnostic feature manifest from a training
sample. The manifest separates structural keys from model features and makes
cutoff-availability decisions explicit.

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

node.columns = list(manifest.graph_columns)
categorical_columns = manifest.categorical_columns
```

SQL nodes provide the same operation directly:

```python
manifest = node.infer_feature_manifest(
    training_sample,
    foreign_keys=("sponsor_id",),
    target_columns="outcome",
    unsafe_columns=("future_duration",),
    apply_columns=True,
)
```

Profiling must use training data. Validation and test frames may be aligned to
the resulting manifest, but must not be used to decide types, coverage,
cardinality, or feature inclusion.

GraphReduce cannot infer when a value became available from its dtype or name.
Callers must declare mutable or future-derived columns through
`unsafe_columns`. This is dataset metadata, not task-specific feature logic.

The manifest complements the existing generic SQL feature primitives. In
particular, `auto_annotate_features=True` already creates bounded categorical
indicators and category-gated numeric values. Those values feed normal
reductions, so a categorical `event_type` and numerical `amount` can produce
aggregates such as the minimum amount for each frequent event type without
hardcoding domain values.
