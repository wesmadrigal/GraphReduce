# Verify Point-in-Time Correctness

Time leakage occurs when a feature uses information that would not have been
available at prediction time. GraphReduce prevents this by applying separate
historical and future windows around a cutoff.

## Window contract

For cutoff `T`:

```text
features: T - compute period < timestamp < T
labels:   T                  < timestamp < T + label period
```

For non-null timestamps, the comparisons are strict. Rows on any boundary are
excluded.

## Inspect one node

Run this example from the repository root:

```python
import datetime as dt
from pathlib import Path

import pandas as pd

from graphreduce.enum import ComputeLayerEnum, PeriodUnit
from graphreduce.node import DynamicNode

orders = DynamicNode(
    fpath=str(Path("tests/data/cust_data/orders.csv")),
    fmt="csv",
    pk="id",
    prefix="ord",
    date_key="ts",
    compute_layer=ComputeLayerEnum.pandas,
    cut_date=dt.datetime(2023, 7, 1),
    compute_period_val=365,
    compute_period_unit=PeriodUnit.day,
    label_period_val=60,
    label_period_unit=PeriodUnit.day,
)

orders.do_data()

feature_rows = orders.prep_for_features()
label_rows = orders.prep_for_labels()

print(feature_rows[["ord_id", "ord_ts"]])
print(label_rows[["ord_id", "ord_ts"]])

feature_times = pd.to_datetime(feature_rows["ord_ts"])
label_times = pd.to_datetime(label_rows["ord_ts"])
cutoff = pd.Timestamp("2023-07-01")

assert (feature_times < cutoff).all()
assert (label_times > cutoff).all()
```

`do_data()` loads the CSV and adds the `ord_` prefix. The two helper methods
return filtered frames; they do not mix feature and label rows. The explicit
`pd.to_datetime()` calls above normalize the result before the verification
assertions.

## Configure the whole graph

Set shared time controls on `GraphReduce`:

```python
graph = GraphReduce(
    parent_node=customers,
    compute_layer=ComputeLayerEnum.pandas,
    cut_date=dt.datetime(2023, 7, 1),
    compute_period_val=365,
    compute_period_unit=PeriodUnit.day,
    label_node=orders,
    label_field="id",
    label_operation="count",
    label_period_val=60,
    label_period_unit=PeriodUnit.day,
    auto_features=True,
)
```

During graph hydration, these values are pushed to the nodes so one execution
uses a consistent reference time.

## Checklist before training

- Give every time-varying node the correct `date_key`.
- Confirm timestamps represent availability time, not merely when an event was
  later recorded or updated.
- Treat static dimensions without a `date_key` carefully if their attributes
  can change over time.
- Decide whether rows with null timestamps should be retained. The dataframe
  feature helper currently retains them, while SQL comparisons normally do not.
- Test both helper windows on a small known sample.
- Assert the output grain after all joins.
- Reserve validation and test cutoffs that are later than training cutoffs.

GraphReduce can enforce configured timestamps; it cannot infer whether a source
column itself contains future-derived information. Declare or remove such
columns before feature generation. See
[Feature schema profiles](feature_schema_profiles.md) for explicit unsafe-column
handling.
