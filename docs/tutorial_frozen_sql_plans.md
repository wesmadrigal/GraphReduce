# Freezing and Replaying SQL Execution Plans

This page shows how to:

1. run a SQL-backed `GraphReduce` graph with `train=True` and `auto_features=True`
2. freeze the exact executed `sqlop` instances
3. replay those same ops into a fresh `train=True` graph
4. replay those same ops into a fresh `train=False` graph

The important point is that `GraphReduce.freeze_execution_plan()` freezes the
executed SQL operation program, not just the final dataframe. That means a later
graph can reuse the same per-method `sqlop` instances instead of dynamically
recomputing feature SQL.

## What Gets Frozen

After `do_transformations_sql()` runs, GraphReduce stores structured execution
records for methods such as:

* `do_data`
* `do_annotate`
* `do_filters`
* `do_normalize`
* `do_reduce`
* `do_labels`
* `do_post_join_annotate`
* `do_post_join_filters`

`freeze_execution_plan()` returns those executed ops grouped by method and also
preserves the execution order by node and edge. `apply_execution_plan(plan)`
loads that frozen plan into a fresh graph so `do_transformations_sql()` can
replay the same method outputs.

When replaying into `train=False`:

* label ops are not applied
* label-dependent post-join ops are not applied
* date-node joins are not attempted
* date literals in frozen SQL ops are rebound to the receiving graph's current
  `cut_date`

If you do not pass `cut_date` to the scoring graph, that graph keeps its normal
default behavior, which is `datetime.datetime.now()`.

## Simple Dataset

This example uses the same sqlite dataset across all three runs:

* `cust`
* `orders`
* `notifications`

The customer node also has a post-join annotation that depends on the label
column created from the `orders` node during training.

```python
import datetime
import sqlite3
import pandas as pd

from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import SQLNode
from graphreduce.enum import ComputeLayerEnum, PeriodUnit, SQLOpType
from graphreduce.models import sqlop
```

```python
def setup_sqlite():
    conn = sqlite3.connect(":memory:")

    pd.DataFrame(
        [
            {"id": 1, "name": "alex"},
            {"id": 2, "name": "blair"},
            {"id": 3, "name": "casey"},
            {"id": 4, "name": "devon"},
        ]
    ).to_sql("cust", conn, index=False, if_exists="replace")

    pd.DataFrame(
        [
            {"id": 1, "customer_id": 1, "ts": "2023-01-05", "amount": 10.0},
            {"id": 2, "customer_id": 1, "ts": "2023-06-01", "amount": 25.0},
            {"id": 3, "customer_id": 2, "ts": "2023-04-15", "amount": 5.0},
            {"id": 4, "customer_id": 4, "ts": "2023-05-20", "amount": 99.0},
        ]
    ).to_sql("orders", conn, index=False, if_exists="replace")

    pd.DataFrame(
        [
            {"id": 1, "customer_id": 1, "ts": "2023-06-10"},
            {"id": 2, "customer_id": 1, "ts": "2023-06-11"},
            {"id": 3, "customer_id": 3, "ts": "2023-03-01"},
        ]
    ).to_sql("notifications", conn, index=False, if_exists="replace")

    return conn
```

```python
def build_graph(conn, train, cut_date=None):
    orders = SQLNode(
        fpath="orders",
        pk="id",
        prefix="ord",
        client=conn,
        compute_layer=ComputeLayerEnum.sqlite,
        columns=["id", "customer_id", "ts", "amount"],
        date_key="ts",
    )

    cust = SQLNode(
        fpath="cust",
        pk="id",
        prefix="cust",
        client=conn,
        compute_layer=ComputeLayerEnum.sqlite,
        columns=["id", "name"],
        do_post_join_annotate_ops=[
            sqlop(optype=SQLOpType.select, opval="*"),
            sqlop(
                optype=SQLOpType.select,
                opval="case when ord_id_label > 0 then 1 else 0 end as has_order_label",
            ),
        ] if train else None,
        do_post_join_annotate_requires=[orders],
    )

    notifications = SQLNode(
        fpath="notifications",
        pk="id",
        prefix="not",
        client=conn,
        compute_layer=ComputeLayerEnum.sqlite,
        columns=["id", "customer_id", "ts"],
        date_key="ts",
    )

    kwargs = dict(
        name=f"cust_sql_plan_train_{train}",
        parent_node=cust,
        compute_period_unit=PeriodUnit.day,
        compute_period_val=730,
        label_node=orders,
        label_field="id",
        label_operation="bool",
        label_period_unit=PeriodUnit.day,
        label_period_val=90,
        compute_layer=ComputeLayerEnum.sqlite,
        use_temp_tables=True,
        lazy_execution=False,
        auto_features=True,
        auto_feature_hops_back=3,
        auto_feature_hops_front=0,
        sql_client=conn,
        date_filters_on_agg=True,
        train=train,
    )
    if cut_date is not None:
        kwargs["cut_date"] = cut_date

    gr = GraphReduce(**kwargs)
    gr.add_node(cust)
    gr.add_node(orders)
    gr.add_node(notifications)
    gr.add_entity_edge(cust, notifications, parent_key="id", relation_key="customer_id", reduce=True)
    gr.add_entity_edge(cust, orders, parent_key="id", relation_key="customer_id", reduce=True)
    return gr
```

## Example 1: `train=True` with Dynamic Auto-Generated Features

This is the normal training path. GraphReduce inspects the graph, auto-generates
SQL feature ops where needed, executes labels, and records the executed SQL ops.

```python
train_cut_date = datetime.datetime(2023, 6, 30)

train_conn = setup_sqlite()
train_gr = build_graph(train_conn, train=True, cut_date=train_cut_date)
train_gr.do_transformations_sql()

train_df = pd.read_sql_query(
    f"select * from {train_gr.parent_node._cur_data_ref}",
    train_conn,
).sort_values("cust_id")

frozen_plan = train_gr.freeze_execution_plan()
sql_ops_by_method = train_gr.get_executed_sqlops_by_method(exclude_date_filters=True)

print(sorted(sql_ops_by_method))
print(train_df.columns.tolist())
```

What happened in this run:

* `do_reduce` may include auto-generated SQL ops from `sql_auto_features`
* `do_labels` executes because this is a training graph
* `do_post_join_annotate` executes after the label join because it depends on
  `ord_id_label`
* `freeze_execution_plan()` captures the exact executed `sqlop` sequences

## Example 2: `train=True` Replaying a Previously Frozen Plan

Now create a brand-new graph against the same dataset and same training cut date.
Instead of dynamically generating features again, load the frozen plan first.

```python
replay_train_conn = setup_sqlite()
replay_train_gr = build_graph(replay_train_conn, train=True, cut_date=train_cut_date)

replay_train_gr.apply_execution_plan(frozen_plan)
replay_train_gr.do_transformations_sql()

replay_train_df = pd.read_sql_query(
    f"select * from {replay_train_gr.parent_node._cur_data_ref}",
    replay_train_conn,
).sort_values("cust_id")

replay_train_sql_ops = replay_train_gr.get_executed_sqlops_by_method(
    exclude_date_filters=True
)

print("do_labels" in replay_train_sql_ops)
print(train_df.equals(replay_train_df))
```

What changed in this run:

* the graph shape is the same
* the executed SQL ops are loaded from `frozen_plan`
* `do_transformations_sql()` replays the frozen `sqlop` outputs by node, method,
  and edge
* `do_labels` still executes because this graph is still `train=True`
* the output should match the original training run

This is useful when you want reproducible SQL feature generation across repeated
training jobs.

## Example 3: `train=False` Replaying Frozen Features on the Same Dataset

Now build the same graph again, but this time for scoring.

Do not pass a scoring `cut_date` here if you want the graph to keep its normal
`train=False` behavior. In that case the graph uses its current default
`cut_date`, which is `datetime.datetime.now()`.

```python
score_conn = setup_sqlite()
score_gr = build_graph(score_conn, train=False)

score_gr.apply_execution_plan(frozen_plan)
score_gr.do_transformations_sql()

score_df = pd.read_sql_query(
    f"select * from {score_gr.parent_node._cur_data_ref}",
    score_conn,
).sort_values("cust_id")

score_sql_ops = score_gr.get_executed_sqlops_by_method(exclude_date_filters=True)

print("do_labels" in score_sql_ops)
print("do_post_join_annotate" in score_sql_ops)
print(score_df.columns.tolist())
```

What changed in this run:

* the frozen feature ops are replayed
* date literals from the frozen training plan are rebound to the scoring graph's
  current `cut_date`
* `do_labels` is omitted
* label-dependent `do_post_join_annotate` and `do_post_join_filters` are omitted
* date-node joins are skipped

This lets you keep feature logic fixed between training and scoring while still
allowing the scoring graph to use its own current cutoff date.

## Mental Model

The three runs are doing different things with the same graph structure:

1. `train=True` dynamic run: generate SQL ops and execute them
2. `train=True` frozen replay: reuse the exact executed SQL ops from step 1
3. `train=False` frozen replay: reuse the frozen feature ops but omit labels and
   label-dependent post-join logic

The core API is:

```python
plan = train_gr.freeze_execution_plan()

new_gr.apply_execution_plan(plan)
new_gr.do_transformations_sql()
```

Use `get_executed_sqlops_by_method()` or `get_executed_op_records()` if you want
to inspect exactly what ran after execution.
