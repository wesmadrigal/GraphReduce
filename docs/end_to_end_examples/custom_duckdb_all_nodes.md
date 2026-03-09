# Custom DuckDB Graph: All `cust_data` Nodes

This example runs the same custom all-nodes graph in DuckDB SQL across:

* `cust.csv`
* `orders.csv`
* `order_products.csv`
* `notifications.csv`
* `notification_interactions.csv`
* `notification_interaction_types.csv`

## Complete Example

```python
#!/usr/bin/env python
"""Custom duckdb GraphReduce example using all cust_data nodes."""

from __future__ import annotations

import datetime

try:
    import duckdb
except Exception as exc:  # pragma: no cover
    print(f"duckdb not available: {exc}", flush=True)
    raise SystemExit(0)

from graphreduce.enum import ComputeLayerEnum, SQLOpType
from graphreduce.graph_reduce import GraphReduce
from graphreduce.models import sqlop
from graphreduce.node import SQLNode


class CustNode(SQLNode):
    def do_annotate(self):
        return [
            sqlop(
                optype=SQLOpType.select,
                opval=f"*, length(coalesce({self.colabbr('name')}, '')) as {self.colabbr('name_length')}",
            )
        ]

    def do_filters(self):
        return [sqlop(optype=SQLOpType.where, opval=f"{self.colabbr('id')} is not null")]

    def do_normalize(self):
        return [
            sqlop(
                optype=SQLOpType.select,
                opval=f"*, lower(trim(coalesce({self.colabbr('name')}, ''))) as {self.colabbr('name')}",
            )
        ]

    def do_reduce(self, reduce_key):
        return None

    def do_post_join_annotate(self):
        return [
            sqlop(
                optype=SQLOpType.select,
                opval="*"
                + ", coalesce(ord_num_orders, 0) + coalesce(not_num_notifications, 0) + coalesce(not_num_engaged_interactions, 0) as cust_total_events"
                + ", case"
                + " when (coalesce(ord_num_orders, 0) + coalesce(not_num_notifications, 0) + coalesce(not_num_engaged_interactions, 0)) >= 5 then 'high'"
                + " when (coalesce(ord_num_orders, 0) + coalesce(not_num_notifications, 0) + coalesce(not_num_engaged_interactions, 0)) >= 2 then 'medium'"
                + " else 'low' end as cust_activity_tier",
            )
        ]

    def do_post_join_filters(self):
        return [sqlop(optype=SQLOpType.where, opval="cust_total_events >= 0")]


class OrderNode(SQLNode):
    def do_annotate(self):
        return [
            sqlop(
                optype=SQLOpType.select,
                opval=f"*, cast({self.colabbr('amount')} as double) as {self.colabbr('amount_dbl')}",
            )
        ]

    def do_filters(self):
        return [sqlop(optype=SQLOpType.where, opval=f"{self.colabbr('ts')} >= '2022-01-01'")]

    def do_normalize(self):
        return [
            sqlop(
                optype=SQLOpType.select,
                opval=f"*, coalesce({self.colabbr('amount_dbl')}, 0.0) as {self.colabbr('amount_dbl')}",
            )
        ]

    def do_reduce(self, reduce_key):
        return [
            sqlop(optype=SQLOpType.aggfunc, opval=f"count(distinct {self.colabbr(self.pk)}) as {self.colabbr('num_orders')}"),
            sqlop(optype=SQLOpType.aggfunc, opval=f"sum({self.colabbr('amount_dbl')}) as {self.colabbr('sum_amount')}"),
            sqlop(optype=SQLOpType.aggfunc, opval=f"avg({self.colabbr('amount_dbl')}) as {self.colabbr('avg_amount')}"),
            sqlop(optype=SQLOpType.agg, opval=f"{self.colabbr(reduce_key)}"),
        ]


class OrderProductsNode(SQLNode):
    def do_annotate(self):
        return [
            sqlop(
                optype=SQLOpType.select,
                opval=f"*, cast({self.colabbr('product_id')} as int) as {self.colabbr('product_id_int')}",
            )
        ]

    def do_filters(self):
        return [sqlop(optype=SQLOpType.where, opval=f"{self.colabbr('product_id_int')} is not null")]

    def do_normalize(self):
        return None

    def do_reduce(self, reduce_key):
        return [
            sqlop(optype=SQLOpType.aggfunc, opval=f"count({self.colabbr(self.pk)}) as {self.colabbr('num_order_products')}"),
            sqlop(
                optype=SQLOpType.aggfunc,
                opval=f"count(distinct {self.colabbr('product_id_int')}) as {self.colabbr('num_distinct_products')}",
            ),
            sqlop(optype=SQLOpType.agg, opval=f"{self.colabbr(reduce_key)}"),
        ]


class NotificationNode(SQLNode):
    def do_annotate(self):
        return [
            sqlop(
                optype=SQLOpType.select,
                opval=f"*, strftime({self.colabbr('ts')}, '%m') as {self.colabbr('ts_month')}",
            )
        ]

    def do_filters(self):
        return [sqlop(optype=SQLOpType.where, opval=f"{self.colabbr('ts')} >= '2022-01-01'")]

    def do_normalize(self):
        return None

    def do_reduce(self, reduce_key):
        return [
            sqlop(optype=SQLOpType.aggfunc, opval=f"count(distinct {self.colabbr(self.pk)}) as {self.colabbr('num_notifications')}"),
            sqlop(optype=SQLOpType.aggfunc, opval=f"max({self.colabbr('ts')}) as {self.colabbr('max_notification_ts')}"),
            sqlop(optype=SQLOpType.aggfunc, opval=f"sum(coalesce(ni_num_interactions, 0)) as {self.colabbr('num_interactions')}"),
            sqlop(
                optype=SQLOpType.aggfunc,
                opval=f"sum(coalesce(ni_num_engaged_interactions, 0)) as {self.colabbr('num_engaged_interactions')}",
            ),
            sqlop(optype=SQLOpType.agg, opval=f"{self.colabbr(reduce_key)}"),
        ]


class NotificationInteractionsNode(SQLNode):
    def do_annotate(self):
        return [
            sqlop(
                optype=SQLOpType.select,
                opval=f"*, strftime({self.colabbr('ts')}, '%Y-%m-%d') as {self.colabbr('ts_day')}",
            )
        ]

    def do_filters(self):
        return [sqlop(optype=SQLOpType.where, opval=f"{self.colabbr('ts')} >= '2022-01-01'")]

    def do_normalize(self):
        return None

    def do_reduce(self, reduce_key):
        return [
            sqlop(optype=SQLOpType.aggfunc, opval=f"count({self.colabbr(self.pk)}) as {self.colabbr('num_interactions')}"),
            sqlop(
                optype=SQLOpType.aggfunc,
                opval=f"count(distinct {self.colabbr('interaction_type_id')}) as {self.colabbr('num_interaction_types')}",
            ),
            sqlop(
                optype=SQLOpType.aggfunc,
                opval=f"sum(coalesce(nit_is_engagement_type, 0)) as {self.colabbr('num_engaged_interactions')}",
            ),
            sqlop(optype=SQLOpType.agg, opval=f"{self.colabbr(reduce_key)}"),
        ]


class NotificationInteractionTypeNode(SQLNode):
    def do_annotate(self):
        return [
            sqlop(
                optype=SQLOpType.select,
                opval="*"
                + f", lower(trim({self.colabbr('name')})) as {self.colabbr('name')}"
                + f", case when lower(trim({self.colabbr('name')})) in ('clicked', 'dismissed') then 1 else 0 end as {self.colabbr('is_engagement_type')}"
                + f", case when lower(trim({self.colabbr('name')})) like '%view%' then 1 else 0 end as {self.colabbr('is_view_event')}",
            )
        ]

    def do_filters(self):
        return [sqlop(optype=SQLOpType.where, opval=f"{self.colabbr('id')} is not null")]

    def do_normalize(self):
        return None

    def do_reduce(self, reduce_key):
        return [
            sqlop(optype=SQLOpType.aggfunc, opval=f"min({self.colabbr('name')}) as {self.colabbr('name')}"),
            sqlop(
                optype=SQLOpType.aggfunc,
                opval=f"max({self.colabbr('is_engagement_type')}) as {self.colabbr('is_engagement_type')}",
            ),
            sqlop(optype=SQLOpType.aggfunc, opval=f"max({self.colabbr('is_view_event')}) as {self.colabbr('is_view_event')}"),
            sqlop(optype=SQLOpType.agg, opval=f"{self.colabbr(reduce_key)}"),
        ]


def build_custom_duckdb_graph(con: duckdb.DuckDBPyConnection) -> GraphReduce:
    con.sql("CREATE OR REPLACE VIEW cust AS SELECT * FROM read_csv_auto('tests/data/cust_data/cust.csv', header=true)")
    con.sql("CREATE OR REPLACE VIEW orders AS SELECT * FROM read_csv_auto('tests/data/cust_data/orders.csv', header=true)")
    con.sql("CREATE OR REPLACE VIEW order_products AS SELECT * FROM read_csv_auto('tests/data/cust_data/order_products.csv', header=true)")
    con.sql("CREATE OR REPLACE VIEW notifications AS SELECT * FROM read_csv_auto('tests/data/cust_data/notifications.csv', header=true)")
    con.sql("CREATE OR REPLACE VIEW notification_interactions AS SELECT * FROM read_csv_auto('tests/data/cust_data/notification_interactions.csv', header=true)")
    con.sql("CREATE OR REPLACE VIEW notification_interaction_types AS SELECT * FROM read_csv_auto('tests/data/cust_data/notification_interaction_types.csv', header=true)")

    cust = CustNode(fpath="cust", prefix="cust", pk="id", client=con, compute_layer=ComputeLayerEnum.duckdb, columns=["id", "name"])
    orders = OrderNode(
        fpath="orders",
        prefix="ord",
        pk="id",
        date_key="ts",
        client=con,
        compute_layer=ComputeLayerEnum.duckdb,
        columns=["id", "customer_id", "ts", "amount"],
    )
    order_products = OrderProductsNode(
        fpath="order_products",
        prefix="op",
        pk="id",
        client=con,
        compute_layer=ComputeLayerEnum.duckdb,
        columns=["id", "order_id", "product_id"],
    )
    notifications = NotificationNode(
        fpath="notifications",
        prefix="not",
        pk="id",
        date_key="ts",
        client=con,
        compute_layer=ComputeLayerEnum.duckdb,
        columns=["id", "customer_id", "ts"],
    )
    notification_interactions = NotificationInteractionsNode(
        fpath="notification_interactions",
        prefix="ni",
        pk="id",
        date_key="ts",
        client=con,
        compute_layer=ComputeLayerEnum.duckdb,
        columns=["id", "notification_id", "interaction_type_id", "ts"],
    )
    interaction_types = NotificationInteractionTypeNode(
        fpath="notification_interaction_types",
        prefix="nit",
        pk="id",
        client=con,
        compute_layer=ComputeLayerEnum.duckdb,
        columns=["id", "name"],
    )

    gr = GraphReduce(
        name="custom_duckdb_all_nodes",
        parent_node=cust,
        compute_layer=ComputeLayerEnum.duckdb,
        sql_client=con,
        use_temp_tables=True,
        cut_date=datetime.datetime(2023, 6, 30),
    )

    for node in [cust, orders, order_products, notifications, notification_interactions, interaction_types]:
        gr.add_node(node)

    gr.add_entity_edge(cust, orders, parent_key="id", relation_key="customer_id", reduce=True)
    gr.add_entity_edge(orders, order_products, parent_key="id", relation_key="order_id", reduce=True)
    gr.add_entity_edge(cust, notifications, parent_key="id", relation_key="customer_id", reduce=True)
    gr.add_entity_edge(notifications, notification_interactions, parent_key="id", relation_key="notification_id", reduce=True)
    gr.add_entity_edge(notification_interactions, interaction_types, parent_key="interaction_type_id", relation_key="id", reduce=True)

    return gr


def run_custom_duckdb_all_nodes() -> "duckdb.DuckDBPyRelation":
    con = duckdb.connect()
    gr = build_custom_duckdb_graph(con)
    gr.do_transformations_sql()
    return con.sql(f"select * from {gr.parent_node._cur_data_ref}").to_df()


def main() -> None:
    out_df = run_custom_duckdb_all_nodes()
    print("rows:", len(out_df), flush=True)
    print("columns:", len(out_df.columns), flush=True)
    print("column_names:", out_df.columns.tolist(), flush=True)


if __name__ == "__main__":
    main()
```

Full runnable script:

* `examples/custom_duckdb_all_nodes.py`

## Run Interactive

<div class="modal-runner" data-modal-runner data-api-base="https://runner.13.218.155.128.sslip.io" data-example="custom_duckdb_all_nodes">
  <div class="modal-runner-controls">
    <input class="modal-runner-input" data-api-input value="https://runner.13.218.155.128.sslip.io" />
    <button data-save-api-btn>Save API URL</button>
    <button data-run-btn>Run custom duckdb all-nodes</button>
  </div>
  <div class="modal-runner-status" data-status>Idle</div>
  <pre class="modal-runner-log" data-log></pre>
</div>
