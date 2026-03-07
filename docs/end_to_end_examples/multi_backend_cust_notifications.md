# Multi-Backend Example: Same Graph, Multiple Compute Layers

This section mirrors the RelBench task organization, but focuses on **one
identical graph spec** across multiple compute backends:

* parent node: `cust`
* relation node: `notifications`
* edge: `cust.id -> notifications.customer_id` with `reduce=True`
* `CustNode` filter: `id < 3`
* `NotificationNode` filter: `ts > '2022-06-01'`
* `NotificationNode` reduction: count notifications per customer

Dataset source used in all examples: `tests/data/cust_data/*`

## Expandable Backend Examples

<details>
<summary><strong>pandas backend</strong></summary>

```python
import os
import datetime
import pandas as pd

from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import GraphReduceNode
from graphreduce.enum import ComputeLayerEnum

DATA_PATH = "tests/data/cust_data"


class CustNode(GraphReduceNode):
    def do_annotate(self):
        self.df[self.colabbr("name_length")] = self.df[self.colabbr("name")].fillna("").str.len()
        return self.df

    def do_filters(self):
        self.df = self.df[self.df[self.colabbr("id")] < 3]
        return self.df

    def do_normalize(self):
        pass

    def do_reduce(self, reduce_key):
        pass

    def do_labels(self, reduce_key):
        pass

    def do_post_join_annotate(self):
        pass

    def do_post_join_filters(self):
        pass


class NotificationNode(GraphReduceNode):
    def do_annotate(self):
        self.df[self.colabbr("ts_month")] = self.df[self.colabbr("ts")].astype(str).str.slice(5, 7)
        return self.df

    def do_filters(self):
        self.df = self.df[self.df[self.colabbr("ts")] > "2022-06-01"]
        return self.df

    def do_normalize(self):
        pass

    def do_reduce(self, reduce_key):
        return (
            self.prep_for_features()
            .groupby(self.colabbr(reduce_key))
            .agg(**{self.colabbr("num_notifications"): pd.NamedAgg(column=self.colabbr(self.pk), aggfunc="count")})
            .reset_index()
        )

    def do_labels(self, reduce_key):
        pass

    def do_post_join_annotate(self):
        pass

    def do_post_join_filters(self):
        pass


cust = CustNode(
    fpath=os.path.join(DATA_PATH, "cust.csv"),
    fmt="csv",
    prefix="cust",
    pk="id",
    compute_layer=ComputeLayerEnum.pandas,
    columns=["id", "name"],
)
notif = NotificationNode(
    fpath=os.path.join(DATA_PATH, "notifications.csv"),
    fmt="csv",
    prefix="not",
    pk="id",
    date_key="ts",
    compute_layer=ComputeLayerEnum.pandas,
    columns=["id", "customer_id", "ts"],
)

gr = GraphReduce(
    name="cust_notif_pandas",
    parent_node=cust,
    compute_layer=ComputeLayerEnum.pandas,
    cut_date=datetime.datetime(2023, 6, 30),
)
gr.add_node(cust)
gr.add_node(notif)
gr.add_entity_edge(cust, notif, parent_key="id", relation_key="customer_id", reduce=True)
gr.do_transformations()

print(gr.parent_node.df.head())
```

</details>

<details>
<summary><strong>sqlite backend (from tests/test_graph_reduce.py::test_sql_node_definition)</strong></summary>

```python
import sqlite3
import pandas as pd

from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import SQLNode
from graphreduce.enum import ComputeLayerEnum, SQLOpType
from graphreduce.models import sqlop

DATA_PATH = "tests/data/cust_data"
conn = sqlite3.connect(":memory:")

for table in ["cust", "notifications"]:
    pd.read_csv(f"{DATA_PATH}/{table}.csv").to_sql(table, conn, if_exists="replace", index=False)


class CustNode(SQLNode):
    def do_annotate(self):
        return [sqlop(optype=SQLOpType.select, opval=f"*, LENGTH({self.colabbr('name')}) as {self.colabbr('name_length')}")]

    def do_filters(self):
        return [sqlop(optype=SQLOpType.where, opval=f"{self.colabbr('id')} < 3")]

    def do_normalize(self):
        pass

    def do_reduce(self, reduce_key):
        pass

    def do_post_join_annotate(self):
        pass

    def do_post_join_filters(self):
        pass


class NotificationNode(SQLNode):
    def do_annotate(self):
        return [sqlop(optype=SQLOpType.select, opval=f"*, strftime('%m', {self.colabbr('ts')}) as {self.colabbr('ts_month')}")]

    def do_filters(self):
        return [sqlop(optype=SQLOpType.where, opval=f"{self.colabbr('ts')} > '2022-06-01'")]

    def do_normalize(self):
        pass

    def do_reduce(self, reduce_key):
        return [
            sqlop(optype=SQLOpType.aggfunc, opval=f"count(*) as {self.colabbr('num_notifications')}"),
            sqlop(optype=SQLOpType.agg, opval=f"{self.colabbr(reduce_key)}"),
        ]


cust = CustNode(
    fpath="cust",
    prefix="cust",
    pk="id",
    client=conn,
    compute_layer=ComputeLayerEnum.sqlite,
    columns=["id", "name"],
)
notif = NotificationNode(
    fpath="notifications",
    prefix="not",
    pk="id",
    date_key="ts",
    client=conn,
    compute_layer=ComputeLayerEnum.sqlite,
    columns=["id", "customer_id", "ts"],
)

gr = GraphReduce(
    name="cust_notif_sqlite",
    parent_node=cust,
    compute_layer=ComputeLayerEnum.sqlite,
    sql_client=conn,
    use_temp_tables=True,
)
gr.add_node(cust)
gr.add_node(notif)
gr.add_entity_edge(cust, notif, parent_key="id", relation_key="customer_id", reduce=True)
gr.do_transformations_sql()
```

</details>

<details>
<summary><strong>duckdb backend</strong></summary>

```python
import duckdb
import pandas as pd

from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import SQLNode
from graphreduce.enum import ComputeLayerEnum, SQLOpType
from graphreduce.models import sqlop

DATA_PATH = "tests/data/cust_data"
con = duckdb.connect()
con.sql("CREATE OR REPLACE VIEW cust AS SELECT * FROM read_csv_auto('tests/data/cust_data/cust.csv', header=true)")
con.sql("CREATE OR REPLACE VIEW notifications AS SELECT * FROM read_csv_auto('tests/data/cust_data/notifications.csv', header=true)")


class CustNode(SQLNode):
    def do_annotate(self):
        return [sqlop(optype=SQLOpType.select, opval=f"*, LENGTH({self.colabbr('name')}) as {self.colabbr('name_length')}")]

    def do_filters(self):
        return [sqlop(optype=SQLOpType.where, opval=f"{self.colabbr('id')} < 3")]

    def do_normalize(self):
        pass

    def do_reduce(self, reduce_key):
        pass

    def do_post_join_annotate(self):
        pass

    def do_post_join_filters(self):
        pass


class NotificationNode(SQLNode):
    def do_annotate(self):
        return [sqlop(optype=SQLOpType.select, opval=f"*, strftime({self.colabbr('ts')}, '%m') as {self.colabbr('ts_month')}")]

    def do_filters(self):
        return [sqlop(optype=SQLOpType.where, opval=f"{self.colabbr('ts')} > '2022-06-01'")]

    def do_normalize(self):
        pass

    def do_reduce(self, reduce_key):
        return [
            sqlop(optype=SQLOpType.aggfunc, opval=f"count(*) as {self.colabbr('num_notifications')}"),
            sqlop(optype=SQLOpType.agg, opval=f"{self.colabbr(reduce_key)}"),
        ]


cust = CustNode(
    fpath="cust",
    prefix="cust",
    pk="id",
    client=con,
    compute_layer=ComputeLayerEnum.duckdb,
    columns=["id", "name"],
)
notif = NotificationNode(
    fpath="notifications",
    prefix="not",
    pk="id",
    date_key="ts",
    client=con,
    compute_layer=ComputeLayerEnum.duckdb,
    columns=["id", "customer_id", "ts"],
)

gr = GraphReduce(
    name="cust_notif_duckdb",
    parent_node=cust,
    compute_layer=ComputeLayerEnum.duckdb,
    sql_client=con,
    use_temp_tables=True,
)
gr.add_node(cust)
gr.add_node(notif)
gr.add_entity_edge(cust, notif, parent_key="id", relation_key="customer_id", reduce=True)
gr.do_transformations_sql()
```

</details>

<details>
<summary><strong>pyspark backend</strong></summary>

```python
import os
import datetime
from pyspark.sql import SparkSession, functions as F

from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import GraphReduceNode
from graphreduce.enum import ComputeLayerEnum

DATA_PATH = "tests/data/cust_data"
spark = SparkSession.builder.appName("graphreduce-cust-notif").getOrCreate()


class CustNode(GraphReduceNode):
    def do_annotate(self):
        self.df = self.df.withColumn(self.colabbr("name_length"), F.length(F.col(self.colabbr("name"))))
        return self.df

    def do_filters(self):
        self.df = self.df.filter(F.col(self.colabbr("id")) < 3)
        return self.df

    def do_normalize(self):
        pass

    def do_reduce(self, reduce_key):
        pass

    def do_labels(self, reduce_key):
        pass

    def do_post_join_annotate(self):
        pass

    def do_post_join_filters(self):
        pass


class NotificationNode(GraphReduceNode):
    def do_annotate(self):
        self.df = self.df.withColumn(self.colabbr("ts_month"), F.date_format(F.col(self.colabbr("ts")), "MM"))
        return self.df

    def do_filters(self):
        self.df = self.df.filter(F.col(self.colabbr("ts")) > F.lit("2022-06-01"))
        return self.df

    def do_normalize(self):
        pass

    def do_reduce(self, reduce_key):
        return (
            self.prep_for_features()
            .groupBy(self.colabbr(reduce_key))
            .agg(F.count(F.col(self.colabbr(self.pk))).alias(self.colabbr("num_notifications")))
        )

    def do_labels(self, reduce_key):
        pass

    def do_post_join_annotate(self):
        pass

    def do_post_join_filters(self):
        pass


cust = CustNode(
    fpath=os.path.join(DATA_PATH, "cust.csv"),
    fmt="csv",
    prefix="cust",
    pk="id",
    compute_layer=ComputeLayerEnum.spark,
    columns=["id", "name"],
    spark_sqlctx=spark,
)
notif = NotificationNode(
    fpath=os.path.join(DATA_PATH, "notifications.csv"),
    fmt="csv",
    prefix="not",
    pk="id",
    date_key="ts",
    compute_layer=ComputeLayerEnum.spark,
    columns=["id", "customer_id", "ts"],
    spark_sqlctx=spark,
)

gr = GraphReduce(
    name="cust_notif_spark",
    parent_node=cust,
    compute_layer=ComputeLayerEnum.spark,
    spark_sqlctx=spark,
    cut_date=datetime.datetime(2023, 6, 30),
)
gr.add_node(cust)
gr.add_node(notif)
gr.add_entity_edge(cust, notif, parent_key="id", relation_key="customer_id", reduce=True)
gr.do_transformations()
```

</details>
