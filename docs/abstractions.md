# Abstractions

## Node abstraction
We represent files and tables as nodes.  A node could be a csv file
on your laptop, a parquet file in s3, or a Snowflake table in the cloud.

We parameterize the data location, a string prefix so we know where the data originates,
a primary key, a date key (if any), a compute layer (e.g., `pandas`, `dask`), and
some other optional parameters to intantiate these.

### GraphReduceNode
The base `GraphReduceNode` requires the following abstract methods be defined

* `do_filters` - all filter operations for this node go here (e.g., `df.filter...`)

* `do_annotate` - all annotations go here (e.g., `df['zip'] = df.zipfull.apply(lambda x: x.split('-')[0])`)

* `do_post_join_annotate` - annotations that require data from a child be joined in (e.g., need a delta between dates from two tables)

* `do_normalize` - all anomaly filtering, data normalization, etc. go here (e.g., `df['val_norm'] = df['val'].apply(lambda x: x/df['val'].max())`)

* `do_post_join_filters` - all filters requiring data from more than 1 table go here

* `do_reduce` - all aggregation operations for features (e.g., `df.groupby(key).agg(...)`)

* `do_labels` - any label-specific aggregation operations (e.g., `df.groupby(key).agg(had_order = 1)`)


### DynamicNode
A dynamic node is any node that is instantiated without defined methods.  These are typically used for doing automated feature engineering.

### SQLNode
A SQL node is an abstraction for SQL dialects and backends.  This allows us to
go beyond the dataframe API that a typical `GraphReduceNode` or `DynamicNode` 
is built for and leverage a number of SQL backends.  There is more detail
about how to use these in [the SQL backends tutorial](tutorial_sql_dialects.md).


## Edge
An edge is a relationship between two nodes.  This is typically a foreign key.  For 
example if we had a `customers` table and `orders` table we would add an edge between the 
`customers` node and the `orders` node:

```Python
gr.add_entity_edge(
    parent_node=customer_node,
    relation_node=orders_node,
    parent_key='id',
    relation_key='customer_id',
    reduce=True
```

The `reduce` parameter tells `graphreduce` whether or not to execute 
aggregation operations.


## GraphReduce Graph
The top-level `GraphReduce` class inherits directly from `networkx.DiGraph`
to take advantage of many graph algorithms implemented in `networkx`.  The instances
house to shared parameters for the entire graph of computation across all nodes and edges.

Things such as the node to which to aggregate the data, the date for splitting the data, the compute layer (e.g., `pandas`, `dask`), the amount of history to include (365 days), the label period, whether or not to automate feature engineering, the label/target node and label/target column, etc.  All of these parameters
get pushed down through the graph so we can do things like point in time correctness, etc.

Since we inherit from `networkx` the API for adding nodes is unchanged:
```Python
import datetime
from graphreduce.graph_reduce import GraphReduce
from grpahreduce.enum import PeriodUnit, ComputeLayerEnum

gr = GraphReduce(
    name='test',
    parent_node=customer_node,
    fmt='parquet',
    compute_layer=ComputeLayerEnum.spark,
    cut_date=datetime.datetime(2024, 7, 1),
    compute_period_val=365,
    compute_period_unit=PeriodUnit.day
    auto_features=False,
    label_node=order_node,
    label_operation='sum',
    label_field='order_total',
    label_period_val=60,
    label_period_unit=PeriodUnit.day,
    spark_sqlctx=sqlCtx
)

gr.add_node(customer_node)
gr.add_node(order_node)
gr.add_node(notification_node)
gr.add_node(...)
...
```
