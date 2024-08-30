# Welcome to graphreduce

GraphReduce is an abstraction layer for doing batch feature engineering spanning many tables.  The library
abstracts away much of the tedious, repetitive work such as point in time correctness, dealing with cardinality,
prefixing columns, joins, swapping between compute layers, and more.  We use graphs with [networkx](https://github.com/networkx/networkx) to represent tables as
nodes and relationships as edges, allowing most data storage formats and compute layers to be modeled. 


## Key features
* <b>Cutomizable</b>: abstractions allow feature implementations to be customized.  While many will opt for automated feature engineering, deduplication, anomalies, etc. may need custom or third-party library support.
* <b>Interoperable</b>: supports `pandas`, `pyspark`, and `dask` dataframe APIs and SQL dialects for Redshift, Athena, SQLite, Databricks SQL (Snowflake in progress).
* <b>Composable</b>: by using graphs as the underpinning data structure with `networkx` we allow arbitrarily large feature engineering pipelines by doing depth first traversal based on cardinality.
* <b>Scalable</b>: support for different computational backends and checkpointing allow for massive feature engineering graphs to be constructed and executed with a compute push down paradigm
* <b>Automated</b>: by leveraging and extending ideas from research such as [Deep Feature Synthesis](https://www.researchgate.net/publication/308808004_Deep_feature_synthesis_Towards_automating_data_science_endeavors) we support fully automated feature engineering.
* <b>Point in time correctness</b>: `graphreduce` requires time-series data to have a date key, allowing for point in time correctness filtering to be applied across all data / nodes in the compute graph.
* <b>Production-ready</b>: since batch feature engineering pipelines require a computational graph defined, this makes transition to production deployments 
* <b>Cardinality awareness</b>: in most tabular datasets cardinality needs to be handled carefully to avoid duplication and proper aggregation - `graphreduce` makes  this a breeze.

## Installation

### with pip
```bash
pip install graphreduce
```

### from source
```bash
git clone https://github.com/wesmadrigal/graphreduce
cd graphreduce && pip install -e .
```

## Basic usage

```Python
import datetime
import pandas as pd
from graphreduce.node import GraphReduceNode, DynamicNode
from graphreduce.enum import ComputeLayerEnum, PeriodUnit
from graphreduce.graph_reduce import GraphReduce

# source from a csv file with the relationships
# using the file at: https://github.com/wesmadrigal/GraphReduce/blob/master/examples/cust_graph_labels.csv
reldf = pd.read_csv('cust_graph_labels.csv')

# using the data from: https://github.com/wesmadrigal/GraphReduce/tree/master/tests/data/cust_data
files = {
    'cust.csv' : {'prefix':'cu'},
    'orders.csv':{'prefix':'ord'},
    'order_products.csv': {'prefix':'op'},
    'notifications.csv':{'prefix':'notif'},
    'notification_interactions.csv':{'prefix':'ni'},
    'notification_interaction_types.csv':{'prefix':'nit'}

}
# create graph reduce nodes
gr_nodes = {
    f.split('/')[-1]: DynamicNode(
        fpath=f,
        fmt='csv',
        pk='id',
        prefix=files[f]['prefix'],
        date_key=None,
        compute_layer=GraphReduceComputeLayerEnum.pandas,
        compute_period_val=730,
        compute_period_unit=PeriodUnit.day,
    )
    for f in files.keys()
}
gr = GraphReduce(
    name='cust_dynamic_graph',
    parent_node=gr_nodes['cust.csv'],
    fmt='csv',
    cut_date=datetime.datetime(2023,9,1),
    compute_layer=GraphReduceComputeLayerEnum.pandas,
    auto_features=True,
    auto_feature_hops_front=1,
    auto_feature_hops_back=2,
    label_node=gr_nodes['orders.csv'],
    label_operation='count',
    label_field='id',
    label_period_val=60,
    label_period_unit=PeriodUnit.day
)
# Add graph edges
for ix, row in reldf.iterrows():
    gr.add_entity_edge(
        parent_node=gr_nodes[row['to_name']],
        relation_node=gr_nodes[row['from_name']],
        parent_key=row['to_key'],
        relation_key=row['from_key'],
        reduce=True
    )


gr.do_transformations()
2024-04-23 13:49:41 [info     ] hydrating graph attributes
2024-04-23 13:49:41 [info     ] hydrating attributes for DynamicNode
2024-04-23 13:49:41 [info     ] hydrating attributes for DynamicNode
2024-04-23 13:49:41 [info     ] hydrating attributes for DynamicNode
2024-04-23 13:49:41 [info     ] hydrating attributes for DynamicNode
2024-04-23 13:49:41 [info     ] hydrating attributes for DynamicNode
2024-04-23 13:49:41 [info     ] hydrating attributes for DynamicNode
2024-04-23 13:49:41 [info     ] hydrating graph data
2024-04-23 13:49:41 [info     ] checking for prefix uniqueness
2024-04-23 13:49:41 [info     ] running filters, normalize, and annotations for <GraphReduceNode: fpath=notification_interaction_types.csv fmt=csv>
2024-04-23 13:49:41 [info     ] running filters, normalize, and annotations for <GraphReduceNode: fpath=notification_interactions.csv fmt=csv>
2024-04-23 13:49:41 [info     ] running filters, normalize, and annotations for <GraphReduceNode: fpath=notifications.csv fmt=csv>
2024-04-23 13:49:41 [info     ] running filters, normalize, and annotations for <GraphReduceNode: fpath=orders.csv fmt=csv>
2024-04-23 13:49:41 [info     ] running filters, normalize, and annotations for <GraphReduceNode: fpath=order_products.csv fmt=csv>
2024-04-23 13:49:41 [info     ] running filters, normalize, and annotations for <GraphReduceNode: fpath=cust.csv fmt=csv>
2024-04-23 13:49:41 [info     ] depth-first traversal through the graph from source: <GraphReduceNode: fpath=cust.csv fmt=csv>
2024-04-23 13:49:41 [info     ] reducing relation <GraphReduceNode: fpath=notification_interactions.csv fmt=csv>
2024-04-23 13:49:41 [info     ] performing auto_features on node <GraphReduceNode: fpath=notification_interactions.csv fmt=csv>
2024-04-23 13:49:41 [info     ] joining <GraphReduceNode: fpath=notification_interactions.csv fmt=csv> to <GraphReduceNode: fpath=notifications.csv fmt=csv>
2024-04-23 13:49:41 [info     ] reducing relation <GraphReduceNode: fpath=notifications.csv fmt=csv>
2024-04-23 13:49:41 [info     ] performing auto_features on node <GraphReduceNode: fpath=notifications.csv fmt=csv>
2024-04-23 13:49:41 [info     ] joining <GraphReduceNode: fpath=notifications.csv fmt=csv> to <GraphReduceNode: fpath=cust.csv fmt=csv>
2024-04-23 13:49:41 [info     ] reducing relation <GraphReduceNode: fpath=order_products.csv fmt=csv>
2024-04-23 13:49:41 [info     ] performing auto_features on node <GraphReduceNode: fpath=order_products.csv fmt=csv>
2024-04-23 13:49:41 [info     ] joining <GraphReduceNode: fpath=order_products.csv fmt=csv> to <GraphReduceNode: fpath=orders.csv fmt=csv>
2024-04-23 13:49:41 [info     ] reducing relation <GraphReduceNode: fpath=orders.csv fmt=csv>
2024-04-23 13:49:41 [info     ] performing auto_features on node <GraphReduceNode: fpath=orders.csv fmt=csv>
2024-04-23 13:49:41 [info     ] joining <GraphReduceNode: fpath=orders.csv fmt=csv> to <GraphReduceNode: fpath=cust.csv fmt=csv>
2024-04-23 13:49:41 [info     ] Had label node <GraphReduceNode: fpath=orders.csv fmt=csv>
2024-04-23 13:49:41 [info     ] computed labels for <GraphReduceNode: fpath=orders.csv fmt=csv>

gr.parent_node.df
cu_id	cu_name	notif_customer_id	notif_id_count	notif_customer_id_count	notif_ts_first	notif_ts_min	notif_ts_max	ni_notification_id_min	ni_notification_id_max	ni_notification_id_sum	ni_id_count_min	ni_id_count_max	ni_id_count_sum	ni_notification_id_count_min	ni_notification_id_count_max	ni_notification_id_count_sum	ni_interaction_type_id_count_min	ni_interaction_type_id_count_max	ni_interaction_type_id_count_sum	ni_ts_first_first	ni_ts_first_min	ni_ts_first_max	ni_ts_min_first	ni_ts_min_min	ni_ts_min_max	ni_ts_max_first	ni_ts_max_min	ni_ts_max_max	ord_customer_id	ord_id_count	ord_customer_id_count	ord_ts_first	ord_ts_min	ord_ts_max	op_order_id_min	op_order_id_max	op_order_id_sum	op_id_count_min	op_id_count_max	op_id_count_sum	op_order_id_count_min	op_order_id_count_max	op_order_id_count_sum	op_product_id_count_min	op_product_id_count_max	op_product_id_count_sum	ord_customer_id_dupe	ord_id_label
0	1	wes	1	6	6	2022-08-05	2022-08-05	2023-06-23	101.0	106.0	621.0	1.0	3.0	14.0	1.0	3.0	14.0	1.0	3.0	14.0	2022-08-06	2022-08-06	2023-05-15	2022-08-06	2022-08-06	2023-05-15	2022-08-08	2022-08-08	2023-05-15	1.0	2.0	2.0	2023-05-12	2023-05-12	2023-06-01	1.0	2.0	3.0	4.0	4.0	8.0	4.0	4.0	8.0	4.0	4.0	8.0	1.0	1.0
1	2	john	2	7	7	2022-09-05	2022-09-05	2023-05-22	107.0	110.0	434.0	1.0	1.0	4.0	1.0	1.0	4.0	1.0	1.0	4.0	2023-06-01	2023-06-01	2023-06-04	2023-06-01	2023-06-01	2023-06-04	2023-06-01	2023-06-01	2023-06-04	2.0	1.0	1.0	2023-01-01	2023-01-01	2023-01-01	3.0	3.0	3.0	4.0	4.0	4.0	4.0	4.0	4.0	4.0	4.0	4.0	NaN	NaN
2	3	ryan	3	2	2	2023-06-12	2023-06-12	2023-09-01	NaN	NaN	0.0	NaN	NaN	0.0	NaN	NaN	0.0	NaN	NaN	0.0	NaT	NaT	NaT	NaT	NaT	NaT	NaT	NaT	NaT	3.0	1.0	1.0	2023-06-01	2023-06-01	2023-06-01	5.0	5.0	5.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	NaN	NaN
3	4	tianji	4	2	2	2024-02-01	2024-02-01	2024-02-15	NaN	NaN	0.0	NaN	NaN	0.0	NaN	NaN	0.0	NaN	NaN	0.0
```
