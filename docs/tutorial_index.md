# Quickstart

The most simple graph is two nodes with no time component, no aggregations,
and no labels.  We are using customer (`cust.csv`) and order data (`orders.csv`).  
To see the full code go here [link](https://github.com/wesmadrigal/graphreduce/tree/master/examples/tutorial_ex1.ipynb)

In plain terms what the below code does is as follows:

1. Create prefixes for each node so we always know where the column originated from after joinining the two datasets.
2. Instantiate two `DynamicNode` instances: one node for `cust.csv` and another for `orders.csv`.
3. Instantiate the `GraphReduce` object to house the compute graph.  We are specifying that the `cust.csv` node is the `parent_node`, which means all data will be joined to and aggregated to the `cust.csv` node.  In cases where we reduce the data the resulting dataset should be at the ganularity of the `parent_node` dimension.
4. Add the nodes.
5. Add the edge between the nodes.
6. Execute the computation with `GraphReduce.do_transformations()` the primary entrypoint to execution.
7. Dump out the head of the computed dataframe. 

```Python
from graphreduce.node import GraphReduceNode, DynamicNode
from graphreduce.graph_reduce import GraphReduce
from graphreduce.enum import ComputeLayerEnum as GraphReduceComputeLayerEnum, PeriodUnit

# Need unique prefixes for all nodes
# so when columns are merged we know
# where they originate from.
prefixes = {
    'cust.csv' : {'prefix':'cu'},
    'orders.csv':{'prefix':'ord'}
}

# create graph reduce nodes
gr_nodes = {
    f.split('/')[-1]: DynamicNode(
        fpath=f,
        fmt='csv',
        pk='id',
        prefix=prefixes[f]['prefix'],
        date_key=None,
        compute_layer=GraphReduceComputeLayerEnum.pandas,
        compute_period_val=730,
        compute_period_unit=PeriodUnit.day,
    )
    for f in files.keys()
}

# Instantiate GraphReduce with params.
# We are using 'cust.csv' as parent node
# so the granularity should be at the customer
# dimension.
gr = GraphReduce(
    name='starter_graph',
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

# Add the nodes to the GraphReduce instance.
gr.add_node(gr_nodes['cust.csv'])
gr.add_node(gr_nodes['orders.csv'])

gr.add_entity_edge(
    parent_node=gr_nodes['cust.csv'],
    relation_node=gr_nodes['orders.csv'],
    parent_key='id',
    relation_key='customer_id',
    reduce=True
)

gr.do_transformations()

gr.parent_node.df.head()
	cu_id	cu_name	ord_customer_id	ord_id_count	ord_customer_id_count	ord_ts_min	ord_ts_max	ord_amount_count	ord_customer_id_dupe	ord_id_label
0	1	wes	1	3	3	2023-05-12	2023-09-02	3	1	3
1	2	ana	2	3	3	2022-08-05	2023-10-15	3	2	3
2	3	caleb	3	1	1	2023-06-01	2023-06-01	1	3	1
3	4	luly	4	2	2	2024-01-01	2024-02-01	2	4	2

```
