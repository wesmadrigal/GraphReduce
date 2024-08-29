# Point in time correctness

To handle point in time correctness properly all nodes 
in the graph need to share the same date parameters.

Using the example from before with `cust.csv` and `orders.csv`
let's say we want to only compute features within 6 months
and compute a label for 45 days.  

In the `GraphReduce` instance we specify `compute_period_val` and `label_period_val`.  
These parameters control how much history is included during execution.  For this
graph data from `2023/9/1` going back 180 days will be included in feature preparation
and data from `2023/9/1` going forward 45 days will be included in label preparation.

```Python
gr = GraphReduce(
    name='starter_graph',
    parent_node=gr_nodes['cust.csv'],
    fmt='csv',
    cut_date=datetime.datetime(2023,9,1),
    compute_layer=GraphReduceComputeLayerEnum.pandas,
    compute_period_val=180,
    compute_period_unit=PeriodUnit.day,
    auto_features=True,
    label_node=gr_nodes['orders.csv'],
    label_operation='count',
    label_field='id',
    label_period_val=45,
    label_period_unit=PeriodUnit.day
)
```

As you may gather, this allows us to simply change a couple of parameters
to regenerate datasets with different time periods.  
