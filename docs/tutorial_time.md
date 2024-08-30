# Point in time correctness

Full code [here](https://github.com/wesmadrigal/GraphReduce/blob/master/examples/tutorial_ex2.ipynb)

To handle point in time correctness properly all nodes 
in the graph need to share the same date parameters.  Also,
all nodes must have a `date_key` set if they are to take
advantage of the date filtering provided out of the box.


## A single node date filter
A good way to test if things are working properly is instantiate
a single node and test the following functions:

* `prep_for_features()` - this filters all data prior to a `cut_date` for generating features
* `prep_for_labels()` - this filters all data after a `cut_date` for generating labels / targets

```Python
from graphreduce.node import DynamicNode
from graphreduce.enum import ComputeLayerEnum, PeriodUnit
# Only works in jupyter notebook 
!cat orders.csv

id,customer_id,ts,amount
1,1,2023-05-12,10
2,1,2023-06-01,12
3,2,2023-01-01,13
4,2,2022-08-05,150
5,3,2023-06-01,220
6,1,2023-09-02,1200
7,2,2023-10-15,47
8,4,2024-01-01,42
9,4,2024-02-01,42

order_node = DynamicNode(
    fpath='./orders.csv',
    fmt='csv',
    pk='id',
    prefix='ord',
    date_key='ts',
    compute_layer=ComputeLayerEnum.pandas,
    compute_period_val=180,
    compute_period_unit=PeriodUnit.day,
    cut_date=datetime.datetime(2023, 10, 1)
)

# loads the data
order_node.do_data()

print(len(order_node.df))
9

print(len(order_node.prep_for_features()))
4

print(len(order_node.prep_for_labels()))
1
```

In the above snippet we used a `cut_date` of October 1, 2023 and a
`compute_period_val` of 180, so we want 180 day of history prior
to October 1, 2023.  There are exactly 4 records that satisfy that
criteria, so we can see the `prep_for_features` function is working as expected.


## Putting it all together
Additionally, for the labels we see there is 1 record within 30
days of October 1, 2023 so we can see the `prep_for_labels`
function is working as expected.
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
This interface allows us to simply change a couple of parameters
to regenerate datasets with different time periods.


