# Defining node-level operations

Full code [here](https://github.com/wesmadrigal/GraphReduce/blob/master/examples/tutorial_ex3.ipynb)

Many times the automated primitives aren't enough and we want
custom aggregation, filtering, normalization, and annotation.  In these
cases we need to define operations somewhere.  Graphreduce takes the approach
of centralizing operations in the node to which they pertain.

For example, if we are defining a filter operation on the `orders.csv` feature
definition that will live in a node defined for that dataset:

```Python
from graphreduce.node import GraphReduceNode

class OrderNode(GraphReduceNode):
    def do_filters(self):
        self.df = self.df[self.df[self.colabbr('amount')] < 100000]    
```

By defining a node per dataset we can implement custom logic
and focus only on the data of interest versus line 355 of a 2000 line SQL statement.

## Full node implementation
`graphreduce` prioritizes convention over configuration, so all `GraphReduceNode` subclasses must define the 7 required abstract methods, even if they do nothing.  One of the main reasons for enforcing this is so that as feature definitions evolve the location in which a particular operation needs to go should be clear.

```Python
class OrderNode(GraphReduceNode):
    def do_filters(self):
        self.df = self.df[self.df[self.colabbr('amount')] < 100000]
    
    def do_annotate(self):
        pass

    def do_normalize(self):
        pass

    def do_reduce(self, reduce_key):
        return self.prep_for_features().groupby(self.colabbr(reduce_key)).agg(
            **{
                self.colabbr('max_amount'): pd.NamedAgg(column=self.colabbr('amount'), aggfunc='max'),
                self.colabbr('min_amount'): pd.NamedAgg(column=self.colabbr('amount'), aggfunc='min')
            }
        )

    def do_labels(self, reduce_key):
        pass

    def do_post_join_annotate(self):
        pass

    def do_post_join_filters(self):
        pass
```

## Reduce operations
If we want any aggregation to happen on this node we need to define `do_reduce`.  In this case we are computing the `min` and `max` of the column called `amount`.
There are two helper methods used in the above code snippet that deserve elaboration:

* `self.colabbr` which is `GraphReduceNode.colabbr` - this method just uses the prefix parameterized for this node so a column like `'amount'` will now be `'ord_amount'` if the prefix is `'ord'`
* `self.prep_for_features` which is `GraphReduceNode.prep_for_features` - this method filters the dataframe by the `cut_date` and `compute_period_val` if the data is time series.  If the data is not time series it just returns the full dataframe.

```Python
    # By letting the `reduce_key` be 
    # a parameter we can aggregate to
    # any arbitrary parent dimension.
    def do_reduce(self, reduce_key):
        return self.prep_for_features().groupby(self.colabbr(reduce_key)).agg(
            **{
                self.colabbr('num_orders'): pd.NamedAgg(column=self.colabbr(self.pk), aggfunc='count'),

                self.colabbr('max_amount'): pd.NamedAgg(column=self.colabbr('amount'), aggfunc='max'),
                self.colabbr('min_amount'): pd.NamedAgg(column=self.colabbr('amount'), aggfunc='min')
            }
        )
```

### Feature generation output
We can test these operations individually on an instantiated node.
Recall that the `do_data` method just loads the data.  When we
call the `do_reduce` method it will filter the dates.  We can
see that `ord_customer_id` 1 had 3 orders in the time period
and `ord_customer_id` 2 had 1 order in the time period.
```Python
order_node = OrderNode(
    fpath='orders.csv',
    prefix='ord',
    date_key='ts',
    fmt='csv',
    pk='id',
    compute_layer=ComputeLayerEnum.pandas,
    compute_period_val=180,
    compute_period_unit=PeriodUnit.day,
    cut_date=datetime.datetime(2023,10,1),
    label_period_val=30,
    label_period_unit=PeriodUnit.day
)
order_node.do_data()
print(order.do_reduce('customer_id'))

            	ord_num_orders	ord_max_amount	ord_min_amount
ord_customer_id			
              1	             3	         1200	         10
              3	             1	          220	        220


	             ord_max_amount	ord_min_amount
ord_customer_id		
              1	          1200	10
              3	          220	220
```

### Label / target generation
When not using automated feature engineering we need to specify
the label generation logic.  This can simply be selecting the
label/target column and returning it, or something more complicated
like a boolean flag of whether an event happened or another aggregation
function.  

To continue with this example, we'll be generating a label on the `orders.csv`
file for whether or not a customer had an order in the future 30 days relative
to a `cut_date` of October 1, 2023.

The definition of the `do_labels` function now becomes
```Python
class OrderNode(GraphReduceNode):
    ...
    ...

    def do_labels(self, reduce_key):
        label_df = self.prep_for_labels().groupby(self.colabbr(reduce_key)).agg(
            **{
                # We can subsequently turn this into a boolean
                self.colabbr('label_orders'): pd.NamedAgg(column=self.colabbr(self.pk), aggfunc='count')
            }
        )
        label_df[self.colabbr('label_had_order')] = label_df[self.colabbr('label_orders')].apply(lambda x: 1 if x > 0 else 0)
        return label_df

    ...
    ...
```

Now we can test the `do_labels` method with an instance:
```Python
print(order_node.do_labels('customer_id'))

             	ord_label_orders	ord_label_had_order
ord_customer_id		
              2	           1	               1
```
