# Defining node-level operations

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

## Label / target generation
If this node happens to be the node from which the label originates, we need to implement the 
