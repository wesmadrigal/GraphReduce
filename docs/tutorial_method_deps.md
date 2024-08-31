# Post-join method execution
There is a method decorator for specifying that
a node instance method can only be executed
once other nodes have been joined to the node.

For example, using the `cust.csv` and `orders.csv`
datasets from prior examples, let's say we want
to compute the difference between an order date
and the signup date of the customer.  In this
case we need the timestamp of the order and the
timestamp of the customer.  

```Python
# In the node definition of the customer
# specify this dependency
from graphreduce.node import GraphReduceNode
from graphreduce.context import method_requires

class CustomerNode(GraphReduceNode):
    def do_filters(self):
        pass
    ...
    ...
    
    @method_requires(nodes=[OrderNode])
    def do_post_join_annotate(self):
        self.df[self.colabbr('signup_order_diff_seconds')] = self.df.apply(
            lambda x: (x['ord_ts'] - x[self.colabbr('signup_date')]).total_seconds(),
            axis=1
        )
```

By using the `method_requires` decorator we've told graphreduce
that the `CustomerNode.do_post_join_annotate` method can only
be executed once the `OrderNode` has been joined.  

There are cases where we need multiple child nodes' data merged
to a parent node before certain operations can be executed, such
as annotations and filters.  

This would look like the following snippet:
```Python

class ParentNode(GraphReduceNode):
    ...

    @method_requies(nodes=[ChildNodeOne, ChildNodeThree])
    def do_post_join_filters(self):
        self.df[self.colabbr('newcol')] = self.df.apply(lambda x: f"{x['col1']}-{x['child_col']}-{x['child3_col']}", axis=1)

    
    @method_requires(nodes=[ChildNodeOne, ChildNodeTwo, ChildNodeThree])
    def do_post_join_filters(self):
        self.df = self.df[..]

```
