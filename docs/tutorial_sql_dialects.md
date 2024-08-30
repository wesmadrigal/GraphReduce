# SQL backends and dialects

To use SQL dialects we need to use `SQLNode` instances
or a subclass of `SQLNode`.  The following are implemented
and available for use out of the box:

* Databricks SQL
* SQLite
* Redshift
* AWs Athena
* Snowflake (in progress)


## SQL operations
We provide an abstraction for defining individual
SQL operations so that they become stackable.  Graphreduce
compiles all of the SQL operations at runtime and pushes
down to the provider.  

The parent class `SQLNode` defines a method `build_query`,
which takes an arbitrary number of `sqlop` instances
and builds the query.  This abstraction allows for individual
select statements, where clauses, group bys, etc. to be defined
individually and at runtime `build_query` compiles them into
an executable statement.

An example `sqlop` would be a simple select:
```Python
from graphreduce.models import sqlop
from graphreduce.enum import SQLOpType

sel = sqlop(optype=SQLOpType, opval="*")
```

This select just says to `select *`.  We can stack a number
of these and have `build_query` compile them for us.  We will
continue with the `cust.csv` data but now it is stored in `sqlite`.

```Python
cust = SQLNode(fpath='cust',
                prefix='cust',
                client=conn, 
                compute_layer=ComputeLayerEnum.sqlite, 
                columns=['id','name'])

print(cust.build_query(
    ops=[
        sqlop(optype=SQLOpType.select, opval='id'),
        sqlop(optype=SQLOpType.select, opval='name')
    ]
))

SELECT id,name
    FROM cust
WHERE true
        
```

Since we parameterized the `SQLNode` instance with the `cust`
table, our instance already knows which table to select from
during every `sqlop` instance.  The `sqlop` tries to be the
smallest unit of operation in SQL, allowing for stacking
as many of them as you want.

You can also chain the graphreduce methods and dynamically build up
SQL like this:

```Python
# Define the order node
class OrderNode(SQLNode):
    def do_filters(self) -> typing.List[sqlop]:
        return [
            sqlop(optype=SQLOpType.where, opval=f"{self.colabbr(self.date_key)} > '2022-12-01'")
        ]
    
    def do_annotate(self) -> typing.List[sqlop]:
        pass
    
    def do_normalize(self):
        pass
    
    def do_reduce(self, reduce_key):
        return [
            # Shouldn't this just be a select?
            sqlop(optype=SQLOpType.aggfunc, opval=f"count(*) as {self.colabbr('num_orders')}"),
            sqlop(optype=SQLOpType.agg, opval=f"{self.colabbr(reduce_key)}")
        ]
    
    
    def do_labels(self, reduce_key):
        return [
            sqlop(optype=SQLOpType.aggfunc, opval=f"count(*) as {self.colabbr('num_orders_label')}"),
            sqlop(optype=SQLOpType.agg, opval=f"{self.colabbr(reduce_key)}")
        ]

# Instantiate
order = OrderNode(
    fpath='orders',
    prefix='ord',
    client=conn,
    compute_layer=ComputeLayerEnum.sqlite,
    columns=['id','customer_id','ts','amount'],
    date_key='ts'
)

# build a query
print(order.build_query(
    ops=order.do_filters() + order.do_reduce('customer_id')
))

SELECT ord_customer_id,
        count(*) as ord_num_orders
        FROM orders
        WHERE ord_ts > '2022-12-01'
        GROUP BY ord_customer_id
```

## More examples
There are more examples on [github](https://github.com/wesmadrigal/GraphReduce)

* [example 1](https://github.com/wesmadrigal/GraphReduce/blob/master/examples/sql_dialects_ex1.ipynb)
* [example 2](https://github.com/wesmadrigal/GraphReduce/blob/master/examples/sql_dialects_ex2.ipynb)
* [example 3](https://github.com/wesmadrigal/GraphReduce/blob/master/examples/sql_dialects_ex3.ipynb)
* [example 4 automated feature engineering](https://github.com/wesmadrigal/GraphReduce/blob/master/examples/sql_dialects_ex4.ipynb)
* [example 5 automated feature engineering con't](https://github.com/wesmadrigal/GraphReduce/blob/master/examples/sql_dialects_ex4.ipynb)
