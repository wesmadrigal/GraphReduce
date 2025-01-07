# Swapping compute layers

Full code [here](https://github.com/wesmadrigal/GraphReduce/blob/master/examples/tutorial_ex4.ipynb)

There are API differences between a lot of compute layers
but between `pandas` and `dask` the API is mostly the same.  This
makes swapping between these two compute layers a breeze.

Let's say we're using the `orders.csv` and `cust.csv`
from prior examples and using automated feature engineering
with `DynamicNode` instances.


## Pandas
We'll use the same `orders.csv` and `cust.csv` datasets
and start with a `pandas` backend, which is specified
with the `compute_layer` parameter.

```
cust_node = DynamicNode(
    fpath='cust.csv',
    fmt='csv',
    pk='id',
    prefix='cu',
    date_key=None,
    compute_layer=ComputeLayerEnum.pandas
)

cust_node.do_data()
type(cust_node.df)
pandas.core.frame.DataFrame
```


## Dask
Now we can instantiate the same nodes with `dask`:
```Python
cust_node = DynamicNode(
    fpath='cust.csv',
    fmt='csv',
    pk='id',
    prefix='cu',
    date_key=None,
    compute_layer=ComputeLayerEnum.dask
)

cust_node.do_data()
type(cust_node.df)
dask.dataframe.core.DataFrame
```

## Spark
For `spark` you will need access to a SparkContext
and can instantiate as follows:
```Python
cloud_node = DynamicNode(
    fpath='s3://mybucket/path/to/file.parquet',
    fmt='parquet',
    pk='id',
    prefix='fi',
    date_key='updated_at',
    compute_layer=ComputeLayerEnum.spark,
    compute_period_val=365,
    compute_period_unit=PeriodUnit.day,
    spark_sqlctx=sqlCtx
)
cloud_node.do_data()
type(cloud_node.df)
pyspark.sql.dataframe.DataFrame
```

## Daft
For `daft` you will just need to specify
the compute layer.
```Python
daft_node = DynamicNode(
   fpath='cusv.csv',
   fmt='csv',
   pk='id',
   prefix='cu',
   date_key=None,
   compute_layer=ComputeLayerEnum.daft
)

cust_node.do_data()
type(cust_node.df)
daft.dataframe.dataframe.DataFrame
```

## SQL compute engines
To use SQL dialect we need to use the `SQLNode` class
and it's subclasses.  These are instantiated as follows:
```Python
from graphreduce.node import SQLNode

cust_node = SQLNode(
    fpath='schema.customers',
    fmt='sql',
    pk='id',
    prefix='cu',
    date_key='signup_date',
    compute_layer=ComputeLayerEnum.sql
)
```
