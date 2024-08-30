# Swapping compute layers
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
        compute_layer=ComputeLayerEnum.pandas,
        compute_period_val=730,
        compute_period_unit=PeriodUnit.day,
    )
    for f in prefixes.keys()
}

gr_nodes['cust.csv'].do_data()
type(gr_nodes['cust.csv'].df)
pandas.core.frame.DataFrame
```


## Dask
Now we can instantiate the same nodes with `dask`:
```Python
# create graph reduce nodes
gr_nodes = {
    f.split('/')[-1]: DynamicNode(
        fpath=f,
        fmt='csv',
        pk='id',
        prefix=prefixes[f]['prefix'],
        date_key=None,
        compute_layer=ComputeLayerEnum.dask,
        compute_period_val=730,
        compute_period_unit=PeriodUnit.day,
    )
    for f in prefixes.keys()
}

gr_nodes['cust.csv'].do_data()
type(gr_nodes['cust.csv'].df)
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

To use SQL dialect we need to use the `SQLNode` class
and it's subclasses.
