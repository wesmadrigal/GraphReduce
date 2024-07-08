#!/usr/bin/env python

# std lib
import abc
import datetime
import typing
import json
import enum
import time

# third party
import pandas as pd
from dask import dataframe as dd
import pyspark
from structlog import get_logger
from dateutil.parser import parse as date_parse

# internal
from graphreduce.enum import ComputeLayerEnum, PeriodUnit, SQLOpType
from graphreduce.storage import StorageClient
from graphreduce.models import sqlop


logger = get_logger('Node')


class GraphReduceNode(metaclass=abc.ABCMeta): 
    """
Base node class, which can be used directly
or subclassed for further customization.

Many helpful methods are implemented and can
be used as is, but for different engines
and dialects (e.g., SQL vs. python) it can
be necessary to implement an engine-specific
methods (e.g., `do_data` to get data from Snowflake)

The classes `do_annotate`, `do_filters`,
`do_normalize`, `do_reduce`, `do_labels`,
`do_post_join_annotate`, and `do_post_join_filters`
are abstractmethods which must be defined.
    """
    fpath : str
    fmt : str
    pk : str 
    prefix : str
    date_key : str
    compute_layer : ComputeLayerEnum
    cut_date : typing.Optional[datetime.datetime]
    compute_period_val : typing.Union[int, float]
    compute_period_unit : PeriodUnit
    reduce: bool
    label_period_val : typing.Optional[typing.Union[int, float]]
    label_period_unit : typing.Optional[PeriodUnit] 
    label_field: typing.Optional[str]
    spark_sqlctx : typing.Optional[pyspark.sql.SQLContext]
    columns: typing.List
    storage_client: typing.Optional[StorageClient]
    # Only for SQL dialects at the moment.
    lazy_execution: bool

    def __init__ (
            self,
            # IF is SQL dialect this should be a table name.
            fpath : str = '',
            # If is SQL dialect "sql" is fine here.
            fmt : str = '',
            pk : str = None,
            prefix : str = None,
            date_key : str = None,
            compute_layer : ComputeLayerEnum = None,
            cut_date : datetime.datetime = datetime.datetime.now(),
            compute_period_val : typing.Union[int, float] = 365,
            compute_period_unit : PeriodUnit  = PeriodUnit.day,
            reduce : bool = True,
            label_period_val : typing.Optional[typing.Union[int, float]] = None,
            label_period_unit : typing.Optional[PeriodUnit] = None,
            label_field : typing.Optional[str] = None,
            spark_sqlctx : pyspark.sql.SQLContext = None,
            columns : list = [],
            storage_client: typing.Optional[StorageClient] = None,
            checkpoints: list = [],
            # Only for SQL dialects at the moment.
            lazy_execution: bool = False,
            ):
        """
Constructor
        """
        # For when this is already set on the class definition.
        if not hasattr(self, 'pk'):
            self.pk = pk
        # For when this is already set on the class definition.
        if not hasattr(self, 'prefix'):
            self.prefix = prefix
        # For when this is already set on the class definition.
        if not hasattr(self, 'date_key'):
            self.date_key = date_key
        self.fpath = fpath
        self.fmt = fmt
        self.compute_layer = compute_layer
        self.cut_date = cut_date
        self.compute_period_val = compute_period_val
        self.compute_period_unit = compute_period_unit
        self.reduce = reduce
        self.label_period_val = label_period_val
        self.label_period_unit = label_period_unit
        self.label_field = label_field
        self.spark_sqlctx = spark_sqlctx
        self.columns = columns

        # Lazy execution for the SQL nodes.
        self._lazy_execution = lazy_execution
        self._storage_client = storage_client
        # List of merged neighbor classes.
        self._merged = []
        # List of checkpoints.

        # Logical types of the original columns from `woodwork`.
        self._logical_types = {}

        if not self.date_key:
            logger.warning(f"no `date_key` set for {self}")


    def __repr__ (
            self
            ):
        """
Instance representation
        """
        return f"<GraphReduceNode: fpath={self.fpath} fmt={self.fmt}>"

    def __str__ (
            self
            ):
        """
Instances string
        """
        return f"<GraphReduceNode: fpath={self.fpath} fmt={self.fmt}>"


    def reload (
            self
            ):
        """
Refresh the node.
        """
        self._merged = []
        self._checkpoints = []
        self.df = None
        self._logical_types = {}

    
    def do_data (
        self
    ) -> typing.Union[
        pd.DataFrame,
        dd.DataFrame,
        pyspark.sql.dataframe.DataFrame
    ]:
        """
Get some data
        """

        if self.compute_layer.value == 'pandas':
            if not hasattr(self, 'df') or (hasattr(self,'df') and not isinstance(self.df, pd.DataFrame)):
                self.df = getattr(pd, f"read_{self.fmt}")(self.fpath)

                # Initialize woodwork.
                self.df.ww.init()
                self._logical_types = self.df.ww.logical_types

                # Rename columns with prefixes.
                if len(self.columns):
                    self.df = self.df[[c for c in self.columns]]
                self.columns = list(self.df.columns)
                self.df.columns = [f"{self.prefix}_{c}" for c in self.df.columns]
        elif self.compute_layer.value == 'dask':
            if not hasattr(self, 'df') or (hasattr(self, 'df') and not isinstance(self.df, dd.DataFrame
)):
                self.df = getattr(dd, f"read_{self.fmt}")(self.fpath)

                # Initialize woodwork.
                self.df.ww.init()
                self._logical_types = self.df.ww.logical_types

                # Rename columns with prefixes.
                if len(self.columns):
                    self.df = self.df[[c for c in self.columns]]
                self.columns = list(self.df.columns)
                self.df.columns = [f"{self.prefix}_{c}" for c in self.df.columns]
        elif self.compute_layer.value == 'spark':
            if not hasattr(self, 'df') or (hasattr(self, 'df') and not isinstance(self.df, pyspark.sql.DataFrame)):
                self.df = getattr(self.spark_sqlctx.read, f"{self.fmt}")(self.fpath)
                if self.columns:
                    self.df = self.df.select(self.columns)
                for c in self.df.columns:
                    self.df = self.df.withColumnRenamed(c, f"{self.prefix}_{c}")

        # at this point of connectors we may want to try integrating
        # with something like fugue: https://github.com/fugue-project/fugue
        elif self.compute_layer.value == 'ray':
            pass
 
        elif self.compute_layer.value == 'snowflake':
            pass

        elif self.compute_layer.value == 'postgres':
            pass

        elif self.compute_layer.value == 'redshift':
            pass


    @abc.abstractmethod
    def do_filters (
        self
    ):
        """
do some filters on the data
        """
        pass
    

    @abc.abstractmethod
    def do_annotate(self):
        """
Implement custom annotation functionality
for annotating this particular data
        """
        pass

    
    @abc.abstractmethod
    def do_post_join_annotate(self):
        """
Implement custom annotation functionality
for annotating data after joining with 
child data
        """
        pass


    @abc.abstractmethod
    def do_normalize(self):
        pass


    def do_post_join_filters(self):
        """
Filter operations that require some
additional relational data to perform.
        """
        pass


    def auto_features (
            self,
            reduce_key : str,
            type_func_map : dict = {},
            compute_layer : ComputeLayerEnum = ComputeLayerEnum.pandas,
            ):
        """
If we're doing automatic features
this function will run a series of
automatic aggregations.  The top-level
`GraphReduce` object will handle joining
the results together.
        """
        if compute_layer == ComputeLayerEnum.pandas:
            return self.pandas_auto_features(reduce_key=reduce_key, type_func_map=type_func_map)
        elif compute_layer == ComputeLayerEnum.dask:
            return self.dask_auto_features(reduce_key=reduce_key, type_func_map=type_func_map)
        elif compute_layer == ComputeLayerEnum.spark:
            return self.spark_auto_features(reduce_key=reduce_key, type_func_map=type_func_map)
        elif self.compute_layer in [ComputeLayerEnum.snowflake, ComputeLayerEnum.sqlite, ComputeLayerEnum.mysql, ComputeLayerEnum.postgres, ComputeLayerEnum.redshift, ComputeLayerEnum.databricks]:
            # Assumes `SQLNode.get_sample` is implemented to get
            # a sample of the data in pandas dataframe form.
            sample_df = self.get_sample()
            return self.sql_auto_features(sample_df, reduce_key=reduce_key, type_func_map=type_func_map)


    def auto_labels (
            self,
            reduce_key : str,
            type_func_map : dict = {},
            compute_layer : ComputeLayerEnum = ComputeLayerEnum.pandas,
            ):
        """
If we're doing automatic features
this function will run a series of
automatic aggregations.  The top-level
`GraphReduce` object will handle joining
the results together.
        """
        if compute_layer == ComputeLayerEnum.pandas:
            return self.pandas_auto_labels(reduce_key=reduce_key, type_func_map=type_func_map)
        elif compute_layer == ComputeLayerEnum.dask:
            return self.dask_auto_labels(reduce_key=reduce_key, type_func_map=type_func_map)
        elif compute_layer == ComputeLayerEnum.spark:
            return self.spark_auto_labels(reduce_key=reduce_key, type_func_map=type_func_map)


    def pandas_auto_features (
            self,
            reduce_key : str,
            type_func_map : dict = {}
            ) -> pd.DataFrame:
        """
Pandas implementation of dynamic propagation of features.
This is basically automated feature engineering but suffixed
with `_propagation` to indicate that we are propagating data
upward through the graph from child nodes with no feature
definitions.
        """
        agg_funcs = {}
        for col, _type in dict(self.df.dtypes).items():
            _type = str(_type)
            if type_func_map.get(_type):
                for func in type_func_map[_type]:
                    col_new = f"{col}_{func}"
                    agg_funcs[col_new] = pd.NamedAgg(column=col, aggfunc=func)
        return self.prep_for_features().groupby(self.colabbr(reduce_key)).agg(
                **agg_funcs
                ).reset_index()


    def dask_auto_features (
            self,
            reduce_key : str,
            type_func_map : dict = {},
            ) -> dd.DataFrame:
        """
Dask implementation of dynamic propagation of features.
This is basically automated feature engineering but suffixed
with `_propagation` to indicate that we are propagating data
upward through the graph from child nodes with no feature
definitions.
        """
        agg_funcs = {}
        for col, _type in dict(self.df.dtypes).items():
            _type = str(_type)
            if type_func_map.get(_type):
                for func in type_func_map[_type]:
                    col_new = f"{col}_{func}"
                    agg_funcs[col_new] = pd.NamedAgg(column=col, aggfunc=func)
        return self.prep_for_features().groupby(self.colabbr(reduce_key)).agg(
                **agg_funcs
                ).reset_index()


    def spark_auto_features (
            self,
            reduce_key : str,
            type_func_map : dict = {},
            ) -> pyspark.sql.DataFrame:
        """
Spark implementation of dynamic propagation of features.
This is basically automated feature engineering but suffixed
with `_propagation` to indicate that we are propagating data
upward through the graph from child nodes with no feature
definitions.
        """
        agg_funcs = []
        for field in self.df.schema.fields:
            field_meta = json.loads(field.json())
            col = field_meta['name']
            _type = field_meta['type']
            if type_func_map.get(_type):
                for func in type_func_map[_type]:
                    col_new = f"{col}_{func}"
                    agg_funcs.append(getattr(F, func)(F.col(col)).alias(col_new))
        return self.prep_for_features().groupby(self.colabbr(reduce_key)).agg(
                *agg_funcs
                )


    def sql_auto_features (
            self,
            table_df_sample: typing.Union[pd.DataFrame, dd.DataFrame],
            reduce_key: str,
            type_func_map: dict = {},
            ) -> typing.List[sqlop]:
        """
SQL dialect implementation of automated
feature engineering.

At the moment we're just using `pandas` inferred
data types for these operations.  This is an
area that can benefit from `woodwork` and other
data type inference libraries.
        """
        agg_funcs = []
        for col, _type in dict(table_df_sample.dtypes).items():
            _type = str(_type)
            if type_func_map.get(_type):
                for func in type_func_map[_type]:
                    col_new = f"{col}_{func}"
                    agg_funcs.append(
                            sqlop(
                                optype=SQLOpType.aggfunc,
                                opval=f"{func}" + f"({col}) as {col_new}"
                                )
                            )
        # Need the aggregation and time-based filtering.
        agg = sqlop(optype=SQLOpType.agg, opval=f"{self.colabbr(reduce_key)}")

        tfilt = self.prep_for_features() if self.prep_for_features() else []

        return tfilt + agg_funcs + [agg]


    def sql_auto_labels (
            self,
            table_df_sample: typing.Union[pd.DataFrame, dd.DataFrame],
            reduce_key : str,
            type_func_map : dict = {}
            ) -> pd.DataFrame:
        """
Pandas implementation of auto labeling based on
provided columns.
        """
        agg_funcs = {}
        for col, _type in dict(table_df_sample.dtypes).items():
            if col.endswith('_label'):
                _type = str(_type)
                if type_func_map.get(_type):
                    for func in type_func_map[_type]:
                        col_new = f"{col}_{func}_label"
                        agg_funcs.append(
                                sqlop(
                                    optype=SQLOpType.aggfunc,
                                    opval=f"{func}" + f"({col}) as {col_new}"
                                    )
                                )
        # Need the aggregation and time-based filtering.
        agg = sqlop(optype=SQLOpType.agg, opval=f"{self.colabbr(reduce_key)}")
        tfilt = self.prep_for_labels() if self.prep_for_labels() else []
        return tfilt + agg_funcs + [agg]


    def pandas_auto_labels (
            self,
            reduce_key : str,
            type_func_map : dict = {}
            ) -> pd.DataFrame:
        """
Pandas implementation of auto labeling based on
provided columns.
        """
        agg_funcs = {}
        for col, _type in dict(self.df.dtypes).items():
            if col.endswith('_label'):
                _type = str(_type)
                if type_func_map.get(_type):
                    for func in type_func_map[_type]:
                        col_new = f"{col}_{func}_label"
                        agg_funcs[col_new] = pd.NamedAgg(column=col, aggfunc=func)
        return self.prep_for_labels().groupby(self.colabbr(reduce_key)).agg(
                **agg_funcs
                ).reset_index()


    def dask_auto_labels (
            self,
            reduce_key : str,
            type_func_map : dict = {},
            ) -> dd.DataFrame:
        """
Dask implementation of auto labeling based on
provided columns.
        """
        agg_funcs = {}
        for col, _type in dict(self.df.dtypes).items():
            if col.endswith('_label'):
                _type = str(_type)
                if type_func_map.get(_type):
                    for func in type_func_map[_type]:
                        col_new = f"{col}_{func}_label"
                        agg_funcs[col_new] = pd.NamedAgg(column=col, aggfunc=func)
        return self.prep_for_labels().groupby(self.colabbr(reduce_key)).agg(
                **agg_funcs
                ).reset_index()


    def spark_auto_labels (
            self,
            reduce_key : str,
            type_func_map : dict = {},
            ) -> pyspark.sql.DataFrame:
        """
Spark implementation of auto labeling based on
provided columns.
        """
        agg_funcs = []
        for field in self.df.schema.fields:
            field_meta = json.loads(field.json())
            col = field_meta['name']
            _type = field_meta['type']
            if col.endswith('_label'):
                if type_func_map.get(_type):
                    for func in type_func_map[_type]:
                        col_new = f"{col}_{func}_label"
                        agg_funcs.append(getattr(F, func)(F.col(col)).alias(col_new))
        return self.prep_for_labels().groupby(self.colabbr(reduce_key)).agg(
                *agg_funcs
                )


    @abc.abstractmethod
    def do_reduce (
            self, 
            reduce_key
            ):
        """
Reduce operation or the node

Args
    reduce_key : key to use to perform the reduce operation
    children : list of children nodes
        """
        pass
    
    
    @abc.abstractmethod
    def do_labels (
            self,
            reduce_key: typing.Optional[str] = None,
            ):
        """
Generate labels
        """
        pass
    
        
    def colabbr(self, col: str) -> str:
        return f"{self.prefix}_{col}"


    def compute_period_minutes (
            self,
            ) -> int:
        """
Convert the compute period to minutes
        """
        if self.compute_period_unit == PeriodUnit.second:
            return self.compute_period_val / 60
        elif self.compute_period_unit == PeriodUnit.minute:
            return self.compute_period_val
        elif self.compute_period_unit == PeriodUnit.hour:
            return self.compute_period_val * 60
        elif self.compute_period_unit == PeriodUnit.day:
            return self.compute_period_val * 1440
        elif self.compute_period_unit == PeriodUnit.week:
            return (self.compute_period_val * 7)*1440
        elif self.compute_period_unit == PeriodUnit.month:
            return (self.compute_period_val * 30.417)*1440


    def label_period_minutes (
            self,
            ) -> int:
        """
Convert the label period to minutes
        """
        if self.label_period_unit == PeriodUnit.second:
            return self.label_period_val / 60
        elif self.label_period_unit == PeriodUnit.minute:
            return self.label_period_val
        elif self.label_period_unit == PeriodUnit.hour:
            return self.label_period_val * 60
        elif self.label_period_unit == PeriodUnit.day:
            return self.label_period_val * 1440
        elif self.label_period_unit == PeriodUnit.week:
            return (self.label_period_val * 7)*1440
        elif self.label_period_unit == PeriodUnit.month:
            return (self.label_period_val * 30.417)*1440
    
    
    def prep_for_features (
            self,
            allow_null: bool = False
            ) -> typing.Union[pd.DataFrame, dd.DataFrame, pyspark.sql.dataframe.DataFrame, typing.List[sqlop]]:
        """
Prepare the dataset for feature aggregations / reduce
        """
        if self.date_key:           
            if self.cut_date and isinstance(self.cut_date, str) or isinstance(self.cut_date, datetime.datetime):
                # Using a SQL engine so need to return `sqlop` instances.
                if self.compute_layer in [ComputeLayerEnum.sqlite, ComputeLayerEnum.postgres, ComputeLayerEnum.snowflake, ComputeLayerEnum.redshift, ComputeLayerEnum.mysql, ComputeLayerEnum.athena, ComputeLayerEnum.databricks]:
                    return [
                            sqlop(optype=SQLOpType.where, opval=f"{self.colabbr(self.date_key)} < '{str(self.cut_date)}'"),
                            sqlop(optype=SQLOpType.where, opval=f"{self.colabbr(self.date_key)} > '{str(self.cut_date - datetime.timedelta(minutes=self.compute_period_minutes()))}'")
                            ]

                elif isinstance(self.df, pd.DataFrame) or isinstance(self.df, dd.DataFrame):
                    return self.df[
                        (
                            (self.df[self.colabbr(self.date_key)] < str(self.cut_date))
                            &
                            (self.df[self.colabbr(self.date_key)] > str(self.cut_date - datetime.timedelta(minutes=self.compute_period_minutes())))
                        )
                        |
                        (self.df[self.colabbr(self.date_key)].isnull())
                    ]
                elif isinstance(self.df, pyspark.sql.dataframe.DataFrame):
                    return self.df.filter(
                        (
                            (self.df[self.colabbr(self.date_key)] < self.cut_date)
                            &
                            (self.df[self.colabbr(self.date_key)] > (self.cut_date - datetime.timedelta(minutes=self.compute_period_minutes())))
                        )
                        |
                        (self.df[self.colabbr(self.date_key)].isNull())
                    )

            else:
                # Using a SQL engine so need to return `sqlop` instances.
                if self.compute_layer in [ComputeLayerEnum.sqlite, ComputeLayerEnum.postgres, ComputeLayerEnum.snowflake, ComputeLayerEnum.redshift, ComputeLayerEnum.mysql, ComputeLayerEnum.athena, ComputeLayerEnum.databricks]:
                    return [
                            sqlop(optype=SQLOpType.where, opval=f"{self.colabbr(self.date_key)} < '{str(datetime.datetime.now())}'"),
                            sqlop(optype=SQLOpType.where, opval=f"{self.colabbr(self.date_key)} > '{str(datetime.datetime.now() - datetime.timedelta(minutes=self.compute_period_minutes()))}'")
                            ]

                elif isinstance(self.df, pd.DataFrame) or isinstance(self.df, dd.DataFrame):
                    return self.df[
                        (
                            (self.df[self.colabbr(self.date_key)] < datetime.datetime.now())
                            &
                            (self.df[self.colabbr(self.date_key)] > (datetime.datetime.now() - datetime.timedelta(minutes=self.compute_period_minutes())))
                        )
                        |
                        (self.df[self.colabbr(self.date_key)].isnull())
                    ]
                elif isinstance(self.df, pyspark.sql.dataframe.DataFrame):
                    return self.df.filter(
                            (self.df[self.colabbr(self.date_key)] > (datetime.datetime.now() - datetime.timedelta(minutes=self.compute_period_minutes())))
                            |
                            (self.df[self.colabbr(self.date_key)].isNull())
                )

        # SQL engine.
        elif not hasattr(self, 'df'):
            return None
        # no-op
        return self.df
    
    
    def prep_for_labels (
            self
            ) -> typing.Union[pd.DataFrame, dd.DataFrame, pyspark.sql.dataframe.DataFrame]:
        """
Prepare the dataset for labels
        """
        if self.date_key:
            if self.cut_date and isinstance(self.cut_date, str) or isinstance(self.cut_date, datetime.datetime):
                # Using a SQL engine so need to return `sqlop` instances.
                if self.compute_layer in [ComputeLayerEnum.sqlite, ComputeLayerEnum.postgres, ComputeLayerEnum.snowflake, ComputeLayerEnum.redshift, ComputeLayerEnum.mysql, ComputeLayerEnum.athena, ComputeLayerEnum.databricks]:
                    return [
                            sqlop(optype=SQLOpType.where, opval=f"{self.colabbr(self.date_key)} > '{str(self.cut_date)}'"),
                            sqlop(optype=SQLOpType.where, opval=f"{self.colabbr(self.date_key)} < '{str(self.cut_date + datetime.timedelta(minutes=self.label_period_minutes()))}'")
                            ]

                elif isinstance(self.df, pd.DataFrame):
                    return self.df[
                        (self.df[self.colabbr(self.date_key)] > str(self.cut_date))
                        &
                        (self.df[self.colabbr(self.date_key)] < str(self.cut_date + datetime.timedelta(minutes=self.label_period_minutes())))
                    ]
                elif isinstance(self.df, pyspark.sql.dataframe.DataFrame):
                    return self.df.filter(
                        (self.df[self.colabbr(self.date_key)] > str(self.cut_date))
                        &
                        (self.df[self.colabbr(self.date_key)] < str(self.cut_date + datetime.timedelta(minutes=self.label_period_minutes())))
                    )
            else:
                # Using a SQL engine so need to return `sqlop` instances.
                if self.compute_layer in [ComputeLayerEnum.sqlite, ComputeLayerEnum.postgres, ComputeLayerEnum.snowflake, ComputeLayerEnum.redshift, ComputeLayerEnum.mysql, ComputeLayerEnum.athena, ComputeLayerEnum.databricks]:
                    return [
                            sqlop(optype=SQLOpType.where, opval=f"{self.colabbr(self.date_key)} > '{str(datetime.datetime.now() - datetime.timedelta(minutes=self.label_period_minutes()))}'")
                            ]

                elif isinstance(self.df, pd.DataFrame):
                    return self.df[
                        self.df[self.colabbr(self.date_key)] > (datetime.datetime.now() - datetime.timedelta(minutes=self.label_period_minutes()))
                    ]
                elif isinstance(self.df, pyspark.sql.dataframe.DataFrame):
                    return self.df.filter(
                    self.df[self.colabbr(self.date_key)] > (datetime.datetime.now() - datetime.timedelta(minutes=self.label_period_minutes()))
                )

        elif not hasattr(self, 'df'):
            return None
        # no-op
        return self.df



    def default_label (
            self,
            op: typing.Union[str, callable],
            field: str,
            reduce_key: typing.Optional[str] = None,
            ) -> typing.Union[
                    pd.DataFrame, dd.DataFrame, pyspark.sql.dataframe.DataFrame,
                    typing.List[sqlop]
                    ]:
        """
Default label operation.

        Arguments
        ----------
        op: operation to call for label
        field: str label field to call operation on
        reduce: bool whether or not to reduce
        """
        if hasattr(self, 'df') and self.colabbr(field) in self.df.columns:
            if self.compute_layer in [ComputeLayerEnum.pandas, ComputeLayerEnum.dask]:
                if self.reduce:
                    if callable(op):
                        return self.prep_for_labels().groupby(self.colabbr(reduce_key)).agg(**{
                            self.colabbr(field+'_label') : pd.NamedAgg(column=self.colabbr(field), aggfunc=op)
                        }).reset_index()
                    else:
                        return self.prep_for_labels().groupby(self.colabbr(reduce_key)).agg(**{
                            self.colabbr(field+'_label') : pd.NamedAgg(column=self.colabbr(field), aggfunc=op)
                            }).reset_index()
                else:
                    label_df = self.prep_for_labels()
                    if callable(op):
                        label_df[self.colabbr(field)+'_label'] = label_df[self.colabbr(field)].apply(op)
                    else:
                        label_df[self.colabbr(field)+'_label'] = label_df[self.colabbr(field)].apply(lambda x: getattr(x, op)())
                    return label_df[[self.colabbr(self.pk), self.colabbr(field)+'_label']]

            elif self.compute_layer == ComputeLayerEnum.spark:
                pass
        elif self.compute_layer in [ComputeLayerEnum.snowflake, ComputeLayerEnum.sqlite, ComputeLayerEnum.mysql, ComputeLayerEnum.postgres, ComputeLayerEnum.redshift, ComputeLayerEnum.athena, ComputeLayerEnum.databricks]:
            if self.reduce:
                return self.prep_for_labels() + [
                            sqlop(optype=SQLOpType.agg, opval=f"{self.colabbr(reduce_key)}"),
                            sqlop(optype=SQLOpType.aggfunc, opval=f"{op}"+ f"({self.colabbr(field)}) as {self.colabbr(field)}_label")
                            ]
            else:
                return self.prep_for_labels() + [
                            sqlop(optype=SQLOpType.select, opval=f"{op}" + f"({self.colabbr(field)}) as {self.colabbr(field)}_label")
                            ]
        else:
            pass


    def online_features (
            self,
            ):
        """
Define online features.
        """
        pass


    def on_demand_features (
            self,
            ):
        """
Define on demand features for this node.
        """
        pass




class DynamicNode(GraphReduceNode):
    """
A dynamic architecture for entities with no logic 
needed in addition to the top-level GraphReduceNode
parameters.  The required abstract methods:
    `do_annotate`
    `do_filters`
    `do_normalize`
    `do_post_join_filters`
    `do_post_join_annotate`
   
    """
    def __init__ (
            self,
            *args,
            **kwargs,
            ):
        """
Constructor
        """
        super().__init__(*args, **kwargs)

    def do_filters(self):
        pass

    def do_annotate(self):
        pass

    def do_post_join_annotate(self):
        pass

    def do_normalize(self):
        pass

    def do_post_join_filters(self):
        pass

    def do_reduce(self, reduce_key: str):
        pass

    def do_labels(self, reduce_key: str):
        pass





class GraphReduceQueryException(Exception): pass


class SQLNode(GraphReduceNode):
    """
Base node for SQL engines.  Makes some common 
operations available.  This class should be
extended for engines that do not conform
to a single `client` interface, such as 
AWS Athena, which requires additional params.

Subclasses should simply extend the `SQLNode` interface:

    """
    def __init__ (
        self,
        *args,
        client: typing.Any = None,
        lazy_execution: bool = False,
        **kwargs
    ):
        """
Constructor.
        """
        self.client = client
        self.lazy_execution = lazy_execution

        # The current data ref.
        self._cur_data_ref = None
        # A place to store temporary tables or views.
        self._temp_refs = {}
        self._removed_refs = []

        super().__init__(*args, **kwargs)


    def _clean_refs(self):
        """
Cleanup tables created during execution.
        """
        for k, v in self._temp_refs.items():
            if v not in self._removed_refs:
                sql = f"DROP VIEW {v}"
                self.execute_query(sql)
                self._removed_refs.append(v)
                logger.info(f"dropped {v}")


    def get_ref_name (
            self,
            fn: typing.Union[callable, str] = None,
            lookup: bool = False,
            ) -> str:
        """
Get a reference name for the function.
        """
        func_name = fn if isinstance(fn, str) else fn.__name__
        # If this ref name is already in 
        # the _temp_refs dict create a new ref.

        # IF there is a schema in the fpath
        # we need to remove the schema.
        if '.' in self.fpath:
            fpath = self.fpath.split('.')[-1]
        else:
            fpath = self.fpath
        ref_name = f"{fpath}_{func_name}_grtemp"
        if self._temp_refs.get(func_name):
            if lookup:
                return self._temp_refs[func_name]
            if self._temp_refs[func_name] == ref_name:
                i = 1
                ref_name = ref_name + str(i)
                while self._temp_refs.get(func_name) == ref_name:
                    i += 1
                    ref_name = ref_name + str(i)
                return ref_name
            else:
                return ref_name
        else:
            return ref_name


    def create_ref (
            self,
            sql: str = '',
            fn: typing.Union[callable, str] = None,
            # Overwrite the 
            overwrite: bool = False,
        ) -> str:
        """
Gets a temporary table or view name
based on the method being called.
        """


        # No reference has been created for this method.
        fn = fn if isinstance(fn, str) else fn.__name__

        # If no SQL was provided use the current reference.
        if not sql:
            logger.info(f"no sql was provided for {fn} so using current data ref")
            self._temp_refs[fn] = self._cur_data_ref
            return self._cur_data_ref

        if not self._temp_refs.get(fn) or overwrite: 
            ref_name = self.get_ref_name(fn)   
            self.create_temp_view(sql, ref_name)
            self._temp_refs[fn] = ref_name
            return ref_name
        # Reference for this method already created
        # so we will just retrieve.
        else:
            return self._temp_refs[fn]


    def get_current_ref (
            self
        ) -> str:
        """
Returns the name of the current
reference to the nodes data.
        """
        if not self._cur_data_ref:
            return self.fpath
        else:
            return self._cur_data_ref


    def create_temp_view (
            self,
            qry: str,
            view_name: str,
            ) -> str:
        """
Create a view with the results of
the query.
        """

        try:
            sql = f"""
            CREATE VIEW {view_name} AS
            {qry}
            """
            self.execute_query(sql, ret_df=False)
            self._cur_data_ref = view_name
            return view_name
        except Exception as e:
            logger.error(e)
            return None
   

    def get_sample (
        self,
        n: int = 100,
        table: str = None,
    ) -> pd.DataFrame:
        """
Gets a sample of rows for the current
table or a parameterized table.
        """
        
        samp_query = """
        SELECT * 
        FROM {table}
        LIMIT {n}
        """
        if not table:
            qry = samp_query.format(
                table=self._cur_data_ref if self._cur_data_ref else self.fpath,
                n=n
            )
        else:
            qry = samp_query.format(
                table=table,
                n=n
            )
        return self.execute_query(qry) 
                
    
    def build_query (
        self,
        ops: typing.List[sqlop],
        data_ref: str = None,
    ) -> str:
        """
Builds a SQL query given a list of `sqlop` instances.

     Parameters
     ----------
     ops: List of `sqlop` instances
     data_ref: (optional) str of the data reference to use
        """

        if not ops:
            return None
        
        if isinstance(ops, list):
            pass
        elif isinstance(ops, sqlop):
            ops = [ops]
        
        select_anatomy = """
        SELECT {selects}
        FROM {from_}
        WHERE {wheres}
        """

        group_anatomy = """
        SELECT {selects},
        {aggfuncs}
        FROM {from_}
        WHERE {wheres}
        GROUP BY {group}
        """

        # If a data reference is passed as 
        # a parameter we will use that.
        if data_ref:
            dr = data_ref
        else:
            dr = self._cur_data_ref if self._cur_data_ref else self.fpath

        # Table to select from.
        from_ = sqlop(
            optype=SQLOpType.from_,         
            opval=dr
        )
        
        # Boolean if this is an aggregation function.
        if len([_x for _x in ops if _x.optype == SQLOpType.agg]):
            aggfuncs = [_x for _x in ops if _x.optype == SQLOpType.aggfunc]
            if not len(aggfuncs):
                raise GraphReduceQueryException("Aggregation queries must have at least 1 `sqlop` of type SQLOpType.aggfunc")
            aggfuncs = ','.join([_x.opval for _x in aggfuncs])

            # Can only be one aggregation per query build.
            agg = [_x for _x in ops if _x.optype == SQLOpType.agg][0].opval

            wheres = [_x for _x in ops if _x.optype == SQLOpType.where]
            if not len(wheres):
                wheres = "true"
            else:
                wheres = ' and '.join([_x.opval for _x in wheres])

            return group_anatomy.format(
                selects=agg,
                aggfuncs=aggfuncs,
                from_=from_.opval,
                group=agg,
                wheres=wheres
            )

        # Otherwise go with standard select anatomy.
        else:
            selects = [_x for _x in ops if _x.optype == SQLOpType.select]
            # IF not select statements, select *.
            if not len(selects):
                selects = [sqlop(optype=SQLOpType.select, opval="*")]
            
            qry_selects = ','.join([_x.opval for _x in selects])
            #froms = [_x for _x in ops if _x.optype == SQLOpType.from_][0].opval
            wheres = [_x for _x in ops if _x.optype == SQLOpType.where]
            if len(wheres):
                qry_wheres = ' and '.join([_x.opval for _x in wheres])
            else:
                qry_wheres = "true"

            return select_anatomy.format(
                selects=qry_selects,
                from_=from_.opval,
                wheres=qry_wheres
            )


    def get_client(self) -> typing.Any:
        return self.client


    def execute_query (
            self,
            qry: str,
            ret_df: bool = True,
            ) -> typing.Optional[typing.Union[None, pd.DataFrame]]:
        """
Execute a query and get back a dataframe.
        """
        
        client = self.get_client()
        if not ret_df:
            cur = client.cursor()
            cur.execute(qry)
        else:
            return pd.read_sql_query(qry, client)


    def do_data(self) -> typing.Union[sqlop, typing.List[sqlop]]:
        """
Load the data.
        """

        col_renames = [
            f"{col} as {self.colabbr(col)}"
            for col in self.columns
        ]
        sel = sqlop(optype=SQLOpType.select, opval=f"{','.join(col_renames)}")
        return [sel]
 

    def do_annotate(self) -> typing.Union[sqlop, typing.List[sqlop]]:
        """
Should return a list of SQL statements 
casting columns as different types.
        """
        pass


    def do_normalize(self):
        pass
    
    
    def do_filters(self) -> typing.Union[sqlop, typing.List[sqlop]]:
        # Example:
        #return [
        #    sqlop(optype=SQLOpType.where, opval=f"{self.colabbr('id')} < 1000"),
        #]
        return None

    
    # Returns aggregate functions
    # Returns aggregate 
    def do_reduce(self, reduce_key) -> typing.Union[sqlop, typing.List[sqlop]]:
        # Example:
        #return [
        #    sqlop(optype=SQLOpType.aggfunc, opval=f"count(*) as {self.colabbr('num_dupes')}"),
        #    sqlop(optype=SQLOpType.agg, opval=f"{self.colabbr(reduce_key)}")
        #]
        pass
        
    
    def do_post_join_annotate(self):
        pass
   

    def do_labels(self):
        pass
   

    def do_post_join_filters(self):
        pass


    def do_sql(self) -> str:
        """
One function to compute this entire node.
        """
        pass


class AthenaNode(SQLNode):
    def __init__ (
            self,
            *args,
            s3_output_location: str = None,
            **kwargs,
            ):
        """
Constructor.
        """

        self.s3_output_location = s3_output_location

        super(AthenaNode, self).__init__(*args, **kwargs)
        
    
    def prep_for_features(self):
        if self.cut_date:
            if isinstance(self.cut_date, str):
                self.cut_date = parser.parse(self.cut_date)
        else:
            self.cut_date = datetime.datetime.now()
        
        if self.cut_date and self.date_key:
            return [
                sqlop(optype=SQLOpType.where, 
                      opval=f"{self.colabbr(self.date_key)} > timestamp '{str(self.cut_date - datetime.timedelta(minutes=self.compute_period_minutes()))}'"),
                sqlop(optype=SQLOpType.where,
                      opval=f"{self.colabbr(self.date_key)} < timestamp '{str(self.cut_date)}'")
            ]
        else:
            # do nothing
            return None
    
    
    def prep_for_labels(self):
        if self.cut_date:
            if isinstance(self.cut_date, str):
                self.cut_date = parser.parse(self.cut_date)
        else:
            self.cut_date = datetime.datetime.now()
        
        if self.cut_date and self.date_key:
            return [
                sqlop(optype=SQLOpType.where, 
                      opval=f"{self.colabbr(self.date_key)} < timestamp '{str(self.cut_date + datetime.timedelta(minutes=self.label_period_minutes()))}'"),
                sqlop(optype=SQLOpType.where,
                      opval=f"{self.colabbr(self.date_key)} > timestamp '{str(self.cut_date)}'")
            ]
        else:
            # do nothing
            return None
        

    def execute_query (
        self,
        qry: str,
        *args,
        **kwargs,
    ) -> typing.Optional[pd.DataFrame]:
        """
Execute a query and get back a dataframe.
        """
        
        logger.info(f"attempting to execute: {qry}")
        client = self.get_client()
        resp = client.start_query_execution(
            QueryString=qry,
            ResultConfiguration={
                'OutputLocation': self.s3_output_location
            }
        )
        # While query is executing sleep
        qry_id = resp['QueryExecutionId']

        qry_status = client.get_query_execution(QueryExecutionId=qry_id)
        if qry_status['QueryExecution']['Status']['State'] == 'FAILED':
            raise Exception(f"Query {qry} FAILED")

        else:
            while qry_status['QueryExecution']['Status']['State'] not in ['SUCCEEDED', 'FAILED']:
                logger.info("sleeping and waiting for query to finish")
                time.sleep(1)
                qry_status = client.get_query_execution(QueryExecutionId=qry_id)

        if qry_status['QueryExecution']['Status']['State'] == 'FAILED':
            logger.error("query FAILED")
            print(qry_status)
            return None

        results = client.get_query_results(QueryExecutionId=qry_id)
        colinfo = results['ResultSet']['ResultSetMetadata']['ColumnInfo']
        rows = results['ResultSet']['Rows']
        while results.get('NextToken'):
            results = client.get_query_results(QueryExecutionId=qry_id, NextToken=results['NextToken'])
            for row in results['ResultSet']['Rows']:
                rows.append(row)
        # create a dataframe ready version of the data
        dfdata = []
        for row in rows[1:]:
            newrow = {}
            for i in range(len(row['Data'])):
                col = colinfo[i]['Name']
                if row['Data'][i]:
                    val = row['Data'][i]['VarCharValue']
                else:
                    val = None
                newrow.update({col: val})
            dfdata.append(newrow)
        return pd.DataFrame(dfdata)



class DatabricksNode(SQLNode):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        
    def create_temp_view (
        self,
        qry: str,
        view_name: str
    ) -> str:
        """
Create a view with the results
of the query.
        """
        try:
            sql = f"""
            CREATE TEMPORARY VIEW {view_name} AS 
            {qry}
            """
            self.execute_query(sql, ret_df=False)
            self._cur_data_ref = view_name
        except Exception as e:
            logger.error(e)
            return None
