#!/usr/bin/env python

# std lib
import abc
import datetime
import typing

# third party
import pandas as pd
from dask import dataframe as dd
import pyspark

# internal
from graphreduce.enum import ComputeLayerEnum, PeriodUnit



class GraphReduceNode(metaclass=abc.ABCMeta): 
    fpath : str
    fmt : str
    prefix : str
    date_key : str
    pk : str 
    feature_function : str
    compute_layer : ComputeLayerEnum
    spark_sqlctx : typing.Optional[pyspark.sql.SQLContext]
    cut_date : datetime.datetime
    compute_period_val : typing.Union[int, float]
    compute_period_unit : PeriodUnit
    has_labels : bool
    label_period_val : typing.Optional[typing.Union[int, float]]
    label_period_unit : typing.Optional[PeriodUnit] 

    def __init__ (
            self,
            fpath : str,
            fmt : str,
            pk : str = None,
            prefix : str = None,
            date_key : str = None,
            compute_layer : ComputeLayerEnum = None,
            cut_date : datetime.datetime = datetime.datetime.now(),
            compute_period_val : typing.Union[int, float] = 365,
            compute_period_unit : PeriodUnit  = PeriodUnit.day,
            reduce : bool = True,
            has_labels : bool = False,
            label_period_val : typing.Optional[typing.Union[int, float]] = None,
            label_period_unit : typing.Optional[PeriodUnit] = None,
            feature_function : typing.Optional[str] = None,
            spark_sqlctx : pyspark.sql.SQLContext = None,
            columns : list = [],
            ):
        """
Constructor

Args
    fpath : file path
    fmt : format of file
    pk : primary key
    prefix : prefix to use for columns
    date_key : column containing the date - if there isn't one leave blank
    compute_layer : compute layer to use (e.g, ComputeLayerEnum.pandas)
    cut_date : date around which to orient the data
    compute_period_val : amount of time to consider
    compute_period_unit : unit of measure for compute period (e.g., PeriodUnit.day)
    reduce : whether or not to reduce the data
    has_labels : whether or not the node has labels to compute
    label_period_val : optional period of time to compute labels
    label_period_unit : optional unit of measure for label period (e.g., PeriodUnit.day)
    feature_function : optional feature function, usually used when reduce is false
    columns : optional list of columns to include
        """
        self.fpath = fpath
        self.fmt = fmt
        self.pk = pk
        self.prefix = prefix
        self.date_key = date_key
        self.compute_layer = compute_layer
        self.cut_date = cut_date
        self.compute_period_val = compute_period_val
        self.compute_period_unit = compute_period_unit
        self.reduce = reduce
        self.has_labels = has_labels
        self.label_period_val = label_period_val
        self.label_period_unit = label_period_unit
        self.feature_function = feature_function
        self.spark_sqlctx = spark_sqlctx

        self.columns = []
        

    
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
                if len(self.columns):
                    self.df = self.df[[c for c in self.columns]]
                self.columns = list(self.df.columns)
                self.df.columns = [f"{self.prefix}_{c}" for c in self.df.columns]
        elif self.compute_layer.value == 'dask':
            if not hasattr(self, 'df') or (hasattr(self, 'df') and not isinstance(self.df, dd.DataFrame
)):
                self.df = getattr(dd, f"read_{self.fmt}")(self.fpath)
                if len(self.columns):
                    self.df = self.df[[c for c in self.columns]]
                self.columns = list(self.df.columns)
                self.df.columns = [f"{self.prefix}_{c}" for c in self.df.columns]
        elif self.compute_layer.value == 'spark':
            if not hasattr(self, 'df') or (hasattr(self, 'df') and not isinstance(self.df, pyspark.sql.DataFrame)):
                self.df = getattr(self.spark_sqlctx.read, {self.fmt})(self.fpath)
                for c in self.df.columns:
                    self.df = self.df.withColumnRenamed(c, f"{self.prefix}_{c}")


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
        '''
        Implement custom annotation functionality
        for annotating this particular data
        '''
        return

    
    @abc.abstractmethod
    def do_post_join_annotate(self):
        '''
        Implement custom annotation functionality
        for annotating data after joining with 
        child data
        '''
        pass

     
    @abc.abstractmethod
    def do_clip_cols(self):
        return
            

    @abc.abstractmethod
    def do_reduce(self, reduce_key, children : list = []):
        """
Reduce operation or the node

Args
    reduce_key : key to use to perform the reduce operation
    children : list of children nodes
        """
        pass
    
    
    @abc.abstractmethod
    def do_labels(self, reduce_key):
        pass
    
        
    def colabbr(self, col: str) -> str:
        return f"{self.prefix}_{col}"
    
    
    def prep_for_features(self):
        """
Prepare the dataset for feature aggregations / reduce
        """
        if self.date_key:
            if self.cut_date and isinstance(self.cut_date, str) or isinstance(self.cut_date, datetime.datetime):
                if isinstance(self.df, pd.DataFrame) or isinstance(self.df, dd.DataFrame):
                    return self.df[
                        (self.df[self.colabbr(self.date_key)] < self.cut_date)
                        &
                        (self.df[self.colabbr(self.date_key)] > (self.cut_date - datetime.timedelta(days=self.compute_period_val)))
                    ]
                elif isinstance(self.df, pyspark.sql.dataframe.DataFrame):
                    return self.df.filter(
                        (self.df[self.colabbr(self.date_key)] < self.cut_date)
                        &
                        (self.df[self.colabbr(self.date_key)] > (self.cut_date - datetime.timedelta(days=self.compute_period_val)))
                    )
            else:
                if isinstance(self.df, pd.DataFrame) or isinstance(self.df, dd.DataFrame):
                    return self.df[
                        (self.df[self.colabbr(self.date_key)] < datetime.datetime.now())
                        &
                        (self.df[self.colabbr(self.date_key)] > (datetime.datetime.now() - datetime.timedelta(days=self.compute_period_val)))
                    ]
                elif isinstance(self.df, pyspark.sql.dataframe.DataFrame):
                    return self.df.filter(
                        self.df[self.colabbr(self.date_key)] > (datetime.datetime.now() - datetime.timedelta(days=self.compute_period_val))
                )
        # no-op
        return self.df
    
    
    def prep_for_labels(self):
        """
        Prepare the dataset for labels
        """
        if self.date_key:
            if self.cut_date and isinstance(self.cut_date, str) or isinstance(self.cut_date, datetime.datetime):
                if isinstance(self.df, pd.DataFrame):
                    return self.df[
                        (self.df[self.colabbr(self.date_key)] > (self.cut_date))
                        &
                        (self.df[self.colabbr(self.date_key)] < (self.cut_date + datetime.timedelta(days=self.label_period_val)))
                    ]
                elif isinstance(self.df, pyspark.sql.dataframe.DataFrame):
                    return self.df.filter(
                        (self.df[self.colabbr(self.date_key)] > (self.cut_date))
                        &
                        (self.df[self.colabbr(self.date_key)] < (self.cutDate + datetime.timedelta(days=self.label_period_val)))
                    )
            else:
                if isinstance(self.df, pd.DataFrame):
                    return self.df[
                        self.df[self.colabbr(self.date_key)] > (datetime.datetime.now() - datetime.timedelta(days=self.label_period_val))
                    ]
                elif isinstance(self.df, pyspark.sql.dataframe.DataFrame):
                    return self.df.filter(
                    self.df[self.colabbr(self.date_key)] > (datetime.datetime.now() - datetime.timedelta(days=self.label_period_val))
                )
        return self.df
