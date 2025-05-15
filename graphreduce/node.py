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
from pyspark.sql import functions as F, types as T
from structlog import get_logger
from dateutil.parser import parse as date_parse
from torch_frame.utils import infer_df_stype
import daft
from daft.unity_catalog import UnityCatalog
from pyiceberg.catalog.rest import RestCatalog
import duckdb


# internal
from graphreduce.enum import ComputeLayerEnum, PeriodUnit, SQLOpType
from graphreduce.storage import StorageClient
from graphreduce.models import sqlop
from graphreduce.common import (
    clean_datetime_pandas,
    clean_datetime_dask,
    clean_datetime_spark,
)


logger = get_logger("Node")


class GraphReduceNode(metaclass=abc.ABCMeta):
    """
    Base node class, which can be used directly
    or subclassed for further customization.

    Many helpful methods are implemented and can
    be used as is, but for different engines
    and dialects (e.g., SQL vs. python) it can
    be necessary to implement an engine-specific
    methods (e.g., `do_data` to get data from Snowflake)

    The methods `do_annotate`, `do_filters`,
    `do_normalize`, `do_reduce`, `do_labels`,
    `do_post_join_annotate`, and `do_post_join_filters`
    are abstractmethods which must be defined.
    """

    fpath: str
    fmt: str
    pk: str
    prefix: str
    date_key: str
    compute_layer: ComputeLayerEnum
    cut_date: typing.Optional[datetime.datetime]
    compute_period_val: typing.Union[int, float]
    compute_period_unit: PeriodUnit
    reduce: bool
    label_period_val: typing.Optional[typing.Union[int, float]]
    label_period_unit: typing.Optional[PeriodUnit]
    label_field: typing.Optional[str]
    spark_sqlctx: typing.Optional[pyspark.sql.SQLContext]
    columns: typing.List
    storage_client: typing.Optional[StorageClient]
    # Only for SQL dialects at the moment.
    lazy_execution: bool

    def __init__(
        self,
        # IF is SQL dialect this should be a table name.
        fpath: str = "",
        # If is SQL dialect "sql" is fine here.
        fmt: str = "",
        pk: str = None,
        prefix: str = None,
        date_key: str = None,
        compute_layer: ComputeLayerEnum = None,
        # 'python' or 'sql'
        dialect: str = "python",
        cut_date: datetime.datetime = datetime.datetime.now(),
        compute_period_val: typing.Union[int, float] = 365,
        compute_period_unit: PeriodUnit = PeriodUnit.day,
        reduce: bool = True,
        label_period_val: typing.Optional[typing.Union[int, float]] = None,
        label_period_unit: typing.Optional[PeriodUnit] = None,
        label_field: typing.Optional[str] = None,
        label_operation: typing.Optional[typing.Union[str, callable]] = None,
        spark_sqlctx: pyspark.sql.SQLContext = None,
        columns: list = [],
        storage_client: typing.Optional[StorageClient] = None,
        checkpoints: list = [],
        # Only for SQL dialects at the moment.
        lazy_execution: bool = False,
        # Read encoding.
        delimiter: str = None,
        encoding: str = None,
        catalog_client: typing.Any = None,
        # The time-series period in days to use.
        ts_periods: list = [30, 60, 90, 180, 365, 730],
    ):
        """
        Constructor
        """
        # For when this is already set on the class definition.
        if not hasattr(self, "pk"):
            self.pk = pk
        # For when this is already set on the class definition.
        if not hasattr(self, "prefix"):
            self.prefix = prefix
        # For when this is already set on the class definition.
        if not hasattr(self, "date_key"):
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
        self.label_operation = label_operation
        self.spark_sqlctx = spark_sqlctx
        self.columns = columns

        # Read options
        self.delimiter = delimiter if delimiter else ","
        self.encoding = encoding

        # Lazy execution for the SQL nodes.
        self._lazy_execution = lazy_execution
        self._storage_client = storage_client
        # List of merged neighbor classes.
        self._merged = []
        # List of checkpoints.

        # Logical types of the original columns from `woodwork`.
        self._logical_types = {}

        self._stypes = {}

        if not self.date_key:
            logger.warning(f"no `date_key` set for {self}")

        self._catalog_client = catalog_client
        self.ts_periods = ts_periods

    def __repr__(self):
        """
        Instance representation
        """
        return (
            f"<GraphReduceNode: fpath={self.fpath} fmt={self.fmt} prefix={self.prefix}>"
        )

    def __str__(self):
        """
        Instances string
        """
        return (
            f"<GraphReduceNode: fpath={self.fpath} fmt={self.fmt} prefix={self.prefix}>"
        )

    def _is_identifier(self, col: str) -> bool:
        """
        Check if a column is an identifier.
        """
        if col.lower() == "id":
            return True
        elif col.lower().split("_")[-1].endswith("id"):
            return True
        elif col.lower() == "uuid":
            return True
        elif col.lower() == "guid":
            return True
        elif col.lower() == "identifier":
            return True
        elif col.lower().endswith("key"):
            return True

    def _is_bool(
        self,
        col: str,
    ) -> bool:
        pass

    def is_ts_data(
        self,
        reduce_key: str = None,
    ) -> bool:
        """
        Determines if the data is timeseries.
        """
        if self.date_key:
            if (
                self.compute_layer == ComputeLayerEnum.pandas
                or self.compute_layer == ComputeLayerEnum.dask
            ):
                grouped = self.df.groupby(self.colabbr(reduce_key)).agg(
                    {self.colabbr(self.pk): "count"}
                )
                if len(grouped) / len(self.df) < 0.9:
                    return True
            elif self.compute_layer == ComputeLayerEnum.spark:
                grouped = (
                    self.df.groupBy(self.colabbr(reduce_key))
                    .agg(F.count(self.colabbr(self.pk)))
                    .count()
                )
                n = self.df.count()
                if float(grouped) / float(n) < 0.9:
                    return True
            # TODO(wes): define the SQL logic.
            elif self.compute_layer in [
                ComputeLayerEnum.sqlite,
                ComputeLayerEnum.snowflake,
                ComputeLayerEnum.databricks,
                ComputeLayerEnum.athena,
                ComputeLayerEnum.redshift,
                ComputeLayerEnum.duckdb,
            ]:
                # run a group by and get the value.
                grp_qry = f"""
                select count(*) as grouped_rows
                from (
                    select {reduce_key}, count({self.pk})
                    FROM {self.fpath}
                    group by {reduce_key}
                ) t;
                """
                row_qry = f"""
                select count(*) as row_count from {self.fpath}
                """
                grp_df = self.execute_query(grp_qry, ret_df=True)
                grp_df.columns = [c.lower() for c in grp_df.columns]
                row_df = self.execute_query(row_qry, ret_df=True)
                row_df.columns = [c.lower() for c in row_df.columns]
                grp_count = grp_df["grouped_rows"].values[0]
                row_count = row_df["row_count"].values[0]
                if float(grp_count) / float(row_count) < 0.9:
                    return True
            elif self.compute_layer == ComputeLayerEnum.daft:
                grouped = (
                    self.df.groupby(self.colabbr(reduce_key))
                    .agg(self.df[self.colabbr(self.pk)].count())
                    .count_rows()
                )
                n = self.df.count_rows()
                if float(grouped) / float(n) < 0.9:
                    return True
        return False

    def reload(self):
        """
        Refresh the node.
        """
        self._merged = []
        self._checkpoints = []
        self.df = None
        self._logical_types = {}

    def do_data(
        self,
    ) -> typing.Union[pd.DataFrame, dd.DataFrame, pyspark.sql.dataframe.DataFrame]:
        """
        Get some data
        """

        if self.compute_layer.value == "pandas":
            if not hasattr(self, "df") or (
                hasattr(self, "df") and not isinstance(self.df, pd.DataFrame)
            ):
                if self.encoding and self.delimiter:
                    self.df = getattr(pd, f"read_{self.fmt}")(
                        self.fpath, encoding=self.encoding, delimiter=self.delimiter
                    )
                else:
                    self.df = getattr(pd, f"read_{self.fmt}")(self.fpath)
                # Initialize woodwork.
                # self.df.ww.init()
                # self._logical_types = self.df.ww.logical_types

                # Rename columns with prefixes.
                if len(self.columns):
                    self.df = self.df[[c for c in self.columns]]
                self.columns = list(self.df.columns)
                self.df.columns = [f"{self.prefix}_{c}" for c in self.df.columns]
                # Infer the semantic type with `torch_frame`.
                self._stypes = infer_df_stype(self.df.head(100))
        elif self.compute_layer.value == "dask":
            if not hasattr(self, "df") or (
                hasattr(self, "df") and not isinstance(self.df, dd.DataFrame)
            ):
                self.df = getattr(dd, f"read_{self.fmt}")(self.fpath)
                # Initialize woodwork.
                # self.df.ww.init()
                # self._logical_types = self.df.ww.logical_types

                # Rename columns with prefixes.
                if len(self.columns):
                    self.df = self.df[[c for c in self.columns]]
                self.columns = list(self.df.columns)
                self.df.columns = [f"{self.prefix}_{c}" for c in self.df.columns]
                # Infer the semantic type with `torch_frame`.
                self._stypes = infer_df_stype(self.df.head())
        elif self.compute_layer.value == "spark":
            if not hasattr(self, "df") or (
                hasattr(self, "df") and not isinstance(self.df, pyspark.sql.DataFrame)
            ):
                if self.fmt != "sql":
                    self.df = getattr(self.spark_sqlctx.read, f"{self.fmt}")(self.fpath)
                elif self.fmt == "sql":
                    self.df = self.spark_sqlctx.sql(f"select * from {self.fpath}")

                if self.columns:
                    self.df = self.df.select(self.columns)
                for c in self.df.columns:
                    self.df = self.df.withColumnRenamed(c, f"{self.prefix}_{c}")

                # Infer the semantic type with `torch_frame`.
                self._stypes = infer_df_stype(self.df.sample(0.5).limit(10).toPandas())
        elif self.compute_layer.value == "daft":
            if not hasattr(self, "df") or (
                hasattr(self, "df")
                and not isinstance(self.df, daft.dataframe.dataframe.DataFrame)
            ):
                # Iceberg.
                if self._catalog_client:
                    if isinstance(self._catalog_client, RestCatalog):
                        tbl = self._catalog_client.load_table(self.fpath)
                        self.df = daft.read_iceberg(tbl)
                    elif isinstance(self._catalog_client, UnityCatalog):
                        tbl = self._catalog_client.load_table(self.fpath)
                        # TODO(wes): support more than just deltalake.
                        self.df = daft.read_deltalake(tbl)
                else:
                    self.df = getattr(daft, f"read_{self.fmt}")(self.fpath)
                self.columns = [c.name() for c in self.df.columns]
                for col in self.df.columns:
                    self.df = self.df.with_column(f"{self.prefix}_{col.name()}", col)
                newcols = [
                    c for c in self.df.columns if c.name().startswith(self.prefix)
                ]
                self.df = self.df.select(*newcols)

                # Infer the semantic type with `torch_frame`.
                n = self.df.count_rows()
                m = 100
                frac = float(m / n) if m / n < 1 else 1.0
                self._stypes = infer_df_stype(self.df.sample(frac).to_pandas())
        # at this point of connectors we may want to try integrating
        # with something like fugue: https://github.com/fugue-project/fugue
        elif self.compute_layer.value == "ray":
            pass

        elif self.compute_layer.value == "snowflake":
            pass

        elif self.compute_layer.value == "postgres":
            pass

        elif self.compute_layer.value == "redshift":
            pass

    @abc.abstractmethod
    def do_filters(self):
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

    def do_post_join_reduce(self, reduce_key: str):
        """
        Implementation for reduce operations
        after a join.
        """
        pass

    def auto_features(
        self,
        reduce_key: str,
        type_func_map: dict = {},
        compute_layer: ComputeLayerEnum = ComputeLayerEnum.pandas,
    ):
        """
        If we're doing automatic features
        this function will run a series of
        automatic aggregations.  The top-level
        `GraphReduce` object will handle joining
        the results together.
        """
        if compute_layer == ComputeLayerEnum.pandas:
            return self.pandas_auto_features(
                reduce_key=reduce_key, type_func_map=type_func_map
            )
        elif compute_layer == ComputeLayerEnum.dask:
            return self.dask_auto_features(
                reduce_key=reduce_key, type_func_map=type_func_map
            )
        elif compute_layer == ComputeLayerEnum.spark:
            return self.spark_auto_features(
                reduce_key=reduce_key, type_func_map=type_func_map
            )
        elif self.compute_layer in [
            ComputeLayerEnum.snowflake,
            ComputeLayerEnum.sqlite,
            ComputeLayerEnum.mysql,
            ComputeLayerEnum.postgres,
            ComputeLayerEnum.redshift,
            ComputeLayerEnum.databricks,
            ComputeLayerEnum.duckdb,
        ]:
            # Assumes `SQLNode.get_sample` is implemented to get
            # a sample of the data in pandas dataframe form.
            sample_df = self.get_sample()
            return self.sql_auto_features(
                sample_df, reduce_key=reduce_key, type_func_map=type_func_map
            )
        elif self.compute_layer == ComputeLayerEnum.daft:
            return self.daft_auto_features(
                reduce_key=reduce_key, type_func_map=type_func_map
            )

    def auto_labels(
        self,
        reduce_key: str,
        type_func_map: dict = {},
        compute_layer: ComputeLayerEnum = ComputeLayerEnum.pandas,
    ):
        """
        If we're doing automatic features
        this function will run a series of
        automatic aggregations.  The top-level
        `GraphReduce` object will handle joining
        the results together.
        """
        if compute_layer == ComputeLayerEnum.pandas:
            return self.pandas_auto_labels(
                reduce_key=reduce_key, type_func_map=type_func_map
            )
        elif compute_layer == ComputeLayerEnum.dask:
            return self.dask_auto_labels(
                reduce_key=reduce_key, type_func_map=type_func_map
            )
        elif compute_layer == ComputeLayerEnum.spark:
            return self.spark_auto_labels(
                reduce_key=reduce_key, type_func_map=type_func_map
            )
        elif compute_layer == ComputeLayerEnum.daft:
            return self.daft_auto_labels(
                reduce_key=reduce_key, type_func_map=type_func_map
            )

    def pandas_auto_features(
        self, reduce_key: str, type_func_map: dict = {}
    ) -> pd.DataFrame:
        """
        Pandas implementation of dynamic propagation of features.
        This is basically automated feature engineering but suffixed
        with `_propagation` to indicate that we are propagating data
        upward through the graph from child nodes with no feature
        definitions.
        """
        agg_funcs = {}

        ts_data = self.is_ts_data(reduce_key)
        if ts_data:
            # Make sure the dates are cleaned.
            self.df = clean_datetime_pandas(self.df, self.colabbr(self.date_key))
            # First sort the data by dates.
            self.df = self.df.sort_values(self.colabbr(self.date_key), ascending=True)
            self.df[f"prev_{self.colabbr(self.date_key)}"] = self.df.groupby(
                self.colabbr(reduce_key)
            )[self.colabbr(self.date_key)].shift(1)
            # Get the time between the two different records.
            self.df[self.colabbr("time_between_records")] = self.df.apply(
                lambda x: (
                    x[self.colabbr(self.date_key)]
                    - x[f"prev_{self.colabbr(self.date_key)}"]
                ).total_seconds(),
                axis=1,
            )

        # Make sure `self._stypes` is up to date.
        self._stypes = infer_df_stype(self.df.sample(min(1000, len(self.df))))
        for col, stype in self._stypes.items():
            _type = str(stype)
            if self._is_identifier(col) and col != reduce_key:
                # We only perform counts for identifiers.
                agg_funcs[f"{col}_count"] = pd.NamedAgg(column=col, aggfunc="count")
            elif self._is_identifier(col) and col == reduce_key:
                continue
            elif type_func_map.get(_type):
                for func in type_func_map[_type]:
                    if (
                        (_type == "numerical" or "timestamp")
                        and dict(self.df.dtypes)[col].__str__() == "object"
                        and func in ["min", "max", "median", "mean"]
                    ):
                        logger.info(
                            f"skipped aggregation on {col} because semantic numerical but physical object"
                        )
                        continue
                    col_new = f"{col}_{func}"
                    agg_funcs[col_new] = pd.NamedAgg(column=col, aggfunc=func)
        if not len(agg_funcs):
            logger.info(f"No aggregations for {self}")
            return self.df

        grouped = (
            self.prep_for_features()
            .groupby(self.colabbr(reduce_key))
            .agg(**agg_funcs)
            .reset_index()
        )
        if not len(grouped):
            return None
        # If we have time-series data take the time
        # since the last event and the cut date.
        if ts_data:
            logger.info(f"computed post-aggregation features for {self}")

            def is_tz_aware(series):
                return series.dt.tz is not None

            if is_tz_aware(grouped[f"{self.colabbr(self.date_key)}_max"]):
                grouped[f"{self.colabbr(self.date_key)}_max"] = grouped[
                    f"{self.colabbr(self.date_key)}_max"
                ].dt.tz_localize(None)

            grouped[self.colabbr("time_since_last_event")] = grouped.apply(
                lambda x: (
                    self.cut_date - x[f"{self.colabbr(self.date_key)}_max"]
                ).total_seconds(),
                axis=1,
            )

            # Number of events in last strata of time
            days = [30, 60, 90, 365, 730]
            for d in days:
                if d > self.compute_period_val:
                    continue
                feat_prepped = self.prep_for_features()
                if is_tz_aware(feat_prepped[self.colabbr(self.date_key)]):
                    feat_prepped[self.colabbr(self.date_key)] = feat_prepped[
                        self.colabbr(self.date_key)
                    ].dt.tz_localize(None)

                feat_prepped[self.colabbr("time_since_cut")] = feat_prepped.apply(
                    lambda x: (
                        self.cut_date - x[self.colabbr(self.date_key)]
                    ).total_seconds()
                    / 86400,
                    axis=1,
                )
                sub = feat_prepped[
                    (feat_prepped[self.colabbr("time_since_cut")] >= 0)
                    & (feat_prepped[self.colabbr("time_since_cut")] <= d)
                ]
                days_group = (
                    sub.groupby(self.colabbr(reduce_key))
                    .agg(
                        **{
                            self.colabbr(f"{d}d_num_events"): pd.NamedAgg(
                                aggfunc="count", column=self.colabbr(self.pk)
                            )
                        }
                    )
                    .reset_index()
                )
                # join this back to the main dataset.
                grouped = grouped.merge(
                    days_group, on=self.colabbr(reduce_key), how="left"
                )
            logger.info(f"merged all ts groupings to {self}")
        return grouped

    def daft_auto_features(
        self, reduce_key: str, type_func_map: dict = {}
    ) -> pd.DataFrame:
        """
        Daft implementation of dynamic propagation of features.
        This is basically automated feature engineering but suffixed
        with `_propagation` to indicate that we are propagating data
        upward through the graph from child nodes with no feature
        definitions.
        """

        # Temporary hack until a `daft` implementation
        # of window functions is available.  This will
        # also, unfortunately, limit us to single machine
        # data sizes with daft until then.
        original_df = self.df
        self.compute_layer = ComputeLayerEnum.pandas
        self.df = self.df.to_pandas()
        grouped = self.pandas_auto_features(reduce_key, type_func_map=type_func_map)
        grouped = daft.from_pandas(grouped)
        self.df = original_df
        self.compute_layer = ComputeLayerEnum.daft
        return grouped

    def dask_auto_features(
        self,
        reduce_key: str,
        type_func_map: dict = {},
    ) -> dd.DataFrame:
        """
        Dask implementation of dynamic propagation of features.
        This is basically automated feature engineering but suffixed
        with `_propagation` to indicate that we are propagating data
        upward through the graph from child nodes with no feature
        definitions.
        """
        agg_funcs = {}
        for col, stype in self._stypes.items():
            _type = str(stype)
            if type_func_map.get(_type):
                for func in type_func_map[_type]:
                    col_new = f"{col}_{func}"
                    agg_funcs[col_new] = pd.NamedAgg(column=col, aggfunc=func)
        return (
            self.prep_for_features()
            .groupby(self.colabbr(reduce_key))
            .agg(**agg_funcs)
            .reset_index()
        )

    def spark_auto_features(
        self,
        reduce_key: str,
        type_func_map: dict = {},
    ) -> pyspark.sql.DataFrame:
        """
        Spark implementation of dynamic propagation of features.
        This is basically automated feature engineering but suffixed
        with `_propagation` to indicate that we are propagating data
        upward through the graph from child nodes with no feature
        definitions.
        """

        self._stypes = infer_df_stype(self.df.sample(0.5).limit(10).toPandas())
        agg_funcs = []
        ts_data = self.is_ts_data(reduce_key)
        if ts_data:
            logger.info(f"{self} is time-series data")
        for col, stype in self._stypes.items():
            _type = str(stype)

            if self._is_identifier(col) and col != reduce_key:
                func = "count"
                col_new = f"{col}_{func}"
                agg_funcs.append(F.count(F.col(col)).alias(col_new))
            elif self._is_identifier(col) and col == reduce_key:
                continue
            elif type_func_map.get(_type):
                for func in type_func_map[_type]:
                    if func == "nunique":
                        func = "count_distinct"
                    col_new = f"{col}_{func}"
                    agg_funcs.append(getattr(F, func)(F.col(col)).alias(col_new))
        grouped = (
            self.prep_for_features().groupby(self.colabbr(reduce_key)).agg(*agg_funcs)
        )
        # If we have time-series data take the time
        # since the last event and the cut date.
        if ts_data:
            # convert the date key to a timestamp
            date_key_field = [
                x
                for x in self.df.schema.fields
                if x.name == self.colabbr(self.date_key)
            ][0]
            if date_key_field.dataType not in [T.TimestampType(), T.DateType()]:
                logger.info(
                    f"{self} date key was {date_key_field.dataType} - converting to Timestamp"
                )
                self.df = self.df.withColumn(
                    self.colabbr(self.date_key),
                    F.to_timestamp(F.col(self.colabbr(self.date_key))),
                )
            logger.info(f"computed post-aggregation features for {self}")
            spark_datetime = self.spark_sqlctx.sql(
                f"SELECT TO_DATE('{self.cut_date.strftime('%Y-%m-%d')}') as cut_date"
            )
            if "cut_date" not in grouped.columns:
                grouped = grouped.crossJoin(spark_datetime)

            grouped = grouped.withColumn(
                self.colabbr("time_since_last_event"),
                F.unix_timestamp(F.col("cut_date"))
                - F.unix_timestamp(F.col(f"{self.colabbr(self.date_key)}_max")),
            ).drop(F.col("cut_date"))
            if "cut_date" not in self.df.columns:
                self.df = self.df.crossJoin(spark_datetime)

            # Number of events in last strata of time
            days = [30, 60, 90, 365, 730]
            for d in days:
                if d > self.compute_period_val:
                    continue
                feat_prepped = self.prep_for_features()
                feat_prepped = feat_prepped.withColumn(
                    self.colabbr("time_since_cut"),
                    F.unix_timestamp(F.col("cut_date"))
                    - F.unix_timestamp(F.col(self.colabbr(self.date_key))),
                ).drop(F.col("cut_date"))
                sub = feat_prepped.filter(
                    (feat_prepped[self.colabbr("time_since_cut")] >= 0)
                    & (feat_prepped[self.colabbr("time_since_cut")] <= (d * 86400))
                )
                days_group = sub.groupBy(self.colabbr(reduce_key)).agg(
                    F.count(self.colabbr(self.pk)).alias(
                        self.colabbr(f"{d}d_num_events")
                    )
                )
                # join this back to the main dataset.
                grouped = grouped.join(
                    days_group, on=self.colabbr(reduce_key), how="left"
                )
            logger.info(f"merged all ts groupings to {self}")
        if "cut_date" in grouped.columns:
            grouped = grouped.drop(F.col("cut_date"))
        return grouped

    def sql_auto_features(
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
        # Always need to update this
        # because we never know if
        # the original columns comprise all
        # of the columns currently in the df.
        self._stypes = infer_df_stype(table_df_sample)

        # Physical types.
        ptypes = {col: str(t) for col, t in table_df_sample.dtypes.to_dict().items()}

        ts_data = self.is_ts_data(reduce_key)
        for col, stype in self._stypes.items():
            # Check if it is a label first.
            if "_label" in col:
                label_func_map = {
                    "count": "sum",
                    "sum": "sum",
                    "min": "min",
                    "max": "max",
                }

            _type = str(stype)
            if self._is_identifier(col) and col != reduce_key:
                # We only perform counts for identifiers.
                func = "count"
                col_new = f"{col}_{func}"
                agg_funcs.append(
                    sqlop(
                        optype=SQLOpType.aggfunc,
                        opval=f"{func}" + f"({col}) as {col_new}",
                    )
                )

            elif self._is_identifier(col) and col == reduce_key:
                continue

            elif type_func_map.get(_type):
                if ptypes[col] == "bool":
                    col_new = f"{col}_sum"
                    op = sqlop(
                        optype=SQLOpType.aggfunc,
                        opval=f"sum(case when {col} = 1 then 1 else 0 end) as {col_new}",
                    )
                    if op not in agg_funcs:
                        agg_funcs.append(op)
                    continue
                for func in type_func_map[_type]:
                    # There should be a better top-level mapping
                    # but for now this will do.  SQL engines typically
                    # don't have 'median' and 'mean'.  'mean' is typically
                    # just called 'avg'.
                    if (
                        (_type == "numerical" or "timestamp")
                        and dict(table_df_sample)[col].__str__() == "object"
                        and func in ["min", "max", "mean", "median"]
                    ):
                        logger.info(
                            f"skipped aggregation on {col} because semantic numerical but physical object"
                        )
                        continue
                    elif func in self.FUNCTION_MAPPING:
                        func = self.FUNCTION_MAPPING.get(func)

                    elif not func or func == "nunique":
                        continue

                    if (
                        _type == "categorical"
                        and len(
                            table_df_sample[~table_df_sample[col].isnull()][
                                col
                            ].unique()
                        )
                        <= 2
                        and str(
                            table_df_sample[~table_df_sample[col].isnull()][col]
                            .head()
                            .values[0]
                        ).isdigit()
                    ):
                        func = "sum"

                    if func:
                        col_new = f"{col}_{func}"
                        op = sqlop(
                            optype=SQLOpType.aggfunc,
                            opval=f"{func}" + f"({col}) as {col_new}",
                        )
                        if op not in agg_funcs:
                            agg_funcs.append(op)

        # If we have time-series data we want to
        # do historical counts over the last periods.
        if ts_data:
            logger.info(f"had time-series aggregations for {self}")
            for period in self.ts_periods:
                # count the number of identifiers in this period.
                delt = self.cut_date - datetime.timedelta(days=period)
                aggfunc = sqlop(
                    optype=SQLOpType.aggfunc,
                    opval=f"SUM(CASE WHEN {self.colabbr(self.date_key)} >= '{str(delt)}' then 1 else 0 end) as {self.prefix}_num_events_{period}d",
                )
                agg_funcs.append(aggfunc)

        if not len(agg_funcs):
            logger.info(f"No aggregations for {self}")
            return self.df
        agg = sqlop(optype=SQLOpType.agg, opval=f"{self.colabbr(reduce_key)}")
        # Need the aggregation and time-based filtering.
        tfilt = self.prep_for_features() if self.prep_for_features() else []

        return tfilt + agg_funcs + [agg]

    def sql_auto_labels(
        self,
        table_df_sample: typing.Union[pd.DataFrame, dd.DataFrame],
        reduce_key: str,
        type_func_map: dict = {},
    ) -> pd.DataFrame:
        """
        Pandas implementation of auto labeling based on
        provided columns.
        """
        agg_funcs = {}
        if not self._stypes:
            self._stypes = infer_df_stype(table_df_samp)
        for col, stype in self._stypes.items():
            if col.endswith("_label"):
                _type = str(stype)
                if type_func_map.get(_type):
                    for func in type_func_map[_type]:
                        col_new = f"{col}_{func}_label"
                        agg_funcs.append(
                            sqlop(
                                optype=SQLOpType.aggfunc,
                                opval=f"{func}" + f"({col}) as {col_new}",
                            )
                        )
        # Need the aggregation and time-based filtering.
        agg = sqlop(optype=SQLOpType.agg, opval=f"{self.colabbr(reduce_key)}")
        tfilt = self.prep_for_labels() if self.prep_for_labels() else []
        return tfilt + agg_funcs + [agg]

    def pandas_auto_labels(
        self, reduce_key: str, type_func_map: dict = {}
    ) -> pd.DataFrame:
        """
        Pandas implementation of auto labeling based on
        provided columns.
        """
        agg_funcs = {}

        for col, stype in self._stypes.items():
            _type = str(stype)
            if (
                col.endswith("_label")
                or col == self.label_field
                or col == f"{self.colabbr(self.label_field)}"
            ):
                if type_func_map.get(_type):
                    for func in type_func_map[_type]:
                        col_new = f"{col}_{func}_label"
                        agg_funcs[col_new] = pd.NamedAgg(column=col, aggfunc=func)
        return (
            self.prep_for_labels()
            .groupby(self.colabbr(reduce_key))
            .agg(**agg_funcs)
            .reset_index()
        )

    def daft_auto_labels(
        self,
        reduce_key: str,
        type_func_map: dict = {},
    ) -> pd.DataFrame:
        """
        Daft implementation of auto labeling based
        on provided columns.
        """
        # Temporary hack until a `daft` implementation
        # of window functions is available.  This will
        # also, unfortunately, limit us to single machine
        # data sizes with daft until then.
        original_df = self.df
        self.compute_layer = ComputeLayerEnum.pandas
        self.df = self.df.to_pandas()
        grouped = self.pandas_auto_labels(reduce_key, type_func_map=type_func_map)
        self.df = original_df
        self.compute_layer = ComputeLayerEnum.daft
        return grouped

    def dask_auto_labels(
        self,
        reduce_key: str,
        type_func_map: dict = {},
    ) -> dd.DataFrame:
        """
        Dask implementation of auto labeling based on
        provided columns.
        """
        agg_funcs = {}
        for col, stype in self._stypes.items():
            if col.endswith("_label"):
                _type = str(stype)
                if type_func_map.get(_type):
                    for func in type_func_map[_type]:
                        col_new = f"{col}_{func}_label"
                        agg_funcs[col_new] = pd.NamedAgg(column=col, aggfunc=func)
        return (
            self.prep_for_labels()
            .groupby(self.colabbr(reduce_key))
            .agg(**agg_funcs)
            .reset_index()
        )

    def spark_auto_labels(
        self,
        reduce_key: str,
        type_func_map: dict = {},
    ) -> pyspark.sql.DataFrame:
        """
        Spark implementation of auto labeling based on
        provided columns.
        """
        agg_funcs = []
        for col, stype in self._stypes.items():
            _type = str(stype)
            if col.endswith("_label"):
                if type_func_map.get(_type):
                    for func in type_func_map[_type]:
                        if func == "nunique":
                            func = "count_distinct"
                        col_new = f"{col}_{func}_label"
                        agg_funcs.append(getattr(F, func)(F.col(col)).alias(col_new))
        return self.prep_for_labels().groupby(self.colabbr(reduce_key)).agg(*agg_funcs)

    @abc.abstractmethod
    def do_reduce(self, reduce_key):
        """
        Reduce operation or the node

        Args
            reduce_key : key to use to perform the reduce operation
            children : list of children nodes
        """
        pass

    @abc.abstractmethod
    def do_labels(
        self,
        reduce_key: typing.Optional[str] = None,
    ):
        """
        Generate labels
        """
        pass

    def colabbr(self, col: str) -> str:
        return f"{self.prefix}_{col}"

    def compute_period_minutes(
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
            return (self.compute_period_val * 7) * 1440
        elif self.compute_period_unit == PeriodUnit.month:
            return (self.compute_period_val * 30.417) * 1440

    def label_period_minutes(
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
            return (self.label_period_val * 7) * 1440
        elif self.label_period_unit == PeriodUnit.month:
            return (self.label_period_val * 30.417) * 1440

    def prep_for_features(
        self, allow_null: bool = False
    ) -> typing.Union[
        pd.DataFrame, dd.DataFrame, pyspark.sql.dataframe.DataFrame, typing.List[sqlop]
    ]:
        """
        Prepare the dataset for feature aggregations / reduce
        """
        if self.date_key:
            if (
                self.cut_date
                and isinstance(self.cut_date, str)
                or isinstance(self.cut_date, datetime.datetime)
            ):
                # Using a SQL engine so need to return `sqlop` instances.
                if self.compute_layer in [
                    ComputeLayerEnum.sqlite,
                    ComputeLayerEnum.postgres,
                    ComputeLayerEnum.snowflake,
                    ComputeLayerEnum.redshift,
                    ComputeLayerEnum.mysql,
                    ComputeLayerEnum.athena,
                    ComputeLayerEnum.databricks,
                    ComputeLayerEnum.duckdb,
                ]:
                    return [
                        sqlop(
                            optype=SQLOpType.where,
                            opval=f"{self.colabbr(self.date_key)} < '{str(self.cut_date)}'",
                        ),
                        sqlop(
                            optype=SQLOpType.where,
                            opval=f"{self.colabbr(self.date_key)} > '{str(self.cut_date - datetime.timedelta(minutes=self.compute_period_minutes()))}'",
                        ),
                    ]

                elif isinstance(self.df, pd.DataFrame) or isinstance(
                    self.df, dd.DataFrame
                ):
                    return self.df[
                        (
                            (self.df[self.colabbr(self.date_key)] < str(self.cut_date))
                            & (
                                self.df[self.colabbr(self.date_key)]
                                > str(
                                    self.cut_date
                                    - datetime.timedelta(
                                        minutes=self.compute_period_minutes()
                                    )
                                )
                            )
                        )
                        | (self.df[self.colabbr(self.date_key)].isnull())
                    ]
                elif isinstance(self.df, pyspark.sql.dataframe.DataFrame):
                    return self.df.filter(
                        (
                            (self.df[self.colabbr(self.date_key)] < self.cut_date)
                            & (
                                self.df[self.colabbr(self.date_key)]
                                > (
                                    self.cut_date
                                    - datetime.timedelta(
                                        minutes=self.compute_period_minutes()
                                    )
                                )
                            )
                        )
                        | (self.df[self.colabbr(self.date_key)].isNull())
                    )
                elif isinstance(self.df, daft.dataframe.dataframe.DataFrame):
                    return self.df.filter(
                        (
                            (self.df[self.colabbr(self.date_key)] < str(self.cut_date))
                            & (
                                self.df[self.colabbr(self.date_key)]
                                > str(
                                    (
                                        self.cut_date
                                        - datetime.timedelta(
                                            minutes=self.compute_period_minutes()
                                        )
                                    )
                                )
                            )
                        )
                        | (self.df[self.colabbr(self.date_key)].is_null())
                    )

            else:
                # Using a SQL engine so need to return `sqlop` instances.
                if self.compute_layer in [
                    ComputeLayerEnum.sqlite,
                    ComputeLayerEnum.postgres,
                    ComputeLayerEnum.snowflake,
                    ComputeLayerEnum.redshift,
                    ComputeLayerEnum.mysql,
                    ComputeLayerEnum.athena,
                    ComputeLayerEnum.databricks,
                    ComputeLayerEnum.duckdb,
                ]:
                    return [
                        sqlop(
                            optype=SQLOpType.where,
                            opval=f"{self.colabbr(self.date_key)} < '{str(datetime.datetime.now())}'",
                        ),
                        sqlop(
                            optype=SQLOpType.where,
                            opval=f"{self.colabbr(self.date_key)} > '{str(datetime.datetime.now() - datetime.timedelta(minutes=self.compute_period_minutes()))}'",
                        ),
                    ]

                elif isinstance(self.df, pd.DataFrame) or isinstance(
                    self.df, dd.DataFrame
                ):
                    return self.df[
                        (
                            (
                                self.df[self.colabbr(self.date_key)]
                                < datetime.datetime.now()
                            )
                            & (
                                self.df[self.colabbr(self.date_key)]
                                > (
                                    datetime.datetime.now()
                                    - datetime.timedelta(
                                        minutes=self.compute_period_minutes()
                                    )
                                )
                            )
                        )
                        | (self.df[self.colabbr(self.date_key)].isnull())
                    ]
                elif isinstance(self.df, pyspark.sql.dataframe.DataFrame):
                    return self.df.filter(
                        (
                            self.df[self.colabbr(self.date_key)]
                            > (
                                datetime.datetime.now()
                                - datetime.timedelta(
                                    minutes=self.compute_period_minutes()
                                )
                            )
                        )
                        | (self.df[self.colabbr(self.date_key)].isNull())
                    )
                elif isinstance(self.df, daft.dataframe.dataframe.DataFrame):
                    return self.df.filter(
                        (
                            self.df[self.colabbr(self.date_key)]
                            > (
                                datetime.datetime.now()
                                - datetime.timedelta(
                                    minutes=self.compute_period_minutes()
                                )
                            )
                        )
                        | (self.df[self.colabbr(self.date_key)].is_null())
                    )

        # SQL engine do nothing.
        elif not hasattr(self, "df"):
            return None
        # no-op
        return self.df

    def prep_for_labels(
        self,
    ) -> typing.Union[pd.DataFrame, dd.DataFrame, pyspark.sql.dataframe.DataFrame]:
        """
        Prepare the dataset for labels
        """
        if self.date_key:
            if (
                self.cut_date
                and isinstance(self.cut_date, str)
                or isinstance(self.cut_date, datetime.datetime)
            ):
                # Using a SQL engine so need to return `sqlop` instances.
                if self.compute_layer in [
                    ComputeLayerEnum.sqlite,
                    ComputeLayerEnum.postgres,
                    ComputeLayerEnum.snowflake,
                    ComputeLayerEnum.redshift,
                    ComputeLayerEnum.mysql,
                    ComputeLayerEnum.athena,
                    ComputeLayerEnum.databricks,
                    ComputeLayerEnum.duckdb,
                ]:
                    return [
                        sqlop(
                            optype=SQLOpType.where,
                            opval=f"{self.colabbr(self.date_key)} > '{str(self.cut_date)}'",
                        ),
                        sqlop(
                            optype=SQLOpType.where,
                            opval=f"{self.colabbr(self.date_key)} < '{str(self.cut_date + datetime.timedelta(minutes=self.label_period_minutes()))}'",
                        ),
                    ]

                elif isinstance(self.df, pd.DataFrame):
                    return self.df[
                        (self.df[self.colabbr(self.date_key)] > str(self.cut_date))
                        & (
                            self.df[self.colabbr(self.date_key)]
                            < str(
                                self.cut_date
                                + datetime.timedelta(
                                    minutes=self.label_period_minutes()
                                )
                            )
                        )
                    ]
                elif isinstance(self.df, pyspark.sql.dataframe.DataFrame):
                    return self.df.filter(
                        (self.df[self.colabbr(self.date_key)] > str(self.cut_date))
                        & (
                            self.df[self.colabbr(self.date_key)]
                            < str(
                                self.cut_date
                                + datetime.timedelta(
                                    minutes=self.label_period_minutes()
                                )
                            )
                        )
                    )
                elif isinstance(self.df, daft.dataframe.dataframe.DataFrame):
                    return self.df.filter(
                        (self.df[self.colabbr(self.date_key)] > str(self.cut_date))
                        & (
                            self.df[self.colabbr(self.date_key)]
                            < str(
                                self.cut_date
                                + datetime.timedelta(
                                    minutes=self.label_period_minutes()
                                )
                            )
                        )
                    )
            else:
                # Using a SQL engine so need to return `sqlop` instances.
                if self.compute_layer in [
                    ComputeLayerEnum.sqlite,
                    ComputeLayerEnum.postgres,
                    ComputeLayerEnum.snowflake,
                    ComputeLayerEnum.redshift,
                    ComputeLayerEnum.mysql,
                    ComputeLayerEnum.athena,
                    ComputeLayerEnum.databricks,
                ]:
                    return [
                        sqlop(
                            optype=SQLOpType.where,
                            opval=f"{self.colabbr(self.date_key)} > '{str(datetime.datetime.now() - datetime.timedelta(minutes=self.label_period_minutes()))}'",
                        )
                    ]

                elif isinstance(self.df, pd.DataFrame):
                    return self.df[
                        self.df[self.colabbr(self.date_key)]
                        > (
                            datetime.datetime.now()
                            - datetime.timedelta(minutes=self.label_period_minutes())
                        )
                    ]
                elif isinstance(self.df, pyspark.sql.dataframe.DataFrame):
                    return self.df.filter(
                        self.df[self.colabbr(self.date_key)]
                        > (
                            datetime.datetime.now()
                            - datetime.timedelta(minutes=self.label_period_minutes())
                        )
                    )
                elif isinstance(self.df, daft.dataframe.dataframe.DataFrame):
                    return self.df.filter(
                        self.df[self.colabbr(self.date_key)]
                        > (
                            datetime.datetime.now()
                            - datetime.timdelta(minutes=self.label_period_minutes())
                        )
                    )

        elif not hasattr(self, "df"):
            return None
        # no-op
        return self.df

    def default_label(
        self,
        op: typing.Union[str, callable],
        field: str,
        reduce_key: typing.Optional[str] = None,
    ) -> typing.Union[
        pd.DataFrame, dd.DataFrame, pyspark.sql.dataframe.DataFrame, typing.List[sqlop]
    ]:
        """
        Default label operation.

                Arguments
                ----------
                op: operation to call for label
                field: str label field to call operation on
                reduce: bool whether or not to reduce
        """
        if hasattr(self, "df"):
            if (
                self.compute_layer in [ComputeLayerEnum.pandas, ComputeLayerEnum.dask]
                and self.colabbr(field) in self.df.columns
            ):
                if self.reduce:
                    if callable(op):
                        return (
                            self.prep_for_labels()
                            .groupby(self.colabbr(reduce_key))
                            .agg(
                                **{
                                    self.colabbr(field + "_label"): pd.NamedAgg(
                                        column=self.colabbr(field), aggfunc=op
                                    )
                                }
                            )
                            .reset_index()
                        )
                    else:
                        if op == "bool":
                            grp = (
                                self.prep_for_labels()
                                .groupby(self.colabbr(reduce_key))
                                .agg(
                                    **{
                                        self.colabbr(field + "_label"): pd.NamedAgg(
                                            column=self.colabbr(field), aggfunc="count"
                                        )
                                    }
                                )
                                .reset_index()
                            )
                            grp[f"{self.colabbr(field)}_label"] = grp[
                                f"{self.colabbr(field)}_label"
                            ].apply(lambda x: 1 if x >= 1 else 0)
                            return grp
                        return (
                            self.prep_for_labels()
                            .groupby(self.colabbr(reduce_key))
                            .agg(
                                **{
                                    self.colabbr(field + "_label"): pd.NamedAgg(
                                        column=self.colabbr(field), aggfunc=op
                                    )
                                }
                            )
                            .reset_index()
                        )

                else:
                    label_df = self.prep_for_labels()
                    if callable(op):
                        label_df[self.colabbr(field) + "_label"] = label_df[
                            self.colabbr(field)
                        ].apply(op)
                    else:
                        label_df[self.colabbr(field) + "_label"] = label_df[
                            self.colabbr(field)
                        ].apply(lambda x: getattr(x, op)())
                    return label_df[
                        [self.colabbr(self.pk), self.colabbr(field) + "_label"]
                    ]

            elif (
                self.compute_layer == ComputeLayerEnum.spark
                and self.colabbr(field) in self.df.columns
            ):
                if self.reduce:
                    return (
                        self.prep_for_labels()
                        .groupBy(self.colabbr(reduce_key))
                        .agg(
                            getattr(F, op)(F.col(self.colabbr(field))).alias(
                                f"{self.colabbr(field)}_label"
                            )
                        )
                    )
                else:
                    pass
            elif (
                self.compute_layer == ComputeLayerEnum.daft
                and self.colabbr(field) in self.df.column_names
            ):
                if self.reduce:
                    aggcol = daft.col(self.colabbr(field))
                    return (
                        self.prep_for_labels()
                        .groupby(self.colabbr(reduce_key))
                        .agg(
                            getattr(aggcol, op)().alias(f"{self.colabbr(field)}_label")
                        )
                    )
                else:
                    pass
        elif self.compute_layer in [
            ComputeLayerEnum.snowflake,
            ComputeLayerEnum.sqlite,
            ComputeLayerEnum.mysql,
            ComputeLayerEnum.postgres,
            ComputeLayerEnum.redshift,
            ComputeLayerEnum.athena,
            ComputeLayerEnum.databricks,
            ComputeLayerEnum.duckdb,
        ]:
            if self.reduce:
                if op == "bool":
                    label_query = self.prep_for_labels() + [
                        sqlop(
                            optype=SQLOpType.agg, opval=f"{self.colabbr(reduce_key)}"
                        ),
                        sqlop(
                            optype=SQLOpType.aggfunc,
                            opval=f"CASE WHEN COUNT({self.colabbr(field)}) >= 1 THEN 1 ELSE 0 END as {self.colabbr(field)}_label",
                        ),
                    ]
                else:
                    label_query = self.prep_for_labels() + [
                        sqlop(
                            optype=SQLOpType.agg, opval=f"{self.colabbr(reduce_key)}"
                        ),
                        sqlop(
                            optype=SQLOpType.aggfunc,
                            opval=f"{op}"
                            + f"({self.colabbr(field)}) as {self.colabbr(field)}_label",
                        ),
                    ]
                logger.info(self.build_query(label_query))
                return label_query
            else:
                label_query = self.prep_for_labels() + [
                    sqlop(
                        optype=SQLOpType.select,
                        opval=f"{op}"
                        + f"({self.colabbr(field)}) as {self.colabbr(field)}_label",
                    )
                ]
                logger.info(self.build_query(label_query))
                return label_query
        else:
            pass

    def online_features(
        self,
    ):
        """
        Define online features.
        """
        pass

    def on_demand_features(
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

    def __init__(
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


class GraphReduceQueryException(Exception):
    pass


class SQLNode(GraphReduceNode):
    """
    Base node for SQL engines.  Makes some common
    operations available.  This class should be
    extended for engines that do not conform
    to a single `client` interface, such as
    AWS Athena, which requires additional params.

    Subclasses should simply extend the `SQLNode` interface:
    """

    FUNCTION_MAPPING = {
        "mean": "avg",
        "median": None,
        "nunique": None,
    }

    def __init__(
        self,
        *args,
        client: typing.Any = None,
        lazy_execution: bool = False,
        dry_run: bool = False,
        do_annotate_ops: typing.Optional[typing.List[sqlop]] = None,
        do_filters_ops: typing.Optional[typing.List[sqlop]] = None,
        do_reduce_ops: typing.Optional[typing.List[sqlop]] = None,
        do_labels_ops: typing.Optional[typing.List[sqlop]] = None,
        do_post_join_annotate_ops: typing.Optional[typing.List[sqlop]] = None,
        do_post_join_filters_ops: typing.Optional[typing.List[sqlop]] = None,
        do_post_join_annotate_requires: typing.Optional[
            typing.List[GraphReduceNode]
        ] = None,
        do_post_join_filters_requires: typing.Optional[
            typing.List[GraphReduceNode]
        ] = None,
        **kwargs,
    ):
        """
        Constructor for `SQLNode`.

                Arguments
                -------------
                client: a SQL engine client that can `.execute()` sql queries
                lazy_execution: bool whether or not to execute lazily
                dry_run: don't execute but log the SQL that would be executed
                do_annotate_ops: list of `sqlop` instances for `do_annotate`
                do_filters_ops: list of `sqlop` instances for `do_filters`
                do_reduce_ops: list of `sqlop` instances for `do_reduce`
                do_labels_ops: list of `sqlop` instances for `do_labels`
                do_post_join_annotate_ops: list of `sqlop` instances for `do_post_join_annotate`
                do_post_join_filters_ops: list of `sqlop` instances for `do_post_join_filters`
                do_post_join_annotate_requires: list of `SQLNode` instances that this method requires be joined prior to executing
                do_post_join_filters_requires: list of `SQLNode` instances that this method requires be joined priot to executing
        """
        self._sql_client = client
        self.lazy_execution = lazy_execution
        self.dry_run = dry_run

        # The current data ref.
        self._cur_data_ref = None
        # A place to store temporary tables or views.
        self._temp_refs = {}
        self._all_refs = []
        self._removed_refs = []
        # A place to store the full SQL for creating temp references.
        # only ever store the current `_ref_sql`.
        self._ref_sql = None

        self.do_annotate_ops = do_annotate_ops
        self.do_filters_ops = do_filters_ops
        self.do_reduce_ops = do_reduce_ops
        self.do_labels_ops = do_labels_ops
        self.do_post_join_annotate_ops = do_post_join_annotate_ops
        self.do_post_join_filters_ops = do_post_join_filters_ops
        self.do_post_join_annotate_requires = do_post_join_annotate_requires
        self.do_post_join_filters_requires = do_post_join_filters_requires

        # SQL operations for this node.
        self.sql_ops = []

        super().__init__(*args, **kwargs)

    def get_temp_refs(self):
        return self._temp_refs

    def _clean_refs(self):
        """
        Cleanup tables created during execution.
        """
        for k, v in self._all_refs:
            if v not in self._removed_refs:
                try:
                    sql = f"DROP VIEW {v}"
                    self.execute_query(sql)
                    self._removed_refs.append(v)
                    logger.info(f"dropped {v}")
                except Exception as e:
                    continue

    def get_ref_name(
        self,
        fn: typing.Union[callable, str] = None,
        lookup: bool = False,
        schema: str = None,
    ) -> str:
        """
        Get a reference name for the function.
        """
        func_name = fn if isinstance(fn, str) else fn.__name__
        # If this ref name is already in
        # the _temp_refs dict create a new ref.

        # If the node has a `.table_name`
        # which is the case for `duckdb`
        # then use that.
        if hasattr(self, "table_name") and self.table_name:
            fpath = self.table_name
        # IF there is a schema in the fpath
        # we need to remove the schema.
        elif "." in self.fpath and "/" not in self.fpath:
            fpath = self.fpath.split(".")[-1]
        else:
            fpath = self.fpath

        if schema:
            ref_name = f"{schema}.{fpath}_{func_name}_grtemp"
        else:
            ref_name = f"{fpath}_{func_name}_grtemp"
        if self._temp_refs.get(func_name):
            if lookup:
                return self._temp_refs[func_name]

            if ref_name in self._all_refs:
                i = 1
                while ref_name in self._all_refs:
                    ref_name = ref_name + str(i)
                    i += 1
            return ref_name

        else:
            if ref_name in self._all_refs:
                i = 1
                while ref_name in self._all_refs:
                    ref_name = ref_name + str(i)
                    i += 1
            return ref_name

    def create_ref(
        self,
        sql: str = "",
        fn: typing.Union[callable, str] = None,
        # Overwrite the
        overwrite: bool = False,
        schema: str = None,
        dry: bool = False,
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
            ref_name = self.get_ref_name(fn, schema=schema)
            self._all_refs.append(ref_name)
            self.create_temp_view(sql, ref_name, dry=dry)
            self._temp_refs[fn] = ref_name
            return ref_name
        # Reference for this method already created
        # so we will just retrieve.
        else:
            return self._temp_refs[fn]

    def get_current_ref(self) -> str:
        """
        Returns the name of the current
        reference to the nodes data.
        """
        if not self._cur_data_ref:
            return self.fpath
        else:
            return self._cur_data_ref

    def create_temp_view(
        self,
        qry: str,
        view_name: str,
        overwrite: bool = False,
        dry: bool = False,
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
            self._ref_sql = sql
            # Only execute when it is not a dry run
            # but always append the SQL.
            if not dry:
                self.execute_query(sql, ret_df=False)
            self._cur_data_ref = view_name
            return view_name
        except Exception as e:
            logger.error(e)
            return None

    # TODO(wes): optimize by storing previously
    # fetch samples.
    def get_sample(
        self,
        n: int = 10000,
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
                table=self._cur_data_ref if self._cur_data_ref else self.fpath, n=n
            )
        else:
            qry = samp_query.format(table=table, n=n)
        samp = self.execute_query(qry)
        if not self._stypes:
            self._stypes = infer_df_stype(samp)
        return samp

    def build_query(
        self,
        ops: typing.Union[typing.List[sqlop], sqlop],
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

        # Custom ops are returned as is.
        if isinstance(ops, sqlop) and ops.optype == SQLOpType.custom:
            return ops.opval

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
        from_ = sqlop(optype=SQLOpType.from_, opval=dr)

        # Boolean if this is an aggregation function.
        if len([_x for _x in ops if _x.optype == SQLOpType.agg]):
            aggfuncs = [_x for _x in ops if _x.optype == SQLOpType.aggfunc]
            if not len(aggfuncs):
                raise GraphReduceQueryException(
                    "Aggregation queries must have at least 1 `sqlop` of type SQLOpType.aggfunc"
                )
            aggfuncs = ",".join([_x.opval for _x in aggfuncs])

            # Can only be one aggregation per query build.
            agg = [_x for _x in ops if _x.optype == SQLOpType.agg][0].opval

            wheres = [_x for _x in ops if _x.optype == SQLOpType.where]
            if not len(wheres):
                wheres = "true"
            else:
                wheres = " and ".join([_x.opval for _x in wheres])

            return group_anatomy.format(
                selects=agg,
                aggfuncs=aggfuncs,
                from_=from_.opval,
                group=agg,
                wheres=wheres,
            )

        # Otherwise go with standard select anatomy.
        else:
            selects = [_x for _x in ops if _x.optype == SQLOpType.select]
            # IF not select statements, select *.
            if not len(selects):
                selects = [sqlop(optype=SQLOpType.select, opval="*")]

            qry_selects = ",".join([_x.opval for _x in selects])
            # froms = [_x for _x in ops if _x.optype == SQLOpType.from_][0].opval
            wheres = [_x for _x in ops if _x.optype == SQLOpType.where]
            if len(wheres):
                qry_wheres = " and ".join([_x.opval for _x in wheres])
            else:
                qry_wheres = "true"

            return select_anatomy.format(
                selects=qry_selects, from_=from_.opval, wheres=qry_wheres
            )

    def get_client(self) -> typing.Any:
        return self._sql_client

    def execute_query(
        self,
        qry: str,
        ret_df: bool = True,
        commit: bool = False,
    ) -> typing.Optional[typing.Union[None, pd.DataFrame]]:
        """
        Execute a query and get back a dataframe.
        """

        client = self.get_client()
        if not ret_df:
            cur = client.cursor()
            cur.execute(qry)
            if commit:
                client.commit()
        else:
            return pd.read_sql_query(qry, client)

    def do_data(self) -> typing.Union[sqlop, typing.List[sqlop]]:
        """
        Load the data.
        """

        col_renames = [f"{col} as {self.colabbr(col)}" for col in self.columns]
        sel = sqlop(optype=SQLOpType.select, opval=f"{','.join(col_renames)}")
        return [sel]

    def do_annotate(self) -> typing.Union[sqlop, typing.List[sqlop]]:
        """
        Should return a list of SQL statements
        casting columns as different types.
        """
        if self.do_annotate_ops:
            return self.do_annotate_ops
        return None

    def do_normalize(self) -> typing.Union[sqlop, typing.List[sqlop]]:
        return None

    def do_filters(self) -> typing.Union[sqlop, typing.List[sqlop]]:
        """
        Example:
        return [
            sqlop(optype=SQLOpType.where, opval=f"{self.colabbr('id')} < 1000"),
        ]
        """
        if self.do_filters_ops:
            return self.do_filters_ops
        return None

    # Returns aggregate functions
    # Returns aggregate
    def do_reduce(self, reduce_key) -> typing.Union[sqlop, typing.List[sqlop]]:
        """
        Example:
        return [
            sqlop(optype=SQLOpType.aggfunc, opval=f"count(*) as {self.colabbr('num_dupes')}"),
            sqlop(optype=SQLOpType.agg, opval=f"{self.colabbr(reduce_key)}")
        ]
        """
        if self.do_reduce_ops:
            return self.do_reduce_ops
        return None

    def do_post_join_annotate(self) -> typing.Union[sqlop, typing.List[sqlop]]:
        if self.do_post_join_annotate_ops:
            if self.do_post_join_annotate_requires:
                # Check if all required nodes are merged.
                samp = self.get_sample(n=100)
                all_merged = True
                for node in self.do_post_join_annotate_requires:
                    if not len(
                        [c for c in samp.columns if c.startswith(f"{node.prefix}_")]
                    ):
                        all_merged = False
                        logger.debug(f"All dependencies not merged")
                if all_merged:
                    logger.debug(f"All dependencies merged")
                    return self.do_post_join_annotate_ops
                else:
                    return None
            else:
                return self.do_post_join_annotate_ops
        return None

    def do_labels(self, reduce_key: str) -> typing.Union[sqlop, typing.List[sqlop]]:
        if self.do_labels_ops:
            return self.do_labels_ops
        return None

    def do_post_join_filters(self) -> typing.Union[sqlop, typing.List[sqlop]]:
        if self.do_post_join_filters_ops:
            if self.do_post_join_filters_requires:
                # Check if all required nodes are merged.
                samp = self.get_sample(n=100)
                all_merged = True
                for node in self.do_post_join_filters_requires:
                    if not len(
                        [c for c in samp.columns if c.startswith(f"{node.prefix}_")]
                    ):
                        all_merged = False
                        logger.debug(f"All dependencies not merged")
                if all_merged:
                    logger.debug(f"All dependencies merged")
                    return self.do_post_join_filters_ops
                else:
                    return None
            else:
                return self.do_post_join_filters_ops

            return self.do_post_join_filters_ops
        return None

    def do_sql(self) -> str:
        """
        One function to compute this entire node.
        """
        pass


class AthenaNode(SQLNode):
    def __init__(
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
                sqlop(
                    optype=SQLOpType.where,
                    opval=f"{self.colabbr(self.date_key)} > timestamp '{str(self.cut_date - datetime.timedelta(minutes=self.compute_period_minutes()))}'",
                ),
                sqlop(
                    optype=SQLOpType.where,
                    opval=f"{self.colabbr(self.date_key)} < timestamp '{str(self.cut_date)}'",
                ),
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
                sqlop(
                    optype=SQLOpType.where,
                    opval=f"{self.colabbr(self.date_key)} < timestamp '{str(self.cut_date + datetime.timedelta(minutes=self.label_period_minutes()))}'",
                ),
                sqlop(
                    optype=SQLOpType.where,
                    opval=f"{self.colabbr(self.date_key)} > timestamp '{str(self.cut_date)}'",
                ),
            ]
        else:
            # do nothing
            return None

    def execute_query(
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
            ResultConfiguration={"OutputLocation": self.s3_output_location},
        )
        # While query is executing sleep
        qry_id = resp["QueryExecutionId"]

        qry_status = client.get_query_execution(QueryExecutionId=qry_id)
        if qry_status["QueryExecution"]["Status"]["State"] == "FAILED":
            raise Exception(f"Query {qry} FAILED")

        else:
            while qry_status["QueryExecution"]["Status"]["State"] not in [
                "SUCCEEDED",
                "FAILED",
            ]:
                logger.info("sleeping and waiting for query to finish")
                time.sleep(1)
                qry_status = client.get_query_execution(QueryExecutionId=qry_id)

        if qry_status["QueryExecution"]["Status"]["State"] == "FAILED":
            logger.error("query FAILED")
            print(qry_status)
            return None

        results = client.get_query_results(QueryExecutionId=qry_id)
        colinfo = results["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]
        rows = results["ResultSet"]["Rows"]
        while results.get("NextToken"):
            results = client.get_query_results(
                QueryExecutionId=qry_id, NextToken=results["NextToken"]
            )
            for row in results["ResultSet"]["Rows"]:
                rows.append(row)
        # create a dataframe ready version of the data
        dfdata = []
        for row in rows[1:]:
            newrow = {}
            for i in range(len(row["Data"])):
                col = colinfo[i]["Name"]
                if row["Data"][i]:
                    val = row["Data"][i]["VarCharValue"]
                else:
                    val = None
                newrow.update({col: val})
            dfdata.append(newrow)
        return pd.DataFrame(dfdata)


class DatabricksNode(SQLNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_temp_view(
        self,
        qry: str,
        view_name: str,
        dry: bool = False,
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
            self._ref_sql = sql
            if not dry:
                self.execute_query(sql, ret_df=False)
            self._cur_data_ref = view_name
        except Exception as e:
            logger.error(e)
            return None


class RedshiftNode(SQLNode):
    def __init__(self, *args, **kwargs):
        """
        Constructor.
        """
        super().__init__(*args, **kwargs)

    def _clean_refs(self):
        for k, v in self._temp_refs.items():
            if v not in self._removed_refs:
                sql = f"DROP TABLE IF EXISTS {v}"
                self.execute_query(sql, ret_df=False, commit=True)
                self._removed_refs.append(v)
                logger.info(f"dropped {v}")

    def create_temp_view(
        self,
        qry: str,
        view_name: str,
        dry: bool = False,
    ) -> str:
        """
        Create a view with the results
        of the query.
        """
        # try:
        self.execute_query(
            f"DROP TABLE IF EXISTS {view_name}", ret_df=False, commit=True
        )
        sql = f"""
            CREATE TABLE {view_name}
            AS {qry}
            """
        self._ref_sql = sql
        if not dry:
            self.execute_query(sql, ret_df=False, commit=True)
        self._cur_data_ref = view_name
        return view_name
        # except Exception as e:
        #    logger.error(e)
        #    return None

    def sql_auto_features(
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

        # Always need to update this
        # because we never know if
        # the original columns comprise all
        # of the columns currently in the df.
        self._stypes = infer_df_stype(table_df_sample)

        # Physical types.
        ptypes = {col: str(t) for col, t in table_df_sample.dtypes.to_dict().items()}

        ts_data = self.is_ts_data(reduce_key)
        for col, stype in self._stypes.items():
            _type = str(stype)
            if self._is_identifier(col) and col != reduce_key:
                # We only perform counts for identifiers.
                func = "count"
                col_new = f"{col}_{func}"
                agg_funcs.append(
                    sqlop(
                        optype=SQLOpType.aggfunc,
                        opval=f"{func}" + f"({col}) as {col_new}",
                    )
                )

            elif self._is_identifier(col) and col == reduce_key:
                continue
            elif type_func_map.get(_type):
                if ptypes[col] == "bool":
                    col_new = f"{col}_sum"
                    op = sqlop(
                        optype=SQLOpType.aggfunc,
                        opval=f"sum(case when {col} = 1 then 1 else 0 end) as {col_new}",
                    )
                    if op not in agg_funcs:
                        agg_funcs.append(op)
                    continue
                for func in type_func_map[_type]:
                    # There should be a better top-level mapping
                    # but for now this will do.  SQL engines typically
                    # don't have 'median' and 'mean'.  'mean' is typically
                    # just called 'avg'.
                    if (
                        (_type == "numerical" or "timestamp")
                        and dict(table_df_sample)[col].__str__() == "object"
                        and func in ["min", "max", "mean", "median"]
                    ):
                        logger.info(
                            f"skipped aggregation on {col} because semantic numerical but physical object"
                        )
                        continue
                    elif func in self.FUNCTION_MAPPING:
                        func = self.FUNCTION_MAPPING.get(func)

                    elif not func or func == "nunique":
                        continue

                    # Redshift-specific
                    elif func == "first":
                        func = "any_value"

                    # For categorical types check if it is
                    # a 0, 1 category and, if so, do a sum.
                    if (
                        _type == "categorical"
                        and len(table_df_sample[col].unique()) <= 2
                        and str(table_df_sample[col].head().values[0]).isdigit()
                    ):
                        func = "sum"

                    if func:
                        col_new = f"{col}_{func}"
                        op = sqlop(
                            optype=SQLOpType.aggfunc,
                            opval=f"{func}" + f"({col}) as {col_new}",
                        )
                        if op not in agg_funcs:
                            agg_funcs.append(op)

        # If we have time-series data we want to
        # do historical counts over the last periods.
        if ts_data:
            # Get the min and max dates.
            logger.info(f"had time-series aggregations for {self}")
            # Get the time since the last event.
            for period in self.ts_periods:
                # count the number of identifiers in this period.
                delt = self.cut_date - datetime.timedelta(days=period)
                aggfunc = sqlop(
                    optype=SQLOpType.aggfunc,
                    opval=f"SUM(CASE WHEN {self.colabbr(self.date_key)} >= '{str(delt)}' then 1 else 0 end) as {self.prefix}_num_events_{period}d",
                )
                agg_funcs.append(aggfunc)

        if not len(agg_funcs):
            logger.info(f"No aggregations for {self}")
            return self.df
        agg = sqlop(optype=SQLOpType.agg, opval=f"{self.colabbr(reduce_key)}")
        # Need the aggregation and time-based filtering.
        tfilt = self.prep_for_features() if self.prep_for_features() else []

        return tfilt + agg_funcs + [agg]

    def sql_auto_labels(
        self,
        table_df_sample: typing.Union[pd.DataFrame, dd.DataFrame],
        reduce_key: str,
        type_func_map: dict = {},
    ) -> pd.DataFrame:
        """
        Pandas implementation of auto labeling based on
        provided columns.
        """
        agg_funcs = {}
        if not self._stypes:
            self._stypes = infer_df_stype(table_df_samp)
        for col, stype in self._stypes.items():
            if col.endswith("_label"):
                _type = str(stype)
                if type_func_map.get(_type):
                    for func in type_func_map[_type]:
                        if func == "first":
                            func = "any_value"
                        col_new = f"{col}_{func}_label"
                        agg_funcs.append(
                            sqlop(
                                optype=SQLOpType.aggfunc,
                                opval=f"{func}" + f"({col}) as {col_new}",
                            )
                        )
        # Need the aggregation and time-based filtering.
        agg = sqlop(optype=SQLOpType.agg, opval=f"{self.colabbr(reduce_key)}")
        tfilt = self.prep_for_labels() if self.prep_for_labels() else []
        return tfilt + agg_funcs + [agg]


class SnowflakeNode(SQLNode):
    def __init__(self, *args, **kwargs):
        """
        Constructor.
        """
        super().__init__(*args, **kwargs)
        # Use an available database.

    def _clean_refs(self):
        # Get all views and find the ones
        # in temp refs that are still active.
        views = self.execute_query("show views")
        active_views = [row["name"] for ix, row in views.iterrows()]
        for k, v in self._temp_refs.items():
            if v not in self._removed_refs and v in active_views:
                sql = f"DROP VIEW {v}"
                self.execute_query(sql, ret_df=False)
                self._removed_refs.append(v)
                logger.info(f"dropped {v}")

    def use_db(
        self,
        db: str,
    ) -> bool:
        try:
            res = self.execute_query(f"use database {db}")
            return True
        except Exception as e:
            return False

    def create_temp_view(
        self,
        qry: str,
        view_name: str,
        dry: bool = False,
    ) -> str:
        """
        Create a view with the results
        of the query.
        """
        try:
            sql = f"""
            CREATE VIEW {view_name}
            AS {qry}
            """
            self._ref_sql = sql
            if not dry:
                self.execute_query(sql, ret_df=False)
            self._cur_data_ref = view_name
            return view_name
        except Exception as e:
            logger.error(e)
            return None


class DuckdbNode(SQLNode):
    def __init__(self, *args, table_name: str = None, **kwargs):
        """
        ...SQLNode

                Arguments:
                ----------
                table_name: (optional) str of table name to use if `fpath` is filesystem
        """
        super().__init__(*args, **kwargs)
        if "/" in self.fpath and not table_name:
            raise Exception("parameter 'table_name' must be set for duckdb files")
        elif "/" in self.fpath and table_name:
            self.table_name = table_name
        else:
            self.table_name = None

    def execute_query(
        self,
        qry: str,
        ret_df: bool = True,
        commit: bool = False,
    ) -> typing.Optional[typing.Union[None, pd.DataFrame]]:
        """
        Execute a query and get back a dataframe.
        """

        if not ret_df:
            self.get_client().sql(qry)
        else:
            return self.get_client().sql(qry).to_df()

    def _clean_refs(self):
        # Get all views and find the ones
        # in temp refs that are still active.
        temp_tables = duckdb.sql("""
        SELECT * FROM temp.sqlite_master;
        """)
        active_tables = list(temp_tables.to_df()["name"])
        for k, v in self._temp_refs.items():
            if v not in self._removed_refs and v in active_tables:
                sql = f"DROP TEMP TABLE {v}"
                self.execute_query(sql, ret_df=False)
                self._removed_refs.append(v)
                logger.info(f"dropped {v}")

    def create_temp_view(
        self,
        qry: str,
        view_name: str,
        dry: bool = False,
    ) -> str:
        """
        Create a view with the results
        of the query.
        """
        try:
            sql = f"""
            CREATE OR REPLACE TEMP TABLE {view_name}
            AS {qry}
            """
            self._ref_sql = sql
            if not dry:
                self.execute_query(sql, ret_df=False)
            self._cur_data_ref = view_name
            return view_name
        except Exception as e:
            logger.exception(e)
            return None
