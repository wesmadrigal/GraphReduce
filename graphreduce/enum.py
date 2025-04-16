#!/usr/bin/env

import enum

class PeriodUnit(enum.Enum):
    second = 'second'
    minute = 'minute'
    hour = 'hour'
    day = 'day'
    week = 'week'
    month = 'month'
    year = 'year'

class ComputeLayerEnum(enum.Enum):
    # File-based compute layers.
    pandas = 'pandas'
    polars = 'polars'
    dask = 'dask'
    spark = 'spark'
    ray = 'ray'
    # SQL dialects.
    athena = 'athena'
    snowflake = 'snowflake'
    redshift = 'redshift'
    postgres = 'postgres'
    mysql = 'mysql'
    sqlite = 'sqlite'
    databricks = 'databricks'
    daft = 'daft'
    duckdb = 'duckdb'


class StorageFormatEnum(enum.Enum):
    csv = 'csv'
    parquet = 'parquet'
    tsv = 'tsv'
    delta = 'delta'
    iceberg = 'iceberg'


class ProviderEnum(enum.Enum):
    local = 'local'
    s3 = 's3'
    blob = 'blob'
    gcs = 'gcs'
    hdfs = 'hdfs'
    databricks = 'databricks'


class SQLOpType(enum.Enum):
    # All things related to selecting
    # data get housed under this op type.
    # Even case statements, if in the select
    # part of the query, should go here.
    select = 'select'
    from_ = 'from'
    # All aspects of where clauses use this.
    where = 'where'
    # The anatomy of a given aggregation
    # typically consists of a select
    # AND a group by statement.
    # This op should be used only for the
    # actual grouping portion and a separate
    # `select` op type should be used
    # for the columns and aggregation functions.
    agg = 'group by'
    aggfunc = 'aggfunc'
    order = 'order by'
    having = 'having'
    window = 'window'
    custom = 'custom'
