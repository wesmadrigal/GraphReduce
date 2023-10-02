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
    pandas = 'pandas'
    dask = 'dask'
    spark = 'spark'


class StorageFormatEnum(enum.Enum):
    csv = 'csv'
    parquet = 'parquet'
    tsv = 'tsv'
    delta = 'delta'

class ProviderEnum(enum.Enum):
    local = 'local'
    s3 = 's3'
    blog = 'blob'
    gcs = 'gcs'
    hdfs = 'hdfs'
