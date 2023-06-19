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
