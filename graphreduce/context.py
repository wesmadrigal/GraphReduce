#!/usr/bin/env python
from __future__ import annotations
"""A module for helping with context
in GraphReduce compute graphs.
"""

# standard library
import typing

# third party
import pandas as pd
import dask.dataframe as dd
from structlog import get_logger
try:
    import pyspark
except Exception:  # pragma: no cover - optional dependency
    pyspark = None

# internal
from graphreduce.node import GraphReduceNode


logger = get_logger('graphreduce.context')
valid_dataframe_classes = [pd.DataFrame, dd.DataFrame]
if pyspark is not None:
    for spark_df_type in [
        getattr(getattr(getattr(pyspark, "sql", None), "dataframe", None), "DataFrame", None),
        getattr(
            getattr(getattr(getattr(pyspark, "sql", None), "connect", None), "dataframe", None),
            "DataFrame",
            None,
        ),
    ]:
        if spark_df_type is not None:
            valid_dataframe_classes.append(spark_df_type)
valid_dataframe_classes = tuple(valid_dataframe_classes)


def method_requires (
        nodes: typing.List[GraphReduceNode] = [],
        checkpoint: bool = False,
        ) -> callable:
    """
A decorator for ensuring the function
only runs when the calling Node has
a merged edge to the required `nodes` list.

Arguments
    nodes: list of GraphReduceNode classes to require
           for function execution

    checkpoint: boolean of whether or not to checkpoint

Usage:
    @requires(nodes=[CustomerNode, OrderNode])
    def do_post_join_annotate(self):
        self.df = self.df.withColumn(...)
    """
    def wrapit(func, nodes=nodes):
        def newfunc(inst, *args, **kwargs):
            for x in nodes:
                if x not in inst._merged:
                    return None
            res = func(inst, *args, **kwargs)
            if hasattr(inst, '_storage_client') and checkpoint:
                if not isinstance(res, None.__class__):
                    if res.__class__ in valid_dataframe_classes:
                        df = res
                else:
                    df = inst.df

                fname = func.__name__
                if hasattr(inst, '_checkpoints') and fname in inst._checkpoints:
                    return res
                else:
                    name = f"{inst.__class__.__name__}_{fname}.{inst.fmt}"
                    # checkpoint 
                    inst._storage_client.offload(
                            df,
                            name
                            )
                    path = inst._storage_client.get_path(name)
                    # reload 
                    inst.df = inst._storage_client.load(path)

                    # add function to list of checkpoints.
                    # inst._checkpoints.append(fname)
                    inst._checkpoints = [fname]
            return res
        return newfunc
    return wrapit
