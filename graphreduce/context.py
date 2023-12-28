#!/usr/bin/env python
"""A module for helping with context
in GraphReduce compute graphs.
"""

# standard library
import typing

# third party
import pandas as pd
import dask.dataframe as dd
import pyspark

# internal
from graphreduce.node import GraphReduceNode


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
                    if res.__class__ in [pd.DataFrame, dd.DataFrame, pyspark.sql.dataframe.DataFrame]:
                        df = res
                else:
                    df = inst.df

                # Some libraries don't like a ton of underscores
                # So rename the function
                fname = func.__name__
                fname = fname.replace('_', '-')
                name = f"{inst.__class__.__name__}_{fname}.{inst.fmt}"
                # checkpoint 
                inst._storage_client.offload(
                        df,
                        name
                        )
                path = inst._storage_client.get_path(name)
                # reload 
                inst.df = inst._storage_client.load(path)
            return res
        return newfunc
    return wrapit
