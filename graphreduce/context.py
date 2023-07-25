#!/usr/bin/env python
"""A module for helping with context
in GraphReduce compute graphs.
"""

import typing

from graphreduce.node import GraphReduceNode


def requires (
        nodes: typing.List[GraphReduceNode]
        ) -> callable:
    """
A decorator for ensuring the function
only runs when the calling Node has
a merged edge to the required `nodes` list.

Arguments
    nodes: list of GraphReduceNode classes to require
           for function execution

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
            func(inst, *args, **kwargs)
        return newfunc
    return wrapit
