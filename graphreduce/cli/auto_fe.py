#!/usr/bin/env python

import sqlite3
import json
import os
import typing
import datetime

import typer
from typer import Argument, Option
import pandas as pd


# examples for using SQL engines and dialects
from graphreduce.node import SQLNode, DynamicNode
from graphreduce.graph_reduce import GraphReduce
from graphreduce.enum import SQLOpType, ComputeLayerEnum, PeriodUnit
from graphreduce.models import sqlop
from graphreduce.context import method_requires


auto_fe_cli = typer.Typer(name="auto_fe", help="Perform automated feature engineering", no_args_is_help=True)




@auto_fe_cli.command("autofefs")
def autofe_filesystem (
            # directory or sqlite db
            data_path: str = Argument(help="Path to data"),
            # 'csv', 'parquet', etc.
            fmt: str = Argument(help="File format"),
            # {fname: 'prefix}
            prefixes: str = Argument(help="json dict of filenames with prefixes (e.g., `{'test.csv':'test'}`)"),
            # {fname: 'ts'}
            date_keys: str = Argument(help="json dict of filenames with associated date key (e.g., `{'test.csv': 'ts'}`)"),
            # [ {'from_node': 'fname', 'from_key', 'to_node': 'fname', 'to_key': key', 'reduce':True} ]
            relationships: str = Argument(
                help="json of relationships (e.g., `[{'from_node':'fname', 'from_key':'cust_id', 'to_node':'tname', 'to_key'}]`)"),
            parent_node: str = Argument(
                help="parent/root node to which to aggregate all of the data"
                ),
            cut_date: str = Argument(str(datetime.datetime.today())),
            # 'pandas', 'dask', 'sql'
            compute_layer: str = Argument("pandas"),
            hops_front: int = Argument(1),
            hops_back: int = Argument(3),
            output_path: str = Option('-op', '--output-path', help='output path for the data')
            ):
    """
Main automated feature engineering function.
    """

    prefixes = json.loads(prefixes)
    date_keys = json.loads(date_keys)
    relationships = json.loads(relationships)

    nodes = {}
    if fmt in ['csv', 'parquet', 'delta', 'iceberg']:
        for f in os.listdir(data_path):
            print(f"adding file {f}")
            nodes[f] = DynamicNode(
                    fpath=f"{data_path}/{f}",
                    fmt=f.split('.')[1],
                    prefix=prefixes.get(f),
                    compute_layer=getattr(ComputeLayerEnum, compute_layer),
                    date_key=date_keys.get(f, None)
                    )

    gr = GraphReduce(
            name='autofe',
            parent_node=nodes[parent_node],
            fmt=fmt,
            cut_date=datetime.datetime.now(),
            compute_layer=getattr(ComputeLayerEnum, compute_layer),
            auto_features=True,
            auto_feature_hops_front=hops_front,
            auto_feature_hops_back=hops_back
            )

    for rel in relationships:
        gr.add_entity_edge(
                parent_node=nodes[rel['to_node']],
                parent_key=rel['to_key'],
                relation_node=nodes[rel['from_node']],
                relation_key=rel['from_key'],
                reduce=rel.get('reduce', True)
                )

    gr.do_transformations()
    if not output_path:
        output_path = os.path.join(
                os.path.expanduser("~"),
                "graphreduce_outputs/test.csv"
                )

    getattr(gr.parent_node.df, f"to_{fmt}")(output_path)
