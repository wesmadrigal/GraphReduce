#!/usr/bin/env python

# std lib
import os
import sys
import abc
import datetime
import enum
import typing

# third party
import pandas as pd
import networkx as nx
from dask import dataframe as dd
from structlog import get_logger
import pyspark
import pyvis

# internal
from graphreduce.node import GraphReduceNode
from graphreduce.enum import ComputeLayerEnum, PeriodUnit

logger = get_logger('GraphReduce')



class GraphReduce(nx.DiGraph):
    def __init__(
        self, 
        name : str = 'graph_reduce',
        parent_node : typing.Optional[GraphReduceNode] = None,
        fmt : str = 'parquet',
        compute_layer : ComputeLayerEnum = None,
        cut_date : datetime.datetime = datetime.datetime.now(),
        compute_period_val : typing.Union[int, float] = 365,
        compute_period_unit : PeriodUnit  = PeriodUnit.day,
        has_labels : bool = False,
        label_period_val : typing.Optional[typing.Union[int, float]] = None,
        label_period_unit : typing.Optional[PeriodUnit] = None,
        spark_sqlCtx : pyspark.sql.SQLContext = None,
        feature_function : typing.Optional[str] = None,
        dynamic_propagation : bool = False,
        type_func_map : typing.Dict[str, typing.List[str]] = {
            'int64' : ['min', 'max', 'sum'],
            'str' : ['first'],
            'object' : ['first'],
            'float64' : ['min', 'max', 'sum'],
            'bool' : ['first'],
            'datetime64' : ['first']
            },
        *args,
        **kwargs
    ):
        """
Constructor for GraphReduce

Args:
    name : the name of the graph reduce
    parent_node : parent-most node in the graph, if doing reductions the granularity to which to reduce the data
    fmt : the format of the dataset 
    compute_layer : compute layer to use (e.g., spark)    
    cut_date : the date to cut off history
    compute_period_val : the amount of time to consider during the compute job
    compute_period_unit : the unit for the compute period value (e.g., day)
    has_labels : whether or not the compute job computes labels, when True `prep_for_labels()` and `compute_labels` will be called
    label_period_val : amount of time to consider when computing labels
    label_period_unit : the unit for the label period value (e.g., day)
    spark_sqlCtx : if compute layer is spark this must be passed
    feature_function : optional custom feature function
    dynamic_propagation : optional to dynamically propagate children data upward, useful for very large compute graphs
    type_func_match : optional mapping from type to a list of functions (e.g., {'int' : ['min', 'max', 'sum'], 'str' : ['first']})
        """
        super(GraphReduce, self).__init__(*args, **kwargs)
        
        self.name = name
        self.parent_node = parent_node
        self.cut_date = cut_date
        self.fmt = fmt
        self.compute_period_val = compute_period_val
        self.compute_period_unit = compute_period_unit
        self.has_labels = has_labels
        self.label_period_val = label_period_val
        self.label_period_unit = label_period_unit
        self.compute_layer = compute_layer
        self.feature_function = feature_function
        self.dynamic_propagation = dynamic_propagation
        self.type_func_map = type_func_map
        
        # if using Spark
        self._sqlCtx = spark_sqlCtx
        
        if self.compute_layer == ComputeLayerEnum.spark and self._sqlCtx is None:
            raise Exception(f"Must provide a `spark_sqlCtx` kwarg if using {self.compute_layer.value} as compute layer")
        
        if self.has_labels and (self.label_period_val is None or self.label_period_unit is None):
            raise Exception(f"If has_labels is True must provide values for `label_period_val` and `label_period_unit`")
        
        # current node being computed over
        self._curnode = None
    

    @property
    def parent(self):
        return self.parent_node


    def assign_parent (
        self,
        parent_node : GraphReduceNode,
    ):
        """
Assign the parent-most node in the graph
        """
        self._parent_node = parent_node
    
        
    def hydrate_graph_attrs (
        self,
        attrs=[
            'cut_date',
            'compute_period_val',
            'compute_period_unit',
            'has_labels',
            'label_period_val',
            'label_period_unit',
            'compute_layer',
            'feature_function'
        ]
    ):
        """
Hydrate the nodes in the graph with parent 
attributes in `attrs`
        """
        for node in self.nodes():
            logger.info(f"hydrating attributes for {node.__class__.__name__}")
            for attr in attrs:
                parent_val = getattr(self, attr)
                if not hasattr(node, attr):
                    setattr(node, attr, parent_val)
                elif hasattr(node, attr):
                    child_val = getattr(node, attr)
                    if parent_val != child_val:
                        setattr(node, attr, getattr(self, attr))
                
                
    def hydrate_graph_data (
        self,
    ):
        """
Hydrate the nodes in the graph with their data
        """
        for node in self.nodes():
            node.do_data()
            
            
    def add_entity_edge (
        self,
        parent_node : GraphReduceNode,
        relation_node : GraphReduceNode,
        parent_key : str,
        relation_key : str,
        # need to enforce this better
        relation_type : str = 'parent_child',
        reduce : bool = True
    ):
        """
Add an entity relation
        """
        if not self.has_edge(parent_node, relation_node):
            self.add_edge(
                parent_node,
                relation_node,
                keys={
                    'parent_key' : parent_key,
                    'relation_key' : relation_key,
                    'relation_type' : relation_type,
                    'reduce' : reduce
                }
            )

 
    def join (
        self,
        parent_node : GraphReduceNode,
        relation_node : GraphReduceNode,
        relation_df = None
        ):
        """
        Join the child or peer nnode to the parent node
        
        Optionally pass the `child_df` directly
        """
        
        meta = self.get_edge_data(parent_node, relation_node)
        
        if not meta:
            raise Exception(f"no edge metadata for {parent_node} and {relation_node}")
            
        if meta.get('keys'):
            meta = meta['keys']
            
        if meta and meta['relation_type'] == 'parent_child':
            parent_pk = meta['parent_key']
            relation_fk = meta['relation_key']
        elif meta and meta['relation_type'] == 'peer':
            parent_pk = meta['parent_key']
            relation_fk = meta['relation_key']
        
        
        if self.compute_layer in [ComputeLayerEnum.pandas, ComputeLayerEnum.dask]:
            if isinstance(relation_df, pd.DataFrame) or isinstance(relation_df, dd.DataFrame):
                joined = parent_node.df.merge(
                    relation_df,
                    left_on=parent_node.df[f"{parent_node.prefix}_{parent_pk}"],
                    right_on=relation_df[f"{relation_node.prefix}_{relation_fk}"],
                    suffixes=('','_dupe'),
                    how="left"
                )
            else:
                joined = parent_node.df.merge(
                    relation_node.df,
                    left_on=parent_node.df[f"{parent_node.prefix}_{parent_pk}"],
                    right_on=relation_node.df[f"{relation_node.prefix}_{relation_fk}"],
                    suffixes=('','_dupe'),
                    how="left"
                )
            if "key_0" in joined.columns:
                joined = joined[[c for c in joined.columns if c != "key_0"]]
                return joined
            else:
                return joined
                            
        elif self.compute_layer == ComputeLayerEnum.spark:     
            if isinstance(relation_df, pyspark.sql.dataframe.DataFrame) and isinstance(parent_node.df, pyspark.sql.dataframe.DataFrame):
                joined = parent_node.df.join(
                        relation_df,
                        on=parent_node.df[f"{parent_node.prefix}_{parent_pk}"] == relation_df[f"{relation_node.prefix}_{relation_fk}"],
                        how="left"
                        )
                return joined
            elif isinstance(parent_node.df, pyspark.sql.dataframe.DataFrame) and isinstance(relation_node.df, pyspark.sql.dataframe.DataFrame):
                joined = parent_node.df.join(
                    relation_node.df,
                    on=parent_node.df[f"{parent_node.prefix}_{parent_pk}"] == relation_node.df[f"{relation_node.prefix}_{relation_fk}"],
                    how="left"
                ) 
                return joined
            else:
                raise Exception(f"Cannot use spark on dataframe of type: {type(parent_node.df)}")
                
        else:
            logger.error('no valid compute layer')
            
        return None

    
    
    def depth_first_generator(self):
        """
Depth-first traversal over the edges
        """
        if not self.parent_node:
            raise Exception("Must have a parent node set to do depth first traversal")
        for edge in list(reversed(list(nx.dfs_edges(self, source=self.parent_node)))):
            yield edge


    def get_children (
            self,
            node : GraphReduceNode
            ) -> typing.List[GraphReduceNode]:
        """
Get the children of a given node
        """
        return [x for x in list(reversed(list(nx.dfs_preorder_nodes(self, source=node)))) if x != node]
    
    
    def plot_graph (
        self,
        fname : str = 'graph.html',
        notebook : bool = False,
    ):
        """
Plot the graph

Args
    fname : file name to save the graph to - should be .html
    notebook : whether or not to render in notebook
        """
        # need to populate a new graph
        # with string representations
        # of the dense object representations
        # we are using right now
        stringG = nx.DiGraph()
        for n in self.nodes():
            stringG.add_node(n.__class__.__name__)

        for edge in self.edges():
            edge_data = self.get_edge_data(edge[0], edge[1])
            edge_data = edge_data['keys']
            edge_title = f"{edge[0].__class__.__name__} key: {edge_data['parent_key']}\n{edge[1].__class__.__name__} key: {edge_data['relation_key']}\nrelation type: {edge_data['relation_type']}\nreduce relation: {edge_data['reduce']}"
            stringG.add_edge(
                edge[0].__class__.__name__, 
                edge[1].__class__.__name__,
                title=edge_title)
        
        nt = pyvis.network.Network(notebook=notebook)
        nt.from_nx(stringG)
        logger.info(f"plotted graph at {fname}")
        nt.show(fname)
   

    def prefix_uniqueness(self):
        """
Identify children with duplicate prefixes, if any
        """
        prefixes = {}
        dupes = []
        for node in self.nodes():
            if not prefixes.get(node.prefix):
                prefixes[node.prefix] = node
            else:
                dupes.append(node)
                dupes.append(prefixes[node.prefix])
        if len(dupes):
            raise Exception(f"duplicate prefix on the following nodes: {dupes}")

    
    def do_transformations(self):
        """
Perform all graph transformations
1) hydrate graph
2) check for duplicate prefixes
3) filter data
4) clip anomalies
5) annotate data
6) depth-first edge traversal to: aggregate / reduce features and labels
6a) optional alternative feature_function mapping
6b) join back to parent edge
6c) post-join annotations if any
7) repeat step 6 on all edges up the hierarchy
        """
        
        # get data, filter data, clip columns, and annotate
        logger.info("hydrating graph attributes")
        self.hydrate_graph_attrs()
        logger.info("hydrating graph data")
        self.hydrate_graph_data()

        logger.info("checking for prefix uniqueness")
        self.prefix_uniqueness()
    
        for node in self.nodes():
            logger.info(f"running filters, clip cols, and annotations for {node.__class__.__name__}")
            node.do_filters()
            node.do_clip_cols()
            node.do_annotate()

        logger.info(f"depth-first traversal through the graph from source: {self.parent_node.__class__.__name__}")
        for edge in self.depth_first_generator():
            parent_node = edge[0]
            relation_node = edge[1]
            edge_data = self.get_edge_data(parent_node, relation_node)
            if edge_data.get('keys'):
                edge_data = edge_data['keys']

            if edge_data['reduce']:
                logger.info(f"reducing relation {relation_node.__class__.__name__}")
                join_df = relation_node.do_reduce(edge_data['relation_key'])
                # only relevant when reducing
                if self.dynamic_propagation:
                    logger.info(f"doing dynamic propagation on node {relation_node.__class__.__name__}")
                    child_df = relation_node.dynamic_propagation(
                            reduce_key=edge_data['relation_key'],
                            type_func_map=self.type_func_map,
                            compute_layer=self.compute_layer
                        )
                    # NOTE: this is pandas specific and will break
                    # on other compute layers for now 
                    if self.compute_layer in [ComputeLayerEnum.pandas, ComputeLayerEnum.dask]:
                        join_df = join_df.merge(
                                child_df,
                                on=relation_node.colabbr(edge_data['relation_key']),
                                suffixes=('', '_dupe')
                        )
                    elif self.compute_layer == ComputeLayerEnum.spark:
                        join_df = join_df.join(
                                child_df,
                                on=join_df[relation_node.colabbr(edge_data['relation_key'])] == child_df[relation_node.colabbr(edge_data['relation_key'])],
                                how="left"
                            )

            elif not edge_data['reduce'] and self.feature_function:
                logger.info(f"not reducing relation {relation_node.__class__.__name__}")
                join_df = getattr(relation_node, self.feature_function)()
            else:
                # in this case we will join the entire relation's dataframe
                logger.info(f"doing nothing with relation node {relation_node.__class__.__name__}")
                join_df = None
                
            logger.info(f"joining {relation_node.__class__.__name__} to {parent_node.__class__.__name__}")
            joined_df = self.join(
                parent_node,
                relation_node,
                relation_df=join_df
            )
            # update the parent dataframe
            parent_node.df = joined_df
            
            if self.has_labels:
                label_df = relation_node.do_labels(edge_data['relation_key'])
                logger.info(f"computed labels for {relation_node.__class__.__name__}")
                if label_df.__class__.__name__ != 'NoneType':
                    joined_with_labels = self.join(
                        parent_node,
                        relation_node,
                        relation_df=label_df
                    )
                    parent_node.df = joined_with_labels

            # post-join annotations (if any)
            parent_node.do_post_join_annotate()

