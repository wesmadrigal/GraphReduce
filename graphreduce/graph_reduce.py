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
from pyspark.sql import functions as F
import daft

# internal
from graphreduce.node import GraphReduceNode, DynamicNode, SQLNode
from graphreduce.enum import ComputeLayerEnum, PeriodUnit
from graphreduce.storage import StorageClient
from graphreduce.models import sqlop

logger = get_logger("GraphReduce")


class GraphReduce(nx.DiGraph):
    def __init__(
        self,
        name: str = "graph_reduce",
        parent_node: typing.Optional[GraphReduceNode] = None,
        fmt: str = "parquet",
        compute_layer: ComputeLayerEnum = None,
        cut_date: datetime.datetime = datetime.datetime.now(),
        compute_period_val: typing.Union[int, float] = 365,
        compute_period_unit: PeriodUnit = PeriodUnit.day,
        auto_features: bool = False,
        auto_feature_hops_back: int = 2,
        auto_feature_hops_front: int = 1,
        feature_typefunc_map: typing.Dict[str, typing.List[str]] = {
            "int64": ["median", "mean", "sum", "min", "max"],
            "str": ["min", "max", "count"],
            #'object' : ['first', 'count'],
            "object": ["count"],
            "float64": ["median", "min", "max", "sum", "mean"],
            "float32": ["median", "min", "max", "sum", "mean"],
            #'bool' : ['first'],
            #'datetime64' : ['first', 'min', 'max'],
            "datetime64": ["min", "max"],
            "datetime64[ns]": ["min", "max"],
        },
        feature_stype_map: typing.Dict[str, typing.List[str]] = {
            "numerical": ["median", "mean", "sum", "min", "max"],
            "categorical": ["count", "nunique"],
            "embedding": ["first"],
            "image_embedded": ["first"],
            "multicategorical": ["mode"],
            "sequence_numerical": ["min", "max"],
            "timestamp": ["min", "max"],
        },
        # Label parameters.
        label_node: typing.Optional[
            typing.Union[GraphReduceNode, typing.List[GraphReduceNode]]
        ] = None,
        label_operation: typing.Optional[typing.Union[callable, str]] = None,
        # Field on the node.
        label_field: typing.Optional[str] = None,
        label_period_val: typing.Optional[typing.Union[int, float]] = None,
        label_period_unit: typing.Optional[PeriodUnit] = None,
        spark_sqlctx: pyspark.sql.SQLContext = None,
        storage_client: typing.Optional[StorageClient] = None,
        catalog_client: typing.Any = None,
        sql_client: typing.Any = None,
        # Only for SQL engines.
        lazy_execution: bool = False,
        dry_run: bool = False,
        # Debug
        debug: bool = False,
        checkpoint_schema: str = None,
        *args,
        **kwargs,
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
            label_period_val : amount of time to consider when computing labels
            label_period_unit : the unit for the label period value (e.g., day)
            spark_sqlctx : if compute layer is spark this must be passed
            auto_features: optional to automatically compute features and propagate child features upward; useful for large compute graphs
            auto_feature_hops_back: optional for automatically computing features
            auto_feature_hops_front: optional for automatically computing features
            feature_typefunc_map : optional mapping from type to a list of functions (e.g., {'int' : ['min', 'max', 'sum'], 'str' : ['first']})
            label_node: optionl GraphReduceNode for the label
            label_operation: optional str or callable operation to call to compute the label
            label_field: optional str field to compute the label
            storage_client: optional `graphreduce.storage.StorageClient` instance to checkpoint compute graphs
            catalog_client: optional Unity or Polaris catalog client instance
            debug: bool whether to run debug logging
        """
        super(GraphReduce, self).__init__(*args, **kwargs)

        self.name = name
        self.parent_node = parent_node
        self.cut_date = cut_date
        self.fmt = fmt
        # Compute period for features.
        self.compute_period_val = compute_period_val
        self.compute_period_unit = compute_period_unit
        self.compute_layer = compute_layer

        # Label parameters.
        self.label_node = label_node
        self.label_field = label_field
        # Options: 'first', 'sum', 'avg', 'median', 'bool'
        self.label_operation = label_operation
        self.label_period_val = label_period_val
        self.label_period_unit = label_period_unit

        # Automatic feature engineering parameters.
        self.auto_features = auto_features
        self.auto_feature_hops_back = auto_feature_hops_back
        self.auto_feature_hops_front = auto_feature_hops_front
        self.feature_typefunc_map = feature_typefunc_map
        self.feature_stype_map = feature_stype_map

        # SQL dialect parameters.
        self._lazy_execution = lazy_execution

        # if using Spark
        self.spark_sqlctx = spark_sqlctx
        self._storage_client = storage_client

        # Catalogs.
        self._catalog_client = catalog_client
        # SQL engine client.
        self._sql_client = sql_client
        self._checkpoint_schema = checkpoint_schema

        self.debug = debug
        self.dry_run = dry_run
        # Keep track of all the SQL queries.
        self.sql_ops = []

        if self.compute_layer == ComputeLayerEnum.spark and self.spark_sqlctx is None:
            raise Exception(
                f"Must provide a `spark_sqlctx` kwarg if using {self.compute_layer.value} as compute layer"
            )

        if self.label_node and (
            self.label_period_val is None or self.label_period_unit is None
        ):
            raise Exception(
                f"If label_node is parameterized must provide values for `label_period_val` and `label_period_unit`"
            )

    def __repr__(self):
        return f"<GraphReduce: parent_node={self.parent_node.__class__}>"

    def __str__(self):
        return f"<GraphReduce num_nodes: {len(self.nodes())} num_edges: {len(self.edges())}>"

    def _mark_merged(
        self, parent_node: GraphReduceNode, relation_node: GraphReduceNode
    ):
        """
        Mark a relation node as merged to the parent.
        """
        if relation_node.__class__ not in parent_node._merged:
            parent_node._merged.append(relation_node.__class__)

    def _clean_refs(self):
        """
        Only for SQL dialect graphs where there are
        views and temporary tables to clean up throughout
        the graph.
        """
        for node in self.nodes():
            if hasattr(node, "_clean_refs"):
                node._clean_refs()

    @property
    def parent(self):
        return self.parent_node

    def assign_parent(
        self,
        parent_node: GraphReduceNode,
    ):
        """
        Assign the parent-most node in the graph
        """
        self._parent_node = parent_node

    def hydrate_graph_attrs(
        self,
        attrs=[
            "cut_date",
            "compute_period_val",
            "compute_period_unit",
            "label_period_val",
            "label_period_unit",
            "compute_layer",
            "spark_sqlctx",
            "_storage_client",
            "_lazy_execution",
            "_catalog_client",
            "_sql_client",
        ],
    ):
        """
        Hydrate the nodes in the graph with parent
        attributes in `attrs`
        """
        for node in self.nodes():
            logger.info(f"hydrating attributes for {node.__class__.__name__}")
            for attr in attrs:
                if hasattr(self, attr):
                    parent_val = getattr(self, attr)
                    if not hasattr(node, attr):
                        setattr(node, attr, parent_val)
                    elif hasattr(node, attr):
                        child_val = getattr(node, attr)
                        if parent_val != child_val:
                            setattr(node, attr, getattr(self, attr))
                    elif attr == "_sql_client":
                        setattr(node, "client", getattr(self, attr))

    def hydrate_graph_data(
        self,
    ):
        """
        Hydrate the nodes in the graph with their data
        """
        for node in self.nodes():
            if self.debug:
                logger.debug(f"hydrating {node} data")
            node.do_data()

    def add_entity_edge(
        self,
        parent_node: GraphReduceNode,
        relation_node: GraphReduceNode,
        parent_key: str,
        relation_key: str,
        # need to enforce this better
        relation_type: str = "parent_child",
        reduce: bool = True,
        reduce_after_join: bool = False,
    ):
        """
        Add an entity relation
        """
        if reduce and reduce_after_join:
            raise Exception(f"only one can be true: `reduce` or `reduce_after_join`")
        if not self.has_edge(parent_node, relation_node):
            self.add_edge(
                parent_node,
                relation_node,
                keys={
                    "parent_key": parent_key,
                    "relation_key": relation_key,
                    "relation_type": relation_type,
                    "reduce": reduce,
                    "reduce_after_join": reduce_after_join,
                },
            )

    def join_any(
        self,
        to_node: GraphReduceNode,
        from_node: GraphReduceNode,
        how: str = "left",
        to_node_key: str = None,
        from_node_key: str = None,
        to_node_df=None,
        from_node_df=None,
    ):
        """
        Join the relations.
        """

        if to_node_key and from_node_key:
            pass
        else:
            meta = self.get_edge_data(to_node, from_node)
            if meta:
                meta = meta["keys"]
                to_node_key = meta["parent_key"]
                from_node_key = meta["relation_key"]

            elif not meta:
                meta = self.get_edge_data(from_node, to_node)
                if meta:
                    meta = meta["keys"]
                    to_node_key = meta["relation_key"]
                    from_node_key = meta["parent_key"]
                else:
                    raise Exception(f"no edge metadata for {to_node} and {from_node}")

        if self.compute_layer in [ComputeLayerEnum.pandas, ComputeLayerEnum.dask]:
            joined = to_node.df.merge(
                from_node.df,
                left_on=to_node.df[f"{to_node.prefix}_{to_node_key}"],
                right_on=from_node.df[f"{from_node.prefix}_{from_node_key}"],
                suffixes=("", "_dupe"),
                how="left",
            )
            self._mark_merged(to_node, from_node)
            if "key_0" in joined.columns:
                joined = joined[[c for c in joined.columns if c != "key_0"]]
                return joined
            else:
                return joined
        elif self.compute_layer == ComputeLayerEnum.daft:
            joined = to_node.df.join(
                from_node.df,
                left_on=to_node.df[f"{to_node.prefix}_{to_node_key}"],
                right_on=from_node.df[f"{from_node.prefix}_{from_node_key}"],
                suffix="_dupe",
                how="left",
            )
            self._mark_merged(to_node, from_node)
            return joined
        elif self.compute_layer == ComputeLayerEnum.spark:
            if isinstance(to_node.df, pyspark.sql.dataframe.DataFrame) and isinstance(
                from_node.df, pyspark.sql.dataframe.DataFrame
            ):
                joined = to_node.df.join(
                    from_node.df,
                    on=to_node.df[f"{to_node.prefix}_{to_node_key}"]
                    == from_node.df[f"{from_node.prefix}_{from_node_key}"],
                    how="left",
                )
                self._mark_merged(to_node, from_node)
                return joined
            else:
                raise Exception(
                    f"Cannot use spark on dataframe of type: {type(to_node.df)}"
                )
        # TODO: make a `DialectEnum.sql` catchall for this.
        elif self.compute_layer in [
            ComputeLayerEnum.athena,
            ComputeLayerEnum.snowflake,
            ComputeLayerEnum.redshift,
            ComputeLayerEnum.postgres,
            ComputeLayerEnum.sqlite,
        ]:
            pass

        else:
            logger.error(f"{self.compute_layer} is not a valid compute layer")

    def join(
        self,
        parent_node: GraphReduceNode,
        relation_node: GraphReduceNode,
        relation_df=None,
    ):
        """
        Join the child or peer nnode to the parent node

        Optionally pass the `child_df` directly
        """

        meta = self.get_edge_data(parent_node, relation_node)

        if not meta:
            meta = self.get_edge_data(relation_node, parent_node)

            raise Exception(f"no edge metadata for {parent_node} and {relation_node}")

        if meta.get("keys"):
            meta = meta["keys"]

        if meta and meta["relation_type"] == "parent_child":
            parent_pk = meta["parent_key"]
            relation_fk = meta["relation_key"]
        elif meta and meta["relation_type"] == "peer":
            parent_pk = meta["parent_key"]
            relation_fk = meta["relation_key"]

        if self.compute_layer in [ComputeLayerEnum.pandas, ComputeLayerEnum.dask]:
            if isinstance(relation_df, pd.DataFrame) or isinstance(
                relation_df, dd.DataFrame
            ):
                joined = parent_node.df.merge(
                    relation_df,
                    left_on=parent_node.df[f"{parent_node.prefix}_{parent_pk}"],
                    right_on=relation_df[f"{relation_node.prefix}_{relation_fk}"],
                    suffixes=("", "_dupe"),
                    how="left",
                )
            else:
                joined = parent_node.df.merge(
                    relation_node.df,
                    left_on=parent_node.df[f"{parent_node.prefix}_{parent_pk}"],
                    right_on=relation_node.df[f"{relation_node.prefix}_{relation_fk}"],
                    suffixes=("", "_dupe"),
                    how="left",
                )
            self._mark_merged(parent_node, relation_node)
            if "key_0" in joined.columns:
                joined = joined[[c for c in joined.columns if c != "key_0"]]
                return joined
            else:
                return joined
        elif self.compute_layer == ComputeLayerEnum.daft:
            if isinstance(relation_df, daft.dataframe.dataframe.DataFrame):
                joined = parent_node.df.join(
                    relation_df,
                    left_on=parent_node.df[f"{parent_node.prefix}_{parent_pk}"],
                    right_on=relation_df[f"{relation_node.prefix}_{relation_fk}"],
                    suffix="_dupe",
                    how="left",
                )
            else:
                joined = parent_node.df.join(
                    relation_node.df,
                    left_on=parent_node.df[f"{parent_node.prefix}_{parent_pk}"],
                    right_on=relation_node.df[f"{relation_node.prefix}_{relation_fk}"],
                    suffix="_dupe",
                    how="left",
                )
            self._mark_merged(parent_node, relation_node)
            return joined
        elif self.compute_layer == ComputeLayerEnum.spark:
            valid_dataframe_types = (
                pyspark.sql.dataframe.DataFrame,
                pyspark.sql.connect.dataframe.DataFrame,
            )
            if isinstance(relation_df, valid_dataframe_types) and isinstance(
                parent_node.df, valid_dataframe_types
            ):
                has_dupe = False
                join_key = f"{relation_node.prefix}_{relation_fk}"
                if join_key in parent_node.df.columns:
                    new = f"{join_key}_dupe"
                    relation_df = relation_df.withColumnRenamed(join_key, new)
                    join_key = new
                    has_dupe = True

                joined = parent_node.df.join(
                    relation_df,
                    on=parent_node.df[f"{parent_node.prefix}_{parent_pk}"]
                    == relation_df[join_key],
                    how="left",
                )
                if has_dupe:
                    joined = joined.drop(F.col(join_key))

                self._mark_merged(parent_node, relation_node)
                return joined
            elif isinstance(parent_node.df, valid_dataframe_types) and isinstance(
                relation_node.df, valid_dataframe_types
            ):
                has_dupe = False
                join_key = f"{relation_node.prefix}_{relation_fk}"
                if join_key in parent_node.df.columns:
                    has_dupe = True
                    new = f"{join_key}_dupe"
                    relation_node.df = relation_node.df.withColumnRenamed(join_key, new)
                    join_key = new

                joined = parent_node.df.join(
                    relation_node.df,
                    on=parent_node.df[f"{parent_node.prefix}_{parent_pk}"]
                    == relation_node.df[join_key],
                    how="left",
                )
                if has_dupe:
                    joined = joined.drop(F.col(join_key))

                self._mark_merged(parent_node, relation_node)
                return joined
            else:
                raise Exception(
                    f"Cannot use spark on dataframe of type: {type(parent_node.df)}"
                )
        else:
            logger.error("no valid compute layer")
        return None

    # Since this is a general SQL implementation
    # it is possible that we need to extend it
    # in the future to be engine-specific.
    # If that is the case we can either extend
    # this method or add engine-specific methods
    # to engine-specific nodes (e.g., `SnowflakeNode.join_sql`)
    def join_sql(
        self,
        parent_node: GraphReduceNode,
        relation_node: GraphReduceNode,
        # Optional keys.
        parent_node_key: str = None,
        relation_node_key: str = None,
    ) -> str:
        """
        Joins two graph reduce nodes of SQL dialect.
        """

        meta = self.get_edge_data(parent_node, relation_node)

        if not meta:
            meta = self.get_edge_data(relation_node, parent_node)
            raise Exception(f"no edge metadata for {parent_node} and {relation_node}")
        if meta.get("keys"):
            meta = meta["keys"]

        if meta and meta["relation_type"] == "parent_child":
            parent_pk = meta["parent_key"]
            relation_fk = meta["relation_key"]
        elif meta and meta["relation_type"] == "peer":
            parent_pk = meta["parent_key"]
            relation_fk = meta["relation_key"]

        parent_table = (
            parent_node._cur_data_ref
            if parent_node._cur_data_ref
            else parent_node.fpath
        )
        relation_table = (
            relation_node._cur_data_ref
            if relation_node._cur_data_ref
            else relation_node.fpath
        )
        logger.info(f"parent table: {parent_table} relation table: {relation_table}")
        # Check if the relation foreign key is already
        # in the parent and, if so, rename it.
        parent_samp = parent_node.get_sample()
        relation_samp = relation_node.get_sample()
        logger.info(f"parent columns: {parent_samp.columns}")
        logger.info(f"relation columns: {relation_samp.columns}")
        relation_fk = f"{relation_node.prefix}_{relation_fk}"
        if relation_fk in parent_samp.columns or relation_fk.lower() in [
            _x.lower() for _x in relation_samp.columns
        ]:
            logger.info(f"removing duplicate column {relation_fk} on join")
            relation_cols = [
                f"relation.{c}"
                for c in relation_samp.columns
                if c.lower() != relation_fk.lower()
            ]
            sel = ",".join(relation_cols)
            JOIN_SQL = f"""
                SELECT parent.*, {sel}
                FROM {parent_table} parent
                LEFT JOIN {relation_table} relation
                ON parent.{parent_node.prefix}_{parent_pk} = relation.{relation_fk}
            """
        else:
            JOIN_SQL = f"""
                SELECT parent.*, relation.*
                FROM {parent_table} parent
                LEFT JOIN {relation_table} relation
                ON parent.{parent_node.prefix}_{parent_pk} = relation.{relation_node.prefix}_{relation_fk}
            """
        # Always overwrite the join reference.
        parent_node.create_ref(
            JOIN_SQL,
            "join",
            overwrite=True,
            schema=self._checkpoint_schema,
            dry=self.dry_run,
        )
        # Get the table after the join.
        parent_samp = parent_node.get_sample()
        self.sql_ops.append(JOIN_SQL)
        if parent_node._ref_sql:
            self.sql_ops.append(parent_node._ref_sql)
            parent_node._ref_sql = None
        self._mark_merged(parent_node, relation_node)

    def depth_first_generator(self):
        """
        Depth-first traversal over the edges
        """
        if not self.parent_node:
            raise Exception("Must have a parent node set to do depth first traversal")
        for edge in list(
            reversed(
                list(
                    nx.dfs_edges(
                        self,
                        source=self.parent_node,
                        depth_limit=self.auto_feature_hops_back,
                    )
                )
            )
        ):
            yield edge

    def traverse_up(self, start: typing.Union[GraphReduceNode, DynamicNode]) -> list:
        """
        Traverses up the graph for merging parents.
        """
        parents = [(start, n, 1) for n in self.predecessors(start)]
        to_traverse = [(n, 1) for n in self.predecessors(start)]
        cur_level = 1
        while len(to_traverse) and cur_level <= self.auto_feature_hops_front:
            cur_node, cur_level = to_traverse[0]
            del to_traverse[0]

            for node in self.predecessors(cur_node):
                if cur_level + 1 <= self.auto_feature_hops_front:
                    parents.append((cur_node, node, cur_level + 1))
                    to_traverse.append((node, cur_level + 1))
        # Returns higher levels first so that
        # when we iterate through these edges
        # we will traverse from top to bottom
        # where the bottom is our `start`.
        parents_ordered = list(reversed(parents))
        if self.debug:
            for ix in range(len(parents_ordered)):
                logger.debug(f"index {ix} is level {parents_ordered[ix][-1]}")
        return parents_ordered

    def get_children(self, node: GraphReduceNode) -> typing.List[GraphReduceNode]:
        """
        Get the children of a given node
        """
        return [
            x
            for x in list(reversed(list(nx.dfs_preorder_nodes(self, source=node))))
            if x != node
        ]

    def plot_graph(
        self,
        fname: str = "graph.html",
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
            if n.__class__.__name__ == "DynamicNode":
                stringG.add_node(n.fpath)
            else:
                stringG.add_node(n.__class__.__name__)

        for edge in self.edges():
            edge_data = self.get_edge_data(edge[0], edge[1])
            edge_data = edge_data["keys"]
            edge_title = f"{edge[0].__class__.__name__} key: {edge_data['parent_key']}\n{edge[1].__class__.__name__} key: {edge_data['relation_key']}\nrelation type: {edge_data['relation_type']}\nreduce relation: {edge_data['reduce']}"
            if n.__class__.__name__ == "DynamicNode":
                stringG.add_edge(edge[0].fpath, edge[1].fpath, title=edge_title)
            else:
                stringG.add_edge(
                    edge[0].__class__.__name__,
                    edge[1].__class__.__name__,
                    title=edge_title,
                )

        nt = pyvis.network.Network()
        nt.from_nx(stringG)
        logger.info(f"plotted graph at {fname}")
        nt.save_graph(fname)

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

    def do_transformations_sql(self):
        """
        Perform all graph transformations
        1) hydrate graph
        2) check for duplicate prefixes
        3) annotate date
        4) filter data
        5) clip anomalies
        6) annotate data
        7) depth-first edge traversal to: aggregate / reduce features and labels
        7a) join back to parent node
        7b) post-join annotations and filters (if any)
        8) repeat step 7 on all edges up the hierarchy
        """
        logger.info("hydrating graph attributes")
        self.hydrate_graph_attrs()

        logger.info("checking for prefix uniqueness")
        self.prefix_uniqueness()

        # Node-level data prep operations.
        for node in self.nodes():
            # `self.do_data` must always return some `sqlop`
            ops = node.do_data()
            if not ops:
                raise Exception(
                    f"{node.__class__.__name__}.do_data must be implemented"
                )

            logger.debug(f"do data: {node.build_query(ops)}")
            self.sql_ops.append(node.build_query(ops))
            node.create_ref(
                node.build_query(ops),
                node.do_data,
                schema=self._checkpoint_schema,
                dry=self.dry_run,
            )
            # Now append the reference SQL.
            if node._ref_sql:
                self.sql_ops.append(node._ref_sql)
                node._ref_sql = None
            logger.debug(f"do annotate: {node.build_query(node.do_annotate())}")
            self.sql_ops.append(node.build_query(node.do_annotate()))
            node.create_ref(
                node.build_query(node.do_annotate()),
                node.do_annotate,
                schema=self._checkpoint_schema,
                dry=self.dry_run,
            )
            if node._ref_sql:
                self.sql_ops.append(node._ref_sql)
                node._ref_sql = None

            logger.debug(f"do annotate: {node.build_query(node.do_filters())}")
            self.sql_ops.append(node.build_query(node.do_filters()))
            node.create_ref(
                node.build_query(node.do_filters()),
                node.do_filters,
                schema=self._checkpoint_schema,
                dry=self.dry_run,
            )
            if node._ref_sql:
                self.sql_ops.append(node._ref_sql)
                node._ref_sql = None

            self.sql_ops.append(node.build_query(node.do_normalize()))
            node.create_ref(
                node.build_query(node.do_normalize()),
                node.do_normalize,
                schema=self._checkpoint_schema,
                dry=self.dry_run,
            )
            if node._ref_sql:
                self.sql_ops.append(node._ref_sql)
                node._ref_sql = None
            #    node.create_ref(node.build_query(ops), node.do_data, schema=self._checkpoint_schema)
            #    node.create_ref(node.build_query(node.do_annotate()), node.do_annotate, schema=self._checkpoint_schema)
            #    node.create_ref(node.build_query(node.do_filters()), node.do_filters, schema=self._checkpoint_schema)
            #    node.create_ref(node.build_query(node.do_normalize()), node.do_normalize, schema=self._checkpoint_schema)

        # Check for automatic feature engineering
        # for forward relationships.  These are
        # assumed to be 1:1, so no aggregation is
        # needed.
        if self.auto_features:
            for to_node, from_node, level in self.traverse_up(start=self.parent_node):
                if (
                    self.auto_feature_hops_front
                    and level <= self.auto_feature_hops_front
                ):
                    logger.info(f"joining {from_node} to {to_node}")
                    self.join_sql(to_node, from_node)

        logger.info(
            f"depth-first traversal through the graph from source: {self.parent_node}"
        )
        for edge in self.depth_first_generator():
            parent_node = edge[0]
            relation_node = edge[1]
            edge_data = self.get_edge_data(parent_node, relation_node)
            if edge_data.get("keys"):
                edge_data = edge_data["keys"]

            if edge_data["reduce"]:
                logger.info(f"reducing relation {relation_node}")

                # Check for automatic feature engineering.
                if self.auto_features and not relation_node.do_reduce(
                    edge_data["relation_key"]
                ):
                    logger.info(f"performing auto_features on node {relation_node}")

                    sql_ops = relation_node.auto_features(
                        reduce_key=edge_data["relation_key"],
                        # type_func_map=self.feature_typefunc_map,
                        type_func_map=self.feature_stype_map,
                        compute_layer=self.compute_layer,
                    )
                    logger.info(f"{sql_ops}")

                    self.sql_ops.append(relation_node.build_query(sql_ops))

                    relation_node.create_ref(
                        relation_node.build_query(
                            relation_node.auto_features(
                                reduce_key=edge_data["relation_key"],
                                # type_func_map=self.feature_typefunc_map,
                                type_func_map=self.feature_stype_map,
                                compute_layer=self.compute_layer,
                            )
                        ),
                        relation_node.do_reduce,
                        schema=self._checkpoint_schema,
                        dry=self.dry_run,
                    )
                    if relation_node._ref_sql:
                        self.sql_ops.append(relation_node._ref_sql)
                        relation_node._ref_sql = None

                # Custom `do_reduce` implementation.
                else:
                    # Table name is stored within the node itself.
                    tfilt = (
                        relation_node.prep_for_features()
                        if relation_node.prep_for_features()
                        else []
                    )
                    # NOTE: we do not automatically do date filtering
                    # here so maybe that should be a top-level parameter
                    # for when we have a custom reduce implementation?
                    reduce_ops = relation_node.do_reduce(edge_data["relation_key"])
                    reduce_sql = relation_node.build_query(reduce_ops)
                    logger.info(f"reduce SQL: {reduce_sql}")
                    self.sql_ops.append(reduce_sql)
                    reduce_ref = relation_node.create_ref(
                        reduce_sql,
                        relation_node.do_reduce,
                        schema=self._checkpoint_schema,
                        dry=self.dry_run,
                    )
                    if relation_node._ref_sql:
                        self.sql_ops.append(relation_node._ref_sql)
                        relation_node._ref_sql = None
            else:
                # in this case we will join the entire relation's dataframe
                logger.info(f"doing nothing with relation node {relation_node}")

            logger.info(f"joining {relation_node} to {parent_node}")

            # This should be executed inside of the function.
            self.join_sql(
                parent_node,
                relation_node,
            )

            # Target variables.
            if (
                self.label_node
                and (
                    self.label_node == relation_node
                    or relation_node.label_field is not None
                )
                or relation_node.do_labels(edge_data["relation_key"]) is not None
            ):
                logger.info(f"Had label node {self.label_node}")

                # Get the reference right before `do_reduce`
                # so the records are not aggregated yet.
                data_ref = relation_node.get_ref_name(
                    relation_node.do_filters,
                    lookup=True,
                    schema=self._checkpoint_schema,
                )

                # TODO: don't need to reduce if it's 1:1 cardinality.
                if self.auto_features and not relation_node.do_labels(
                    edge_data["relation_key"]
                ):
                    self.sql_ops.append(
                        relation_node.build_query(
                            relation_node.default_label(
                                op=self.label_operation,
                                field=self.label_field,
                                reduce_key=edge_data["relation_key"],
                            ),
                            data_ref=data_ref,
                        )
                    )
                    logger.info(f"SQL Ops: {self.sql_ops[-1]}")
                    label_ref = relation_node.create_ref(
                        relation_node.build_query(
                            relation_node.default_label(
                                op=self.label_operation,
                                field=self.label_field,
                                reduce_key=edge_data["relation_key"],
                            ),
                            data_ref=data_ref,
                        ),
                        relation_node.do_labels,
                        schema=self._checkpoint_schema,
                        dry=self.dry_run,
                    )
                    if relation_node._ref_sql:
                        self.sql_ops.append(relation_node._ref_sql)
                        relation_node._ref_sql = None
                else:
                    # We should not default to `prep_for_labels` and instead force
                    # the user to call this helper function.
                    self.sql_ops.append(
                        relation_node.build_query(
                            relation_node.do_labels(edge_data["relation_key"]),
                            data_ref=data_ref,
                        )
                    )
                    logger.info(f"SQL Ops: {self.sql_ops[-1]}")
                    label_ref = relation_node.create_ref(
                        relation_node.build_query(
                            relation_node.do_labels(edge_data["relation_key"]),
                            data_ref=data_ref,
                        ),
                        relation_node.do_labels,
                        schema=self._checkpoint_schema,
                    )
                    if relation_node._ref_sql:
                        self.sql_ops.append(relation_node._ref_sql)
                        relation_node._ref_sql = None

                    # tfilt = relation_node.prep_for_labels() if relation_node.prep_for_labels() else []
                    # label_sql = tfilt + relation_node.do_labels(edge_data['relation_key'])
                    # label_ref = relation_node.create_ref(
                    #        relation_node.build_query(
                    #            label_sql,
                    #            data_ref=data_ref
                    #            ),
                    #        relation_node.do_labels,
                    #        schema=self._checkpoint_schema
                    #        )

                logger.info(f"computed labels for {relation_node}")
                # Since the `SQLNode.build_query` method
                # updates the `SQLNode._cur_data_ref` attribute
                # we can always join "naively" after a fresh call
                # to `SQLNode.create_ref`.
                self.join_sql(
                    parent_node,
                    relation_node,
                )

            # post-join annotations (if any)
            pja_sql = parent_node.build_query(parent_node.do_post_join_annotate())
            logger.info(f"Running do_post_join_annotate")
            logger.info(f"{pja_sql}")
            self.sql_ops.append(pja_sql)
            pja_ref = parent_node.create_ref(
                pja_sql,
                parent_node.do_post_join_annotate,
                schema=self._checkpoint_schema,
                dry=self.dry_run,
                # Need to ensure we're overwriting the
                # reference because previous executions
                # may have tried creating a reference before
                # dependencies were merged.
                overwrite=True,
            )
            # post-join filters (if any)
            if hasattr(parent_node, "do_post_join_filters"):
                pjf_sql = parent_node.build_query(parent_node.do_post_join_filters())
                self.sql_ops.append(pjf_sql)
                pjf_ref = parent_node.create_ref(
                    pjf_sql,
                    parent_node.do_post_join_filters,
                    schema=self._checkpoint_schema,
                    dry=self.dry_run,
                    # Need to ensure we're overwriting the
                    # reference because previous executions
                    # may have tried creating a reference before
                    # dependencies were merged.
                    overwrite=True,
                )

    def do_transformations(self):
        """
        Perform all graph transformations
        1) hydrate graph
        2) check for duplicate prefixes
        3) filter data
        4) clip anomalies
        5) annotate data
        6) depth-first edge traversal to: aggregate / reduce features and labels
        6a) join back to parent edge
        6b) post-join annotations if any
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
            logger.info(f"running filters, normalize, and annotations for {node}")
            # For SQL dialects this just returns a list of `sqlop`
            # instances, so we need to build the query and execute.
            node.do_annotate()
            node.do_filters()
            node.do_normalize()

        if self.auto_features:
            for to_node, from_node, level in self.traverse_up(start=self.parent_node):
                if (
                    self.auto_feature_hops_front
                    and level <= self.auto_feature_hops_front
                ):
                    # It is assumed that front-facing relations
                    # are not one to many and therefore we
                    # won't have duplication on the join.
                    # This may be an incorrect assumption
                    # so this implementation is currently brittle.
                    if self.debug:
                        logger.debug(
                            f"Performing FRONT auto_features front join from {from_node} to {to_node}"
                        )
                    joined_df = self.join_any(to_node, from_node)
                    to_node.df = joined_df

        logger.info(
            f"depth-first traversal through the graph from source: {self.parent_node}"
        )
        for edge in self.depth_first_generator():
            parent_node = edge[0]
            relation_node = edge[1]
            edge_data = self.get_edge_data(parent_node, relation_node)
            if edge_data.get("keys"):
                edge_data = edge_data["keys"]

            if edge_data["reduce"]:
                logger.info(f"reducing relation {relation_node}")
                join_df = relation_node.do_reduce(edge_data["relation_key"])
                # only relevant when reducing
                if self.auto_features:
                    logger.info(
                        f"performing auto_features on node {relation_node} with reduce key {edge_data['relation_key']}"
                    )
                    child_df = relation_node.auto_features(
                        reduce_key=edge_data["relation_key"],
                        # type_func_map=self.feature_typefunc_map,
                        type_func_map=self.feature_stype_map,
                        compute_layer=self.compute_layer,
                    )

                    # NOTE: this is pandas specific and will break
                    # on other compute layers for now
                    if self.compute_layer in [
                        ComputeLayerEnum.pandas,
                        ComputeLayerEnum.dask,
                        ComputeLayerEnum.daft,
                    ]:
                        if isinstance(join_df, pd.DataFrame) or isinstance(
                            join_df, dd.DataFrame
                        ):
                            join_df = join_df.merge(
                                child_df,
                                on=relation_node.colabbr(edge_data["relation_key"]),
                                suffixes=("", "_dupe"),
                            )
                        else:
                            join_df = child_df
                            if self.debug:
                                logger.debug(
                                    f"assigned join_df to be {child_df.columns}"
                                )
                    elif self.compute_layer == ComputeLayerEnum.spark:
                        if isinstance(join_df, pyspark.sql.dataframe.DataFrame):
                            join_df = join_df.join(
                                child_df,
                                on=join_df[
                                    relation_node.colabbr(edge_data["relation_key"])
                                ]
                                == child_df[
                                    relation_node.colabbr(edge_data["relation_key"])
                                ],
                                how="left",
                            )
                        else:
                            join_df = child_df
                            if self.debug:
                                logger.debug(
                                    f"assigned join_df to be {child_df.columns}"
                                )

            else:
                # in this case we will join the entire relation's dataframe
                logger.info(f"doing nothing with relation node {relation_node}")
                join_df = None

            logger.info(f"joining {relation_node} to {parent_node}")
            joined_df = self.join(parent_node, relation_node, relation_df=join_df)

            # Update the parent dataframe.
            parent_node.df = joined_df

            # Target variables.
            if self.label_node and (
                self.label_node == relation_node
                or relation_node.label_field is not None
            ):
                logger.info(f"Had label node {self.label_node}")
                # Automatic label generation.
                if isinstance(relation_node, DynamicNode):
                    if self.label_node == relation_node:
                        label_df = relation_node.default_label(
                            op=self.label_operation,
                            field=self.label_field,
                            reduce_key=edge_data["relation_key"],
                        )
                    elif relation_node.label_field is not None:
                        label_df = relation_node.default_label(
                            op=relation_node.label_operation
                            if relation_node.label_operation
                            else "count",
                            field=relation_node.label_field,
                            reduce_key=edge_data["relation_key"],
                        )
                # There should be an implementation of `do_labels`
                # when the instance is a `GraphReduceNode`.
                elif isinstance(relation_node, GraphReduceNode):
                    label_df = relation_node.do_labels(edge_data["relation_key"])

                logger.info(f"computed labels for {relation_node}")
                if label_df.__class__.__name__ != "NoneType":
                    joined_with_labels = self.join(
                        parent_node, relation_node, relation_df=label_df
                    )
                    parent_node.df = joined_with_labels

            # post-join annotations (if any)
            parent_node.do_post_join_annotate()
            # post-join filters (if any)
            if hasattr(parent_node, "do_post_join_filters"):
                parent_node.do_post_join_filters()

            # post-join aggregation
            if edge_data["reduce_after_join"]:
                parent_node.do_post_join_reduce(
                    edge_data["relation_key"], type_func_map=self.feature_stype_map
                )
