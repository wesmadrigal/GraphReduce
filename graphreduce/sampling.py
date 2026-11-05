"""Root-limited, relationship-preserving loading for graph execution."""

import networkx as nx
import pandas as pd
from dask import dataframe as dd

from graphreduce.enum import ComputeLayerEnum
from graphreduce.node import SQLNode


class RelationalSample:
    """Keep immutable load-stage keys so later transformations cannot shrink a sample."""

    def __init__(self, graph, size):
        self.graph = graph
        self.size = size
        self.sources = {}
        if graph.parent_node not in graph:
            raise ValueError("Sampling requires parent_node to be in the graph")
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Relational sampling requires an acyclic graph")
        reachable = {graph.parent_node, *nx.descendants(graph, graph.parent_node)}
        if any(node not in reachable and not node.is_date_node for node in graph):
            raise ValueError("All sampled nodes must be reachable from parent_node")
        self.nodes = [node for node in nx.topological_sort(graph) if node in reachable]
        for node in self.nodes:
            if not isinstance(node, SQLNode) and graph.compute_layer not in (
                ComputeLayerEnum.pandas,
                ComputeLayerEnum.dask,
            ):
                raise NotImplementedError(
                    "Relational sampling supports SQL, pandas, and Dask nodes"
                )

    def relationships(self, node):
        parents = list(self.graph.predecessors(node)) if node in self.graph else []
        for parent in parents:
            if parent.is_date_node:
                continue
            meta = self.graph.get_edge_data(parent, node)
            meta = meta.get("keys", meta)
            yield parent, meta["parent_key"], meta["relation_key"]
        # A configured date node can also be supplied without an explicit edge.
        if node is self.graph.date_node and not parents:
            yield self.graph.parent_node, self.graph.parent_node.pk, node.pk

    def sql(self, node, query):
        """Restrict the normal do_data query before creating its reference.

        EXISTS preserves duplicate source rows without multiplying them when
        parent keys repeat. Key values never leave the database.
        """
        source = f"({query.strip().rstrip(';')}) AS __gr_sample_row"
        if node is self.graph.parent_node:
            return f"SELECT * FROM {source} LIMIT {self.size}"
        predicates = []
        for parent, parent_key, relation_key in self.relationships(node):
            parent_columns = self.graph._resolve_prefixed_columns(parent, parent_key)
            child_columns = self.graph._resolve_prefixed_columns(node, relation_key)
            comparisons = " AND ".join(
                f"__gr_sample_parent.{node.render_identifier(pk)} = "
                f"__gr_sample_row.{node.render_identifier(fk)}"
                for pk, fk in zip(parent_columns, child_columns)
            )
            predicates.append(
                f"EXISTS (SELECT 1 FROM {self.sources[parent]} AS __gr_sample_parent "
                f"WHERE {comparisons})"
            )
        if not predicates:
            raise ValueError(f"No sampled parent relationship for {node.prefix}")
        return f"SELECT * FROM {source} WHERE " + " OR ".join(predicates)

    def filter_frame(self, node):
        if node is self.graph.parent_node:
            if isinstance(node.df, dd.DataFrame):
                node.df = node.df.head(self.size, npartitions=-1, compute=False)
            else:
                node.df = node.df.head(self.size).copy()
            return
        relationships = []
        for parent, parent_key, relation_key in self.relationships(node):
            parent_columns = self.graph._resolve_prefixed_columns(
                parent, parent_key, self.sources[parent].columns
            )
            child_columns = self.graph._resolve_prefixed_columns(
                node, relation_key, node.df.columns
            )
            keys = self.sources[parent][list(parent_columns)].dropna().drop_duplicates()
            # Only sampled key sets are collected for Dask; child rows stay lazy.
            if isinstance(keys, dd.DataFrame):
                keys = keys.compute()
            relationships.append((list(child_columns), pd.MultiIndex.from_frame(keys)))

        def select_rows(frame):
            keep = pd.Series(False, index=frame.index)
            for columns, keys in relationships:
                keep |= frame[columns].notna().all(axis=1) & pd.MultiIndex.from_frame(
                    frame[columns]
                ).isin(keys)
            return frame.loc[keep].copy()

        if isinstance(node.df, dd.DataFrame):
            node.df = node.df.map_partitions(select_rows, meta=node.df._meta)
        else:
            node.df = select_rows(node.df)

    def remember(self, node):
        if isinstance(node, SQLNode):
            self.sources[node] = node.get_current_ref()
            return
        # Snapshot only keys needed at the next hop, before annotations/filters.
        columns = []
        for child in self.graph.successors(node):
            meta = self.graph.get_edge_data(node, child)
            meta = meta.get("keys", meta)
            columns.extend(
                self.graph._resolve_prefixed_columns(
                    node, meta["parent_key"], node.df.columns
                )
            )
        self.sources[node] = node.df[list(dict.fromkeys(columns))].copy()
