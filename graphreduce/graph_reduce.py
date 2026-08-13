#!/usr/bin/env python
from __future__ import annotations

# std lib
import copy
import datetime
import functools
import operator
import typing
import uuid

# third party
import pandas as pd
import networkx as nx
from dask import dataframe as dd
from structlog import get_logger
import pyvis

try:
    import pyspark
    from pyspark.sql import functions as F
except ImportError:  # pragma: no cover - optional dependency
    pyspark = None
    F = None

try:
    import daft
except ImportError:  # pragma: no cover - optional dependency
    daft = None

# internal
from graphreduce.node import (
    DynamicNode,
    GraphReduceNode,
    KeySpec,
    key_parts,
    normalize_key,
)
from graphreduce.enum import ComputeLayerEnum, PeriodUnit, SQLOpType
from graphreduce.storage import StorageClient
from graphreduce.models import sqlop

logger = get_logger("GraphReduce")


SPARK_DF_TYPES = tuple()
if pyspark is not None:  # pragma: no branch
    SPARK_DF_TYPES = tuple(
        t
        for t in [
            getattr(getattr(pyspark, "sql", None), "DataFrame", None),
            getattr(
                getattr(getattr(pyspark, "sql", None), "dataframe", None),
                "DataFrame",
                None,
            ),
            getattr(
                getattr(
                    getattr(getattr(pyspark, "sql", None), "connect", None),
                    "dataframe",
                    None,
                ),
                "DataFrame",
                None,
            ),
        ]
        if t is not None
    )

DAFT_DF_TYPES = tuple()
if daft is not None:  # pragma: no branch
    daft_df = getattr(
        getattr(getattr(daft, "dataframe", None), "dataframe", None), "DataFrame", None
    )
    DAFT_DF_TYPES = (daft_df,) if daft_df is not None else tuple()


def _require_backend(module: typing.Any, backend: str, extra: str) -> None:
    if module is None:
        raise ImportError(
            f'{backend} backend is not installed. Install with `pip install "graphreduce[{extra}]"`.'
        )


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
            "multicategorical": ["first"],
            "sequence_numerical": ["min", "max"],
            "timestamp": ["min", "max"],
            "text_embedded": ["dummy"],
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
        date_filters_on_agg: bool = False,
        date_node: typing.Optional[GraphReduceNode] = None,
        train: bool = True,
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
            date_filters_on_agg: bool whether or not to automatically filter by dates during custom defined aggregations
            train: bool whether the graph is being built for training. If false, date-node joins are skipped.
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
        self.date_filters_on_agg = date_filters_on_agg
        self.train = train

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
        # SQL references must be unique when multiple graph instances share a
        # database connection, including parallel cutoff-date workers.
        self.execution_namespace = uuid.uuid4().hex[:12]

        self.debug = debug
        self.dry_run = dry_run
        # Keep track of all the SQL queries.
        self.sql_ops = []
        # Structured execution log for SQL-backed runs.
        self.executed_op_records = []
        # Frozen execution plan replay queues keyed by node / method / edge.
        self._execution_plan_queues = {}

        # If we have a date node.
        self.date_node = date_node
        if isinstance(self.date_node, GraphReduceNode) or issubclass(
            self.date_node.__class__, GraphReduceNode
        ):
            if not self.date_node.is_date_node:
                self.date_node.is_date_node = True

        if self.compute_layer == ComputeLayerEnum.spark:
            _require_backend(pyspark, "spark", "spark")
        if self.compute_layer == ComputeLayerEnum.spark and self.spark_sqlctx is None:
            raise Exception(
                f"Must provide a `spark_sqlctx` kwarg if using {self.compute_layer.value} as compute layer"
            )
        if self.compute_layer == ComputeLayerEnum.daft:
            _require_backend(daft, "daft", "daft")

        if self.label_node and (
            self.label_period_val is None or self.label_period_unit is None
        ):
            raise Exception(
                "If label_node is parameterized must provide values for `label_period_val` and `label_period_unit`"
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

    @property
    def params(self):
        return {
            "parent_node": self.parent_node,
            "cut_date": self.cut_date,
            "fmt": self.fmt,
            "compute_period_val": self.compute_period_val,
            "copute_period_unit": self.compute_period_unit,
            "compute_layer": self.compute_layer,
            "label_node": self.label_node,
            "label_field": self.label_field,
            "label_operation": self.label_operation,
            "label_period_val": self.label_period_val,
            "label_period_unit": self.label_period_unit,
            "auto_features": self.auto_features,
            "auto_feature_hops_back": self.auto_feature_hops_back,
            "auto_feature_hops_front": self.auto_feature_hops_front,
            "feature_typefunc_map": self.feature_typefunc_map,
            "feature_stype_map": self.feature_stype_map,
            "date_filters_on_agg": self.date_filters_on_agg,
            "train": self.train,
            "debug": self.debug,
            "lazy_execution": self._lazy_execution,
            "date_node": self.date_node,
        }

    def assign_parent(
        self,
        parent_node: GraphReduceNode,
    ):
        """
        Assign the parent-most node in the graph
        """
        self._parent_node = parent_node

    def _normalize_sqlop_list(
        self, ops: typing.Optional[typing.Union[sqlop, typing.List[sqlop]]]
    ) -> typing.List[sqlop]:
        """
        Normalize a single sqlop or list of sqlops into a list.
        """
        if ops is None:
            return []
        if isinstance(ops, list):
            return [_op for _op in ops if _op is not None]
        return [ops]

    def _ops_reference_labels(self, ops: typing.List[sqlop]) -> bool:
        """
        Check whether a list of sqlops references label-derived columns.
        """
        return any("_label" in getattr(op, "opval", "") for op in ops)

    def _record_executed_ops(
        self,
        node: GraphReduceNode,
        method_name: str,
        ops: typing.Optional[typing.Union[sqlop, typing.List[sqlop]]] = None,
        date_filter_ops: typing.Optional[typing.Union[sqlop, typing.List[sqlop]]] = None,
        sql: typing.Optional[str] = None,
        auto_generated: bool = False,
        edge: typing.Optional[typing.Tuple[GraphReduceNode, GraphReduceNode]] = None,
        reduce_key: typing.Optional[KeySpec] = None,
    ) -> typing.Optional[typing.Dict[str, typing.Any]]:
        """
        Record structured sqlop execution metadata while preserving the raw sqlops.
        """
        base_ops = self._normalize_sqlop_list(ops)
        extra_date_filter_ops = self._normalize_sqlop_list(date_filter_ops)
        combined_ops = [*base_ops, *extra_date_filter_ops]
        if not combined_ops:
            return None

        record = {
            "node": node,
            "node_prefix": getattr(node, "prefix", None),
            "node_name": node.__class__.__name__,
            "method_name": method_name,
            "ops": combined_ops,
            "method_ops": base_ops,
            "date_filter_ops": extra_date_filter_ops,
            "ops_excluding_date_filters": base_ops,
            "sql": sql,
            "auto_generated": auto_generated,
            "edge": edge,
            "reduce_key": reduce_key,
        }
        self.executed_op_records.append(record)
        return record

    def _serialize_edge(
        self,
        edge: typing.Optional[typing.Tuple[GraphReduceNode, GraphReduceNode]],
    ) -> typing.Optional[typing.Tuple[typing.Optional[str], typing.Optional[str]]]:
        """
        Convert an edge into a stable prefix tuple for plan replay.
        """
        if not edge:
            return None
        return tuple(getattr(node, "prefix", None) for node in edge)

    def _planned_queue_key(
        self,
        node: GraphReduceNode,
        method_name: str,
        edge: typing.Optional[typing.Tuple[GraphReduceNode, GraphReduceNode]] = None,
    ) -> typing.Tuple[
        typing.Optional[str],
        str,
        typing.Optional[typing.Tuple[typing.Optional[str], typing.Optional[str]]],
    ]:
        """
        Build the lookup key for a frozen execution plan record.
        """
        return (
            getattr(node, "prefix", None),
            method_name,
            self._serialize_edge(edge),
        )

    def _consume_planned_record(
        self,
        node: GraphReduceNode,
        method_name: str,
        edge: typing.Optional[typing.Tuple[GraphReduceNode, GraphReduceNode]] = None,
    ) -> typing.Optional[typing.Dict[str, typing.Any]]:
        """
        Pop the next frozen execution record for a node / method / edge combination.
        """
        queue_key = self._planned_queue_key(node=node, method_name=method_name, edge=edge)
        queue = self._execution_plan_queues.get(queue_key)
        if queue:
            return queue.pop(0)
        return None

    def get_executed_op_records(
        self,
        method_name: typing.Optional[str] = None,
        exclude_date_filters: bool = False,
        auto_generated: typing.Optional[bool] = None,
    ) -> typing.List[typing.Dict[str, typing.Any]]:
        """
        Return execution records, optionally filtered by method and auto-generated state.
        """
        records = self.executed_op_records
        if method_name is not None:
            records = [record for record in records if record["method_name"] == method_name]
        if auto_generated is not None:
            records = [
                record for record in records if record["auto_generated"] == auto_generated
            ]
        if not exclude_date_filters:
            return list(records)

        filtered_records = []
        for record in records:
            filtered_record = dict(record)
            filtered_record["ops"] = list(record["ops_excluding_date_filters"])
            filtered_records.append(filtered_record)
        return filtered_records

    def get_executed_sqlops_by_method(
        self,
        exclude_date_filters: bool = False,
        auto_generated: typing.Optional[bool] = None,
    ) -> typing.Dict[str, typing.List[sqlop]]:
        """
        Return the executed sqlops grouped by method name.
        """
        grouped = {}
        for record in self.get_executed_op_records(
            exclude_date_filters=exclude_date_filters,
            auto_generated=auto_generated,
        ):
            grouped.setdefault(record["method_name"], []).extend(record["ops"])
        return grouped

    def _rebind_sqlop_dates_for_node(
        self,
        node: GraphReduceNode,
        ops: typing.List[sqlop],
        original_cut_date: typing.Optional[datetime.datetime],
    ) -> typing.List[sqlop]:
        """
        Rebind date literals in frozen sqlops from the original cut date to this graph's cut date.
        """
        if not original_cut_date or not self.cut_date:
            return copy.deepcopy(ops)

        rebound_ops = copy.deepcopy(ops)
        original_cut_date = (
            original_cut_date
            if isinstance(original_cut_date, datetime.datetime)
            else datetime.datetime.fromisoformat(str(original_cut_date))
        )
        current_cut_date = (
            self.cut_date
            if isinstance(self.cut_date, datetime.datetime)
            else datetime.datetime.fromisoformat(str(self.cut_date))
        )

        replacements = {
            str(original_cut_date): str(current_cut_date),
        }

        if hasattr(node, "compute_period_minutes"):
            replacements[
                str(
                    original_cut_date
                    - datetime.timedelta(minutes=node.compute_period_minutes())
                )
            ] = str(
                current_cut_date
                - datetime.timedelta(minutes=node.compute_period_minutes())
            )

        if hasattr(node, "label_period_minutes") and getattr(
            node, "label_period_val", None
        ) is not None:
            replacements[
                str(
                    original_cut_date
                    + datetime.timedelta(minutes=node.label_period_minutes())
                )
            ] = str(
                current_cut_date
                + datetime.timedelta(minutes=node.label_period_minutes())
            )

        for period in getattr(node, "ts_periods", []) or []:
            replacements[
                str(original_cut_date - datetime.timedelta(days=period))
            ] = str(current_cut_date - datetime.timedelta(days=period))

        for op in rebound_ops:
            for old_val, new_val in sorted(
                replacements.items(), key=lambda item: len(item[0]), reverse=True
            ):
                op.opval = op.opval.replace(old_val, new_val)

        return rebound_ops

    def freeze_execution_plan(self) -> typing.Dict[str, typing.Any]:
        """
        Freeze the executed SQL plan so it can be replayed on a later graph instance.
        """
        grouped_by_method = {}
        frozen_records = []

        for record in self.executed_op_records:
            method_ops = copy.deepcopy(record["ops"])
            grouped_by_method.setdefault(record["method_name"], []).extend(
                copy.deepcopy(record["ops"])
            )
            frozen_records.append(
                {
                    "node_prefix": record["node_prefix"],
                    "method_name": record["method_name"],
                    "ops": method_ops,
                    "method_ops": copy.deepcopy(record["method_ops"]),
                    "date_filter_ops": copy.deepcopy(record["date_filter_ops"]),
                    "auto_generated": record["auto_generated"],
                    "reduce_key": record["reduce_key"],
                    "edge": self._serialize_edge(record["edge"]),
                }
            )

        return {
            "cut_date": self.cut_date,
            "records": frozen_records,
            "ops_by_method": grouped_by_method,
        }

    def apply_execution_plan(self, plan: typing.Dict[str, typing.Any]) -> None:
        """
        Apply a frozen execution plan by loading sqlop queues per node / method / edge.
        """
        original_cut_date = plan.get("cut_date")
        node_lookup = {getattr(node, "prefix", None): node for node in self.nodes()}
        replay_queues = {}

        for record in plan.get("records", []):
            if not self.train and record["method_name"] == "do_labels":
                continue
            node = node_lookup.get(record["node_prefix"])
            if node is None:
                continue
            rebound_method_ops = self._rebind_sqlop_dates_for_node(
                node=node,
                ops=record.get("method_ops", record["ops"]),
                original_cut_date=original_cut_date,
            )
            rebound_date_filter_ops = self._rebind_sqlop_dates_for_node(
                node=node,
                ops=record.get("date_filter_ops", []),
                original_cut_date=original_cut_date,
            )
            rebound_ops = [*rebound_method_ops, *rebound_date_filter_ops]
            if (
                not self.train
                and record["method_name"]
                in ["do_post_join_annotate", "do_post_join_filters"]
                and self._ops_reference_labels(rebound_ops)
            ):
                continue
            queue_key = (
                record["node_prefix"],
                record["method_name"],
                tuple(record["edge"]) if record.get("edge") else None,
            )
            planned_reduce_key = record.get("reduce_key")
            if planned_reduce_key is not None:
                planned_reduce_key = normalize_key(
                    planned_reduce_key,
                    name="reduce_key",
                )
            replay_queues.setdefault(queue_key, []).append(
                {
                    "node_prefix": record["node_prefix"],
                    "method_name": record["method_name"],
                    "ops": rebound_ops,
                    "method_ops": rebound_method_ops,
                    "date_filter_ops": rebound_date_filter_ops,
                    "auto_generated": record["auto_generated"],
                    "reduce_key": planned_reduce_key,
                    "edge": tuple(record["edge"]) if record.get("edge") else None,
                }
            )

        self._execution_plan_queues = replay_queues

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
            "execution_namespace",
            # "date_node"
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
        parent_key: KeySpec,
        relation_key: KeySpec,
        # need to enforce this better
        relation_type: str = "parent_child",
        reduce: bool = True,
        reduce_after_join: bool = False,
    ):
        """
        Add an entity relation
        """

        if parent_node.is_date_node:
            raise Exception(
                "Date nodes can only be relation nodes in relationships like this `.add_entity_edge(parent_node, date_node, ...)`"
            )

        if reduce and reduce_after_join:
            raise Exception("only one can be true: `reduce` or `reduce_after_join`")

        parent_key = normalize_key(parent_key, name="parent_key")
        relation_key = normalize_key(relation_key, name="relation_key")
        parent_key_parts = key_parts(parent_key, name="parent_key")
        relation_key_parts = key_parts(relation_key, name="relation_key")
        if len(parent_key_parts) != len(relation_key_parts):
            raise ValueError(
                "parent_key and relation_key must contain the same number of "
                f"columns; got {len(parent_key_parts)} and "
                f"{len(relation_key_parts)}"
            )
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

    def _node_needs_date_node_propagation(self, node: GraphReduceNode) -> bool:
        """
        Return whether a node needs the dynamic date reference joined into it.

        Nodes with no date key still need the reference when dated descendants
        depend on them as the next propagation hop.
        """
        if node.date_key:
            return True
        return any(
            descendant.date_key and not descendant.is_date_node
            for descendant in nx.descendants(self, node)
        )

    def _date_node_propagation_edge(
        self, node: GraphReduceNode
    ) -> typing.Optional[typing.Tuple[GraphReduceNode, GraphReduceNode]]:
        """
        Find an inbound edge whose parent already has a propagated date node.
        """
        for parent, child in self.in_edges(node):
            if parent.is_date_node:
                continue
            if getattr(parent, "date_node", None):
                return parent, child
        return None

    def _resolve_prefixed_column(
        self,
        node: GraphReduceNode,
        column: str,
        columns: typing.Optional[typing.Iterable[typing.Any]] = None,
    ) -> str:
        """
        Normalize a join key to the concrete column name used on a node.

        Some callers pass raw keys like `UserID`, while others pass already-
        prefixed keys like `cd_UserID`. SQL join generation should accept both.
        """

        prefixed = f"{node.prefix}_{column}"
        if isinstance(column, str) and column.startswith(f"{node.prefix}_"):
            return column

        if columns is not None:
            column_names = {str(col).lower() for col in columns}
            if str(column).lower() in column_names:
                return str(column)
            if prefixed.lower() in column_names:
                return prefixed

        return prefixed

    def _resolve_prefixed_columns(
        self,
        node: GraphReduceNode,
        key: KeySpec,
        columns: typing.Optional[typing.Iterable[typing.Any]] = None,
    ) -> typing.Tuple[str, ...]:
        """Resolve every ordered component of a scalar or composite key."""

        resolved = tuple(
            self._resolve_prefixed_column(node, part, columns)
            for part in key_parts(key)
        )
        if columns is not None:
            available = {str(column).lower() for column in columns}
            missing = [column for column in resolved if column.lower() not in available]
            if missing:
                raise KeyError(
                    f"missing key columns on {node}: {missing}; "
                    f"available columns: {list(columns)}"
                )
        return resolved

    @staticmethod
    def _frame_columns(frame: typing.Any) -> typing.Optional[typing.Iterable[str]]:
        """Return backend dataframe column names when available."""

        if hasattr(frame, "columns"):
            return frame.columns
        if hasattr(frame, "column_names"):
            return frame.column_names
        return None

    def _spark_join_frames(
        self,
        left_df: typing.Any,
        right_df: typing.Any,
        left_columns: typing.Sequence[str],
        right_columns: typing.Sequence[str],
        how: str = "left",
    ) -> typing.Any:
        """Join Spark frames on ordered keys, isolating colliding right keys."""

        resolved_right_columns = list(right_columns)
        renamed_right_columns = []
        for ix, right_column in enumerate(resolved_right_columns):
            if right_column not in left_df.columns:
                continue
            candidate = f"{right_column}_dupe"
            while candidate in left_df.columns or candidate in right_df.columns:
                candidate = f"{candidate}_dupe"
            right_df = right_df.withColumnRenamed(right_column, candidate)
            resolved_right_columns[ix] = candidate
            renamed_right_columns.append(candidate)

        conditions = [
            left_df[left_column] == right_df[right_column]
            for left_column, right_column in zip(
                left_columns, resolved_right_columns
            )
        ]
        condition = functools.reduce(operator.and_, conditions)
        joined = left_df.join(right_df, on=condition, how=how)
        for renamed_column in renamed_right_columns:
            joined = joined.drop(F.col(renamed_column))
        return joined

    def join_any(
        self,
        to_node: GraphReduceNode,
        from_node: GraphReduceNode,
        how: str = "left",
        to_node_key: typing.Optional[KeySpec] = None,
        from_node_key: typing.Optional[KeySpec] = None,
        to_node_df=None,
        from_node_df=None,
    ):
        """
        Join the relations.
        """

        if (to_node_key is None) != (from_node_key is None):
            raise ValueError("to_node_key and from_node_key must be provided together")
        if to_node_key is None:
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

        to_frame = to_node.df if to_node_df is None else to_node_df
        from_frame = from_node.df if from_node_df is None else from_node_df
        if len(key_parts(to_node_key, name="to_node_key")) != len(
            key_parts(from_node_key, name="from_node_key")
        ):
            raise ValueError(
                "to_node_key and from_node_key must contain the same number "
                "of columns"
            )
        to_columns = self._resolve_prefixed_columns(
            to_node,
            to_node_key,
            self._frame_columns(to_frame),
        )
        from_columns = self._resolve_prefixed_columns(
            from_node,
            from_node_key,
            self._frame_columns(from_frame),
        )

        if self.compute_layer in [ComputeLayerEnum.pandas, ComputeLayerEnum.dask]:
            joined = to_frame.merge(
                from_frame,
                left_on=list(to_columns),
                right_on=list(from_columns),
                suffixes=("", "_dupe"),
                how=how,
            )
            self._mark_merged(to_node, from_node)
            if "key_0" in joined.columns:
                joined = joined[[c for c in joined.columns if c != "key_0"]]
                return joined
            else:
                return joined
        elif self.compute_layer == ComputeLayerEnum.daft:
            _require_backend(daft, "daft", "daft")
            joined = to_frame.join(
                from_frame,
                left_on=list(to_columns),
                right_on=list(from_columns),
                suffix="_dupe",
                how=how,
            )
            self._mark_merged(to_node, from_node)
            return joined
        elif self.compute_layer == ComputeLayerEnum.spark:
            _require_backend(pyspark, "spark", "spark")
            if isinstance(to_frame, SPARK_DF_TYPES) and isinstance(
                from_frame, SPARK_DF_TYPES
            ):
                joined = self._spark_join_frames(
                    to_frame,
                    from_frame,
                    to_columns,
                    from_columns,
                    how=how,
                )
                self._mark_merged(to_node, from_node)
                return joined
            else:
                raise Exception(
                    f"Cannot use spark on dataframe of type: {type(to_frame)}"
                )
        # TODO: make a `DialectEnum.sql` catchall for this.
        elif self.compute_layer in [
            ComputeLayerEnum.athena,
            ComputeLayerEnum.snowflake,
            ComputeLayerEnum.redshift,
            ComputeLayerEnum.postgres,
            ComputeLayerEnum.sqlite,
            ComputeLayerEnum.trino,
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
        reverse_edge = False

        if not meta:
            meta = self.get_edge_data(relation_node, parent_node)
            if not meta:
                raise Exception(
                    f"no edge metadata for {parent_node} and {relation_node}"
                )
            reverse_edge = True

        if meta.get("keys"):
            meta = meta["keys"]

        if meta and meta["relation_type"] in ["parent_child", "peer"]:
            if reverse_edge:
                parent_pk = meta["relation_key"]
                relation_fk = meta["parent_key"]
            else:
                parent_pk = meta["parent_key"]
                relation_fk = meta["relation_key"]

        relation_frame = relation_node.df if relation_df is None else relation_df
        parent_columns = self._resolve_prefixed_columns(
            parent_node,
            parent_pk,
            self._frame_columns(parent_node.df),
        )
        relation_columns = self._resolve_prefixed_columns(
            relation_node,
            relation_fk,
            self._frame_columns(relation_frame),
        )

        if self.compute_layer in [ComputeLayerEnum.pandas, ComputeLayerEnum.dask]:
            if isinstance(relation_df, pd.DataFrame) or isinstance(
                relation_df, dd.DataFrame
            ):
                joined = parent_node.df.merge(
                    relation_df,
                    left_on=list(parent_columns),
                    right_on=list(relation_columns),
                    suffixes=("", "_dupe"),
                    how="left",
                )
            else:
                joined = parent_node.df.merge(
                    relation_node.df,
                    left_on=list(parent_columns),
                    right_on=list(relation_columns),
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
            _require_backend(daft, "daft", "daft")
            if isinstance(relation_df, DAFT_DF_TYPES):
                joined = parent_node.df.join(
                    relation_df,
                    left_on=list(parent_columns),
                    right_on=list(relation_columns),
                    suffix="_dupe",
                    how="left",
                )
            else:
                joined = parent_node.df.join(
                    relation_node.df,
                    left_on=list(parent_columns),
                    right_on=list(relation_columns),
                    suffix="_dupe",
                    how="left",
                )
            self._mark_merged(parent_node, relation_node)
            return joined
        elif self.compute_layer == ComputeLayerEnum.spark:
            _require_backend(pyspark, "spark", "spark")
            valid_dataframe_types = SPARK_DF_TYPES
            if isinstance(relation_df, valid_dataframe_types) and isinstance(
                parent_node.df, valid_dataframe_types
            ):
                joined = self._spark_join_frames(
                    parent_node.df,
                    relation_df,
                    parent_columns,
                    relation_columns,
                    how="left",
                )
                self._mark_merged(parent_node, relation_node)
                return joined
            elif isinstance(parent_node.df, valid_dataframe_types) and isinstance(
                relation_node.df, valid_dataframe_types
            ):
                joined = self._spark_join_frames(
                    parent_node.df,
                    relation_node.df,
                    parent_columns,
                    relation_columns,
                    how="left",
                )
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
        parent_node_key: typing.Optional[KeySpec] = None,
        relation_node_key: typing.Optional[KeySpec] = None,
    ) -> str:
        """
        Joins two graph reduce nodes of SQL dialect.
        """

        meta = self.get_edge_data(parent_node, relation_node)
        reverse_edge = False

        if (parent_node_key is None) != (relation_node_key is None):
            raise ValueError(
                "parent_node_key and relation_node_key must be provided together"
            )
        if not meta and parent_node_key is None:
            meta = self.get_edge_data(relation_node, parent_node)
            if not meta:
                raise Exception(
                    f"no edge metadata for {parent_node} and {relation_node}"
                )
            reverse_edge = True
        if meta and meta.get("keys"):
            meta = meta["keys"]

        if meta and meta["relation_type"] in ["parent_child", "peer"]:
            if reverse_edge:
                parent_pk = meta["relation_key"]
                relation_fk = meta["parent_key"]
            else:
                parent_pk = meta["parent_key"]
                relation_fk = meta["relation_key"]
        elif not meta and parent_node_key is not None:
            parent_pk = parent_node_key
            relation_fk = relation_node_key

        if len(key_parts(parent_pk, name="parent_key")) != len(
            key_parts(relation_fk, name="relation_key")
        ):
            raise ValueError(
                "parent and relation join keys must contain the same number "
                "of columns"
            )

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
        parent_pk_cols = self._resolve_prefixed_columns(
            parent_node, parent_pk, parent_samp.columns
        )
        relation_fk_cols = self._resolve_prefixed_columns(
            relation_node, relation_fk, relation_samp.columns
        )
        join_predicate = " AND ".join(
            f"parent.{parent_col} = relation.{relation_col}"
            for parent_col, relation_col in zip(
                parent_pk_cols, relation_fk_cols
            )
        )
        parent_cols_lower = {_x.lower() for _x in parent_samp.columns}
        duplicate_relation_cols = [
            c for c in relation_samp.columns if c.lower() in parent_cols_lower
        ]
        if duplicate_relation_cols:
            logger.info(
                "removing duplicate columns on join",
                columns=duplicate_relation_cols,
            )
            relation_cols = [
                f"relation.{c}"
                for c in relation_samp.columns
                if c.lower() not in parent_cols_lower
            ]
            sel = ",".join(relation_cols)
            relation_select = f", {sel}" if sel else ""
            JOIN_SQL = f"""
                SELECT parent.*{relation_select}
                FROM {parent_table} parent
                LEFT JOIN {relation_table} relation
                ON {join_predicate}
            """
        else:
            JOIN_SQL = f"""
                SELECT parent.*, relation.*
                FROM {parent_table} parent
                LEFT JOIN {relation_table} relation
                ON {join_predicate}
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

    def do_transformations_sql(self, dry: bool = False):
        """
        Perform all graph transformations
        1) hydrate graph
        2) check for duplicate prefixes
         2a) if there is a `date_node` push it
             down to the whole graph
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
        self.executed_op_records = []

        logger.info("checking for prefix uniqueness")
        self.prefix_uniqueness()

        # Node-level data prep operations.
        for node in nx.bfs_tree(self, source=self.parent_node):
            if node.is_date_node:
                continue
            # for node in self.nodes():
            # `self.do_data` must always return some `sqlop`
            planned_data = self._consume_planned_record(node, "do_data")
            ops = (
                list(planned_data["ops"])
                if planned_data
                else self._normalize_sqlop_list(node.do_data())
            )
            if not ops:
                raise Exception(
                    f"{node.__class__.__name__}.do_data must be implemented"
                )
            data_sql = node.build_query(ops)
            self._record_executed_ops(
                node=node,
                method_name="do_data",
                ops=planned_data["method_ops"] if planned_data else ops,
                date_filter_ops=planned_data["date_filter_ops"] if planned_data else None,
                sql=data_sql,
                auto_generated=planned_data["auto_generated"] if planned_data else False,
            )

            logger.debug(f"do data: {data_sql}")
            self.sql_ops.append(data_sql)
            node.create_ref(
                data_sql,
                node.do_data,
                schema=self._checkpoint_schema,
                dry=self.dry_run,
            )
            # Now append the reference SQL.
            if node._ref_sql:
                self.sql_ops.append(node._ref_sql)
                node._ref_sql = None

            planned_annotate = self._consume_planned_record(node, "do_annotate")
            annotate_ops = (
                list(planned_annotate["ops"])
                if planned_annotate
                else self._normalize_sqlop_list(node.do_annotate())
            )
            annotate_sql = node.build_query(annotate_ops)
            self._record_executed_ops(
                node=node,
                method_name="do_annotate",
                ops=planned_annotate["method_ops"] if planned_annotate else annotate_ops,
                date_filter_ops=(
                    planned_annotate["date_filter_ops"] if planned_annotate else None
                ),
                sql=annotate_sql,
                auto_generated=planned_annotate["auto_generated"] if planned_annotate else False,
            )
            logger.debug(f"do annotate: {annotate_sql}")
            self.sql_ops.append(annotate_sql)
            node.create_ref(
                annotate_sql,
                node.do_annotate,
                schema=self._checkpoint_schema,
                dry=self.dry_run,
            )
            if node._ref_sql:
                self.sql_ops.append(node._ref_sql)
                node._ref_sql = None

            planned_filters = self._consume_planned_record(node, "do_filters")
            filter_ops = (
                list(planned_filters["ops"])
                if planned_filters
                else self._normalize_sqlop_list(node.do_filters())
            )
            filter_sql = node.build_query(filter_ops)
            self._record_executed_ops(
                node=node,
                method_name="do_filters",
                ops=planned_filters["method_ops"] if planned_filters else filter_ops,
                date_filter_ops=planned_filters["date_filter_ops"] if planned_filters else None,
                sql=filter_sql,
                auto_generated=planned_filters["auto_generated"] if planned_filters else False,
            )
            logger.debug(f"do filters: {filter_sql}")
            self.sql_ops.append(filter_sql)
            node.create_ref(
                filter_sql,
                node.do_filters,
                schema=self._checkpoint_schema,
                dry=self.dry_run,
            )
            if node._ref_sql:
                self.sql_ops.append(node._ref_sql)
                node._ref_sql = None

            planned_normalize = self._consume_planned_record(node, "do_normalize")
            normalize_ops = (
                list(planned_normalize["ops"])
                if planned_normalize
                else self._normalize_sqlop_list(node.do_normalize())
            )
            normalize_sql = node.build_query(normalize_ops)
            self._record_executed_ops(
                node=node,
                method_name="do_normalize",
                ops=planned_normalize["method_ops"] if planned_normalize else normalize_ops,
                date_filter_ops=(
                    planned_normalize["date_filter_ops"] if planned_normalize else None
                ),
                sql=normalize_sql,
                auto_generated=planned_normalize["auto_generated"] if planned_normalize else False,
            )
            self.sql_ops.append(normalize_sql)
            node.create_ref(
                normalize_sql,
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

            # If there is a `date_node` then we need
            # to push it down to all of the relationships.
            # For now we require that the `date_node` be
            # at the `parent_node` granularity and linked
            # to it.
            # The first iteration of this will always be
            # the `parent_node`.
            if self.date_node and self.train:
                logger.info(f"Found date node {self.date_node}")
                if node == self.parent_node:
                    # Load the data into the date node.
                    self.date_node.create_ref(
                        self.date_node.build_query(self.date_node.do_data()),
                        self.date_node.do_data,
                        schema=self._checkpoint_schema,
                        dry=self.dry_run,
                    )
                    if self.date_node._ref_sql:
                        self.sql_ops.append(self.date_node._ref_sql)
                    # Merge the date table with the parent node.
                    self.join_sql(
                        node,
                        self.date_node,
                        parent_node_key=node.pk,
                        relation_node_key=self.date_node.pk,
                    )
                    node.date_node = self.date_node
                # Needs a date key
                elif self._node_needs_date_node_propagation(node):
                    # For all other nodes we need to leverage
                    # leverage the existing relationship paths
                    # to push the date_node down through the graph.
                    # Parent:pk -> DateNode:pk = Child:fk -> DateNode:pk
                    parent_edge = self._date_node_propagation_edge(node)
                    if parent_edge is None:
                        raise ValueError(
                            f"Could not propagate date_node to {node}: no inbound parent has a propagated date_node"
                        )
                    my_parent = parent_edge[0]
                    parent_date_node = my_parent.date_node
                    meta = self.get_edge_data(parent_edge[0], parent_edge[1])
                    if meta.get("keys"):
                        meta = meta["keys"]
                    if meta and meta["relation_type"] == "parent_child":
                        parent_pk = meta["parent_key"]
                        relation_fk = meta["relation_key"]
                    elif meta and meta["relation_type"] == "peer":
                        parent_pk = meta["parent_key"]
                        relation_fk = meta["relation_key"]
                    date_prefix = f"{parent_date_node.prefix}_"
                    date_key = parent_date_node.date_key
                    date_agg_func = my_parent.get_pick_one_value_agg()
                    if date_key.startswith(date_prefix):
                        propagated_date_col = date_key
                        propagated_date_key = date_key[len(date_prefix) :]
                    else:
                        propagated_date_col = f"{date_prefix}{date_key}"
                        propagated_date_key = date_key
                    parent_pk_parts = key_parts(parent_pk, name="parent_key")
                    propagated_parent_key_cols = self._resolve_prefixed_columns(
                        my_parent, parent_pk
                    )
                    propagated_key_select = ",\n                                ".join(
                        f"{parent_col} as "
                        f"{parent_date_node.prefix}_{parent_key_part}"
                        for parent_col, parent_key_part in zip(
                            propagated_parent_key_cols, parent_pk_parts
                        )
                    )
                    propagated_key_group = ", ".join(
                        propagated_parent_key_cols
                    )
                    # Grab the date data from the parent and merge
                    # it.
                    dn = my_parent.__class__(
                        fpath=parent_date_node.prefix,
                        prefix=parent_date_node.prefix,
                        date_key=propagated_date_key,
                        table_name=parent_date_node.table_name,
                        # Use the key from the edge being propagated.
                        pk=parent_pk,
                        do_data_ops=sqlop(
                            optype=SQLOpType.custom,
                            opval=f"""
                                select {propagated_key_select},
                                {date_agg_func}({propagated_date_col}) as {propagated_date_col}
                                from {my_parent._cur_data_ref}
                                group by {propagated_key_group}
                                """,
                        ),
                        client=self._sql_client,
                        execution_namespace=self.execution_namespace,
                    )
                    dn.create_ref(
                        dn.build_query(dn.do_data()),
                        dn.do_data,
                        schema=self._checkpoint_schema,
                        dry=self.dry_run,
                    )
                    if dn._ref_sql:
                        self.sql_ops.append(dn._ref_sql)
                    # merge these now.
                    self.join_sql(
                        node, dn, parent_node_key=relation_fk, relation_node_key=dn.pk
                    )
                    # Go ahead and add the date node
                    # to this node as a reference for
                    # future.
                    node.date_node = dn
            elif self.date_node and not self.train:
                logger.info(
                    "Skipping date node joins because GraphReduce was initialized with train=False"
                )

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

            if relation_node.is_date_node:
                if self.train:
                    logger.info(
                        f"Skipping date-node edge {relation_node} during depth-first traversal because it was handled earlier"
                    )
                else:
                    logger.info(
                        f"Skipping date-node edge {relation_node} because GraphReduce was initialized with train=False"
                    )
                continue

            planned_reduce = None
            reduce_method_ops = None
            if edge_data["reduce"] and not relation_node.is_date_node:
                planned_reduce = self._consume_planned_record(
                    relation_node,
                    "do_reduce",
                    edge=edge,
                )
                reduce_method_ops = (
                    planned_reduce["method_ops"]
                    if planned_reduce
                    else relation_node.do_reduce(edge_data["relation_key"])
                )

            if edge_data["reduce"] and not relation_node.is_date_node:
                logger.info(f"reducing relation {relation_node}")

                # Check for automatic feature engineering.
                if planned_reduce:
                    reduce_sql = relation_node.build_query(planned_reduce["ops"])
                    self._record_executed_ops(
                        node=relation_node,
                        method_name="do_reduce",
                        ops=planned_reduce["method_ops"],
                        date_filter_ops=planned_reduce["date_filter_ops"],
                        sql=reduce_sql,
                        auto_generated=planned_reduce["auto_generated"],
                        edge=edge,
                        reduce_key=edge_data["relation_key"],
                    )
                    self.sql_ops.append(reduce_sql)
                    logger.info(f"{reduce_sql}")
                    relation_node.create_ref(
                        reduce_sql,
                        relation_node.do_reduce,
                        schema=self._checkpoint_schema,
                        dry=self.dry_run,
                    )
                    if relation_node._ref_sql:
                        self.sql_ops.append(relation_node._ref_sql)
                        relation_node._ref_sql = None

                elif self.auto_features and not reduce_method_ops:
                    logger.info(f"performing auto_features on node {relation_node}")
                    auto_feature_ops = self._normalize_sqlop_list(
                        relation_node.auto_features(
                        reduce_key=edge_data["relation_key"],
                        # type_func_map=self.feature_typefunc_map,
                        type_func_map=self.feature_stype_map,
                        compute_layer=self.compute_layer,
                    ))
                    auto_feature_date_filters = [
                        op for op in auto_feature_ops if op.optype == SQLOpType.where
                    ]
                    auto_feature_base_ops = [
                        op for op in auto_feature_ops if op.optype != SQLOpType.where
                    ]
                    auto_feature_sql = relation_node.build_query(auto_feature_ops)
                    self._record_executed_ops(
                        node=relation_node,
                        method_name="do_reduce",
                        ops=auto_feature_base_ops,
                        date_filter_ops=auto_feature_date_filters,
                        sql=auto_feature_sql,
                        auto_generated=True,
                        edge=edge,
                        reduce_key=edge_data["relation_key"],
                    )
                    self.sql_ops.append(auto_feature_sql)
                    logger.info(f"{auto_feature_sql}")
                    relation_node.create_ref(
                        auto_feature_sql,
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
                    tfilt = self._normalize_sqlop_list(tfilt)
                    # NOTE: we do not automatically do date filtering
                    # here so maybe that should be a top-level parameter
                    # for when we have a custom reduce implementation?
                    reduce_base_ops = self._normalize_sqlop_list(
                        reduce_method_ops
                    )
                    reduce_ops = list(reduce_base_ops)
                    if self.date_filters_on_agg:
                        reduce_ops = reduce_ops + tfilt
                        logger.info(f"Added in date filtering ops: {tfilt}")
                    reduce_sql = relation_node.build_query(reduce_ops)
                    self._record_executed_ops(
                        node=relation_node,
                        method_name="do_reduce",
                        ops=reduce_base_ops,
                        date_filter_ops=tfilt if self.date_filters_on_agg else None,
                        sql=reduce_sql,
                        auto_generated=False,
                        edge=edge,
                        reduce_key=edge_data["relation_key"],
                    )
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

            planned_label = None
            label_method_ops = None
            if self.train:
                planned_label = self._consume_planned_record(
                    relation_node,
                    "do_labels",
                    edge=edge,
                )
                label_method_ops = (
                    planned_label["method_ops"]
                    if planned_label
                    else relation_node.do_labels(edge_data["relation_key"])
                )
            # Target variables.
            if (
                self.train
                and (
                    (
                        self.label_node
                        and (
                            self.label_node == relation_node
                            or relation_node.label_field is not None
                        )
                    )
                    or label_method_ops is not None
                )
            ):
                logger.info(f"Had label node {self.label_node}")

                # Get the reference right before `do_reduce`
                # so the records are not aggregated yet.
                if relation_node._merged:
                    data_ref = relation_node.get_ref_name(
                        "join", lookup=True, schema=self._checkpoint_schema
                    )
                else:
                    data_ref = relation_node.get_ref_name(
                        relation_node.do_filters,
                        lookup=True,
                        schema=self._checkpoint_schema,
                    )

                # TODO: don't need to reduce if it's 1:1 cardinality.
                if planned_label:
                    label_sql = relation_node.build_query(
                        planned_label["ops"],
                        data_ref=data_ref,
                    )
                    self._record_executed_ops(
                        node=relation_node,
                        method_name="do_labels",
                        ops=planned_label["method_ops"],
                        date_filter_ops=planned_label["date_filter_ops"],
                        sql=label_sql,
                        auto_generated=planned_label["auto_generated"],
                        edge=edge,
                        reduce_key=edge_data["relation_key"],
                    )
                    self.sql_ops.append(label_sql)
                    logger.info(f"SQL Ops: {self.sql_ops[-1]}")
                    label_ref = relation_node.create_ref(
                        label_sql,
                        relation_node.do_labels,
                        schema=self._checkpoint_schema,
                        dry=self.dry_run,
                    )
                    if relation_node._ref_sql:
                        self.sql_ops.append(relation_node._ref_sql)
                        relation_node._ref_sql = None
                elif self.auto_features and not label_method_ops:
                    label_ops = self._normalize_sqlop_list(
                        relation_node.default_label(
                            op=self.label_operation,
                            field=self.label_field,
                            reduce_key=edge_data["relation_key"],
                        )
                    )
                    label_sql = relation_node.build_query(
                        label_ops,
                        data_ref=data_ref,
                    )
                    self._record_executed_ops(
                        node=relation_node,
                        method_name="do_labels",
                        ops=label_ops,
                        sql=label_sql,
                        auto_generated=True,
                        edge=edge,
                        reduce_key=edge_data["relation_key"],
                    )
                    self.sql_ops.append(label_sql)
                    logger.info(f"SQL Ops: {self.sql_ops[-1]}")
                    label_ref = relation_node.create_ref(
                        label_sql,
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
                    if self.date_filters_on_agg:
                        tfilt = relation_node.prep_for_labels()
                    else:
                        tfilt = None
                    tfilt = self._normalize_sqlop_list(tfilt)

                    label_base_ops = self._normalize_sqlop_list(
                        label_method_ops
                    )
                    ops = list(label_base_ops)
                    if tfilt:
                        ops = ops + tfilt

                    label_sql = relation_node.build_query(ops, data_ref=data_ref)
                    self._record_executed_ops(
                        node=relation_node,
                        method_name="do_labels",
                        ops=label_base_ops,
                        date_filter_ops=tfilt if self.date_filters_on_agg else None,
                        sql=label_sql,
                        auto_generated=False,
                        edge=edge,
                        reduce_key=edge_data["relation_key"],
                    )
                    self.sql_ops.append(label_sql)
                    logger.info(f"SQL Ops: {self.sql_ops[-1]}")
                    label_ref = relation_node.create_ref(
                        label_sql,
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
            planned_post_join_annotate = self._consume_planned_record(
                parent_node,
                "do_post_join_annotate",
                edge=edge,
            )
            post_join_annotate_ops = (
                list(planned_post_join_annotate["ops"])
                if planned_post_join_annotate
                else self._normalize_sqlop_list(parent_node.do_post_join_annotate())
            )
            pja_sql = parent_node.build_query(post_join_annotate_ops)
            self._record_executed_ops(
                node=parent_node,
                method_name="do_post_join_annotate",
                ops=(
                    planned_post_join_annotate["method_ops"]
                    if planned_post_join_annotate
                    else post_join_annotate_ops
                ),
                date_filter_ops=(
                    planned_post_join_annotate["date_filter_ops"]
                    if planned_post_join_annotate
                    else None
                ),
                sql=pja_sql,
                auto_generated=(
                    planned_post_join_annotate["auto_generated"]
                    if planned_post_join_annotate
                    else False
                ),
                edge=edge,
            )
            logger.info("Running do_post_join_annotate")
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
                planned_post_join_filters = self._consume_planned_record(
                    parent_node,
                    "do_post_join_filters",
                    edge=edge,
                )
                post_join_filter_ops = (
                    list(planned_post_join_filters["ops"])
                    if planned_post_join_filters
                    else self._normalize_sqlop_list(parent_node.do_post_join_filters())
                )
                pjf_sql = parent_node.build_query(post_join_filter_ops)
                self._record_executed_ops(
                    node=parent_node,
                    method_name="do_post_join_filters",
                    ops=(
                        planned_post_join_filters["method_ops"]
                        if planned_post_join_filters
                        else post_join_filter_ops
                    ),
                    date_filter_ops=(
                        planned_post_join_filters["date_filter_ops"]
                        if planned_post_join_filters
                        else None
                    ),
                    sql=pjf_sql,
                    auto_generated=(
                        planned_post_join_filters["auto_generated"]
                        if planned_post_join_filters
                        else False
                    ),
                    edge=edge,
                )
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

            # post-join reduce (if any)
            if hasattr(parent_node, "do_post_join_reduce"):
                pass

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
                        relation_reduce_columns = list(
                            relation_node.colabbrs(edge_data["relation_key"])
                        )
                        if isinstance(join_df, pd.DataFrame) or isinstance(
                            join_df, dd.DataFrame
                        ):
                            join_df = join_df.merge(
                                child_df,
                                on=relation_reduce_columns,
                                suffixes=("", "_dupe"),
                            )
                        else:
                            join_df = child_df
                            if self.debug:
                                logger.debug(
                                    f"assigned join_df to be {child_df.columns}"
                                )
                    elif self.compute_layer == ComputeLayerEnum.spark:
                        if isinstance(join_df, SPARK_DF_TYPES):
                            join_df = join_df.join(
                                child_df,
                                on=list(
                                    relation_node.colabbrs(
                                        edge_data["relation_key"]
                                    )
                                ),
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
            if self.train and self.label_node and (
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
