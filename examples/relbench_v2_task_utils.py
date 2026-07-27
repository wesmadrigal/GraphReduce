#!/usr/bin/env python
"""Shared GraphReduce runners for the RelBench v2 datasets.

RelBench owns task construction. This module only builds features at the
timestamps and for the entities present in the official task tables.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Sequence

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from relbench.base import AutoCompleteTask, TaskType
from relbench.tasks import get_task as get_official_task

from graphreduce.enum import ComputeLayerEnum, PeriodUnit
from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import DuckdbNode
from relbench_catboost_utils import (
    CLASSIFIER_CONFIGS,
    REGRESSOR_CONFIGS,
    fit_tuned_classifier,
    fit_tuned_regressor,
)
from relbench_dataset_utils import (
    get_training_frame_workers,
    iter_training_frames,
    target_table_from_frame,
)


DEFAULT_MAX_TRAIN_TIMESTAMPS = 10
DEFAULT_MAX_TRAIN_ROWS = 500_000
MODEL_CONFIG_COUNT = 1
TEMPORAL_FAMILIES = ("base", "temporal", "episode")
EVENT_FAMILIES = ("base", "episode")
DEFAULT_TS_PERIODS = (1, 3, 4, 7, 14, 30, 60, 90, 180, 365, 730)
COMPACT_TS_PERIODS = (7, 30, 90, 365)


@dataclass(frozen=True)
class NodeDefinition:
    name: str
    view: str
    prefix: str
    pk: str
    date_key: str | None
    feature_families: tuple[str, ...] = ("base",)
    feature_family_max_columns: int = 6
    ts_periods: tuple[int, ...] = DEFAULT_TS_PERIODS
    categorical_top_k: int = 5


@dataclass(frozen=True)
class EdgeDefinition:
    parent: str
    relation: str
    parent_key: str
    relation_key: str
    reduce: bool = True


@dataclass(frozen=True)
class GraphDefinition:
    root: str
    nodes: tuple[NodeDefinition, ...]
    edges: tuple[EdgeDefinition, ...]

    @property
    def root_node(self) -> NodeDefinition:
        return next(node for node in self.nodes if node.name == self.root)


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _register_table(
    con: duckdb.DuckDBPyConnection,
    db,
    table_name: str,
    view_name: str,
) -> None:
    """Register an official RelBench table without copying its dataframe."""

    ref_name = f"_official_{view_name}_df"
    con.register(ref_name, db.table_dict[table_name].df)
    source_ref = ref_name
    if get_training_frame_workers() != 1:
        source_ref = f"{ref_name}_table"
        con.sql(
            f"CREATE OR REPLACE TABLE {_quote(source_ref)} AS "
            f"SELECT * FROM {_quote(ref_name)}"
        )
    con.sql(
        f"CREATE OR REPLACE VIEW {_quote(view_name)} AS "
        f"SELECT * FROM {_quote(source_ref)}"
    )


def _columns(con: duckdb.DuckDBPyConnection, view_name: str) -> list[str]:
    return con.sql(f"SELECT * FROM {_quote(view_name)} LIMIT 0").to_df().columns.tolist()


def _row_id_view(
    con: duckdb.DuckDBPyConnection,
    source_view: str,
    output_view: str,
    row_id: str,
    select_sql: str = "*",
) -> None:
    con.sql(
        f"""
        CREATE OR REPLACE VIEW {_quote(output_view)} AS
        SELECT row_number() OVER () AS {_quote(row_id)}, {select_sql}
        FROM {_quote(source_view)}
        """
    )


def _node(
    con: duckdb.DuckDBPyConnection,
    definition: NodeDefinition,
) -> DuckdbNode:
    return DuckdbNode(
        fpath=definition.view,
        prefix=definition.prefix,
        pk=definition.pk,
        date_key=definition.date_key,
        columns=_columns(con, definition.view),
        categorical_top_k=definition.categorical_top_k,
        categorical_cardinality_threshold=20,
        feature_families=definition.feature_families,
        feature_family_max_columns=definition.feature_family_max_columns,
        ts_periods=list(definition.ts_periods),
        auto_text_features=True,
    )


def _official_task_tables(task) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        table = task.get_table(split, mask_input_cols=False)
        frame = table.df
        if not pd.api.types.is_datetime64_any_dtype(frame[task.time_col]):
            frame = frame.copy()
            frame[task.time_col] = pd.to_datetime(frame[task.time_col])
        tables[split] = frame
    return tables


def _sample_timestamps(
    timestamps: Sequence[pd.Timestamp],
    maximum: int,
) -> list[pd.Timestamp]:
    ordered = sorted(pd.Timestamp(timestamp) for timestamp in timestamps)
    if len(ordered) <= maximum:
        return ordered
    indexes = np.linspace(0, len(ordered) - 1, maximum, dtype=int)
    return [ordered[index] for index in sorted(set(indexes))]


def _limit_training_rows(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    return frame.sample(n=maximum, random_state=42).reset_index(drop=True)


def _configured_model_config(config: dict[str, object]) -> dict[str, object]:
    configured = dict(config)
    iteration_override = os.environ.get("RELBENCH_MODEL_ITERATIONS")
    if iteration_override is not None:
        configured["iterations"] = max(1, int(iteration_override))
    return configured


def _scope_root_view(
    con: duckdb.DuckDBPyConnection,
    definition: GraphDefinition,
    entity_ids: pd.Series,
    entity_col: str,
    suffix: str,
) -> GraphDefinition:
    root = definition.root_node
    id_ref = f"_official_scope_ids_{suffix}"
    scoped_view = f"{root.view}_{suffix}_scope"
    ids = pd.DataFrame({entity_col: entity_ids.drop_duplicates()})
    con.register(id_ref, ids)
    con.sql(
        f"""
        CREATE OR REPLACE VIEW {_quote(scoped_view)} AS
        SELECT root.*
        FROM {_quote(root.view)} root
        INNER JOIN {_quote(id_ref)} ids
          ON root.{_quote(root.pk)} = ids.{_quote(entity_col)}
        """
    )
    scoped_node_views = {definition.root: scoped_view}
    nodes_by_name = {node.name: node for node in definition.nodes}
    for edge in definition.edges:
        if edge.parent != definition.root or not edge.reduce:
            continue
        relation = nodes_by_name[edge.relation]
        relation_scoped_view = f"{relation.view}_{suffix}_scope"
        con.sql(
            f"""
            CREATE OR REPLACE VIEW {_quote(relation_scoped_view)} AS
            SELECT relation.*
            FROM {_quote(relation.view)} relation
            INNER JOIN {_quote(scoped_view)} root
              ON relation.{_quote(edge.relation_key)}
               = root.{_quote(edge.parent_key)}
            """
        )
        scoped_node_views[edge.relation] = relation_scoped_view
    nodes = tuple(
        replace(node, view=scoped_node_views[node.name])
        if node.name in scoped_node_views
        else node
        for node in definition.nodes
    )
    return replace(definition, nodes=nodes)


def _materialize_features(
    con: duckdb.DuckDBPyConnection,
    definition: GraphDefinition,
    cut_timestamp: pd.Timestamp,
    min_timestamp: pd.Timestamp,
    graph_name: str,
) -> pd.DataFrame:
    node_by_name = {item.name: _node(con, item) for item in definition.nodes}
    root = node_by_name[definition.root]
    inclusive_cut = pd.Timestamp(cut_timestamp) + pd.Timedelta(microseconds=1)
    lookback_days = max(1, int(np.ceil((inclusive_cut - min_timestamp) / pd.Timedelta(days=1))))
    graph = GraphReduce(
        name=graph_name,
        parent_node=root,
        compute_layer=ComputeLayerEnum.duckdb,
        sql_client=con,
        cut_date=inclusive_cut.to_pydatetime(),
        compute_period_val=lookback_days,
        compute_period_unit=PeriodUnit.day,
        auto_features=True,
        auto_labels=False,
        date_filters_on_agg=True,
        auto_feature_hops_back=3,
        auto_feature_hops_front=0,
        use_temp_tables=True,
    )
    for node in node_by_name.values():
        graph.add_node(node)
    for edge in definition.edges:
        graph.add_entity_edge(
            node_by_name[edge.parent],
            node_by_name[edge.relation],
            edge.parent_key,
            edge.relation_key,
            reduce=edge.reduce,
        )
    graph.do_transformations_sql()
    frame = con.sql(f"SELECT * FROM {graph.parent_node._cur_data_ref}").to_df()
    graph._clean_refs()
    return frame


def _prepare_arxiv_graph(
    con: duckdb.DuckDBPyConnection,
    db,
    entity_table: str,
) -> GraphDefinition:
    for table_name in (
        "papers",
        "authors",
        "paperAuthors",
        "paperCategories",
        "citations",
    ):
        _register_table(con, db, table_name, f"{table_name}_src")

    _row_id_view(con, "paperAuthors_src", "paper_authors_src", "paper_author_id")
    _row_id_view(
        con,
        "paperCategories_src",
        "paper_categories_src",
        "paper_category_id",
    )
    con.sql(
        """
        CREATE OR REPLACE VIEW citations_out_src AS
        SELECT
            row_number() OVER () AS citation_out_id,
            Paper_ID,
            References_Paper_ID,
            Submission_Date
        FROM citations_src
        """
    )
    con.sql(
        """
        CREATE OR REPLACE VIEW citations_in_src AS
        SELECT
            row_number() OVER () AS citation_in_id,
            References_Paper_ID AS Paper_ID,
            citations_src.Paper_ID AS Citing_Paper_ID,
            Submission_Date
        FROM citations_src
        """
    )

    if entity_table == "papers":
        nodes = (
            NodeDefinition("root", "papers_src", "pap", "Paper_ID", "Submission_Date"),
            NodeDefinition("authors", "paper_authors_src", "pau", "paper_author_id", "Submission_Date", TEMPORAL_FAMILIES),
            NodeDefinition("categories", "paper_categories_src", "pca", "paper_category_id", "Submission_Date", TEMPORAL_FAMILIES),
            NodeDefinition("citations_out", "citations_out_src", "cout", "citation_out_id", "Submission_Date", TEMPORAL_FAMILIES),
            NodeDefinition("citations_in", "citations_in_src", "cin", "citation_in_id", "Submission_Date", TEMPORAL_FAMILIES),
        )
        edges = (
            EdgeDefinition("root", "authors", "Paper_ID", "Paper_ID"),
            EdgeDefinition("root", "categories", "Paper_ID", "Paper_ID"),
            EdgeDefinition("root", "citations_out", "Paper_ID", "Paper_ID"),
            EdgeDefinition("root", "citations_in", "Paper_ID", "Paper_ID"),
        )
        return GraphDefinition("root", nodes, edges)

    if entity_table != "authors":
        raise ValueError(f"Unsupported rel-arxiv entity table: {entity_table}")

    con.sql(
        """
        CREATE OR REPLACE VIEW author_papers_src AS
        SELECT
            row_number() OVER () AS author_paper_id,
            pa.Author_ID,
            pa.Paper_ID,
            pa.Submission_Date,
            p.Primary_Category_ID,
            p.Title,
            p.Abstract
        FROM paperAuthors_src pa
        INNER JOIN papers_src p ON pa.Paper_ID = p.Paper_ID
        """
    )
    con.sql(
        """
        CREATE OR REPLACE VIEW author_paper_categories_src AS
        SELECT
            row_number() OVER () AS author_paper_category_id,
            pa.Author_ID,
            pc.Paper_ID,
            pc.Category_ID,
            pc.Submission_Date
        FROM paperAuthors_src pa
        INNER JOIN paperCategories_src pc ON pa.Paper_ID = pc.Paper_ID
        """
    )
    con.sql(
        """
        CREATE OR REPLACE VIEW author_citations_out_src AS
        SELECT
            row_number() OVER () AS author_citation_out_id,
            pa.Author_ID,
            c.Paper_ID,
            c.References_Paper_ID,
            c.Submission_Date
        FROM paperAuthors_src pa
        INNER JOIN citations_src c ON pa.Paper_ID = c.Paper_ID
        """
    )
    con.sql(
        """
        CREATE OR REPLACE VIEW author_citations_in_src AS
        SELECT
            row_number() OVER () AS author_citation_in_id,
            pa.Author_ID,
            c.References_Paper_ID AS Paper_ID,
            c.Paper_ID AS Citing_Paper_ID,
            c.Submission_Date
        FROM paperAuthors_src pa
        INNER JOIN citations_src c ON pa.Paper_ID = c.References_Paper_ID
        """
    )
    nodes = (
        NodeDefinition("root", "authors_src", "aut", "Author_ID", None),
        NodeDefinition("papers", "author_papers_src", "ap", "author_paper_id", "Submission_Date", TEMPORAL_FAMILIES),
        NodeDefinition("categories", "author_paper_categories_src", "apc", "author_paper_category_id", "Submission_Date", TEMPORAL_FAMILIES),
        NodeDefinition("citations_out", "author_citations_out_src", "aco", "author_citation_out_id", "Submission_Date", TEMPORAL_FAMILIES),
        NodeDefinition("citations_in", "author_citations_in_src", "aci", "author_citation_in_id", "Submission_Date", TEMPORAL_FAMILIES),
    )
    edges = tuple(
        EdgeDefinition("root", node.name, "Author_ID", "Author_ID")
        for node in nodes
        if node.name != "root"
    )
    return GraphDefinition("root", nodes, edges)


def _prepare_ratebeer_graph(
    con: duckdb.DuckDBPyConnection,
    db,
    entity_table: str,
) -> GraphDefinition:
    required = {
        "beers",
        "brewers",
        "beer_styles",
        "users",
        "beer_ratings",
        "favorites",
        "place_ratings",
        "availability",
        "places",
    }
    for table_name in required:
        _register_table(con, db, table_name, f"{table_name}_src")

    if entity_table == "users":
        nodes = (
            NodeDefinition("root", "users_src", "usr", "user_id", "created_at"),
            NodeDefinition("ratings", "beer_ratings_src", "brt", "rating_id", "created_at", TEMPORAL_FAMILIES),
            NodeDefinition("favorites", "favorites_src", "fav", "favorite_id", "created_at", TEMPORAL_FAMILIES),
            NodeDefinition("place_ratings", "place_ratings_src", "prt", "rating_id", "created_at", TEMPORAL_FAMILIES),
            NodeDefinition("availability", "availability_src", "avl", "avail_id", "created_at", TEMPORAL_FAMILIES),
        )
        edges = (
            EdgeDefinition("root", "ratings", "user_id", "user_id"),
            EdgeDefinition("root", "favorites", "user_id", "user_id"),
            EdgeDefinition("root", "place_ratings", "user_id", "user_id"),
            EdgeDefinition("root", "availability", "user_id", "user_id"),
        )
        return GraphDefinition("root", nodes, edges)

    if entity_table == "beers":
        con.sql(
            """
            CREATE OR REPLACE VIEW beer_ratings_for_beers_src AS
            SELECT
                rating_id,
                beer_id,
                overall,
                total_score,
                description_score,
                language,
                created_at
            FROM beer_ratings_src
            """
        )
        nodes = (
            NodeDefinition("root", "beers_src", "ber", "beer_id", "created_at"),
            NodeDefinition(
                "ratings",
                "beer_ratings_for_beers_src",
                "brt",
                "rating_id",
                "created_at",
                TEMPORAL_FAMILIES,
                ts_periods=COMPACT_TS_PERIODS,
            ),
            NodeDefinition("favorites", "favorites_src", "fav", "favorite_id", "created_at", TEMPORAL_FAMILIES),
            NodeDefinition("availability", "availability_src", "avl", "avail_id", "created_at", TEMPORAL_FAMILIES),
            NodeDefinition("brewer", "brewers_src", "brw", "brewer_id", None),
            NodeDefinition("style", "beer_styles_src", "sty", "style_id", None),
        )
        edges = (
            EdgeDefinition("root", "ratings", "beer_id", "beer_id"),
            EdgeDefinition("root", "favorites", "beer_id", "beer_id"),
            EdgeDefinition("root", "availability", "beer_id", "beer_id"),
            EdgeDefinition("root", "brewer", "brewer_id", "brewer_id", reduce=False),
            EdgeDefinition("root", "style", "style_id", "style_id", reduce=False),
        )
        return GraphDefinition("root", nodes, edges)

    if entity_table == "brewers":
        con.sql(
            """
            CREATE OR REPLACE VIEW brewer_ratings_src AS
            SELECT br.*, b.brewer_id
            FROM beer_ratings_src br
            INNER JOIN beers_src b ON br.beer_id = b.beer_id
            """
        )
        nodes = (
            NodeDefinition("root", "brewers_src", "brw", "brewer_id", None),
            NodeDefinition("beers", "beers_src", "ber", "beer_id", "created_at", TEMPORAL_FAMILIES),
            NodeDefinition("ratings", "brewer_ratings_src", "brr", "rating_id", "created_at", TEMPORAL_FAMILIES, 4),
        )
        edges = (
            EdgeDefinition("root", "beers", "brewer_id", "brewer_id"),
            EdgeDefinition("root", "ratings", "brewer_id", "brewer_id"),
        )
        return GraphDefinition("root", nodes, edges)

    if entity_table == "beer_ratings":
        nodes = (
            NodeDefinition("root", "beer_ratings_src", "rat", "rating_id", "created_at"),
            NodeDefinition("user", "users_src", "usr", "user_id", "created_at"),
            NodeDefinition("beer", "beers_src", "ber", "beer_id", "created_at"),
            NodeDefinition("brewer", "brewers_src", "brw", "brewer_id", None),
            NodeDefinition("style", "beer_styles_src", "sty", "style_id", None),
        )
        edges = (
            EdgeDefinition("root", "user", "user_id", "user_id", reduce=False),
            EdgeDefinition("root", "beer", "beer_id", "beer_id", reduce=False),
            EdgeDefinition("beer", "brewer", "brewer_id", "brewer_id", reduce=False),
            EdgeDefinition("beer", "style", "style_id", "style_id", reduce=False),
        )
        return GraphDefinition("root", nodes, edges)

    raise ValueError(f"Unsupported rel-ratebeer entity table: {entity_table}")


def _prepare_salt_graph(
    con: duckdb.DuckDBPyConnection,
    db,
    entity_table: str,
) -> GraphDefinition:
    for table_name in ("salesdocumentitem", "salesdocument", "customer", "address"):
        _register_table(con, db, table_name, f"{table_name}_src")

    nodes: list[NodeDefinition] = []
    edges: list[EdgeDefinition] = []
    if entity_table == "salesdocumentitem":
        nodes.append(
            NodeDefinition(
                "root",
                "salesdocumentitem_src",
                "itm",
                "ID",
                "CREATIONTIMESTAMP",
            )
        )
        nodes.append(
            NodeDefinition(
                "header",
                "salesdocument_src",
                "hdr",
                "SALESDOCUMENT",
                "CREATIONTIMESTAMP",
            )
        )
        edges.append(
            EdgeDefinition(
                "root",
                "header",
                "SALESDOCUMENT",
                "SALESDOCUMENT",
                reduce=False,
            )
        )
    elif entity_table == "salesdocument":
        con.sql(
            """
            CREATE OR REPLACE VIEW salesdocumentitem_for_sales_src AS
            SELECT
                SALESDOCUMENT,
                SALESDOCUMENTITEM,
                SALESDOCUMENTITEMCATEGORY,
                PRODUCT,
                CAST(SOLDTOPARTY AS VARCHAR) AS SOLDTOPARTY,
                CAST(SHIPTOPARTY AS VARCHAR) AS SHIPTOPARTY,
                CAST(BILLTOPARTY AS VARCHAR) AS BILLTOPARTY,
                CAST(PAYERPARTY AS VARCHAR) AS PAYERPARTY,
                CREATIONTIMESTAMP,
                ID
            FROM salesdocumentitem_src
            """
        )
        nodes.extend(
            (
                NodeDefinition(
                    "root",
                    "salesdocument_src",
                    "hdr",
                    "SALESDOCUMENT",
                    "CREATIONTIMESTAMP",
                ),
                NodeDefinition(
                    "items",
                    "salesdocumentitem_for_sales_src",
                    "itm",
                    "ID",
                    "CREATIONTIMESTAMP",
                    EVENT_FAMILIES,
                    ts_periods=COMPACT_TS_PERIODS,
                    categorical_top_k=3,
                ),
            )
        )
        edges.append(
            EdgeDefinition("root", "items", "SALESDOCUMENT", "SALESDOCUMENT")
        )
    else:
        raise ValueError(f"Unsupported rel-salt entity table: {entity_table}")

    item_parent = "root" if entity_table == "salesdocumentitem" else "items"
    for role, key in (
        ("sold", "SOLDTOPARTY"),
        ("ship", "SHIPTOPARTY"),
        ("bill", "BILLTOPARTY"),
        ("payer", "PAYERPARTY"),
    ):
        customer_view = f"{role}_customer_src"
        address_view = f"{role}_address_src"
        customer_projection = (
            "SELECT * FROM customer_src"
            if entity_table == "salesdocumentitem"
            else "SELECT CAST(CUSTOMER AS VARCHAR) AS CUSTOMER, ADDRESSID FROM customer_src"
        )
        con.sql(
            f"CREATE OR REPLACE VIEW {_quote(customer_view)} AS "
            f"{customer_projection}"
        )
        con.sql(
            f"CREATE OR REPLACE VIEW {_quote(address_view)} AS SELECT * FROM address_src"
        )
        nodes.extend(
            (
                NodeDefinition(
                    f"{role}_customer",
                    customer_view,
                    f"{role[:2]}c",
                    "CUSTOMER",
                    None,
                ),
                NodeDefinition(
                    f"{role}_address",
                    address_view,
                    f"{role[:2]}a",
                    "ADDRESSID",
                    None,
                ),
            )
        )
        edges.extend(
            (
                EdgeDefinition(
                    item_parent,
                    f"{role}_customer",
                    key,
                    "CUSTOMER",
                    reduce=False,
                ),
                EdgeDefinition(
                    f"{role}_customer",
                    f"{role}_address",
                    "ADDRESSID",
                    "ADDRESSID",
                    reduce=False,
                ),
            )
        )
    return GraphDefinition("root", tuple(nodes), tuple(edges))


def _prepare_mimic_graph(
    con: duckdb.DuckDBPyConnection,
    db,
) -> GraphDefinition:
    for table_name in db.table_dict:
        _register_table(con, db, table_name, f"{table_name}_src")

    nodes: list[NodeDefinition] = [
        NodeDefinition("root", "patients_src", "pat", "subject_id", None)
    ]
    edges: list[EdgeDefinition] = []
    event_specs = (
        ("icustays", "icu", "stay_id", "intime"),
        ("chartevents", "chr", "chart_event_id", "charttime"),
        ("procedureevents", "pro", "orderid", "starttime"),
    )
    for table_name, prefix, pk, expected_date in event_specs:
        if table_name not in db.table_dict:
            continue
        table = db.table_dict[table_name]
        if "subject_id" not in table.df.columns:
            continue
        view_name = f"{table_name}_src"
        actual_pk = table.pkey_col
        if actual_pk is None:
            output_view = f"{table_name}_with_id_src"
            _row_id_view(con, view_name, output_view, pk)
            view_name = output_view
            actual_pk = pk
        date_key = expected_date if expected_date in table.df.columns else table.time_col
        nodes.append(
            NodeDefinition(
                table_name,
                view_name,
                prefix,
                actual_pk,
                date_key,
                TEMPORAL_FAMILIES,
                4,
            )
        )
        edges.append(
            EdgeDefinition("root", table_name, "subject_id", "subject_id")
        )
    return GraphDefinition("root", tuple(nodes), tuple(edges))


def _prepare_graph(
    con: duckdb.DuckDBPyConnection,
    db,
    dataset_name: str,
    entity_table: str,
) -> GraphDefinition:
    if dataset_name == "rel-arxiv":
        return _prepare_arxiv_graph(con, db, entity_table)
    if dataset_name == "rel-ratebeer":
        return _prepare_ratebeer_graph(con, db, entity_table)
    if dataset_name == "rel-salt":
        return _prepare_salt_graph(con, db, entity_table)
    if dataset_name == "rel-mimic":
        return _prepare_mimic_graph(con, db)
    raise ValueError(f"Unsupported RelBench v2 dataset: {dataset_name}")


def _build_entity_frames(
    task,
    dataset_name: str,
    task_tables: dict[str, pd.DataFrame],
    *,
    max_train_timestamps: int,
    max_train_rows: int,
) -> tuple[dict[str, pd.DataFrame], int, int]:
    db = task.dataset.get_db(upto_test_timestamp=False)
    min_timestamp = pd.Timestamp(db.min_timestamp)
    con = duckdb.connect()
    output: dict[str, pd.DataFrame] = {}
    official_train_timestamps = int(
        task_tables["train"][task.time_col].nunique(dropna=True)
    )
    selected_train_timestamps = official_train_timestamps

    try:
        definition = _prepare_graph(con, db, dataset_name, task.entity_table)
        root = definition.root_node
        root_feature_id = f"{root.prefix}_{root.pk}"

        if isinstance(task, AutoCompleteTask):
            for split, official_labels in task_tables.items():
                labels = official_labels
                if split == "train":
                    labels = _limit_training_rows(labels, max_train_rows)
                if labels.empty:
                    output[split] = labels.copy()
                    continue
                scoped_definition = _scope_root_view(
                    con,
                    definition,
                    labels[task.entity_col],
                    task.entity_col,
                    split,
                )
                cut_timestamp = pd.Timestamp(labels[task.time_col].max())
                features = _materialize_features(
                    con,
                    scoped_definition,
                    cut_timestamp,
                    min_timestamp,
                    f"{dataset_name}_{task.target_col}_{split}",
                )
                output[split] = features.merge(
                    labels,
                    left_on=root_feature_id,
                    right_on=task.entity_col,
                    how="right",
                    validate="one_to_one",
                )
            return output, official_train_timestamps, 1

        for split, labels in task_tables.items():
            timestamps = sorted(
                pd.Timestamp(timestamp)
                for timestamp in labels[task.time_col].dropna().unique()
            )
            if split == "train":
                timestamps = _sample_timestamps(
                    timestamps,
                    max_train_timestamps,
                )
                selected_train_timestamps = len(timestamps)

            frame_jobs = []
            for snapshot_index, cut_timestamp in enumerate(timestamps):
                snapshot_labels = labels[
                    labels[task.time_col] == cut_timestamp
                ].copy()
                if snapshot_labels.empty:
                    continue
                frame_jobs.append((snapshot_index, cut_timestamp, snapshot_labels))

            def build_frame(frame_con, frame_job):
                con = frame_con
                snapshot_index, cut_timestamp, snapshot_labels = frame_job
                scoped_definition = _scope_root_view(
                    con,
                    definition,
                    snapshot_labels[task.entity_col],
                    task.entity_col,
                    f"{split}_{snapshot_index}",
                )
                features = _materialize_features(
                    con,
                    scoped_definition,
                    cut_timestamp,
                    min_timestamp,
                    f"{dataset_name}_{task.target_col}_{split}_{cut_timestamp:%Y%m%d}",
                )
                features[task.time_col] = cut_timestamp
                frame = features.merge(
                    snapshot_labels,
                    left_on=[root_feature_id, task.time_col],
                    right_on=[task.entity_col, task.time_col],
                    how="right",
                    validate="one_to_one",
                )
                return frame

            frame_workers = None if split == "train" else 1
            frames = list(
                iter_training_frames(
                    con,
                    frame_jobs,
                    build_frame,
                    workers=frame_workers,
                )
            )
            output[split] = (
                pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            )

        output["train"] = _limit_training_rows(output["train"], max_train_rows)
        return output, official_train_timestamps, selected_train_timestamps
    finally:
        con.close()


def _is_identifier_like(column: str, series: pd.Series) -> bool:
    lowered = column.lower()
    if not (
        lowered == "id"
        or lowered.endswith("_id")
        or lowered.endswith("id")
        or "guid" in lowered
        or "code" in lowered
    ):
        return False
    non_null = series.dropna()
    if non_null.empty:
        return True
    return non_null.nunique() > min(512, max(20, int(len(non_null) * 0.05)))


def _prepare_model_inputs(
    frames: dict[str, pd.DataFrame],
    task,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    prepared = {split: frame.copy() for split, frame in frames.items()}
    derived_columns: dict[str, dict[str, pd.Series]] = {
        split: {} for split in prepared
    }
    common_columns = set.intersection(
        *(set(frame.columns) for frame in prepared.values())
    )
    feature_columns: list[str] = []

    for split, frame in prepared.items():
        timestamp = pd.to_datetime(frame[task.time_col], errors="coerce")
        derived_columns[split]["snapshot_unix"] = (
            timestamp.astype("int64") / 1_000_000_000
        )
    feature_columns.append("snapshot_unix")

    excluded = {task.target_col, task.time_col, task.entity_col}
    for column in sorted(common_columns):
        if column in excluded or "label" in column.lower():
            continue
        train_series = prepared["train"][column]
        if pd.api.types.is_datetime64_any_dtype(train_series):
            output_column = f"{column}_age_seconds"
            for split, frame in prepared.items():
                derived_columns[split][output_column] = (
                    pd.to_datetime(frame[task.time_col], errors="coerce")
                    - pd.to_datetime(frame[column], errors="coerce")
                ).dt.total_seconds()
            feature_columns.append(output_column)
            continue
        if pd.api.types.is_bool_dtype(train_series):
            for frame in prepared.values():
                frame[column] = frame[column].astype("float64")
            feature_columns.append(column)
            continue
        if pd.api.types.is_numeric_dtype(train_series):
            if _is_identifier_like(column, train_series):
                continue
            for frame in prepared.values():
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            feature_columns.append(column)
            continue

        sample = train_series.dropna().astype(str)
        if sample.empty:
            continue
        sample = sample.sample(min(len(sample), 10_000), random_state=42)
        if sample.nunique() > 256 or sample.str.len().median() > 64:
            continue
        categories = {
            value: index
            for index, value in enumerate(sorted(sample.unique().tolist()))
        }
        for frame in prepared.values():
            frame[column] = (
                frame[column].astype("string").map(categories).fillna(-1).astype("int32")
            )
        feature_columns.append(column)

    feature_columns = list(dict.fromkeys(feature_columns))
    for split, frame in prepared.items():
        if derived_columns[split]:
            prepared[split] = frame = pd.concat(
                [frame, pd.DataFrame(derived_columns[split], index=frame.index)],
                axis=1,
            )
        frame[feature_columns] = (
            frame[feature_columns]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )
    return prepared, feature_columns


def _multiclass_predictions(
    train_inputs: pd.DataFrame,
    train_target: pd.Series,
    val_inputs: pd.DataFrame,
    val_target: pd.Series,
    test_inputs: pd.DataFrame,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    config = _configured_model_config(CLASSIFIER_CONFIGS[-1])
    model = CatBoostClassifier(
        iterations=int(config["iterations"]),
        depth=int(config["depth"]),
        learning_rate=float(config["learning_rate"]),
        l2_leaf_reg=float(config["l2_leaf_reg"]),
        loss_function="MultiClass",
        eval_metric="MultiClass",
        classes_count=num_classes,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        train_inputs,
        train_target.astype("int64"),
        eval_set=(val_inputs, val_target.astype("int64")),
        use_best_model=True,
        early_stopping_rounds=100,
    )
    val_prediction = np.asarray(model.predict_proba(val_inputs), dtype="float64")
    test_prediction = np.asarray(model.predict_proba(test_inputs), dtype="float64")
    return val_prediction, test_prediction, config


def _evaluate_entity_task(
    task,
    frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, float], dict[str, float], int, dict[str, object]]:
    prepared, feature_columns = _prepare_model_inputs(frames, task)
    if not feature_columns:
        raise RuntimeError("GraphReduce produced no common model features")

    train = prepared["train"]
    val = prepared["val"]
    test = prepared["test"]
    train_inputs = train[feature_columns]
    val_inputs = val[feature_columns]
    test_inputs = test[feature_columns]
    train_target = train[task.target_col]
    val_target = val[task.target_col]

    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        configs = tuple(
            _configured_model_config(config)
            for config in CLASSIFIER_CONFIGS[-MODEL_CONFIG_COUNT:]
        )
        model, config, _ = fit_tuned_classifier(
            train_inputs,
            train_target,
            val_inputs,
            val_target,
            configs=configs,
        )
        val_prediction = np.asarray(
            model.predict_proba(val_inputs)[:, 1],
            dtype="float64",
        )
        test_prediction = np.asarray(
            model.predict_proba(test_inputs)[:, 1],
            dtype="float64",
        )
    elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        num_classes = int(
            getattr(
                task,
                "num_classes",
                max(train_target.max(), val_target.max()) + 1,
            )
        )
        val_prediction, test_prediction, config = _multiclass_predictions(
            train_inputs,
            train_target,
            val_inputs,
            val_target,
            test_inputs,
            num_classes,
        )
    elif task.task_type == TaskType.REGRESSION:
        configs = tuple(
            _configured_model_config(config)
            for config in REGRESSOR_CONFIGS[-MODEL_CONFIG_COUNT:]
        )
        model, config, _ = fit_tuned_regressor(
            train_inputs,
            train_target.astype("float64"),
            val_inputs,
            val_target.astype("float64"),
            configs=configs,
        )
        val_prediction = np.asarray(model.predict(val_inputs), dtype="float64")
        test_prediction = np.asarray(model.predict(test_inputs), dtype="float64")
    else:
        raise ValueError(f"Unsupported entity task type: {task.task_type}")

    val_metrics = task.evaluate(
        val_prediction,
        target_table_from_frame(task, val),
    )
    test_metrics = task.evaluate(
        test_prediction,
        target_table_from_frame(task, test),
    )
    return val_metrics, test_metrics, len(feature_columns), config


def run_relbench_v2_entity_task(
    dataset_name: str,
    task_name: str,
    *,
    max_train_timestamps: int = DEFAULT_MAX_TRAIN_TIMESTAMPS,
    max_train_rows: int | None = None,
) -> dict[str, object]:
    """Run one official RelBench v2 entity or autocomplete task."""

    task = get_official_task(dataset_name, task_name, download=True)
    if task.task_type == TaskType.LINK_PREDICTION:
        raise ValueError(f"{dataset_name}/{task_name} is a link-prediction task")
    task_tables = _official_task_tables(task)
    if max_train_rows is None:
        max_train_rows = int(
            os.environ.get("RELBENCH_MAX_TRAIN_ROWS", DEFAULT_MAX_TRAIN_ROWS)
        )
    frames, official_timestamp_count, selected_timestamp_count = _build_entity_frames(
        task,
        dataset_name,
        task_tables,
        max_train_timestamps=max_train_timestamps,
        max_train_rows=max_train_rows,
    )
    for split, frame in frames.items():
        if frame.empty:
            raise RuntimeError(f"No rows materialized for {dataset_name}/{task_name} {split}")
    validation_metrics, test_metrics, feature_count, model_config = (
        _evaluate_entity_task(task, frames)
    )
    result = {
        "dataset": dataset_name,
        "task": task_name,
        "task_type": task.task_type.value,
        "validation_timestamp": pd.Timestamp(task.dataset.val_timestamp),
        "test_timestamp": pd.Timestamp(task.dataset.test_timestamp),
        "official_train_timestamps": official_timestamp_count,
        "selected_train_timestamps": selected_timestamp_count,
        "target": task.target_col,
        "train_rows": len(frames["train"]),
        "validation_rows": len(frames["val"]),
        "test_rows": len(frames["test"]),
        "feature_count": feature_count,
        "model_config": model_config,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    for key, value in result.items():
        print(f"{key}: {value}", flush=True)
    return result


def _destination_popularity(
    frame: pd.DataFrame,
    src_col: str,
    dst_col: str,
) -> tuple[dict[int, Counter], Counter]:
    by_source: dict[int, Counter] = defaultdict(Counter)
    global_counts: Counter = Counter()
    for source, destinations in zip(frame[src_col], frame[dst_col]):
        values = [int(value) for value in destinations]
        by_source[int(source)].update(values)
        global_counts.update(values)
    return dict(by_source), global_counts


def _popularity_predictions(
    frame: pd.DataFrame,
    src_col: str,
    by_source: dict[int, Counter],
    global_counts: Counter,
    *,
    eval_k: int,
    num_destinations: int,
) -> np.ndarray:
    global_order = [item for item, _ in global_counts.most_common(eval_k)]
    predictions: list[list[int]] = []
    for source in frame[src_col]:
        local_order = [
            item
            for item, _ in by_source.get(int(source), Counter()).most_common(eval_k)
        ]
        candidates = list(dict.fromkeys(local_order + global_order))
        if len(candidates) < eval_k:
            selected = set(candidates)
            candidates.extend(
                item
                for item in range(num_destinations)
                if item not in selected
            )
        predictions.append(candidates[:eval_k])
    return np.asarray(predictions, dtype="int64")


def run_relbench_v2_link_task(
    dataset_name: str,
    task_name: str,
) -> dict[str, object]:
    """Run an official link task with a history-aware popularity baseline.

    GraphReduce currently produces entity-grain features, while RelBench link
    evaluation requires ranked destination IDs. This baseline keeps task
    loading and evaluation official and provides a runnable starting point for
    a future pair scorer.
    """

    task = get_official_task(dataset_name, task_name, download=True)
    if task.task_type != TaskType.LINK_PREDICTION:
        raise ValueError(f"{dataset_name}/{task_name} is not a link-prediction task")
    task_tables = _official_task_tables(task)
    by_source, global_counts = _destination_popularity(
        task_tables["train"],
        task.src_entity_col,
        task.dst_entity_col,
    )
    num_destinations = len(
        task.dataset.get_db().table_dict[task.dst_entity_table]
    )
    validation_prediction = _popularity_predictions(
        task_tables["val"],
        task.src_entity_col,
        by_source,
        global_counts,
        eval_k=task.eval_k,
        num_destinations=num_destinations,
    )
    test_prediction = _popularity_predictions(
        task_tables["test"],
        task.src_entity_col,
        by_source,
        global_counts,
        eval_k=task.eval_k,
        num_destinations=num_destinations,
    )
    validation_table = task.get_table("val", mask_input_cols=False)
    test_table = task.get_table("test", mask_input_cols=False)
    result = {
        "dataset": dataset_name,
        "task": task_name,
        "task_type": task.task_type.value,
        "validation_timestamp": pd.Timestamp(task.dataset.val_timestamp),
        "test_timestamp": pd.Timestamp(task.dataset.test_timestamp),
        "target": task.dst_entity_col,
        "train_rows": len(task_tables["train"]),
        "validation_rows": len(task_tables["val"]),
        "test_rows": len(task_tables["test"]),
        "feature_count": 0,
        "baseline": "per-source-then-global-destination-popularity",
        "validation_metrics": task.evaluate(
            validation_prediction,
            validation_table,
        ),
        "test_metrics": task.evaluate(test_prediction, test_table),
    }
    for key, value in result.items():
        print(f"{key}: {value}", flush=True)
    return result
