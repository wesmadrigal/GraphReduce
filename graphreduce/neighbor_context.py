"""Build a neighbor's own historical context before learned message pooling.

This operation re-roots native GraphReduce at the terminal entity of a path.
It consumes no labels, follows declared relationships, and retains the original
neighbor grain. The caller can subsequently learn a row transformation.
"""

from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
import io

import duckdb
import pandas as pd


def enrich_neighbors(
    rows,
    tables,
    relationships,
    *,
    terminal,
    depth=1,
    max_columns=4,
    max_relations=12,
    max_neighbors=128,
    seed=0,
    memory_limit="4GB",
    plan_cache=None,
):
    """Return rows plus native base/state features at each neighbor's cutoff.

    ``rows`` carries private ``__cutoff`` and ``__neighbor_key_N`` metadata.
    These keys are used only for joining and are removed by ``extract_path``.
    Each context edge is capped before GraphReduce reduction. Multiple child
    relations are reduced separately, preventing Cartesian-product inflation.
    """
    from .enum import ComputeLayerEnum, PeriodUnit
    from .graph_reduce import GraphReduce
    from .node import DuckdbNode

    def parts(key):
        return (key,) if isinstance(key, str) else tuple(key or ())

    def q(value):
        return '"' + str(value).replace('"', '""') + '"'

    if min(depth, max_columns, max_relations, max_neighbors) < 1:
        raise ValueError("Neighbor context budgets must be positive")
    selected, edges, frontier = [terminal], [], [terminal]
    for _ in range(depth):
        next_frontier = []
        for edge in sorted(relationships, key=lambda r: (r.parent, r.child)):
            if (
                edge.parent in frontier
                and edge.child not in selected
                and len(edges) < max_relations
            ):
                edges.append(edge)
                selected.append(edge.child)
                next_frontier.append(edge.child)
        frontier = next_frontier
    if not edges or rows.empty:
        return rows
    structural = {
        name: set(parts(table.key)) | {table.date} for name, table in tables.items()
    }
    for edge in relationships:
        structural[edge.parent].update(parts(edge.parent_key))
        structural[edge.child].update(parts(edge.child_key))
    root_keys = parts(tables[terminal].key)
    if not root_keys:
        raise ValueError("Neighbor context requires a declared terminal primary key")
    metadata = [f"__neighbor_key_{i}" for i in range(len(root_keys))]
    chunks = []
    # A caller can persist this dictionary alongside its supervised scorer.
    # Freeze feature discovery on the first historical training cutoff, then
    # replay the same operations for every later cutoff and inference batch.
    plan_cache = {} if plan_cache is None else plan_cache
    for cutoff, block in rows.groupby("__cutoff", sort=True):
        con = duckdb.connect(
            config={
                "memory_limit": memory_limit,
                "max_temp_directory_size": "16GB",
                "threads": 2,
            }
        )
        try:
            cutoff_sql = pd.Timestamp(cutoff).isoformat(sep=" ")
            ids = (
                block[metadata]
                .drop_duplicates()
                .rename(columns=dict(zip(metadata, root_keys)))
            )
            con.register("neighbor_ids", ids)
            nodes, views, excluded, origins = {}, {}, set(), {}
            for index, name in enumerate(selected):
                table = tables[name]
                source = f"context_source_{index}"
                view = f"context_view_{index}"
                con.register(source, table.source)
                date_filter = (
                    f"AND s.{q(table.date)} < TIMESTAMP '{cutoff_sql}'"
                    if table.date
                    else ""
                )
                if name == terminal:
                    parent_view = "neighbor_ids"
                    join_keys = list(zip(root_keys, root_keys))
                else:
                    edge = next(e for e in edges if e.child == name)
                    parent_view = views[edge.parent]
                    join_keys = list(zip(parts(edge.parent_key), parts(edge.child_key)))
                predicate = " AND ".join(f"p.{q(a)} = s.{q(b)}" for a, b in join_keys)
                query = (
                    f"SELECT s.* FROM {q(source)} s WHERE EXISTS "
                    f"(SELECT 1 FROM {q(parent_view)} p WHERE {predicate}) {date_filter}"
                )
                if name != terminal:
                    sample_keys = parts(table.key) or parts(edge.child_key)
                    partition = ", ".join(f"s.{q(k)}" for k in parts(edge.child_key))
                    keys = ", ".join(f"s.{q(k)}" for k in sample_keys)
                    query += (
                        f" QUALIFY row_number() OVER (PARTITION BY {partition} "
                        f"ORDER BY hash({keys}, {int(seed)}), {keys}) <= {int(max_neighbors)}"
                    )
                con.execute(f"CREATE TEMP TABLE {q(view)} AS {query}")
                views[name] = view
                candidates = (
                    table.columns if table.columns is not None else table.source.columns
                )
                payload = (
                    []
                    if name == terminal
                    else [c for c in candidates if c not in structural[name]][
                        :max_columns
                    ]
                )
                required = [c for c in table.source.columns if c in structural[name]]
                prefix = f"ctx{index}"
                nodes[name] = DuckdbNode(
                    fpath=view,
                    prefix=prefix,
                    pk=table.key,
                    date_key=table.date,
                    columns=list(dict.fromkeys([*required, *payload])),
                    cutoff_boundary="strict",
                    auto_annotate_features=False,
                    auto_text_features=False,
                    auto_base_predicate_max=0,
                    feature_families=("base", "state"),
                    feature_family_max_columns=max_columns,
                    feature_family_max_features_per_column=16,
                    feature_propagation_max_functions_per_column=1,
                )
                excluded.update(required)
                excluded.update(f"{prefix}_{c}" for c in required)
                # Native base aggregation also supports undeclared identifier
                # heuristics. Do not let arithmetic on a declared key with an
                # unconventional name (e.g. "record") become scorer payload.
                # Resolve the longest source name, so a key "record" does not
                # accidentally hide a payload column named "record_value".
                for column in table.source.columns:
                    origins[f"{prefix}_{column}"] = (
                        column in structural[name] and column != table.date
                    )
            graph = GraphReduce(
                parent_node=nodes[terminal],
                sql_client=con,
                compute_layer=ComputeLayerEnum.duckdb,
                cut_date=pd.Timestamp(cutoff).to_pydatetime(),
                compute_period_val=3650,
                compute_period_unit=PeriodUnit.day,
                auto_features=True,
                auto_feature_hops_front=depth,
                auto_feature_hops_back=depth,
                feature_families=("base", "state"),
                date_filters_on_agg=True,
                feature_family_max_columns=max_columns,
                feature_family_max_features_per_column=16,
                auto_base_predicate_max=0,
                auto_annotate_features=False,
                auto_text_features=False,
                infer_ts_periods=False,
            )
            graph.add_nodes_from(nodes.values())
            for edge in edges:
                graph.add_entity_edge(
                    nodes[edge.parent],
                    nodes[edge.child],
                    parent_key=edge.parent_key,
                    relation_key=edge.child_key,
                    reduce=edge.reduce,
                )
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                if "plan" in plan_cache:
                    graph.apply_execution_plan(plan_cache["plan"])
                graph.do_transformations_sql()
                if "plan" not in plan_cache:
                    plan_cache["plan"] = graph.freeze_execution_plan()
            frame = con.sql(f"SELECT * FROM {q(graph.parent_node._cur_data_ref)}").df()
            keys = [k if k in frame else nodes[terminal].colabbr(k) for k in root_keys]
            ordered_origins = sorted(origins, key=len, reverse=True)

            def key_value_feature(column):
                origin = next(
                    (c for c in ordered_origins if column.startswith(c + "_")),
                    None,
                )
                if origin is None or not origins[origin]:
                    return False
                suffix = column[len(origin) + 1 :]
                operation = suffix.split("_", 1)[0]
                # Counts and relationship diversity are legitimate structure;
                # the numerical magnitude/category of a key is not payload.
                if suffix == "state_changed" or (
                    suffix.startswith("state_last") and suffix.endswith("_nunique")
                ):
                    return False
                return operation not in {"count", "nunique", "distinct", "repeat"}

            payload = [
                c
                for c in frame
                if c not in excluded
                and c not in keys
                and not key_value_feature(c)
                and pd.api.types.is_numeric_dtype(frame[c])
            ]
            frame = frame[[*keys, *payload]].rename(
                columns={
                    **dict(zip(keys, metadata)),
                    **{c: f"neighbor_context__{c}" for c in payload},
                }
            )
            block = block.copy()
            block["__context_row_order"] = range(len(block))
            enriched = block.merge(
                frame, on=metadata, how="left", validate="many_to_one"
            )
            enriched = enriched.sort_values("__context_row_order").drop(
                columns="__context_row_order"
            )
            enriched.index = block.index
            chunks.append(enriched)
        finally:
            con.close()
    return pd.concat(chunks).reindex(rows.index)
