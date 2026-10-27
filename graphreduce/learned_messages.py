"""Bounded relational extraction and pooling for fitted message encoders.

This module never reads targets or trains estimators. The caller owns training,
label availability, and persistence of the fitted row transformation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pool_messages(
    rows: pd.DataFrame, values: pd.DataFrame, roots: pd.Index
) -> pd.DataFrame:
    """Pool aligned numeric row messages, retaining empty neighborhoods.

    ``rows`` must contain ``__example_id``; ``values`` has the same row order.
    Missing scores remain missing, with explicit count and availability fields.
    """
    if len(rows) != len(values) or not values.columns.is_unique:
        raise ValueError(
            "Messages must align with neighborhood rows and have unique columns"
        )
    work = values.reset_index(drop=True).copy()
    work["__example_id"] = rows["__example_id"].to_numpy()
    grouped = work.groupby("__example_id", sort=True)
    pooled = grouped[list(values)].agg(["mean", "sum", "max", "std"])
    pooled.columns = [f"{column}__{stat}" for column, stat in pooled.columns]
    pooled = pooled.reindex(roots)
    pooled["neighbor_count"] = grouped.size().reindex(roots, fill_value=0)
    pooled["available"] = (pooled.neighbor_count > 0).astype("int8")
    return pooled.replace([np.inf, -np.inf], np.nan)


def extract_path(
    examples,
    tables,
    relationships,
    *,
    root_name,
    entity_key,
    timestamp,
    path,
    max_neighbors=128,
    max_columns=16,
    seed=0,
    memory_limit="4GB",
    context_depth=0,
    context_max_columns=4,
    context_max_relations=12,
    context_plan_cache=None,
    include_terminal_keys=False,
):
    """Extract bounded rows along a declared path of Table-like objects.

    All dated nodes must precede the example timestamp, including the root.
    Structural keys are excluded from payloads. Intermediate fanout is bounded
    at each hop, so a wide cyclic path cannot produce an unbounded final join.
    Sampling is stable by entity, cutoff and keys, independent of input order.
    DataFrame sources are required; the caller materializes other source types.
    ``include_terminal_keys`` exposes keys as ``__neighbor_key_N`` metadata for
    identity-preserving encoders/peer joins. These are not feature payloads.
    """
    import duckdb

    if max_neighbors < 1 or max_columns < 1 or not path:
        raise ValueError("A path and positive neighborhood/column budgets are required")
    if not examples.index.is_unique:
        raise ValueError("Example index must be unique")

    def parts(key):
        return (key,) if isinstance(key, str) else tuple(key or ())

    def q(value):
        return '"' + str(value).replace('"', '""') + '"'

    structural = {
        name: set(parts(table.key)) | {table.date} for name, table in tables.items()
    }
    for edge in relationships:
        structural[edge.parent].update(parts(edge.parent_key))
        structural[edge.child].update(parts(edge.child_key))
    con = duckdb.connect(
        config={
            "memory_limit": memory_limit,
            "max_temp_directory_size": "16GB",
            "threads": 2,
        }
    )
    try:
        ex = examples[[entity_key, timestamp]].copy()
        ex["__example_id"] = examples.index
        con.register("examples", ex)
        for index, name in enumerate((root_name, *path)):
            frame = tables[name].source
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(
                    "extract_path currently requires DataFrame table sources"
                )
            con.register(f"table_{index}", frame)
        root = tables[root_name]
        if len(parts(root.key)) != 1:
            raise ValueError(
                "extract_path currently supports single-column entity keys"
            )
        con.execute(
            f"CREATE TEMP TABLE hop_0 AS SELECT e.__example_id, e.{q(timestamp)} AS __cutoff, "
            f"e.{q(entity_key)} AS __entity, r.* FROM examples e JOIN table_0 r "
            f"ON e.{q(entity_key)} = r.{q(root.key)} "
            + (f"WHERE r.{q(root.date)} < e.{q(timestamp)}" if root.date else "")
        )
        parent = root_name
        for hop, name in enumerate(path, 1):
            matches = [
                r for r in relationships if r.parent == parent and r.child == name
            ]
            if len(matches) != 1:
                raise ValueError("Path must identify exactly one declared edge per hop")
            edge, table = matches[0], tables[name]
            join = " AND ".join(
                f"p.{q(a)} = c.{q(b)}"
                for a, b in zip(
                    parts(edge.parent_key), parts(edge.child_key), strict=True
                )
            )
            date_filter = f"AND c.{q(table.date)} < p.__cutoff" if table.date else ""
            keys = parts(table.key) or parts(edge.child_key)
            # DISTINCT removes duplicate paths to the same terminal row.
            con.execute(
                f"CREATE TEMP TABLE hop_{hop} AS SELECT * FROM ("
                f"SELECT DISTINCT p.__example_id, p.__cutoff, p.__entity, c.* "
                f"FROM hop_{hop - 1} p JOIN table_{hop} c ON {join} {date_filter}) "
                f"QUALIFY row_number() OVER (PARTITION BY __example_id ORDER BY "
                f"hash(__entity, __cutoff, {', '.join(q(k) for k in keys)}, {int(seed)}), "
                f"{', '.join(q(k) for k in keys)}) <= {int(max_neighbors)}"
            )
            con.execute(f"DROP TABLE hop_{hop - 1}")
            parent = name
        table = tables[path[-1]]
        candidates = (
            table.columns if table.columns is not None else table.source.columns
        )
        payload = [c for c in candidates if c not in structural[path[-1]]][:max_columns]
        selected = ["__example_id", *payload]
        query = f"SELECT {', '.join(q(c) for c in selected)}"
        if table.date:
            query += f", epoch(__cutoff - {q(table.date)}) / 86400 AS __age_days"
        metadata = []
        if context_depth or include_terminal_keys:
            if not table.key:
                raise ValueError("Neighbor context requires a terminal primary key")
            if context_depth:
                query += ", __cutoff"
            for index, key in enumerate(parts(table.key)):
                column = f"__neighbor_key_{index}"
                query += f", {q(key)} AS {q(column)}"
                metadata.append(column)
        query += f" FROM hop_{len(path)} ORDER BY __example_id, {', '.join(q(k) for k in keys)}"
        rows = con.sql(query).df()
        if context_depth:
            from .neighbor_context import enrich_neighbors

            con.close()
            rows = enrich_neighbors(
                rows,
                tables,
                relationships,
                terminal=path[-1],
                depth=context_depth,
                max_columns=context_max_columns,
                max_relations=context_max_relations,
                max_neighbors=max_neighbors,
                seed=seed,
                memory_limit=memory_limit,
                plan_cache=context_plan_cache,
            )
            rows = rows.drop(columns=["__cutoff", *([] if include_terminal_keys else metadata)])
        return rows
    finally:
        con.close()


def extract_memberships(examples, tables, relationships, **kwargs):
    """Extract bounded, point-in-time root-to-neighbor identity memberships.

    Project away payloads before any path join, so large text attributes and
    unrelated columns cannot multiply the memory cost of a peer lookup.
    Keys are metadata for relational joins, never numeric model features.
    """
    from types import SimpleNamespace

    def parts(key):
        return (key,) if isinstance(key, str) else tuple(key or ())

    required = {name: set(parts(t.key)) | ({t.date} if t.date else set()) for name, t in tables.items()}
    for edge in relationships:
        required[edge.parent].update(parts(edge.parent_key))
        required[edge.child].update(parts(edge.child_key))
    projected = {
        name: SimpleNamespace(
            source=table.source[[c for c in table.source if c in required[name]]],
            key=table.key, date=table.date, columns=(),
        )
        for name, table in tables.items()
    }
    result = extract_path(
        examples, projected, relationships,
        include_terminal_keys=True, max_columns=1, **kwargs,
    )
    return result[[c for c in result if c == "__example_id" or c.startswith("__neighbor_key_")]]
