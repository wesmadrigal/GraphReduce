#!/usr/bin/env python
"""Utilities for materializing RelBench datasets into local example files."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
from relbench.datasets import get_dataset


def _is_valid_csv(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _is_valid_parquet(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    con = duckdb.connect()
    try:
        con.sql(f"select 1 from read_parquet('{path}') limit 1").fetchall()
        return True
    except Exception:
        return False
    finally:
        con.close()


def _is_valid_target(path: Path) -> bool:
    if path.suffix == ".csv":
        return _is_valid_csv(path)
    if path.suffix == ".parquet":
        return _is_valid_parquet(path)
    return path.exists() and path.stat().st_size > 0


def materialize_relbench_dataset(
    dataset_name: str,
    data_dir: Path,
    table_name_to_filename: dict[str, str],
) -> list[str]:
    """Materialize a RelBench dataset into local files used by the examples."""

    data_dir.mkdir(parents=True, exist_ok=True)
    stamp_path = data_dir / ".relbench_materialized.json"
    expected_files = {table: data_dir / filename for table, filename in table_name_to_filename.items()}

    if stamp_path.exists():
        try:
            stamp = json.loads(stamp_path.read_text())
        except json.JSONDecodeError:
            stamp = None
        if (
            stamp
            and stamp.get("dataset_name") == dataset_name
            and stamp.get("table_name_to_filename") == table_name_to_filename
            and all(_is_valid_target(path) for path in expected_files.values())
        ):
            return []

    dataset = get_dataset(dataset_name, download=True)
    db = dataset.get_db(upto_test_timestamp=False)
    materialized: list[str] = []

    for table_name, filename in table_name_to_filename.items():
        if table_name not in db.table_dict:
            raise ValueError(f"Table '{table_name}' not found in dataset '{dataset_name}'.")
        out_path = expected_files[table_name]
        df = db.table_dict[table_name].df
        if out_path.suffix == ".csv":
            df.to_csv(out_path, index=False)
        elif out_path.suffix == ".parquet":
            df.to_parquet(out_path, index=False)
        else:
            raise ValueError(f"Unsupported output format for {out_path}")
        materialized.append(filename)

    stamp_path.write_text(
        json.dumps(
            {
                "dataset_name": dataset_name,
                "table_name_to_filename": table_name_to_filename,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return materialized
