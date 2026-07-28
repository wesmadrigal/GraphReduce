#!/usr/bin/env python
"""Utilities for loading RelBench datasets for local examples."""

from __future__ import annotations

import inspect
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Iterator, Sequence, TypeVar

import duckdb
import numpy as np
import pandas as pd
from relbench.base import Database, Table
from relbench.datasets import get_dataset
from relbench.tasks import get_task


FrameItem = TypeVar("FrameItem")
FrameResult = TypeVar("FrameResult")
FrameBuilder = Callable[[duckdb.DuckDBPyConnection, FrameItem], FrameResult]
TRAINING_FRAME_WORKERS_ENV = "RELBench_TRAINING_FRAME_WORKERS"
logger = logging.getLogger(__name__)


def get_training_frame_workers(default: int = 1) -> int:
    """Return the configured training-frame concurrency.

    ``1`` keeps the historical sequential behavior. ``0`` or ``all`` means
    one worker per submitted frame, which is useful on large machines.
    """

    raw_value = os.environ.get(TRAINING_FRAME_WORKERS_ENV, str(default)).strip().lower()
    if raw_value in {"0", "all", "max"}:
        return 0
    try:
        workers = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{TRAINING_FRAME_WORKERS_ENV} must be a positive integer, 0, or 'all'"
        ) from exc
    if workers < 0:
        raise ValueError(
            f"{TRAINING_FRAME_WORKERS_ENV} must be a positive integer, 0, or 'all'"
        )
    return workers


def iter_training_frames(
    con: duckdb.DuckDBPyConnection,
    items: Sequence[FrameItem],
    builder: FrameBuilder,
    *,
    workers: int | None = None,
) -> Iterator[FrameResult]:
    """Build cutoff frames concurrently while yielding them in input order.

    Each concurrent builder receives its own DuckDB cursor. This avoids the
    pending-query errors produced by sharing one DuckDB connection between
    threads, while persistent source tables remain visible to every cursor.
    """

    if not items:
        return
    worker_count = get_training_frame_workers() if workers is None else int(workers)
    if worker_count < 0:
        raise ValueError("workers must be non-negative")
    if worker_count == 0:
        worker_count = len(items)
    worker_count = min(worker_count, len(items))

    if worker_count <= 1:
        for item in items:
            yield builder(con, item)
        return

    def build_with_cursor(item: FrameItem) -> FrameResult:
        worker_con = con.cursor()
        try:
            return builder(worker_con, item)
        finally:
            worker_con.close()

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(build_with_cursor, item) for item in items]
        for future in futures:
            yield future.result()


class RelBenchFrameStore:
    """Accumulate per-cutoff frames without retaining every part in RAM.

    Small tasks stay in memory. Once the row limit is crossed, completed parts
    are written as temporary Parquet files and can be read back one part at a
    time. The public examples still return pandas frames, while this class
    prevents the per-timestamp list itself from becoming a second full copy.
    """

    def __init__(
        self,
        name: str,
        spill_row_limit: int | None = None,
        persist_each_frame: bool = False,
    ) -> None:
        configured_limit = os.environ.get("RELBench_FRAME_SPILL_ROWS", "250000")
        self.spill_row_limit = (
            int(configured_limit) if spill_row_limit is None else int(spill_row_limit)
        )
        if self.spill_row_limit < 1:
            raise ValueError("spill_row_limit must be positive")
        self.name = name
        self.persist_each_frame = persist_each_frame
        self._memory_frames: list[pd.DataFrame] = []
        self._memory_rows = 0
        self._row_count = 0
        self._part_paths: list[Path] = []
        self._temporary_directory: TemporaryDirectory[str] | None = None
        self._closed = False

    @property
    def spilled(self) -> bool:
        return bool(self._part_paths)

    @property
    def part_paths(self) -> tuple[Path, ...]:
        return tuple(self._part_paths)

    @property
    def columns(self) -> list[str]:
        if self._memory_frames:
            return self._memory_frames[0].columns.tolist()
        if self._part_paths:
            return pd.read_parquet(self._part_paths[0]).columns.tolist()
        return []

    @property
    def row_count(self) -> int:
        return self._row_count

    def __len__(self) -> int:
        return self._row_count

    def _ensure_directory(self) -> Path:
        if self._temporary_directory is None:
            self._temporary_directory = TemporaryDirectory(prefix=f"{self.name}-")
        return Path(self._temporary_directory.name)

    def _write_part(self, frame: pd.DataFrame) -> None:
        path = self._ensure_directory() / f"part-{len(self._part_paths):05d}.parquet"
        frame.to_parquet(path, index=False)
        self._part_paths.append(path)

    def _flush_memory(self) -> None:
        if not self._memory_frames:
            return
        for frame in self._memory_frames:
            self._write_part(frame)
        self._memory_frames.clear()
        self._memory_rows = 0

    def append(self, frame: pd.DataFrame) -> None:
        if self._closed:
            raise RuntimeError("Cannot append to a closed RelBenchFrameStore")
        if frame.empty:
            return
        self._row_count += len(frame)
        if self.persist_each_frame:
            self._write_part(frame)
            return
        if self._part_paths or self._memory_rows + len(frame) > self.spill_row_limit:
            self._flush_memory()
            self._write_part(frame)
            return
        self._memory_frames.append(frame)
        self._memory_rows += len(frame)

    def column_nunique(self, column: str) -> int:
        values: set[object] = set()
        for batch in self.iter_batches([column]):
            values.update(batch[column].dropna().unique().tolist())
        return len(values)

    def target_nunique(self, column: str) -> int:
        return self.column_nunique(column)

    def target_std(self, column: str) -> float:
        values: list[np.ndarray] = []
        for batch in self.iter_batches([column]):
            values.append(pd.to_numeric(batch[column], errors="coerce").fillna(0).to_numpy(dtype="float64"))
        return float(np.std(np.concatenate(values))) if values else 0.0

    def iter_batches(self, columns: Sequence[str] | None = None) -> Iterator[pd.DataFrame]:
        selected_columns = list(columns) if columns is not None else None
        for frame in self._memory_frames:
            yield frame if selected_columns is None else frame[selected_columns]
        for path in self._part_paths:
            frame = pd.read_parquet(path, columns=selected_columns)
            yield frame

    def sample_frame(self) -> pd.DataFrame:
        return next(self.iter_batches(), pd.DataFrame())

    def to_dataframe(self) -> pd.DataFrame:
        frames = list(self.iter_batches())
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._memory_frames.clear()
        self._row_count = 0
        self._part_paths.clear()
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None


class _TimedeltaSqlDays:
    def __init__(self, days: int):
        self.days = days

    def __str__(self) -> str:
        return str(self.days)

    def __mul__(self, other):
        return pd.Timedelta(days=self.days) * other

    def __rmul__(self, other):
        return other * pd.Timedelta(days=self.days)

    def __rsub__(self, other):
        return other - pd.Timedelta(days=self.days)

    def __radd__(self, other):
        return other + pd.Timedelta(days=self.days)


def get_relbench_dataset_db(
    dataset_name: str,
    download: bool = True,
    upto_test_timestamp: bool = False,
):
    """Load a RelBench dataset through the official getter and return its DB."""

    try:
        dataset = get_dataset(dataset_name, download=download)
    except ValueError as exc:
        if not download or "SHA256 hash of downloaded file" not in str(exc):
            raise
        _download_relbench_archive_without_stale_hash(f"{dataset_name}/db.zip")
        dataset = get_dataset(dataset_name, download=False)
    return dataset, dataset.get_db(upto_test_timestamp=upto_test_timestamp)


def _download_relbench_archive_without_stale_hash(resource: str) -> None:
    """Recover when RelBench's published archive hash is stale.

    RelBench occasionally republishes a database or task archive before
    releasing a package with the matching registry hash. The official
    download has already established the exact HTTPS URL; this fallback only
    bypasses the stale package hash, records the newly observed hash through
    Pooch, and preserves RelBench's normal cache layout and unzip processor.
    """

    import pooch
    from relbench.datasets import DOWNLOAD_REGISTRY as DATASET_DOWNLOAD_REGISTRY
    from relbench.tasks import DOWNLOAD_REGISTRY as TASK_DOWNLOAD_REGISTRY

    resource_path = Path(resource)
    registry = (
        TASK_DOWNLOAD_REGISTRY
        if "tasks" in resource_path.parts
        else DATASET_DOWNLOAD_REGISTRY
    )
    if resource not in registry.registry:
        raise ValueError(f"RelBench has no registered archive for {resource}")

    if resource_path.is_absolute() or ".." in resource_path.parts:
        raise ValueError(f"Invalid RelBench archive path: {resource}")

    cache_dir = Path(registry.abspath).joinpath(*resource_path.parts[:-1])
    archive_path = cache_dir / resource_path.name
    archive_path.unlink(missing_ok=True)
    logger.warning(
        "RelBench registry hash for %s is stale; downloading the current "
        "archive without the old hash and recording its observed SHA256.",
        resource,
    )
    pooch.retrieve(
        registry.get_url(resource),
        known_hash=None,
        fname=archive_path.name,
        path=cache_dir,
        processor=pooch.Unzip(extract_dir="."),
        progressbar=True,
    )


def register_relbench_db_views(
    con: duckdb.DuckDBPyConnection,
    db: Database,
    table_to_view: dict[str, str],
    row_number_ids: dict[str, str] | None = None,
    drop_columns: dict[str, list[str]] | None = None,
) -> list[str]:
    """Register RelBench DB tables as DuckDB views without local parquet files."""

    row_number_ids = row_number_ids or {}
    drop_columns = drop_columns or {}
    parallel_training_frames = get_training_frame_workers() not in {1}
    registered_refs: list[str] = []
    for table_name, view_name in table_to_view.items():
        df = db.table_dict[table_name].df.copy()
        cols_to_drop = [col for col in drop_columns.get(table_name, []) if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        for column in df.columns:
            if str(df[column].dtype) in {"str", "string"}:
                df[column] = df[column].astype("object")
        registered_ref = f"_relbench_{view_name}_df"
        con.register(registered_ref, df)
        registered_refs.append(registered_ref)
        source_ref = registered_ref
        if parallel_training_frames:
            source_ref = f"{registered_ref}_table"
            con.sql(
                f"CREATE OR REPLACE TABLE {source_ref} AS SELECT * FROM {registered_ref}"
            )
        if table_name in row_number_ids:
            con.sql(
                f"""
                CREATE OR REPLACE VIEW {view_name} AS
                SELECT
                    row_number() OVER () AS {row_number_ids[table_name]},
                    *
                FROM {source_ref}
                """
            )
        else:
            con.sql(
                f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM {source_ref}"
            )
    return registered_refs


def get_relbench_task(
    dataset_name: str,
    task_name: str,
    download: bool = True,
):
    """Load a RelBench task through the official task getter."""

    try:
        return get_task(dataset_name, task_name, download=download)
    except PermissionError:
        if not download:
            raise
        return get_task(dataset_name, task_name, download=False)
    except ValueError as exc:
        # RelBench validates hosted task archives against the hash shipped in
        # the installed package. If the server republishes an archive before
        # the package registry is updated, refresh the task archive through
        # the shared stale-hash recovery path instead of failing at startup.
        if not download or "SHA256 hash of downloaded file" not in str(exc):
            raise
        _download_relbench_archive_without_stale_hash(
            f"{dataset_name}/tasks/{task_name}.zip"
        )
        return get_task(dataset_name, task_name, download=False)


def _task_split_timestamp(task, split: str) -> pd.Timestamp:
    if split == "train":
        return task.dataset.val_timestamp - task.timedelta
    if split == "val":
        return task.dataset.val_timestamp
    if split == "test":
        return task.dataset.test_timestamp
    raise ValueError(f"Unknown task split: {split}")


def _task_split_timestamps(task, split: str, db: Database) -> pd.Series:
    if split == "train":
        start = task.dataset.val_timestamp - task.timedelta
        end = db.min_timestamp
        freq = -task.timedelta
    elif split == "val":
        start = task.dataset.val_timestamp
        end = min(
            task.dataset.val_timestamp + task.timedelta * (task.num_eval_timestamps - 1),
            task.dataset.test_timestamp - task.timedelta,
        )
        freq = task.timedelta
    elif split == "test":
        start = task.dataset.test_timestamp
        end = min(
            task.dataset.test_timestamp + task.timedelta * (task.num_eval_timestamps - 1),
            db.max_timestamp - task.timedelta,
        )
        freq = task.timedelta
    else:
        raise ValueError(f"Unknown task split: {split}")
    return pd.Series(pd.date_range(start=start, end=end, freq=freq))


def _make_split_table(task, split: str, db: Database) -> Table:
    """Build a task table against an already-loaded database object."""

    timestamps = _task_split_timestamps(task, split, db)
    original_timedelta = task.timedelta
    if isinstance(original_timedelta, pd.Timedelta):
        task.timedelta = _task_timedelta_sql_value(task)
    try:
        table = task.make_table(db, timestamps)
        return task.filter_dangling_entities(table)
    finally:
        task.timedelta = original_timedelta


def get_relbench_split_timestamps(task, split: str, db: Database | None = None) -> list[pd.Timestamp]:
    """Return the complete official timestamp schedule for a task split.

    The task table is authoritative because RelBench can drop a boundary
    timestamp when its generated label table has no valid rows there (Trial
    has exactly this behavior). The generated dates are still checked against
    the task's formal schedule by ``get_relbench_split_task_table``.
    """

    if db is None:
        original_cache_dir = task.cache_dir
        task.cache_dir = None
        try:
            table = task._get_table(split)
        finally:
            task.cache_dir = original_cache_dir
    else:
        table = _make_split_table(task, split, db)
    timestamps = pd.to_datetime(table.df[task.time_col]).dropna().drop_duplicates().sort_values()
    return [pd.Timestamp(timestamp) for timestamp in timestamps]


def get_relbench_split_task_table(
    dataset_name: str,
    task_name: str,
    split: str,
    *,
    download: bool = True,
    task=None,
    db: Database | None = None,
) -> tuple[object, Table, list[pd.Timestamp]]:
    """Build the full official task table and return all of its timestamps.

    The examples need the full historical training schedule so that every
    label window contributes a GraphReduce frame. ``cache_dir`` is disabled
    here because these examples already own the feature-frame lifecycle and
    should not depend on writing into the global RelBench cache.
    """

    if task is None:
        task = get_relbench_task(dataset_name, task_name, download=download)
    if db is None:
        db = task.dataset.get_db(upto_test_timestamp=split != "test")

    if db is None:
        original_cache_dir = task.cache_dir
        task.cache_dir = None
        try:
            table = task._get_table(split)
        finally:
            task.cache_dir = original_cache_dir
    else:
        table = _make_split_table(task, split, db)

    table.df[task.time_col] = pd.to_datetime(table.df[task.time_col])
    table_timestamps = sorted(
        pd.Timestamp(timestamp) for timestamp in table.df[task.time_col].dropna().unique()
    )
    formal_timestamps = _task_split_timestamps(task, split, db)
    if not set(table_timestamps).issubset(set(formal_timestamps)):
        raise ValueError(
            f"RelBench {dataset_name}/{task_name} {split} task-table timestamps "
            "fall outside the formal schedule"
        )
    return task, table, table_timestamps


def _select_single_timestamp_table(
    table: Table,
    time_col: str,
    split: str,
    timestamp: pd.Timestamp | None = None,
) -> tuple[Table, pd.Timestamp]:
    df = table.df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    if timestamp is not None:
        timestamp = pd.Timestamp(timestamp)
        selected = df[df[time_col] == timestamp].copy()
    else:
        selected = pd.DataFrame()

    if selected.empty:
        timestamp = df[time_col].max() if split == "train" else df[time_col].min()
        selected = df[df[time_col] == timestamp].copy()

    return (
        Table(
            df=selected.reset_index(drop=True),
            fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
            pkey_col=table.pkey_col,
            time_col=table.time_col,
        ),
        pd.Timestamp(timestamp),
    )


def _task_timedelta_sql_value(task) -> str | int:
    if not isinstance(task.timedelta, pd.Timedelta):
        return task.timedelta
    source = inspect.getsource(type(task))
    if "{self.timedelta} days" in source:
        if "timestamps[0] - self.timedelta" in source:
            return _TimedeltaSqlDays(task.timedelta.days)
        return task.timedelta.days
    return f"{task.timedelta.days} days"


def get_single_timestamp_task_table(
    dataset_name: str,
    task_name: str,
    split: str,
    download: bool = True,
    task=None,
    db=None,
) -> tuple[object, Table, pd.Timestamp]:
    """Get one RelBench task timestamp per split for split-level FE examples.

    The public task API returns task supervision tables. For train splits, that
    can include multiple historical timestamps. These examples intentionally use
    one timestamp per split: latest train timestamp, validation timestamp, and
    test timestamp.
    """

    if task is None:
        task = get_relbench_task(dataset_name, task_name, download=download)
    timestamp = _task_split_timestamp(task, split)

    # Build exactly the timestamp these examples use from the official
    # RelBench dataset DB. This intentionally ignores any repo-local parquet
    # materializations.
    if db is None:
        db = task.dataset.get_db(upto_test_timestamp=split != "test")
    split_timestamps = _task_split_timestamps(task, split, db)
    original_timedelta = task.timedelta
    if isinstance(original_timedelta, pd.Timedelta):
        task.timedelta = _task_timedelta_sql_value(task)
    try:
        def make_filtered_table(timestamps: pd.Series) -> Table:
            table = task.make_table(db, timestamps)
            return task.filter_dangling_entities(table)

        table = make_filtered_table(pd.Series([timestamp]))
        selected, selected_timestamp = _select_single_timestamp_table(
            table=table,
            time_col=task.time_col,
            split=split,
            timestamp=timestamp,
        )
        if selected.df.empty:
            table = make_filtered_table(split_timestamps)
            selected, selected_timestamp = _select_single_timestamp_table(
                table=table,
                time_col=task.time_col,
                split=split,
                timestamp=None,
            )
    finally:
        task.timedelta = original_timedelta

    return task, selected, selected_timestamp


def target_table_from_frame(task, frame: pd.DataFrame) -> Table:
    """Build a RelBench target table matching a joined feature/label frame."""

    return Table(
        df=frame[[task.time_col, task.entity_col, task.target_col]].copy(),
        fkey_col_to_pkey_table={task.entity_col: task.entity_table},
        pkey_col=None,
        time_col=task.time_col,
    )
