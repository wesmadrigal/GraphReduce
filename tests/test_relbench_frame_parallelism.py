from __future__ import annotations

import sys
from types import SimpleNamespace

import duckdb
import pandas as pd

from examples import run_all_relbench_tasks
from examples.relbench_dataset_utils import (
    get_training_frame_workers,
    iter_training_frames,
    register_relbench_db_views,
)


def test_training_frame_workers_supports_all_and_explicit_values(monkeypatch):
    monkeypatch.setenv("RELBench_TRAINING_FRAME_WORKERS", "all")
    assert get_training_frame_workers() == 0

    monkeypatch.setenv("RELBench_TRAINING_FRAME_WORKERS", "4")
    assert get_training_frame_workers() == 4


def test_iter_training_frames_runs_concurrently_and_preserves_order():
    con = duckdb.connect()

    def build_frame(frame_con, item):
        assert frame_con is not con
        return pd.DataFrame({"item": [item], "value": [frame_con.sql("SELECT 1").fetchone()[0]]})

    frames = list(iter_training_frames(con, [3, 1, 2], build_frame, workers=0))

    assert [int(frame.loc[0, "item"]) for frame in frames] == [3, 1, 2]
    assert [int(frame.loc[0, "value"]) for frame in frames] == [1, 1, 1]
    con.close()


def test_parallel_registered_views_are_visible_to_worker_cursors(monkeypatch):
    monkeypatch.setenv("RELBench_TRAINING_FRAME_WORKERS", "2")
    con = duckdb.connect()
    db = SimpleNamespace(
        table_dict={
            "events": SimpleNamespace(df=pd.DataFrame({"id": [1, 2]})),
        }
    )

    register_relbench_db_views(con, db, {"events": "events_src"})

    worker_con = con.cursor()
    try:
        assert worker_con.sql("SELECT COUNT(*) FROM events_src").fetchone() == (2,)
    finally:
        worker_con.close()
        con.close()


def test_run_all_exposes_training_frame_workers(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_all", "--training-frame-workers", "all"])

    args = run_all_relbench_tasks.parse_args()

    assert args.training_frame_workers == 0
