import ast
from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace

import duckdb
import pandas as pd
from relbench.tasks import get_task_names


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

from relbench_v2_task_utils import (  # noqa: E402
    EdgeDefinition,
    GraphDefinition,
    NodeDefinition,
    _popularity_predictions,
    _prepare_model_inputs,
    _sample_timestamps,
    _scope_root_view,
)


def test_sample_timestamps_is_even_and_keeps_boundaries():
    timestamps = list(pd.date_range("2020-01-01", periods=40, freq="D"))

    selected = _sample_timestamps(timestamps, 10)

    assert len(selected) == 10
    assert selected[0] == timestamps[0]
    assert selected[-1] == timestamps[-1]
    assert selected == sorted(selected)


def test_popularity_predictions_prefer_source_history_then_global():
    frame = pd.DataFrame({"source": [1, 2]})
    by_source = {
        1: Counter({7: 3, 8: 2}),
        2: Counter({9: 1}),
    }
    global_counts = Counter({8: 10, 6: 8, 5: 7})

    predictions = _popularity_predictions(
        frame,
        "source",
        by_source,
        global_counts,
        eval_k=3,
        num_destinations=10,
    )

    assert predictions.tolist() == [[7, 8, 6], [9, 8, 6]]


def test_prepare_model_inputs_converts_dates_and_low_cardinality_categories():
    task = SimpleNamespace(
        target_col="target",
        time_col="timestamp",
        entity_col="entity_id",
    )
    base = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2022-01-02", "2022-01-03"]),
            "entity_id": [10, 11],
            "created_at": pd.to_datetime(["2022-01-01", "2022-01-01"]),
            "segment": ["a", "b"],
            "value": [1.0, 2.0],
            "target": [0, 1],
        }
    )

    prepared, feature_columns = _prepare_model_inputs(
        {"train": base, "val": base, "test": base},
        task,
    )

    assert "snapshot_unix" in feature_columns
    assert "created_at_age_seconds" in feature_columns
    assert "segment" in feature_columns
    assert prepared["train"]["created_at_age_seconds"].tolist() == [
        86_400.0,
        172_800.0,
    ]


def test_scope_root_view_pushes_ids_into_direct_reduced_children():
    con = duckdb.connect()
    con.sql("CREATE TABLE roots (id INTEGER)")
    con.sql("INSERT INTO roots VALUES (1), (2), (3)")
    con.sql("CREATE TABLE events (event_id INTEGER, root_id INTEGER)")
    con.sql("INSERT INTO events VALUES (10, 1), (11, 1), (20, 2), (30, 3)")
    con.sql("CREATE TABLE dimensions (id INTEGER)")
    definition = GraphDefinition(
        root="root",
        nodes=(
            NodeDefinition("root", "roots", "r", "id", None),
            NodeDefinition("events", "events", "e", "event_id", None),
            NodeDefinition("dimension", "dimensions", "d", "id", None),
        ),
        edges=(
            EdgeDefinition("root", "events", "id", "root_id"),
            EdgeDefinition("root", "dimension", "id", "id", reduce=False),
        ),
    )

    scoped = _scope_root_view(
        con,
        definition,
        pd.Series([1, 3]),
        "id",
        "test",
    )
    scoped_nodes = {node.name: node for node in scoped.nodes}

    assert con.sql(f"SELECT id FROM {scoped_nodes['root'].view} ORDER BY id").fetchall() == [
        (1,),
        (3,),
    ]
    assert con.sql(
        f"SELECT event_id FROM {scoped_nodes['events'].view} ORDER BY event_id"
    ).fetchall() == [(10,), (11,), (30,)]
    assert scoped_nodes["dimension"].view == "dimensions"


def test_every_official_v2_task_has_an_example_entrypoint():
    datasets = ("rel-salt", "rel-arxiv", "rel-ratebeer", "rel-mimic")
    expected = {
        (dataset_name, task_name)
        for dataset_name in datasets
        for task_name in get_task_names(dataset_name)
    }
    implemented: set[tuple[str, str]] = set()

    for path in EXAMPLES_DIR.glob("relbench_*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id not in {
                "run_relbench_v2_entity_task",
                "run_relbench_v2_link_task",
            }:
                continue
            if len(node.args) >= 2:
                implemented.add(
                    (ast.literal_eval(node.args[0]), ast.literal_eval(node.args[1]))
                )

    assert expected <= implemented
