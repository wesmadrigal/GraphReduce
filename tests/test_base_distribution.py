"""CDFs are bounded base aggregates with point-in-time, frozen SQL cutpoints."""

import datetime as dt
import pickle
import sqlite3

import duckdb
import numpy as np
import pandas as pd
import pytest

from graphreduce.base_distribution import (
    BASE_CDF_MAX_COLUMNS,
    BASE_CDF_MAX_FIT_ROWS,
    base_cdf_expressions,
    is_base_cdf_column,
)
from graphreduce.enum import ComputeLayerEnum, PeriodUnit, SQLOpType
from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import DuckdbNode, SQLNode


def node_for(frame, connection, **kwargs):
    sqlite = isinstance(connection, sqlite3.Connection)
    if sqlite:
        frame.to_sql("events", connection, index=False, if_exists="replace")
    else:
        connection.register("events", frame)
    node = (SQLNode if sqlite else DuckdbNode)(
        fpath="events",
        prefix="evt",
        pk="row",
        client=connection,
        compute_layer=ComputeLayerEnum.sqlite if sqlite else ComputeLayerEnum.duckdb,
        columns=list(frame),
        ts_periods=(),
        auto_base_predicate_max=0,
        **kwargs,
    )
    node._cur_data_ref = "events"
    return node


def operations(node, sample=None):
    kwargs = dict(
        reduce_key="entity", type_func_map={"numerical": ["mean", "min", "max", "sum"]}
    )
    return (
        node.sql_auto_features(sample, **kwargs)
        if sample is not None
        else node.auto_features(**kwargs, compute_layer=node.compute_layer)
    )


def cdf_ops(ops):
    return [op.opval for op in ops if "__cdf_" in op.opval]


@pytest.mark.parametrize("backend", ["duckdb", "sqlite"])
def test_cdf_distinguishes_equal_moments_and_handles_missingness(backend):
    a = [-2.0, -1.0, -1.0, 1.0, 1.0, 2.0]
    b = [-2.0, -np.sqrt(2), 0.0, 0.0, np.sqrt(2), 2.0]
    frame = pd.DataFrame(
        {
            "evt_row": range(21),
            "evt_entity": [1] * 6 + [2] * 6 + [3] * 5 + [4] * 4,
            "evt_value": a + b + [np.nan, np.inf, -np.inf, 0.0, 1.0] + [np.nan] * 4,
        }
    )
    conn = duckdb.connect() if backend == "duckdb" else sqlite3.connect(":memory:")
    try:
        node = node_for(frame, conn)
        ops = operations(node)
        result = node.execute_query(node.build_query(ops)).set_index("evt_entity")
        sketches = [c for c in result if "__cdf_q" in c]
        assert sketches and len(sketches) <= 7
        assert (result.loc[1, sketches] != result.loc[2, sketches]).any()
        assert result.loc[3, "evt_value__cdf_coverage_avg"] == pytest.approx(2 / 5)
        assert result.loc[4, "evt_value__cdf_coverage_avg"] == 0
        assert result.loc[4, sketches].isna().all()
        assert result.loc[1:3, sketches].ge(0).all().all()
        assert result.loc[1:3, sketches].le(1).all().all()
    finally:
        conn.close()


def test_skip_structural_binary_constant_string_and_recursive_inputs():
    frame = pd.DataFrame(
        {
            "evt_row": range(5),
            "evt_entity": range(5),
            "evt_peer_id": range(5),
            "evt_zip": range(5),
            "evt_context": range(5),
            "evt_flag": [0, 1, 0, 1, 0],
            "evt_const": [2.0] * 5,
            "evt_text": ["1", "2", "3", "4", "5"],
            "evt_bool": [True] * 5,
            "evt_empty": [np.nan] * 5,
            "evt_value": [1.1, 2.2, 3.3, 4.4, 5.5],
            "evt_prior__cdf_q0_avg": [0.0, 0.1, 0.3, 0.5, 0.7],
            "evt_prior__cdf_coverage_avg_avg": [0.0, 0.1, 0.3, 0.5, 0.7],
        }
    )
    with duckdb.connect() as conn:
        node = node_for(frame, conn, context_keys=("context",))
        cdfs = [op for op in cdf_ops(operations(node)) if op.startswith("AVG(CASE")]
    assert cdfs and all(" as evt_value__cdf_" in op for op in cdfs)


@pytest.mark.parametrize(
    "column_budget,feature_budget", [(0, None), (1, None), (2, 5), (999999, None)]
)
def test_both_column_and_remaining_feature_budgets(column_budget, feature_budget):
    frame = pd.DataFrame({"evt_entity": [1] * 20, "evt_row": range(20)})
    for index in range(20):
        frame[f"evt_value{index}"] = np.arange(20) + 0.15
    with duckdb.connect() as conn:
        node = node_for(
            frame,
            conn,
            feature_family_max_columns=column_budget,
            feature_family_max_features_per_column=feature_budget,
        )
        enabled = operations(node, frame)
        node.base_distribution_features = False
        disabled = operations(node, frame)
    assert all(op in enabled for op in disabled)
    cdfs = cdf_ops(enabled)
    assert len(cdfs) <= min(column_budget, BASE_CDF_MAX_COLUMNS) * 8
    if column_budget:
        assert cdfs
    for column in frame.columns[2:]:
        if feature_budget is not None:
            assert (
                sum(
                    op.optype == SQLOpType.aggfunc and f" as {column}_" in op.opval
                    for op in enabled
                )
                <= feature_budget
            )


def test_no_new_family_or_recursive_expansion_and_propagation_budget():
    frame = pd.DataFrame(
        {
            "evt_entity": [1] * 6,
            "evt_row": range(6),
            "lower__gr_value_amount__cdf_q0_avg": np.arange(6) / 6,
        }
    )
    with duckdb.connect() as conn:
        node = node_for(frame, conn, feature_propagation_max_functions_per_column=1)
        ops = operations(node, frame)
    propagated = cdf_ops(ops)
    assert len(propagated) == 1
    assert propagated[0].startswith("avg(")
    assert propagated[0].count("__cdf_") == 2  # expression and alias, never nested
    assert is_base_cdf_column("lower__gr_value_amount__cdf_q0_avg_avg")


@pytest.mark.parametrize(
    "boundary,expected",
    [
        ("strict", [1.0, 2.0, 3.0]),
        ("inclusive", [1.0, 2.0, 3.0, 100.0]),
        ("exact", [100.0]),
    ],
)
def test_calibration_filters_before_limit_and_uses_same_cutoff(
    boundary, expected, monkeypatch
):
    frame = pd.DataFrame(
        {
            "evt_row": [-1, 1, 2, 3, 4, 5],
            "evt_entity": [1] * 6,
            "evt_date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2023-12-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                ]
            ),
            "evt_value": [999999.0, -999.0, 1.0, 2.0, 3.0, 100.0],
        }
    )
    with duckdb.connect() as conn:
        node = node_for(
            frame,
            conn,
            date_key="date",
            cut_date=dt.datetime(2024, 1, 4),
            cutoff_boundary=boundary,
            compute_period_val=4,
            compute_period_unit=PeriodUnit.day,
        )
        queries = []
        original = node.execute_query

        def capture(query, *args, **kwargs):
            queries.append(query)
            return original(query, *args, **kwargs)

        monkeypatch.setattr(node, "execute_query", capture)
        sample = node.get_base_distribution_sample(
            ["evt_value"], source_columns=frame.columns
        )
        assert sample.evt_value.tolist() == expected
        assert f"LIMIT {BASE_CDF_MAX_FIT_ROWS}" in queries[-1]
        assert (
            queries[-1].index("WHERE")
            < queries[-1].index("ORDER BY")
            < queries[-1].index("LIMIT")
        )


def test_calibration_per_example_cutoffs_and_keyless_ties_are_deterministic():
    frame = pd.DataFrame(
        {
            "evt_row": [1, 2, 3, 4],
            "evt_entity": [1, 1, 2, 2],
            "evt_date": pd.to_datetime(["2024-01-01"] * 4),
            "root_date": pd.to_datetime(["2024-01-01", "2024-01-02"] * 2),
            "evt_value": [999.0, 2.0, -999.0, 1.0],
        }
    )
    with duckdb.connect() as conn:
        node = node_for(frame, conn, date_key="date", compute_period_val=10)
        node.date_node = SQLNode(prefix="root", date_key="date")
        actual = node.get_base_distribution_sample(
            ["evt_value"], source_columns=frame.columns
        )
        assert actual.evt_value.tolist() == [2.0, 1.0]
        node.pk = None
        first = node.get_base_distribution_sample(
            ["evt_value"], source_columns=frame.columns
        )
        conn.register("events", frame.iloc[::-1].reset_index(drop=True))
        second = node.get_base_distribution_sample(
            ["evt_value"], source_columns=frame.columns
        )
        pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize("enabled", [True, False])
def test_frozen_plan_replay_never_recalibrates_including_old_no_cdf_plan(
    enabled, monkeypatch
):
    def build(conn, cutoff):
        root = DuckdbNode(fpath="roots", prefix="root", pk="id", columns=["id"])
        child = DuckdbNode(
            fpath="events",
            prefix="evt",
            pk="row",
            date_key="date",
            columns=["row", "entity", "date", "value"],
        )
        graph = GraphReduce(
            parent_node=root,
            compute_layer=ComputeLayerEnum.duckdb,
            sql_client=conn,
            cut_date=cutoff,
            auto_features=True,
            compute_period_val=3650,
            compute_period_unit=PeriodUnit.day,
            ts_periods=(),
            auto_base_predicate_max=0,
            train=False,
            base_distribution_features=enabled,
        )
        graph.add_node(root)
        graph.add_node(child)
        graph.add_entity_edge(
            root, child, parent_key="id", relation_key="entity", reduce=True
        )
        return graph

    with duckdb.connect() as conn:
        conn.register("roots", pd.DataFrame({"id": [1, 2, 3]}))
        conn.register(
            "events",
            pd.DataFrame(
                {
                    "row": range(14),
                    "entity": [1] * 7 + [2] * 7,
                    "date": pd.to_datetime(
                        ["2024-01-01"] * 6
                        + ["2025-01-01"]
                        + ["2024-01-01"] * 6
                        + ["2025-01-01"]
                    ),
                    "value": [
                        0.0,
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        1e6,
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                        7.0,
                        1e7,
                    ],
                }
            ),
        )
        graph = build(conn, dt.datetime(2024, 6, 1))
        graph.do_transformations_sql()
        plan = pickle.loads(pickle.dumps(graph.freeze_execution_plan()))
        training_ops = [
            record["method_ops"]
            for record in plan["records"]
            if record["method_name"] == "do_reduce"
        ]
        assert bool(cdf_ops(training_ops[0])) == enabled

        def forbid(*args, **kwargs):
            raise AssertionError("Query data must not fit cutpoints")

        monkeypatch.setattr(SQLNode, "get_base_distribution_sample", forbid)
        later = build(conn, dt.datetime(2025, 6, 1))
        later.base_distribution_features = True  # replay of old plans still has no CDF
        later.hydrate_graph_attrs()
        later.apply_execution_plan(plan)
        later.do_transformations_sql()
        result = (
            conn.sql(f"SELECT * FROM {later.parent_node.get_current_ref()}")
            .df()
            .set_index("root_id")
        )
        sketches = [c for c in result if "__cdf_" in c]
        assert bool(sketches) == enabled
        if enabled:
            assert result.loc[3, sketches].isna().all()  # empty neighborhood
            assert result.loc[1, "evt_value__cdf_q0_avg"] == pytest.approx(2 / 7)


def test_extreme_finite_values_have_finite_thresholds():
    expressions = base_cdf_expressions(pd.Series([-1.7e308, 0.0, 1.7e308]), "value")
    assert expressions
    assert all(
        "inf" not in expression and "nan" not in expression
        for _, expression in expressions
    )
