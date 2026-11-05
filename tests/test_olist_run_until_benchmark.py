"""Compare full-data and sampled run_until execution against local Olist CSVs.

Run from the repository root:
    RUN_OLIST_BENCHMARKS=1 venv/bin/python -m pytest -q -s \
        tests/test_olist_run_until_benchmark.py

Optional environment variables:
    OLIST_DATA_DIR: directory containing the original Olist CSVs
    OLIST_BENCHMARK_REPEATS: measured runs per case (default: 3)
    OLIST_BENCHMARK_THREADS: DuckDB threads per connection (default: 1)

Times include CSV loading, annotations, descendant reductions/joins, and the
orders reduction. Graph construction, result fetching, and correctness checks
are outside the timer. Each run uses a fresh graph and in-memory connection.
One warm-up per case precedes alternating measured runs. This measures warm
filesystem-cache performance; it does not flush OS caches or impose a timing
threshold on different machines.
"""

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter

import pandas as pd
import pytest

from graphreduce.enum import ComputeLayerEnum, SQLOpType
from graphreduce.graph_reduce import GraphReduce
from graphreduce.models import sqlop
from graphreduce.node import DuckdbNode

duckdb = pytest.importorskip("duckdb")

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_OLIST_BENCHMARKS") != "1",
    reason="Set RUN_OLIST_BENCHMARKS=1 to benchmark the local Olist dataset.",
)

TABLES = {
    "customers": ("olist_customers_dataset.csv", "cust", "customer_id"),
    "orders": ("olist_orders_dataset.csv", "ord", "order_id"),
    "items": (
        "olist_order_items_dataset.csv",
        "itm",
        ("order_id", "order_item_id"),
    ),
    "payments": (
        "olist_order_payments_dataset.csv",
        "pay",
        ("order_id", "payment_sequential"),
    ),
    "reviews": ("olist_order_reviews_dataset.csv", "rev", "review_id"),
}


@dataclass
class TimedRun:
    seconds: float
    root_ids: frozenset
    loaded_rows: dict[str, int]
    output: pd.DataFrame


@pytest.fixture(scope="module")
def olist_sources():
    data_dir = Path(os.getenv("OLIST_DATA_DIR", "/usr/local/lake/olist"))
    missing = [
        filename
        for filename, _, _ in TABLES.values()
        if not (data_dir / filename).is_file()
    ]
    if missing:
        pytest.skip(f"Missing Olist CSVs in {data_dir}: {', '.join(missing)}")
    sources = {}
    for name, (filename, _, _) in TABLES.items():
        path = data_dir / filename
        with path.open(newline="", encoding="utf-8-sig") as source:
            columns = next(csv.reader(source))
        sources[name] = (path, columns)
    return sources


def _reduction(key, *aggregates):
    return [sqlop(optype=SQLOpType.agg, opval=key)] + [
        sqlop(optype=SQLOpType.aggfunc, opval=expression) for expression in aggregates
    ]


def _build_graph(connection, sources):
    nodes = {}
    for name, (_, prefix, primary_key) in TABLES.items():
        path, columns = sources[name]
        nodes[name] = DuckdbNode(
            fpath="'" + str(path).replace("'", "''") + "'",
            table_name=name,
            prefix=prefix,
            pk=primary_key,
            columns=columns,
            client=connection,
            compute_layer=ComputeLayerEnum.duckdb,
        )

    nodes["orders"].do_annotate_ops = [
        sqlop(optype=SQLOpType.select, opval="*"),
        sqlop(
            optype=SQLOpType.select,
            opval="CASE WHEN ord_order_status = 'delivered' THEN 1 ELSE 0 END AS ord_delivered",
        ),
    ]
    nodes["items"].do_reduce_ops = _reduction(
        "itm_order_id",
        "count(*) AS itm_count",
        "sum(itm_price) AS itm_price_sum",
        "sum(itm_freight_value) AS itm_freight_sum",
    )
    nodes["payments"].do_reduce_ops = _reduction(
        "pay_order_id",
        "count(*) AS pay_count",
        "sum(pay_payment_value) AS pay_value_sum",
    )
    nodes["reviews"].do_reduce_ops = _reduction(
        "rev_order_id",
        "count(*) AS rev_count",
        "sum(rev_review_score) AS rev_score_sum",
    )
    nodes["orders"].do_reduce_ops = _reduction(
        "ord_customer_id",
        "count(*) AS ord_count",
        "sum(ord_delivered) AS ord_delivered_count",
        "sum(coalesce(itm_count, 0)) AS ord_item_count",
        "sum(coalesce(itm_price_sum, 0)) AS ord_item_value",
        "sum(coalesce(itm_freight_sum, 0)) AS ord_freight_value",
        "sum(coalesce(pay_value_sum, 0)) AS ord_payment_value",
        "sum(coalesce(rev_score_sum, 0)) / nullif(sum(rev_count), 0) AS ord_review_score",
    )
    graph = GraphReduce(
        name="olist_run_until_benchmark",
        parent_node=nodes["customers"],
        compute_layer=ComputeLayerEnum.duckdb,
        sql_client=connection,
        auto_feature_hops_back=2,
        auto_features=False,
        auto_annotate_features=False,
        train=False,
    )
    graph.add_entity_edge(
        nodes["customers"],
        nodes["orders"],
        parent_key="customer_id",
        relation_key="customer_id",
    )
    for name in ("items", "payments", "reviews"):
        graph.add_entity_edge(
            nodes["orders"],
            nodes[name],
            parent_key="order_id",
            relation_key="order_id",
        )
    return graph, nodes


def _run_case(sources, sample_size, threads):
    with duckdb.connect(config={"threads": threads}) as connection:
        graph, nodes = _build_graph(connection, sources)
        start = perf_counter()
        if sample_size is None:
            # Exercise the API without a sample_size argument for the baseline.
            result = graph.run_until(nodes["orders"], "do_reduce")
        else:
            result = graph.run_until(
                nodes["orders"], "do_reduce", sample_size=sample_size
            )
        elapsed = perf_counter() - start

        # DuckDB eagerly materializes the reduction inside run_until. Fetching
        # the already-produced output here is for validation, outside timing.
        assert result.node is nodes["orders"]
        assert result.operation == "do_reduce"
        assert graph.executed_op_records[-1]["method_name"] == "do_reduce"
        assert "join" not in nodes["customers"].get_temp_refs()
        refs = {
            name: node.get_ref_name("do_data", lookup=True)
            for name, node in nodes.items()
        }
        loaded_rows = {
            name: connection.sql(f"SELECT count(*) FROM {ref}").fetchone()[0]
            for name, ref in refs.items()
        }
        root_ids = frozenset(
            row[0]
            for row in connection.sql(
                f"SELECT cust_customer_id FROM {refs['customers']}"
            ).fetchall()
        )
        if sample_size is not None:
            assert loaded_rows["customers"] == sample_size
            for parent, child in graph.edges:
                metadata = graph.get_edge_data(parent, child)["keys"]
                parent_ref = parent.get_ref_name("do_data", lookup=True)
                child_ref = child.get_ref_name("do_data", lookup=True)
                orphan_count = connection.sql(
                    f"SELECT count(*) FROM {child_ref} child WHERE NOT EXISTS "
                    f"(SELECT 1 FROM {parent_ref} parent WHERE "
                    f"child.{child.colabbr(metadata['relation_key'])} = "
                    f"parent.{parent.colabbr(metadata['parent_key'])})"
                ).fetchone()[0]
                assert orphan_count == 0
        output = connection.sql(f"SELECT * FROM {result.data}").df()
        return TimedRun(elapsed, root_ids, loaded_rows, output)


def _assert_same_sample_aggregates(full, sampled):
    assert len(full.root_ids) > 100
    assert len(sampled.root_ids) == 100
    assert sampled.root_ids <= full.root_ids
    key = "ord_customer_id"
    expected = full.output.loc[full.output[key].isin(sampled.root_ids)]
    pd.testing.assert_frame_equal(
        sampled.output.sort_values(key).reset_index(drop=True),
        expected.sort_values(key).reset_index(drop=True),
        check_exact=False,
        rtol=1e-9,
        atol=1e-9,
    )
    for name in TABLES:
        assert 0 < sampled.loaded_rows[name] < full.loaded_rows[name]


def test_olist_run_until_full_vs_sample_100(
    olist_sources, capsys, record_testsuite_property
):
    repeats = int(os.getenv("OLIST_BENCHMARK_REPEATS", "3"))
    threads = int(os.getenv("OLIST_BENCHMARK_THREADS", "1"))
    assert repeats > 0 and threads > 0

    # Warm each code path and the filesystem cache before measuring either case.
    warm_full = _run_case(olist_sources, None, threads)
    warm_sample = _run_case(olist_sources, 100, threads)
    _assert_same_sample_aggregates(warm_full, warm_sample)
    capsys.readouterr()

    runs = {None: [], 100: []}
    for iteration in range(repeats):
        for size in (None, 100) if iteration % 2 == 0 else (100, None):
            runs[size].append(_run_case(olist_sources, size, threads))
        _assert_same_sample_aggregates(runs[None][-1], runs[100][-1])
        capsys.readouterr()

    full_seconds = median(run.seconds for run in runs[None])
    sample_seconds = median(run.seconds for run in runs[100])
    report = {
        "data_dir": str(olist_sources["customers"][0].parent),
        "target": "orders.do_reduce",
        "repeats": repeats,
        "duckdb_threads": threads,
        "full_seconds": [run.seconds for run in runs[None]],
        "sample_100_seconds": [run.seconds for run in runs[100]],
        "full_median_seconds": full_seconds,
        "sample_100_median_seconds": sample_seconds,
        "median_seconds_saved": full_seconds - sample_seconds,
        "speedup": full_seconds / sample_seconds,
        "time_reduction_percent": 100 * (1 - sample_seconds / full_seconds),
        "full_loaded_rows": runs[None][-1].loaded_rows,
        "sample_100_loaded_rows": runs[100][-1].loaded_rows,
    }
    record_testsuite_property("olist_run_until_benchmark", json.dumps(report))
    with capsys.disabled():
        print("\nOlist run_until(orders, 'do_reduce'): full data vs sample_size=100")
        print(
            f"Warm-cache medians over {repeats} runs per case; DuckDB threads={threads}"
        )
        print(f"Full:    {full_seconds:.4f}s  runs={report['full_seconds']}")
        print(f"Sample:  {sample_seconds:.4f}s  runs={report['sample_100_seconds']}")
        print(
            f"Saved:   {report['median_seconds_saved']:.4f}s "
            f"({report['time_reduction_percent']:.1f}%); speedup={report['speedup']:.2f}x"
        )
        print(f"{'Loaded table':<16} {'Full':>10} {'Sample 100':>12}")
        for name in TABLES:
            print(
                f"{name:<16} {report['full_loaded_rows'][name]:>10,} {report['sample_100_loaded_rows'][name]:>12,}"
            )
        print(
            "Sampled aggregates match the full-data subset; timings are informational."
        )
