# Partial execution and relational samples

Use `run_until` to inspect an intermediate node without running the remaining
pipeline. It works with SQL nodes and dataframe nodes:

```python
result = graph.run_until(node_2, "do_annotate")
```

The stop is inclusive: node 2's annotation completes, and its filters do not run.
The node can also be identified by its unique prefix:

```python
result = graph.run_until("orders", "do_reduce")
```

A reduction includes any automatic features. Descendant reductions and joins
scheduled earlier in the pipeline complete first. The target's reduced output
has not yet been joined into its parent. Inspect it through `result.data`:

```python
# pandas, Dask, Spark, or Daft: the backend's dataframe
reduced = result.data

# SQL: a table/view reference, queryable with the node's client
reduced = result.node.execute_query(f"SELECT * FROM {result.data}")
```

For dataframe reductions, `result.data` contains the reduction while `node.df`
retains the input used by later label computations. For SQL reductions, the
node's current reference also points at the reduced output.

`ExecutionResult` exposes `node`, `operation`, `data`, and `edge`. The edge is a
`(parent_node, relation_node)` pair for operations performed during edge traversal.
Results refer to live backend data rather than independent copies.

## Operation boundaries

Supported targets are `do_data`, `do_annotate`, `do_filters`, `do_normalize`,
`do_reduce`, `do_labels`, `do_post_join_annotate`, and `do_post_join_filters`.
The dataframe runner also supports `do_post_join_reduce` on edges configured
with `reduce_after_join=True`.

Execution follows the existing pipeline order. For example, the dataframe
pipeline loads every node before starting annotations; the SQL pipeline prepares
each node before moving to the next. Earlier operations on other nodes may run.
A repeated post-join operation stops at its first occurrence. Existing
`auto_feature_hops_back` limits still determine which edges are reduced.

Invalid node names, unsupported operations, and reductions absent from the
configured traversal fail before loading data. A conditional operation such as
label generation must actually execute to be reached. SQL date nodes are
managed by the date-propagation pipeline and cannot be stop targets.

Use a fresh graph and fresh nodes for each independent execution. These methods
do not reset prior transformations or resume a previous partial run.

The general entry point selects the SQL or dataframe runner automatically:

```python
result = graph.run(until_node=node_2, until_operation="do_annotate")
# Or, on a fresh graph, execute the entire pipeline:
result = graph.run()  # result.operation == "complete"; data is the parent output
```

## Load a related subset of the graph

```python
graph.load_data(sample_size=100)
```

This runs only `do_data` and leaves each node ready for inspection. It limits
the root to at most 100 rows, then keeps all matching rows at each subsequent
relationship:

```text
customers: at most 100 source rows
    └── orders: customer_id matches sampled customers.id
          └── items: order_id matches retained orders.id
```

Combine sampling with a stop point on a fresh graph:

```python
result = graph.run_until(node_3, "do_reduce", sample_size=100)
```

Or configure the default when constructing the graph:

```python
graph = GraphReduce(
    parent_node=customers,
    compute_layer=ComputeLayerEnum.duckdb,
    sql_client=connection,
    sample_size=100,
)
# Add the graph's nodes and relationships as usual.
graph.run()
```

The constructor default also applies to `do_transformations()` and
`do_transformations_sql()`. Passing an integer to `run`, `run_until`, or
`load_data` overrides that default for the call. `None` uses the configured
default; a default of `None` loads the full data. Zero produces an empty sample.

Sampling uses each edge's `parent_key` and `relation_key`, including composite
keys and already-prefixed keys. It does not assume that all relationships use
the node's declared primary key. Only the root has a row limit; children retain
all matching rows, including duplicates. Null keys do not match. Empty parent
key sets yield empty descendants. Keys are captured immediately after loading,
before annotations, filters, reductions, or date propagation change the data.

Nodes shared by multiple parents retain rows matching **any** sampled parent
path. Parents load before their descendants regardless of edge insertion order.
Sampling traverses every depth, independently of the reduction hop limit. It
requires an acyclic graph with every ordinary node reachable from the root;
cycles and disconnected or upstream nodes are rejected before loading.

## Backend behavior

SQL sampling wraps each node's normal `do_data` query, preserving its selections
and predicates. The root adds `LIMIT`; descendants use `EXISTS` against parent
load references. The database handles key matching without transferring key
lists into Python. SQL behavior is tested with DuckDB and SQLite.

For pandas, sampling filters the dataframe after `do_data`, so reading a file
can still load the full input into memory. Dask filters child partitions lazily
and collects only the sampled parent key sets for membership checks. Relational
sampling currently supports SQL, pandas, and Dask; other dataframe backends
raise `NotImplementedError` when a sample is requested. Partial execution
without sampling uses their existing runners.

The root limit is a row limit, not a random or statistically representative
sample. SQL row selection is unspecified without an ordered source query.
A root with many descendants can still produce a large sample.

Frozen SQL plans retain the original `do_data` operations, not sampled IDs or
temporary sample references. Set `sample_size` on the replay graph if the replay
should also use a relational sample.

## Compare execution times on Olist

The opt-in pytest at `tests/test_olist_run_until_benchmark.py` builds a five-node
DuckDB graph: customers → orders → items, payments, and reviews. It compares
`run_until(orders, "do_reduce")` with the same call using `sample_size=100`.
Run it from the repository root with the original Olist CSVs available locally:

```bash
RUN_OLIST_BENCHMARKS=1 OLIST_DATA_DIR=/usr/local/lake/olist \
  venv/bin/python -m pytest -q -s tests/test_olist_run_until_benchmark.py
```

Each case has one warm-up and three measured runs, alternating case order with
fresh graphs and DuckDB connections. The timer includes CSV loading and graph
execution; it excludes graph construction, output fetching, and validation.
The test prints median times, time saved, speedup, and loaded row counts. It
checks that sampled aggregates match the full-data subset and that sampled
descendants have matching parents. Timing comparisons are informational.

Set `OLIST_BENCHMARK_REPEATS` or `OLIST_BENCHMARK_THREADS` to override the defaults
of three measured runs and one DuckDB thread. Add `--junitxml=olist-benchmark.xml`
to save the measurements as a JSON-valued test-suite property. The benchmark
skips unless enabled and also skips when the local CSVs are unavailable.
