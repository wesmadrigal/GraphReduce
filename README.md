# GraphReduce

GraphReduce turns related tables into a single, model-ready feature table.
You describe tables as nodes, foreign-key relationships as edges, and the
entity you want to model as the parent node. GraphReduce then filters data to
the correct point in time, aggregates child rows, and joins the results back to
the parent.

It is designed for relational machine-learning workloads where correctness is
as important as scale:

- **Known output grain.** Reduced relations produce one row per parent key.
- **Point-in-time features.** Historical features and future labels use
  separate windows around a cutoff.
- **Traceable columns.** Every node has a unique prefix that records column
  lineage after joins.
- **Automatic or custom features.** Start with schema-aware aggregation, then
  replace any node operation with application-specific logic.
- **Multiple execution engines.** Use dataframe backends or compile the graph
  into SQL.

[Documentation](https://wesmadrigal.github.io/GraphReduce/) ·
[Quickstart](https://wesmadrigal.github.io/GraphReduce/tutorial_index/) ·
[Examples](https://wesmadrigal.github.io/GraphReduce/end_to_end_examples/)

![GraphReduce modeling and feature rollup](./docs/graphreduce_modeling_overview.svg)

## Mental model

Suppose the source data contains customers and orders:

```text
customers (one row per customer)
    id  name

orders (many rows per customer)
    id  customer_id  ts  amount
```

Model the relationship as:

```text
customers.id  1 ───── *  orders.customer_id
     parent                 relation
```

With `customers` as the parent and `reduce=True` on the edge, GraphReduce:

1. loads and prefixes both tables;
2. keeps order rows inside the historical feature window;
3. aggregates orders by `customer_id`;
4. left-joins those features to customers; and
5. optionally computes labels from a separate future window.

The result remains at customer grain: one row per customer.

## Installation

Install the dataframe-oriented core:

```bash
pip install graphreduce
```

Install only the optional integrations you need:

```bash
pip install "graphreduce[duckdb]"
pip install "graphreduce[spark]"
pip install "graphreduce[daft]"
pip install "graphreduce[trino]"
pip install "graphreduce[relbench]"
```

`graphreduce[all]` installs all optional integrations. For local development:

```bash
git clone https://github.com/wesmadrigal/graphreduce.git
cd graphreduce
pip install -e ".[dev]"
```

## Quickstart

The example below expects two local CSV files with the columns shown in the
mental model above.

```python
import datetime as dt

from graphreduce.enum import ComputeLayerEnum, PeriodUnit
from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import DynamicNode

customers = DynamicNode(
    fpath="customers.csv",
    fmt="csv",
    pk="id",
    prefix="cust",
    compute_layer=ComputeLayerEnum.pandas,
)

orders = DynamicNode(
    fpath="orders.csv",
    fmt="csv",
    pk="id",
    prefix="ord",
    date_key="ts",
    compute_layer=ComputeLayerEnum.pandas,
)

graph = GraphReduce(
    name="customer_features",
    parent_node=customers,
    compute_layer=ComputeLayerEnum.pandas,
    cut_date=dt.datetime(2024, 1, 1),
    compute_period_val=365,
    compute_period_unit=PeriodUnit.day,
    auto_features=True,
    label_node=orders,
    label_field="id",
    label_operation="count",
    label_period_val=30,
    label_period_unit=PeriodUnit.day,
)

graph.add_node(customers)
graph.add_node(orders)
graph.add_entity_edge(
    parent_node=customers,
    relation_node=orders,
    parent_key="id",
    relation_key="customer_id",
    reduce=True,
)

graph.do_transformations()
features = graph.parent_node.df
print(features.head())
```

Important details:

- `parent_node=customers` sets the output grain.
- `prefix` values must be unique across the graph.
- `date_key="ts"` makes the orders node eligible for time filtering.
- `compute_period_val=365` uses non-null order timestamps strictly before the
  cutoff.
- `label_period_val=30` uses label rows strictly after the cutoff.
- `reduce=True` aggregates orders before joining them to customers.
- The materialized dataframe is available at `graph.parent_node.df`.

Because labels are joined from an aggregate relation, a parent with no event in
the label window receives a null label. Convert that null to zero only when “no
event” has that meaning for your target.

For a runnable version backed by repository sample data, see
[Hello World](https://wesmadrigal.github.io/GraphReduce/end_to_end_examples/hello_world/).

## Choose an execution path

| Need | Node type | Entry point |
| --- | --- | --- |
| Automatic dataframe features | `DynamicNode` | `do_transformations()` |
| Custom dataframe operations | `GraphReduceNode` subclass | `do_transformations()` |
| SQL compilation and execution | `SQLNode` or backend subclass | `do_transformations_sql()` |

Available compute layers are defined by `ComputeLayerEnum`. The most developed
paths in the current documentation are pandas, Dask, Spark, Daft, DuckDB,
SQLite, Athena, Redshift, Snowflake, Databricks SQL, PostgreSQL, MySQL, and
Trino. Backend-specific clients and optional dependencies still need to be
configured by the application.

## Where to go next

- [Fundamentals](https://wesmadrigal.github.io/GraphReduce/fundamentals/):
  execution order, cardinality, and time semantics.
- [Abstractions](https://wesmadrigal.github.io/GraphReduce/abstractions/): the
  responsibilities of graphs, nodes, and edges.
- [Configuration](https://wesmadrigal.github.io/GraphReduce/configuration/):
  common graph, node, edge, and execution settings.
- [Quickstart](https://wesmadrigal.github.io/GraphReduce/tutorial_index/): a
  runnable two-table walkthrough using repository sample data.
- [SQL auto-feature families](https://wesmadrigal.github.io/GraphReduce/sql_auto_feature_families/):
  schema-aware, temporal, conditional, sequence, episode, semantic, and
  context features.
- [End-to-end examples](https://wesmadrigal.github.io/GraphReduce/end_to_end_examples/):
  complete multi-table and multi-backend workflows.

## Citation

If GraphReduce contributes to your research or production work, please cite
the project:

```bibtex
@software{madrigal_graphreduce_2026,
  author  = {Madrigal, Wes},
  title   = {GraphReduce: Relational Feature Engineering for Tabular Machine Learning},
  year    = {2026},
  url     = {https://github.com/wesmadrigal/GraphReduce}
}
```

For reproducible research, cite the specific GitHub release or commit used.

## Project status

GraphReduce is under active development. The public surface is broad, and
backend behavior is not perfectly uniform. Test a small representative graph
on your chosen backend before scaling a production job. Please report defects
and documentation gaps in the
[issue tracker](https://github.com/wesmadrigal/graphreduce/issues).

## License

GraphReduce is distributed under the MIT License. See
[LICENSE](./LICENSE) for the complete terms.
