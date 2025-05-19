# Abstractions

## Data sources
A source of data, such as a Databricks catalog, a Redshift database and schema,
or a Snowflake database and schema.  Data sources can be one or multiple
schemas / namespaces.  In the Kurve UI they are represented as:

`{provider}://{tenant}/{namespace}?schema={schema}&format={storage_format}`

For an S3 bucket called `bazooka` and a directory called `arms-sales` with
parquet format we would have the following URI:

`s3://bazooka/arms-sales?format=parquet`

For a Snowflake account with a database called `bazooka`,
and a schema called `arms-sales` we would have the following URI:

`snowflake://SOME-SNOWFLAKE-ACCOUNT/bazooka?schema=arms-sales&format=relational`

## Schema graphs
A metadata representation of a data source which represents the
tables or files as nodes and relationships, foreign keys, as edges.

### Nodes
Each node represents a single table or file.

An example might be a table called `bazooka_sales`.

### Edges
Edges define the foreign key between two nodes.

An example would be a table called `bazooka_sales` and another table
called `arms_dealers`.  The two tables, represented as nodes, are connected
by the columns: `bazooka_sales.arms_dealer_id` which points to `arms_dealers.id`.

This relationship is captured in the edge between `bazooka_sales` and `arms_dealers`.

## Compute graphs
A compute graph is a [subgraph](https://en.wikipedia.org/wiki/Subgraph) of a schema
graph and defines computation over the tables and relationships involved.

The top-level compute graph houses the following parameters which are
shared across all nodes:
  - parent node
  - cut date the compute period
  - compute period
  - label period
  - label node
  - label field
  - label operation


### Compute graph execution order of operations
1. load and prefix data
2. annotations
3. filters
4. reduce / aggregate
    1. auto-apply time filters based on cut date, date key, and compute period
    2. apply reduce/aggregation operations
5. compute labels (if any)
    1. auto-apply time filters based on cut date, date key, and label period
6. post-join annotations
    1. annotations to run which require a child/foreign relation to be merged prior to running
7. post-join filters
    1. filters to run which require a child/foreign relation to be merged prior to running

### Compute graph nodes
Compute graph nodes allow us to define the following:

  - the prefix to use for columns
  - the date key to use (if any)
  - the columns to include / exclude
  - annotations
  - filters
  - reduce / aggregation operations
  - labels / target generation operations for ML
  - post join annotations
  - post join filters

#### operations

##### annotations
Annotations are any manipulation of existing columns to create new ones such as
applying a length function to a string column, extracting json to add new columns,
or computing some function over a numerical column in a non-aggregated way.

In plain english we're widening the table by adding new columns.  Let's say
we have a table with 2 columns: `id INT` and `name INT`.  Kurve takes
care of compiling the end to end SQL at runtime we can simply add the select
and not think about the rest of the query:
```sql
select *, length(name) as name_length
```
At runtime this will compile into a full query like:
```sql
create some_stage_name as
select *, length(name) as name_length
from table
```
We added the `name_length` column which is derived from the `name` column, so now we
have 3 columns instead of 2.

##### filters
Filters are any filter applied to the node.  Since order of operation matters, they
can also be combined with annotations.  They can be written in isolation without
the rest of SQL grammar:
```sql
where lower(name) not like '%test%'
```
At runtime this will compile into a full query like:
```sql
select *
from some_stage_name
where lower(name) like '%test%'
```

##### reduce / aggregate
Aggregations are any `group by` applied to the table prior
to joining it to it's parent.  Kurve uses completely automated
reduce operations powered by [graphreduce](https://github.com/wesmadrigal/graphreduce)
but also allows full customizability.

As with filters and annotations, reduce operations can be written
in isolation without the where clauses and annotations as follows:
```sql
select customer_id,
count(*) as num_orders
from 'orders.csv'
group by customer_id
```

##### labels / target variables
Labels are the target variable for machine learning problems.
Based on the parameterized cut date and the label period the label
node will get filtered to compute the label.

For a label period of 90 days and a cut date of Jan 1, 2025 we would
get the following auto-generated SQL for the `orders.csv`:
```sql
select customer_id,
count(*) as num_orders_label
from 'orders.csv'
where ts > cut_date
and ts < cut_date + INTERVAL '90 day'
group by customer_id
```

This tells us if a customer had an order in the next 90 days.

##### post-join annotations (advanced)
Same as annotations above but applied <i>after</i> the specified
relations are merged.  An example might be a `DATEDIFF` applied
to two columns: one from the parent table and the other from the child table.

Check out the [advanced examples](examples.md#post-join-annotation) for more detail.

##### post-join filters (advanced)
Same as filters above but applied <i>after</i> the specified
relations are merged.  An example might be filtering after a `DATEDIFF`
has been applied to two columns from different table after
joining them together.

Check out the [advanced examples](examples.md#post-join-filters) for more detail.

### Compute graph edges
Relationships between table.  The most noteworthy attribute about edges, beyond
the relationship they encode, is whether or not to <b>reduce</b> the edge or not.

For one to many relationships there are times where it is desirable to preserve
all rows in the child relation and still join, knowing it will cause duplication
of the parent relationship.  In these cases we can specify `reduce = False`.


## Decomposing data problems spanning multiple tables with graphs

### Thinking at the node-level
Kurve is built on the realization that most data analytics and AI data discovery
and data preparation involves multiple tables.  The more tables involved the
higher cognitive load due to sheer table counts, but also due to what has traditionally
been spaghetti code or pipeline jungles required to munge all of this data.

By leveraging Kurve's metadata graphs and compute subraphs we can visually see
large subgraphs of multi-table data integration one table / node at a time.
This allows the developer to focus on individual components, nodes, of a larger
compute operation without thinking about every aspect simultaneously.

Kurve abstracts and automates away a lot of the menial boilerplate that also adds to complexity of pipelines and queries, such as prefixing columns in tables, filtering on dates, joining tables, and aggregating / reducing one to many relations prior to a join.  Kurve provides completely automated approaches to this but also allows full cutomization when desired.  Think of it like autonomous driving for data: use the autonamy where you want and take control where you feel you need to.

## How compute graphs get executed

### Compute pushdown
As of writing, Kurve supports connectors to the following.

- amazon s3
- snowflake
- databricks
- amazon Redshift
- unity catalog
- polaris catalog
- amazon RDS


### Depth-first traversal
Kurve uses [depth first traversal](https://en.wikipedia.org/wiki/Depth-first_search) during compute graph
execution, starting at the bottom and recursively working back up to the top.

### Dynamic SQL generation and prefixing
Kurve leverages [sqlglot](https://github.com/tobymao/sqlglot) for SQL parsing
and helping with SQL prefixing based on the specified prefix.

All table in a compute graph must have a prefix and the prefixes must be
unique.  For example a customers table is typically prefixed with something
like `'cust'`.  This allows us to know which table(s) the data originated
from after it has been integrated.

### Temporary references and views
Behind the scenes when a compute graph gets executed Kurve is creating and referencing
incremental temporary references to incremental transformations of data throughout
the execution.  For a 2 node compute graph with no customized operations the
execution flow would be as follows:

1. load and prefix data and create reference for table 1
2. load and prefix data and create reference for table 2
3. aggregate and create reference of aggregated data for table 2
4. join aggregated data from table 2 to table 1 and create updated reference for table 1

This execution flow would have at least 4 temporary incremental references created, which
capture every table-level and join-level maninpuluation to the data.
