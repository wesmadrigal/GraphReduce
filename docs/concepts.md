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

#### annotations
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

#### filters
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

#### reduce / aggregate
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

#### labels / target variables

#### post-join annotations

#### post-join filters


### Compute graph edges


## Compute graph decomposition

### Thinking at the node-level

### Thinking at the neighborhood-level

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

### Dynamic SQL generation and prefixing

### Temporary references and views
