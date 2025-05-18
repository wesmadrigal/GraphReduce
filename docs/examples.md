## Relational schema inference basics
This tutorial covers the basic features of extracting relational metadata
from flat files.

1. Log in to [kurve demo](https://demo.kurve.ai)
2. Look under <b>Sample data sources</b> for `/usr/local/lake/cust_data` and click <b>Create Graph</b>
3. In a few seconds the page should refresh view the graph with <b>View Graph</b>
4. View all of the foreign key relationships between the 6 tables:
    1. `cust.csv` connects with `orders.csv` on `id -> customer_id`
    2. `cust.csv` connects with `notifications.csv` on `id -> customer_id`
    3. `orders.csv` connects with `order_products.csv` on `id -> order_id`
    4. `notifications.csv` connects with `notification_interactions.csv` on `id -> notification_id`
    5. `notification_interactions.csv` connects with `notification_interaction_types.csv` on `interaction_type_id -> id`

In this example Kurve points at the 6 csv files, runs an algorithm, and extracts all of the relational metadata.
Next we'll look at how to run computation on these relational metadata graphs.


## Orienting schema graphs around particular nodes
Most analytics and AI problems are oriented around a particular [dimension](https://en.wikipedia.org/wiki/Dimension_(data_warehouse)).  With a graph structure we can easily manipuate the schema graph
around the dimension of interest.  For this example we'll orient the `/usr/local/lake/cust_data`
sample schema around the `cust.csv`.  To do this we'll assign the `cust.csv` as the <b>parent node</b>:

1. Under <b>Actions</b> in the schema graph viewer click <b>Assign Parent Node</b> and select `cust.csv` with `depth=1`.
2. You should have a subgraph of 3 tables now: `cust.csv`, `orders.csv`, and `notifications.csv`
3. Now that we've oriented the dataset around the `cust.csv` dimension let's get to compute / data integration.
![ex1](images/ex1_step1.jpg)

## Compute graph basics
1. Visit the schema graph for `/usr/local/lake/cust_data`.
2. Click <b>Actions</b>, <b>Assign Parent Node</b>, select `cust.csv` with `depth = 1`.
3. Click <b>Actions</b>, <b>Compute Graph</b> and plug in the following values:
    1. Name: kurve cust demo
    2. Parent Node: cust.csv
    3. Depth limit: 1
    4. Compute period in days: 365
    5. Cut date: 05/01/2023
    6. Label period in days: 90
    7. Label node: notifications.csv
    8. Label field: id
    9. Label operation: count
![ex2](images/ex2_step1.jpg)
4. Notice the color coating of the parent node, cust.csv, and the label node, notifications.csv.
5. Let's make sure our parameters are correct: click <b>Actions</b>, <b>Show compute graph details</b>.  Here is what it should look like:
![ex2s2](images/ex2_step2.jpg)
6. If all of that looks good now let's execute the compute graph:
    1. click <b>Actions</b>
    2. click <b>Execute compute graph</b>
    3. navigate home
    4. you should now see a data source under <b>My data sources</b>, click it and find the `kurve_cust_demo_train` table or whatever name you used and click on the table
    5. notice all of the columns but we should only have 4 rows, this is all 3 tables aggregated and integrated to the `cust.csv` dimension with point-in-time correctness based on the above parameters.
![ex2s3](images/ex2_step3.jpg)



## Node-level operations customization basic
In the compute graph created in the above example we leaned into Kurve's automation, but let's
customize some things and see the effect.

1. Visit the compute graph under `/usr/local/lake/cust_data`

2. Click the parent node `cust.csv`

3. Click <b>Edit Node</b> and under <b>Annotations / special selects</b> enter the following:

    ```
    select *, length(name) as name_length
    ```

4. Under <b>Filters</b> enter the following:

    ```
    where name_length < 4
    ```
![ex3s1](images/ex3_step1.jpg)
5. Execute the compute graph
6. Navigate home and click on the compute graph output table under <b>My data sources</b> and you shuld now see only 3 rows.
![ex3s2](images/ex3_step2.jpg)

In summary, we annotated the `cust.csv` node with a new column called `name_length` and then filtered the dataset to only contain rows where `name_length < 4`, which filtered one row.  This highlights how we can customize compute graphs in a basic way.  The next example will do this in a more advanced way.


## Node-level operations customization advanced
Continuing with the same schema and compute graph from the prior 2 examples let's say we need to customize some of the aggregations on the `orders.csv` table and perform some filters on that table, as well.

1. Visit the compute graph from the prior example
2. Click on the `orders.csv` node
3. Click <b>Edit Node</b> and under <b>Filters</b> enter the following:

    ```
    where total < 2000
    ```
