# Custom PySpark Graph: All `cust_data` Nodes

This example shows a custom PySpark GraphReduce graph that uses all tables in
`tests/data/cust_data`:

* `cust`
* `orders`
* `order_products`
* `notifications`
* `notification_interactions`
* `notification_interaction_types`

It includes custom definitions across `do_annotate`, `do_filters`,
`do_normalize`, `do_reduce`, and parent post-join logic.

Key behaviors:

* Customer name-length annotation via `length(coalesce(name, ''))`
* Order-level amount casting and order spend rollups
* Order-product distinct product count rollups
* Notification interaction engagement rollups driven by interaction-type signals
* Parent-level post-join activity features

Full example code lives in:

* `examples/custom_pyspark_all_nodes.py`

Minimal usage:

```python
from pyspark.sql import SparkSession

from examples.custom_pyspark_all_nodes import build_custom_pyspark_graph

spark = SparkSession.builder.appName("graphreduce-custom-all-nodes").getOrCreate()
gr = build_custom_pyspark_graph(spark)
gr.do_transformations()
print(gr.parent_node.df.columns)
```

## Run Interactive

<div class="modal-runner" data-modal-runner data-api-base="https://runner.13.218.155.128.sslip.io" data-example="custom_pyspark_all_nodes">
  <div class="modal-runner-controls">
    <input class="modal-runner-input" data-api-input value="https://runner.13.218.155.128.sslip.io" />
    <button data-save-api-btn>Save API URL</button>
    <button data-run-btn>Run custom pyspark all-nodes</button>
  </div>
  <div class="modal-runner-status" data-status>Idle</div>
  <pre class="modal-runner-log" data-log></pre>
</div>
