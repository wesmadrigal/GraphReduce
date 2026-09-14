# End-to-End Examples

These examples show complete graph construction, execution, and output
inspection. Start with the smallest example that matches your goal.

## Learn the core workflow

| Example | What it demonstrates | Backend |
| --- | --- | --- |
| [Hello World](hello_world.md) | Parent grain, two child relations, automatic features, and labels | pandas |
| [Preserve child grain](preserve_child_grain.md) | The effect of `reduce=False` | pandas, DuckDB |
| [All tables with labels](all_tables_reduce_with_labels.md) | A deeper graph with complete reduction and ML targets | pandas, DuckDB |

## Compare backends and custom logic

| Example | What it demonstrates |
| --- | --- |
| [Same graph, multiple backends](multi_backend_cust_notifications.md) | Equivalent customer/notification logic across pandas, SQLite, DuckDB, PySpark, Snowflake, and Databricks SQL |
| [Custom pandas graph](custom_pandas_all_nodes.md) | Explicit node operations across all sample tables |
| [Custom DuckDB graph](custom_duckdb_all_nodes.md) | SQL-native custom operations across all sample tables |
| [Custom PySpark graph](custom_pyspark_all_nodes.md) | Distributed dataframe operations across all sample tables |

## Continue into modeling

| Example | Model |
| --- | --- |
| [Predictive AI with XGBoost](predictive_ai_xgboost.md) | `XGBClassifier` on GraphReduce features |
| [Predictive AI with TabPFN](predictive_ai_tabpfn.md) | `TabPFNClassifier` on GraphReduce features |

## RelBench tasks

The [performance comparison](relbench_performance.md) summarizes the available
RelBench runs. Individual task pages provide the corresponding materialization
and evaluation entry points.

Classification:

- [rel-stack: user badges](relbench_rel_stack_badges_duckdb.md)
- [rel-stack: user engagement](relbench_rel_stack_user_engagement_duckdb.md)
- [rel-trial: study outcome](relbench_rel_trial_study_outcome_duckdb.md)
- [rel-hm: user churn](relbench_rel_hm_user_churn_duckdb.md)
- [rel-avito: user clicks](relbench_rel_avito_user_clicks_duckdb.md)
- [rel-avito: user visits](relbench_rel_avito_user_visits_duckdb.md)
- [rel-amazon: user churn](relbench_rel_amazon_user_churn_duckdb.md)
- [rel-amazon: item churn](relbench_rel_amazon_item_churn_duckdb.md)

Regression:

- [rel-stack: post votes](relbench_rel_stack_post_votes_duckdb.md)
- [rel-avito: ad CTR](relbench_rel_avito_ad_ctr_duckdb.md)
- [rel-hm: item sales](relbench_rel_hm_item_sales_duckdb.md)
- [rel-amazon: user LTV](relbench_rel_amazon_user_ltv_duckdb.md)
- [rel-amazon: item LTV](relbench_rel_amazon_item_ltv_duckdb.md)

For smaller SQL building blocks, use the
[SQL backends guide](../tutorial_sql_backends_dynamic_queries.md) before these
full examples.
