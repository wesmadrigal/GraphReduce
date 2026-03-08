#!/usr/bin/env python
"""Custom PySpark GraphReduce example using all cust_data nodes.

Usage:
  python examples/custom_pyspark_all_nodes.py
"""

from __future__ import annotations

import datetime
import os

try:
    from pyspark.sql import SparkSession, functions as F
except Exception as exc:  # pragma: no cover
    print(f"pyspark not available: {exc}", flush=True)
    raise SystemExit(0)

from graphreduce.enum import ComputeLayerEnum
from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import GraphReduceNode

DATA_PATH = "tests/data/cust_data"


class CustNode(GraphReduceNode):
    def do_annotate(self):
        self.df = self.df.withColumn(
            self.colabbr("name_length"),
            F.length(F.coalesce(F.col(self.colabbr("name")), F.lit(""))),
        )
        return self.df

    def do_filters(self):
        self.df = self.df.filter(F.col(self.colabbr("id")).isNotNull())
        return self.df

    def do_normalize(self):
        self.df = self.df.withColumn(
            self.colabbr("name"),
            F.lower(F.trim(F.coalesce(F.col(self.colabbr("name")), F.lit("")))),
        )
        return self.df

    def do_reduce(self, reduce_key):
        return self.df

    def do_labels(self, reduce_key):
        return self.df

    def do_post_join_annotate(self):
        ord_ct = F.coalesce(F.col("ord_num_orders"), F.lit(0))
        not_ct = F.coalesce(F.col("not_num_notifications"), F.lit(0))
        engaged = F.coalesce(F.col("not_num_engaged_interactions"), F.lit(0))

        self.df = self.df.withColumn("cust_total_events", ord_ct + not_ct + engaged).withColumn(
            "cust_activity_tier",
            F.when(F.col("cust_total_events") >= 5, F.lit("high"))
            .when(F.col("cust_total_events") >= 2, F.lit("medium"))
            .otherwise(F.lit("low")),
        )
        return self.df

    def do_post_join_filters(self):
        self.df = self.df.filter(F.col("cust_total_events") >= 0)
        return self.df


class OrderNode(GraphReduceNode):
    def do_annotate(self):
        self.df = self.df.withColumn(self.colabbr("amount_dbl"), F.col(self.colabbr("amount")).cast("double"))
        return self.df

    def do_filters(self):
        self.df = self.df.filter(F.col(self.colabbr("ts")) >= F.lit("2022-01-01"))
        return self.df

    def do_normalize(self):
        self.df = self.df.withColumn(self.colabbr("amount_dbl"), F.coalesce(F.col(self.colabbr("amount_dbl")), F.lit(0.0)))
        return self.df

    def do_reduce(self, reduce_key):
        return (
            self.prep_for_features()
            .groupBy(self.colabbr(reduce_key))
            .agg(
                F.countDistinct(F.col(self.colabbr(self.pk))).alias(self.colabbr("num_orders")),
                F.sum(F.col(self.colabbr("amount_dbl"))).alias(self.colabbr("sum_amount")),
                F.avg(F.col(self.colabbr("amount_dbl"))).alias(self.colabbr("avg_amount")),
            )
        )

    def do_labels(self, reduce_key):
        return self.df

    def do_post_join_annotate(self):
        return self.df

    def do_post_join_filters(self):
        return self.df


class OrderProductsNode(GraphReduceNode):
    def do_annotate(self):
        self.df = self.df.withColumn(
            self.colabbr("product_id_int"),
            F.col(self.colabbr("product_id")).cast("int"),
        )
        return self.df

    def do_filters(self):
        self.df = self.df.filter(F.col(self.colabbr("product_id_int")).isNotNull())
        return self.df

    def do_normalize(self):
        return self.df

    def do_reduce(self, reduce_key):
        return (
            self.prep_for_features()
            .groupBy(self.colabbr(reduce_key))
            .agg(
                F.count(F.col(self.colabbr(self.pk))).alias(self.colabbr("num_order_products")),
                F.countDistinct(F.col(self.colabbr("product_id_int"))).alias(self.colabbr("num_distinct_products")),
            )
        )

    def do_labels(self, reduce_key):
        return self.df

    def do_post_join_annotate(self):
        return self.df

    def do_post_join_filters(self):
        return self.df


class NotificationNode(GraphReduceNode):
    def do_annotate(self):
        self.df = self.df.withColumn(self.colabbr("ts_month"), F.date_format(F.col(self.colabbr("ts")), "MM"))
        return self.df

    def do_filters(self):
        self.df = self.df.filter(F.col(self.colabbr("ts")) >= F.lit("2022-01-01"))
        return self.df

    def do_normalize(self):
        return self.df

    def do_reduce(self, reduce_key):
        return (
            self.prep_for_features()
            .groupBy(self.colabbr(reduce_key))
            .agg(
                F.countDistinct(F.col(self.colabbr(self.pk))).alias(self.colabbr("num_notifications")),
                F.max(F.col(self.colabbr("ts"))).alias(self.colabbr("max_notification_ts")),
                F.sum(F.coalesce(F.col(self.colabbr("ni_num_interactions")), F.lit(0))).alias(
                    self.colabbr("num_interactions")
                ),
                F.sum(F.coalesce(F.col(self.colabbr("ni_num_engaged_interactions")), F.lit(0))).alias(
                    self.colabbr("num_engaged_interactions")
                ),
            )
        )

    def do_labels(self, reduce_key):
        return self.df

    def do_post_join_annotate(self):
        return self.df

    def do_post_join_filters(self):
        return self.df


class NotificationInteractionsNode(GraphReduceNode):
    def do_annotate(self):
        self.df = self.df.withColumn(
            self.colabbr("is_engaged"),
            F.when(F.lower(F.col(self.colabbr("nit_name"))).isin("clicked", "dismissed"), F.lit(1)).otherwise(F.lit(0)),
        )
        return self.df

    def do_filters(self):
        self.df = self.df.filter(F.col(self.colabbr("ts")) >= F.lit("2022-01-01"))
        return self.df

    def do_normalize(self):
        return self.df

    def do_reduce(self, reduce_key):
        return (
            self.prep_for_features()
            .groupBy(self.colabbr(reduce_key))
            .agg(
                F.count(F.col(self.colabbr(self.pk))).alias(self.colabbr("num_interactions")),
                F.countDistinct(F.col(self.colabbr("interaction_type_id"))).alias(self.colabbr("num_interaction_types")),
                F.sum(F.col(self.colabbr("is_engaged"))).alias(self.colabbr("num_engaged_interactions")),
            )
        )

    def do_labels(self, reduce_key):
        return self.df

    def do_post_join_annotate(self):
        return self.df

    def do_post_join_filters(self):
        return self.df


class NotificationInteractionTypeNode(GraphReduceNode):
    def do_annotate(self):
        self.df = self.df.withColumn(self.colabbr("name"), F.lower(F.trim(F.col(self.colabbr("name")))))
        return self.df

    def do_filters(self):
        self.df = self.df.filter(F.col(self.colabbr("id")).isNotNull())
        return self.df

    def do_normalize(self):
        return self.df

    def do_reduce(self, reduce_key):
        return (
            self.prep_for_features()
            .groupBy(self.colabbr(reduce_key))
            .agg(
                F.first(F.col(self.colabbr("name")), ignorenulls=True).alias(self.colabbr("name")),
                F.max(
                    F.when(F.col(self.colabbr("name")).contains("view"), F.lit(1)).otherwise(F.lit(0))
                ).alias(self.colabbr("is_view_event")),
            )
        )

    def do_labels(self, reduce_key):
        return self.df

    def do_post_join_annotate(self):
        return self.df

    def do_post_join_filters(self):
        return self.df


def build_custom_pyspark_graph(spark: SparkSession) -> GraphReduce:
    for table in [
        "cust",
        "orders",
        "order_products",
        "notifications",
        "notification_interactions",
        "notification_interaction_types",
    ]:
        spark.read.option("header", True).option("inferSchema", True).csv(
            os.path.join(DATA_PATH, f"{table}.csv")
        ).createOrReplaceTempView(table)

    cust = CustNode(
        fpath="cust",
        fmt="sql",
        prefix="cust",
        pk="id",
        compute_layer=ComputeLayerEnum.spark,
        columns=["id", "name"],
        spark_sqlctx=spark,
    )
    orders = OrderNode(
        fpath="orders",
        fmt="sql",
        prefix="ord",
        pk="id",
        date_key="ts",
        compute_layer=ComputeLayerEnum.spark,
        columns=["id", "customer_id", "ts", "amount"],
        spark_sqlctx=spark,
    )
    order_products = OrderProductsNode(
        fpath="order_products",
        fmt="sql",
        prefix="op",
        pk="id",
        compute_layer=ComputeLayerEnum.spark,
        columns=["id", "order_id", "product_id"],
        spark_sqlctx=spark,
    )
    notifications = NotificationNode(
        fpath="notifications",
        fmt="sql",
        prefix="not",
        pk="id",
        date_key="ts",
        compute_layer=ComputeLayerEnum.spark,
        columns=["id", "customer_id", "ts"],
        spark_sqlctx=spark,
    )
    notification_interactions = NotificationInteractionsNode(
        fpath="notification_interactions",
        fmt="sql",
        prefix="ni",
        pk="id",
        date_key="ts",
        compute_layer=ComputeLayerEnum.spark,
        columns=["id", "notification_id", "interaction_type_id", "ts"],
        spark_sqlctx=spark,
    )
    interaction_types = NotificationInteractionTypeNode(
        fpath="notification_interaction_types",
        fmt="sql",
        prefix="nit",
        pk="id",
        compute_layer=ComputeLayerEnum.spark,
        columns=["id", "name"],
        spark_sqlctx=spark,
    )

    gr = GraphReduce(
        name="custom_pyspark_all_nodes",
        parent_node=cust,
        compute_layer=ComputeLayerEnum.spark,
        spark_sqlctx=spark,
        cut_date=datetime.datetime(2023, 6, 30),
    )

    for node in [cust, orders, order_products, notifications, notification_interactions, interaction_types]:
        gr.add_node(node)

    gr.add_entity_edge(cust, orders, parent_key="id", relation_key="customer_id", reduce=True)
    gr.add_entity_edge(orders, order_products, parent_key="id", relation_key="order_id", reduce=True)
    gr.add_entity_edge(cust, notifications, parent_key="id", relation_key="customer_id", reduce=True)
    gr.add_entity_edge(notifications, notification_interactions, parent_key="id", relation_key="notification_id", reduce=True)
    gr.add_entity_edge(
        notification_interactions,
        interaction_types,
        parent_key="interaction_type_id",
        relation_key="id",
        reduce=True,
    )

    return gr


def run_custom_pyspark_all_nodes(spark: SparkSession | None = None):
    local_spark = spark or SparkSession.builder.appName("graphreduce-custom-all-nodes").getOrCreate()
    gr = build_custom_pyspark_graph(local_spark)
    gr.do_transformations()
    return gr.parent_node.df


def main() -> None:
    df = run_custom_pyspark_all_nodes()
    print("rows:", df.count(), flush=True)
    print("columns:", len(df.columns), flush=True)
    print("column_names:", df.columns, flush=True)


if __name__ == "__main__":
    main()
