#!/usr/bin/env python
"""Run multi-backend pyspark cust+notifications example."""

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


def _is_interactive_mode() -> bool:
    return os.getenv("GRAPHREDUCE_INTERACTIVE", "0").strip().lower() in {"1", "true", "yes"}


class CustNode(GraphReduceNode):
    def do_annotate(self):
        self.df = self.df.withColumn(self.colabbr("name_length"), F.length(F.col(self.colabbr("name"))))
        return self.df

    def do_filters(self):
        self.df = self.df.filter(F.col(self.colabbr("id")) < 3)
        return self.df

    def do_normalize(self):
        pass

    def do_reduce(self, reduce_key):
        pass

    def do_labels(self, reduce_key):
        pass

    def do_post_join_annotate(self):
        pass

    def do_post_join_filters(self):
        pass


class NotificationNode(GraphReduceNode):
    def do_annotate(self):
        self.df = self.df.withColumn(self.colabbr("ts_month"), F.date_format(F.col(self.colabbr("ts")), "MM"))
        return self.df

    def do_filters(self):
        self.df = self.df.filter(F.col(self.colabbr("ts")) > F.lit("2022-06-01"))
        return self.df

    def do_normalize(self):
        pass

    def do_reduce(self, reduce_key):
        return (
            self.prep_for_features()
            .groupBy(self.colabbr(reduce_key))
            .agg(F.count(F.col(self.colabbr(self.pk))).alias(self.colabbr("num_notifications")))
        )

    def do_labels(self, reduce_key):
        pass

    def do_post_join_annotate(self):
        pass

    def do_post_join_filters(self):
        pass


def main() -> None:
    print("Running pyspark backend...", flush=True)
    spark = SparkSession.builder.appName("graphreduce-cust-notif").getOrCreate()
    spark.read.option("header", True).option("inferSchema", True).csv(
        os.path.join(DATA_PATH, "cust.csv")
    ).createOrReplaceTempView("cust")
    spark.read.option("header", True).option("inferSchema", True).csv(
        os.path.join(DATA_PATH, "notifications.csv")
    ).createOrReplaceTempView("notifications")

    cust = CustNode(
        fpath="cust",
        fmt="sql",
        prefix="cust",
        pk="id",
        compute_layer=ComputeLayerEnum.spark,
        columns=["id", "name"],
        spark_sqlctx=spark,
    )
    notif = NotificationNode(
        fpath="notifications",
        fmt="sql",
        prefix="not",
        pk="id",
        date_key="ts",
        compute_layer=ComputeLayerEnum.spark,
        columns=["id", "customer_id", "ts"],
        spark_sqlctx=spark,
    )

    gr = GraphReduce(
        name="cust_notif_spark",
        parent_node=cust,
        compute_layer=ComputeLayerEnum.spark,
        spark_sqlctx=spark,
        cut_date=datetime.datetime(2023, 6, 30),
    )
    gr.add_node(cust)
    gr.add_node(notif)
    gr.add_entity_edge(cust, notif, parent_key="id", relation_key="customer_id", reduce=True)
    gr.do_transformations()

    rows = gr.parent_node.df.count()
    cols = len(gr.parent_node.df.columns)
    print("rows:", rows, flush=True)
    print("columns:", cols, flush=True)
    print("column_names:", gr.parent_node.df.columns, flush=True)
    print("shape:", (rows, cols), flush=True)
    if _is_interactive_mode():
        print("df.columns:", gr.parent_node.df.columns, flush=True)


if __name__ == "__main__":
    main()
