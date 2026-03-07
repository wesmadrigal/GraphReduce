#!/usr/bin/env python
"""Run GraphReduce + XGBoost example locally and print metrics."""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from graphreduce.enum import ComputeLayerEnum, PeriodUnit
from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import DynamicNode


def main() -> None:
    data_path = Path("tests/data/cust_data")

    cust_node = DynamicNode(fpath=str(data_path / "cust.csv"), fmt="csv", prefix="cust", date_key=None, pk="id")
    orders_node = DynamicNode(fpath=str(data_path / "orders.csv"), fmt="csv", prefix="ord", date_key="ts", pk="id")
    notifications_node = DynamicNode(
        fpath=str(data_path / "notifications.csv"), fmt="csv", prefix="not", date_key="ts", pk="id"
    )

    gr = GraphReduce(
        name="predictive_ai_xgboost_local",
        parent_node=cust_node,
        fmt="csv",
        compute_layer=ComputeLayerEnum.pandas,
        auto_features=True,
        auto_labels=True,
        cut_date=datetime.datetime(2023, 6, 30),
        compute_period_unit=PeriodUnit.day,
        compute_period_val=365,
        label_node=orders_node,
        label_field="id",
        label_operation="count",
        label_period_unit=PeriodUnit.day,
        label_period_val=30,
        auto_feature_hops_back=3,
        auto_feature_hops_front=0,
    )

    gr.add_node(cust_node)
    gr.add_node(orders_node)
    gr.add_node(notifications_node)

    gr.add_entity_edge(cust_node, orders_node, parent_key="id", relation_key="customer_id", relation_type="parent_child", reduce=True)
    gr.add_entity_edge(
        cust_node,
        notifications_node,
        parent_key="id",
        relation_key="customer_id",
        relation_type="parent_child",
        reduce=True,
    )

    print("Starting GraphReduce + XGBoost pipeline...", flush=True)
    gr.do_transformations()
    df = gr.parent_node.df.copy()

    label_candidates = [c for c in df.columns if c.startswith("ord_") and "label" in c]
    if not label_candidates:
        raise ValueError("Could not find label column automatically.")
    target_col = label_candidates[0]

    feature_cols = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols].fillna(0.0)
    y = (df[target_col].fillna(0) > 0).astype(int)

    print(f"rows: {len(df)}", flush=True)
    print(f"columns: {len(df.columns)}", flush=True)
    print("shape:", df.shape, flush=True)
    print(f"target: {target_col}", flush=True)
    print(f"num_features: {len(feature_cols)}", flush=True)

    if y.nunique() < 2:
        print("target has a single class in this sample; skipping model fit", flush=True)
        return

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    test_size = 0.5 if len(df) > 3 else 0.25
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    print(f"accuracy: {acc:.4f}", flush=True)

    if len(set(y_test)) > 1:
        auc = roc_auc_score(y_test, y_proba)
        print(f"roc_auc: {auc:.4f}", flush=True)
    else:
        print("roc_auc: n/a (single-class test split)", flush=True)


if __name__ == "__main__":
    main()
