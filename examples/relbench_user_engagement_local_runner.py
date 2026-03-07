#!/usr/bin/env python
"""Run a lightweight rel-stack user-engagement example with CatBoost."""

from __future__ import annotations

import datetime
from pathlib import Path
from urllib.request import urlretrieve

import duckdb
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from graphreduce.enum import ComputeLayerEnum, PeriodUnit, SQLOpType
from graphreduce.graph_reduce import GraphReduce
from graphreduce.models import sqlop
from graphreduce.node import DuckdbNode
from graphreduce.stypes import infer_df_stype

BASE_URL = "https://open-relbench.s3.us-east-1.amazonaws.com/rel-stack"
TABLES = [
    "Users.csv",
    "Posts.csv",
    "Badges.csv",
    "PostHistory.csv",
    "PostLinks.csv",
    "Votes.csv",
    "Comments.csv",
    "Tags.csv",
]


def _duck_path(path: Path) -> str:
    return f"'{path}'"


def main() -> None:
    data_dir = Path("tests/data/relbench/rel-stack")
    data_dir.mkdir(parents=True, exist_ok=True)
    for table in TABLES:
        out_path = data_dir / table
        if not out_path.exists():
            urlretrieve(f"{BASE_URL}/{table}", out_path)

    cut_date = datetime.datetime(2021, 1, 1)
    con = duckdb.connect()

    user = DuckdbNode(
        fpath=_duck_path(data_dir / "Users.csv"),
        prefix="user",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "DisplayName", "Location", "ProfileImageUrl", "WebsiteUrl", "AboutMe", "CreationDate"],
        table_name="users",
        do_filters_ops=[
            sqlop(
                optype=SQLOpType.where,
                opval=f"user_CreationDate <= '{cut_date.date()}'",
            )
        ],
    )
    post = DuckdbNode(
        fpath=_duck_path(data_dir / "Posts.csv"),
        prefix="post",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "OwnerUserId", "PostTypeId", "AcceptedAnswerId", "ParentId", "Title", "Tags", "Body", "CreationDate"],
        table_name="posts",
    )
    vote = DuckdbNode(
        fpath=_duck_path(data_dir / "Votes.csv"),
        prefix="vote",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "PostId", "VoteTypeId", "UserId", "CreationDate", "BountyAmount"],
        table_name="votes",
        do_labels_ops=[
            sqlop(optype=SQLOpType.aggfunc, opval="count(*) as vote_Id_label"),
            sqlop(optype=SQLOpType.agg, opval="vote_UserId"),
        ],
    )
    comment = DuckdbNode(
        fpath=_duck_path(data_dir / "Comments.csv"),
        prefix="comm",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "PostId", "Score", "Text", "CreationDate", "UserId", "ContentLicense"],
        table_name="comments",
        do_labels_ops=[
            sqlop(optype=SQLOpType.aggfunc, opval="count(*) as comm_Id_label"),
            sqlop(optype=SQLOpType.agg, opval="comm_UserId"),
        ],
    )

    gr = GraphReduce(
        name="relbench-user-engagement-local",
        parent_node=user,
        compute_layer=ComputeLayerEnum.duckdb,
        sql_client=con,
        cut_date=cut_date,
        compute_period_val=3650,
        compute_period_unit=PeriodUnit.day,
        auto_features=True,
        auto_labels=True,
        label_node=post,
        label_field="Id",
        label_operation="count",
        label_period_val=90,
        label_period_unit=PeriodUnit.day,
        auto_feature_hops_back=3,
        auto_feature_hops_front=0,
    )

    for node in [user, post, vote, comment]:
        gr.add_node(node)

    gr.add_entity_edge(user, post, parent_key="Id", relation_key="OwnerUserId", reduce=True)
    gr.add_entity_edge(user, vote, parent_key="Id", relation_key="UserId", reduce=True)
    gr.add_entity_edge(user, comment, parent_key="Id", relation_key="UserId", reduce=True)

    print("Starting relbench user-engagement transformation...", flush=True)
    gr.do_transformations_sql()
    df = con.sql(f"select * from {gr.parent_node._cur_data_ref}").to_df()

    post_label_cols = [c for c in df.columns if c.startswith("post_") and "label" in c.lower()]
    vote_label_cols = [c for c in df.columns if c.startswith("vote_") and "label" in c.lower()]
    comm_label_cols = [c for c in df.columns if c.startswith("comm_") and "label" in c.lower()]
    label_cols = post_label_cols + vote_label_cols + comm_label_cols

    if not label_cols:
        raise ValueError("No engagement label columns found.")

    for c in label_cols:
        df[c] = df[c].fillna(0)
    df["user_had_engagement"] = (df[label_cols].sum(axis=1) > 0).astype("int8")

    stypes = infer_df_stype(df)
    features = [
        k
        for k, v in stypes.items()
        if str(v) == "numerical"
        and k not in ["user_Id", "user_AccountId"]
        and "label" not in k
        and "had_engagement" not in k
    ]
    features = [c for c in features if c in df.columns]

    X = df[features].fillna(0)
    y = df["user_had_engagement"]

    print(f"rows: {len(df)}", flush=True)
    print(f"columns: {len(df.columns)}", flush=True)
    print("shape:", df.shape, flush=True)
    print(f"num_features: {len(features)}", flush=True)

    if y.nunique() < 2:
        print("single-class target; skipping model fit", flush=True)
        return

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    test_preds = np.zeros(len(X_test))
    for idx_tr, idx_va in skf.split(X_train_full, y_train_full):
        X_tr, X_va = X_train_full.iloc[idx_tr], X_train_full.iloc[idx_va]
        y_tr, y_va = y_train_full.iloc[idx_tr], y_train_full.iloc[idx_va]
        mdl = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=300,
            learning_rate=0.05,
            depth=6,
            auto_class_weights="Balanced",
            verbose=False,
        )
        mdl.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True, verbose=False)
        test_preds += mdl.predict_proba(X_test)[:, 1] / 2.0

    auc = roc_auc_score(y_test, test_preds)
    print(f"test_auc: {auc:.4f}", flush=True)


if __name__ == "__main__":
    main()
