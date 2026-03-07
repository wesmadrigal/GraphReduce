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


def _print_steps_summary(downloaded_files: list[str], result_text: str) -> None:
    print("\nSteps completed:", flush=True)
    print(f"1. Downloaded files: {len(downloaded_files)} new file(s).", flush=True)
    print("2. Prepared and aggregated data with GraphReduce.", flush=True)
    print("3. Trained model.", flush=True)
    print("4. Predicted and scored on holdout set.", flush=True)
    print(f"5. Achieved the following result: {result_text}", flush=True)


def _prepare_view(con: duckdb.DuckDBPyConnection, view_name: str, csv_path: Path) -> None:
    con.sql(
        f"""
        CREATE OR REPLACE VIEW {view_name} AS
        SELECT *
        FROM read_csv_auto(
            '{csv_path}',
            header=true,
            strict_mode=false,
            ignore_errors=true
        );
        """
    )


def main() -> None:
    data_dir = Path("tests/data/relbench/rel-stack")
    data_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files: list[str] = []
    for table in TABLES:
        out_path = data_dir / table
        if not out_path.exists():
            urlretrieve(f"{BASE_URL}/{table}", out_path)
            downloaded_files.append(table)

    cut_date = datetime.datetime(2021, 1, 1)
    con = duckdb.connect()
    _prepare_view(con, "users_src", data_dir / "Users.csv")
    _prepare_view(con, "posts_src", data_dir / "Posts.csv")
    _prepare_view(con, "votes_src", data_dir / "Votes.csv")
    _prepare_view(con, "comments_src", data_dir / "Comments.csv")

    user = DuckdbNode(
        fpath="users_src",
        prefix="user",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "DisplayName", "Location", "ProfileImageUrl", "WebsiteUrl", "AboutMe", "CreationDate"],
        do_filters_ops=[
            sqlop(
                optype=SQLOpType.where,
                opval=f"user_CreationDate <= '{cut_date.date()}'",
            )
        ],
    )
    post = DuckdbNode(
        fpath="posts_src",
        prefix="post",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "OwnerUserId", "PostTypeId", "AcceptedAnswerId", "ParentId", "Title", "Tags", "Body", "CreationDate"],
    )
    vote = DuckdbNode(
        fpath="votes_src",
        prefix="vote",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "PostId", "VoteTypeId", "UserId", "CreationDate"],
        do_labels_ops=[
            sqlop(optype=SQLOpType.aggfunc, opval="count(*) as vote_Id_label"),
            sqlop(optype=SQLOpType.agg, opval="vote_UserId"),
        ],
    )
    comment = DuckdbNode(
        fpath="comments_src",
        prefix="comm",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "PostId", "Text", "CreationDate", "UserId", "ContentLicense"],
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
        date_filters_on_agg=True,
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
        _print_steps_summary(downloaded_files, "model fit skipped due to single-class target")
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
    _print_steps_summary(downloaded_files, f"holdout ROC AUC = {auc:.4f}")


if __name__ == "__main__":
    main()
