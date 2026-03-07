#!/usr/bin/env python
"""Run rel-stack post-votes example as a script (no pytest)."""

from __future__ import annotations

import datetime
from pathlib import Path
from urllib.request import urlretrieve

import duckdb
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split

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

    cut_date = datetime.datetime(2020, 1, 1)
    con = duckdb.connect()

    post = DuckdbNode(
        fpath=_duck_path(data_dir / "Posts.csv"),
        prefix="post",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "OwnerUserId", "PostTypeId", "AcceptedAnswerId", "ParentId", "Title", "Tags", "Body", "CreationDate"],
        table_name="posts",
        do_filters_ops=[sqlop(optype=SQLOpType.where, opval=f"post_CreationDate <= '{cut_date.date()}'")],
    )
    vote = DuckdbNode(
        fpath=_duck_path(data_dir / "Votes.csv"),
        prefix="vote",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "PostId", "VoteTypeId", "UserId", "CreationDate", "BountyAmount"],
        table_name="votes",
    )
    comment = DuckdbNode(
        fpath=_duck_path(data_dir / "Comments.csv"),
        prefix="comm",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "PostId", "Score", "Text", "CreationDate", "UserId", "ContentLicense"],
        table_name="comments",
    )
    post_history = DuckdbNode(
        fpath=_duck_path(data_dir / "PostHistory.csv"),
        prefix="ph",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "PostHistoryTypeId", "PostId", "RevisionGUID", "CreationDate", "UserId", "Text", "Comment", "ContentLicense"],
        table_name="post_history",
    )
    post_links = DuckdbNode(
        fpath=_duck_path(data_dir / "PostLinks.csv"),
        prefix="plink",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "CreationDate", "PostId", "RelatedPostId", "LinkTypeId"],
        table_name="post_links",
    )
    tag = DuckdbNode(
        fpath=_duck_path(data_dir / "Tags.csv"),
        prefix="tag",
        pk="Id",
        date_key=None,
        columns=["Id", "TagName", "Count", "ExcerptPostId", "WikiPostId"],
        table_name="tags",
    )
    user = DuckdbNode(
        fpath=_duck_path(data_dir / "Users.csv"),
        prefix="user",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "DisplayName", "Location", "ProfileImageUrl", "WebsiteUrl", "AboutMe", "CreationDate"],
        table_name="users",
    )
    badge = DuckdbNode(
        fpath=_duck_path(data_dir / "Badges.csv"),
        prefix="bad",
        pk="Id",
        date_key="Date",
        columns=["Id", "UserId", "Class", "Name", "Date"],
        table_name="badges",
    )

    gr = GraphReduce(
        name="relbench-post-votes-local",
        parent_node=post,
        compute_layer=ComputeLayerEnum.duckdb,
        sql_client=con,
        cut_date=cut_date,
        compute_period_val=3650,
        compute_period_unit=PeriodUnit.day,
        auto_features=True,
        auto_labels=True,
        label_node=vote,
        label_field="Id",
        label_operation="count",
        label_period_val=90,
        label_period_unit=PeriodUnit.day,
        auto_feature_hops_back=4,
        auto_feature_hops_front=0,
    )

    for node in [post, vote, comment, post_history, post_links, tag, user, badge]:
        gr.add_node(node)

    gr.add_entity_edge(post, vote, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(post, comment, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(post, post_history, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(post, post_links, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(post, tag, parent_key="Id", relation_key="ExcerptPostId", reduce=True)
    gr.add_entity_edge(post, user, parent_key="OwnerUserId", relation_key="Id", reduce=True)
    gr.add_entity_edge(user, badge, parent_key="Id", relation_key="UserId", reduce=True)

    print("Starting rel-stack post votes pipeline...", flush=True)
    gr.do_transformations_sql()
    df = con.sql(f"select * from {gr.parent_node._cur_data_ref}").to_df().copy()

    label_cols = [c for c in df.columns if c.startswith("vote_") and "label" in c.lower()]
    if not label_cols:
        raise ValueError("No vote label columns found.")
    target = label_cols[0]
    df[target] = df[target].fillna(0).astype("float64")

    stypes = infer_df_stype(df)
    features = [
        k for k, v in stypes.items()
        if str(v) == "numerical" and k not in ["post_Id", "post_OwnerUserId"] and "label" not in k and "had_engagement" not in k
    ]
    features = [c for c in features if c in df.columns]

    print(f"rows: {len(df)}", flush=True)
    print(f"columns: {len(df.columns)}", flush=True)
    print("shape:", df.shape, flush=True)
    print(f"target: {target}", flush=True)
    print(f"num_features: {len(features)}", flush=True)

    if len(features) == 0:
        print("no numerical features; skipping model fit", flush=True)
        return

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=42
    )
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    test_preds = np.zeros(len(X_test))

    for idx_tr, idx_va in kf.split(X_train_full):
        X_tr, X_va = X_train_full.iloc[idx_tr], X_train_full.iloc[idx_va]
        y_tr, y_va = y_train_full.iloc[idx_tr], y_train_full.iloc[idx_va]
        mdl = CatBoostRegressor(
            loss_function="MAE",
            eval_metric="MAE",
            iterations=300,
            learning_rate=0.05,
            depth=6,
            verbose=False,
        )
        mdl.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True, verbose=False)
        test_preds += mdl.predict(X_test) / 2.0

    mae = mean_absolute_error(y_test, test_preds)
    print(f"test_mae: {mae:.4f}", flush=True)


if __name__ == "__main__":
    main()
