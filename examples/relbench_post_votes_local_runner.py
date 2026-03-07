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


def _print_steps_summary(downloaded_files: list[str], result_text: str) -> None:
    print("\nSteps completed:", flush=True)
    print(f"1. Downloaded files: {len(downloaded_files)} new file(s).", flush=True)
    print("2. Prepared and aggregated two GraphReduce datasets (2020 train/eval, 2021 out-of-time).", flush=True)
    print("3. Trained model on the 2020 dataset.", flush=True)
    print("4. Predicted and scored on 2020 holdout and 2021 out-of-time datasets.", flush=True)
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


def _build_post_votes_frame(
    con: duckdb.DuckDBPyConnection,
    cut_date: datetime.datetime,
) -> tuple[object, str]:
    post = DuckdbNode(
        fpath="posts_src",
        prefix="post",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "OwnerUserId", "PostTypeId", "AcceptedAnswerId", "ParentId", "Title", "Tags", "Body", "CreationDate"],
        do_filters_ops=[sqlop(optype=SQLOpType.where, opval=f"post_CreationDate <= '{cut_date.date()}'")],
    )
    vote = DuckdbNode(
        fpath="votes_src",
        prefix="vote",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "PostId", "VoteTypeId", "UserId", "CreationDate"],
    )
    comment = DuckdbNode(
        fpath="comments_src",
        prefix="comm",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "PostId", "Text", "CreationDate", "UserId", "ContentLicense"],
    )
    post_history = DuckdbNode(
        fpath="post_history_src",
        prefix="ph",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "PostHistoryTypeId", "PostId", "RevisionGUID", "CreationDate", "UserId", "Text", "Comment", "ContentLicense"],
    )
    post_links = DuckdbNode(
        fpath="post_links_src",
        prefix="plink",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "CreationDate", "PostId", "RelatedPostId", "LinkTypeId"],
    )
    tag = DuckdbNode(
        fpath="tags_src",
        prefix="tag",
        pk="Id",
        date_key=None,
        columns=["Id", "TagName", "Count", "ExcerptPostId", "WikiPostId"],
    )
    user = DuckdbNode(
        fpath="users_src",
        prefix="user",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "DisplayName", "Location", "ProfileImageUrl", "WebsiteUrl", "AboutMe", "CreationDate"],
    )
    badge = DuckdbNode(
        fpath="badges_src",
        prefix="bad",
        pk="Id",
        date_key="Date",
        columns=["Id", "UserId", "Class", "Name", "Date"],
    )

    gr = GraphReduce(
        name=f"relbench-post-votes-local-{cut_date.date()}",
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

    gr.do_transformations_sql()
    df = con.sql(f"select * from {gr.parent_node._cur_data_ref}").to_df().copy()
    label_cols = [c for c in df.columns if c.startswith("vote_") and "label" in c.lower()]
    if not label_cols:
        raise ValueError("No vote label columns found.")
    target = label_cols[0]
    df[target] = df[target].fillna(0).astype("float64")
    return df, target


def main() -> None:
    data_dir = Path("tests/data/relbench/rel-stack")
    data_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files: list[str] = []
    for table in TABLES:
        out_path = data_dir / table
        if not out_path.exists():
            urlretrieve(f"{BASE_URL}/{table}", out_path)
            downloaded_files.append(table)

    train_cut_date = datetime.datetime(2020, 1, 1)
    future_cut_date = datetime.datetime(2021, 1, 1)
    con = duckdb.connect()
    _prepare_view(con, "users_src", data_dir / "Users.csv")
    _prepare_view(con, "posts_src", data_dir / "Posts.csv")
    _prepare_view(con, "badges_src", data_dir / "Badges.csv")
    _prepare_view(con, "post_history_src", data_dir / "PostHistory.csv")
    _prepare_view(con, "post_links_src", data_dir / "PostLinks.csv")
    _prepare_view(con, "votes_src", data_dir / "Votes.csv")
    _prepare_view(con, "comments_src", data_dir / "Comments.csv")
    _prepare_view(con, "tags_src", data_dir / "Tags.csv")

    print("Starting rel-stack post votes pipeline...", flush=True)
    print("Building 2020 training/eval graph...", flush=True)
    df_train, target = _build_post_votes_frame(con, train_cut_date)
    print("Building 2021 out-of-time graph...", flush=True)
    df_future, target_future = _build_post_votes_frame(con, future_cut_date)
    if target != target_future:
        raise ValueError(f"Target mismatch between train ({target}) and future ({target_future})")

    stypes = infer_df_stype(df_train)
    features = [
        k for k, v in stypes.items()
        if str(v) == "numerical" and k not in ["post_Id", "post_OwnerUserId"] and "label" not in k and "had_engagement" not in k
    ]
    features = [c for c in features if c in df_train.columns and c in df_future.columns]

    print(f"train rows: {len(df_train)}", flush=True)
    print(f"train columns: {len(df_train.columns)}", flush=True)
    print("train shape:", df_train.shape, flush=True)
    print(f"future rows: {len(df_future)}", flush=True)
    print(f"future columns: {len(df_future.columns)}", flush=True)
    print("future shape:", df_future.shape, flush=True)
    print(f"target: {target}", flush=True)
    print(f"num_features: {len(features)}", flush=True)

    if len(features) == 0:
        print("no numerical features; skipping model fit", flush=True)
        _print_steps_summary(downloaded_files, "model fit skipped due to no numerical features")
        return

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        df_train[features], df_train[target], test_size=0.2, random_state=42
    )
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    fold_maes: list[float] = []
    test_preds = np.zeros(len(X_test))

    for fold, (idx_tr, idx_va) in enumerate(kf.split(X_train_full), 1):
        print(f"\n=== Fold {fold} ===", flush=True)
        X_tr, X_va = X_train_full.iloc[idx_tr], X_train_full.iloc[idx_va]
        y_tr, y_va = y_train_full.iloc[idx_tr], y_train_full.iloc[idx_va]
        mdl = CatBoostRegressor(
            loss_function="MAE",
            eval_metric="MAE",
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            verbose=200,
        )
        mdl.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True, verbose=200)
        val_pred = mdl.predict(X_va)
        val_mae = mean_absolute_error(y_va, val_pred)
        fold_maes.append(val_mae)
        print(f"Fold {fold} validation MAE : {val_mae:.4f}", flush=True)
        test_preds += mdl.predict(X_test) / 2.0

    print("\n=== CV Summary ===", flush=True)
    print(f"Mean CV MAE : {np.mean(fold_maes):.4f} ± {np.std(fold_maes):.4f}", flush=True)
    print(f"Folds MAE   : {[f'{a:.4f}' for a in fold_maes]}", flush=True)
    holdout_mae = mean_absolute_error(y_test, test_preds)
    print(f"in_time_holdout_mae_2020: {holdout_mae:.4f}", flush=True)

    final_mdl = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=int(mdl.best_iteration_ * 1.1),
        learning_rate=0.05,
        depth=6,
        verbose=200,
    )
    final_mdl.fit(df_train[features], df_train[target], verbose=200)
    future_preds = final_mdl.predict(df_future[features])
    future_mae = mean_absolute_error(df_future[target], future_preds)
    print(f"out_of_time_mae_2021: {future_mae:.4f}", flush=True)
    _print_steps_summary(
        downloaded_files,
        f"in-time holdout MAE (2020 cut date) = {holdout_mae:.4f}; out-of-time MAE (2021 cut date) = {future_mae:.4f}",
    )


if __name__ == "__main__":
    main()
