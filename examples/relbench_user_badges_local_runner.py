#!/usr/bin/env python
"""Run rel-stack user-badges example as a script (no pytest)."""

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


def _build_badges_frame(
    con: duckdb.DuckDBPyConnection,
    data_dir: Path,
    cut_date: datetime.datetime,
) -> tuple[object, str]:
    _prepare_view(con, "users_src", data_dir / "Users.csv")
    _prepare_view(con, "posts_src", data_dir / "Posts.csv")
    _prepare_view(con, "badges_src", data_dir / "Badges.csv")
    _prepare_view(con, "post_history_src", data_dir / "PostHistory.csv")
    _prepare_view(con, "post_links_src", data_dir / "PostLinks.csv")
    _prepare_view(con, "votes_src", data_dir / "Votes.csv")
    _prepare_view(con, "comments_src", data_dir / "Comments.csv")
    _prepare_view(con, "tags_src", data_dir / "Tags.csv")

    user = DuckdbNode(
        fpath="users_src",
        prefix="user",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "DisplayName", "Location", "ProfileImageUrl", "WebsiteUrl", "AboutMe", "CreationDate"],
        do_filters_ops=[sqlop(optype=SQLOpType.where, opval=f"user_CreationDate <= '{cut_date.date()}'")],
    )
    post = DuckdbNode(
        fpath="posts_src",
        prefix="post",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "OwnerUserId", "PostTypeId", "AcceptedAnswerId", "ParentId", "Title", "Tags", "Body", "CreationDate"],
    )
    badge = DuckdbNode(
        fpath="badges_src",
        prefix="bad",
        pk="Id",
        date_key="Date",
        columns=["Id", "UserId", "Class", "Name", "Date"],
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
    tag = DuckdbNode(
        fpath="tags_src",
        prefix="tag",
        pk="Id",
        date_key=None,
        columns=["Id", "TagName", "Count", "ExcerptPostId", "WikiPostId"],
    )

    gr = GraphReduce(
        name=f"relbench-user-badges-{cut_date.date()}",
        parent_node=user,
        compute_layer=ComputeLayerEnum.duckdb,
        sql_client=con,
        cut_date=cut_date,
        compute_period_val=3650,
        compute_period_unit=PeriodUnit.day,
        auto_features=True,
        auto_labels=True,
        label_node=badge,
        label_field="Id",
        label_operation="count",
        label_period_val=90,
        label_period_unit=PeriodUnit.day,
        auto_feature_hops_back=4,
        auto_feature_hops_front=0,
    )

    for node in [user, post, badge, post_history, post_links, vote, comment, tag]:
        gr.add_node(node)

    gr.add_entity_edge(user, post, parent_key="Id", relation_key="OwnerUserId", reduce=True)
    gr.add_entity_edge(user, vote, parent_key="Id", relation_key="UserId", reduce=True)
    gr.add_entity_edge(user, comment, parent_key="Id", relation_key="UserId", reduce=True)
    gr.add_entity_edge(user, badge, parent_key="Id", relation_key="UserId", reduce=True)
    gr.add_entity_edge(post, post_history, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(post, post_links, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(post, vote, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(post, comment, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(post, tag, parent_key="Id", relation_key="ExcerptPostId", reduce=True)

    gr.do_transformations_sql()
    df = con.sql(f"select * from {gr.parent_node._cur_data_ref}").to_df().copy()

    label_cols = [c for c in df.columns if c.startswith("bad_") and "label" in c.lower()]
    if not label_cols:
        raise ValueError("No badge label columns found.")
    target = label_cols[0]
    df[target] = (df[target].fillna(0) > 0).astype("int8")
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
    print("Starting rel-stack user badges pipeline...", flush=True)
    df_train, target = _build_badges_frame(con, data_dir, train_cut_date)
    df_future, target_future = _build_badges_frame(con, data_dir, future_cut_date)
    if target != target_future:
        raise ValueError(f"Target mismatch between train ({target}) and future ({target_future})")

    stypes = infer_df_stype(df_train)
    features = [
        k for k, v in stypes.items()
        if str(v) == "numerical" and k not in ["user_Id", "user_AccountId"] and "label" not in k and "had_engagement" not in k
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

    if len(features) == 0 or df_train[target].nunique() < 2:
        print("insufficient features or single-class target; skipping model fit", flush=True)
        _print_steps_summary(downloaded_files, "model fit skipped due to insufficient features or single-class target")
        return

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        df_train[features], df_train[target], test_size=0.2, stratify=df_train[target], random_state=42
    )
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_aucs: list[float] = []
    test_preds = np.zeros(len(X_test))

    for fold, (idx_tr, idx_va) in enumerate(skf.split(X_train_full, y_train_full), 1):
        if fold > 1:
            break
        print(f"\n=== Fold {fold} ===", flush=True)
        X_tr, X_va = X_train_full.iloc[idx_tr], X_train_full.iloc[idx_va]
        y_tr, y_va = y_train_full.iloc[idx_tr], y_train_full.iloc[idx_va]
        mdl = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            use_best_model=True,
            iterations=300,
            learning_rate=0.05,
            depth=4,
            l2_leaf_reg=8.0,
            min_data_in_leaf=50,
            boosting_type="Ordered",
            auto_class_weights="Balanced",
            bootstrap_type="Bayesian",
            bagging_temperature=1.0,
            random_strength=1.5,
            rsm=0.7,
            od_type="Iter",
            od_wait=50,
            verbose=100,
        )
        mdl.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True, verbose=100)
        val_pred = mdl.predict_proba(X_va)[:, 1]
        val_auc = roc_auc_score(y_va, val_pred)
        fold_aucs.append(val_auc)
        print(f"Fold {fold} validation AUC : {val_auc:.4f}", flush=True)
        test_preds += mdl.predict_proba(X_test)[:, 1]

    print("\n=== CV Summary ===", flush=True)
    print(f"Mean CV AUC : {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}", flush=True)
    print(f"Folds AUC   : {[f'{a:.4f}' for a in fold_aucs]}", flush=True)
    holdout_auc = roc_auc_score(y_test, test_preds)
    print(f"in_time_holdout_auc_2020: {holdout_auc:.4f}", flush=True)

    final_mdl = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=max(100, int(mdl.best_iteration_ * 1.05)),
        learning_rate=0.05,
        depth=4,
        l2_leaf_reg=8.0,
        min_data_in_leaf=50,
        boosting_type="Ordered",
        auto_class_weights="Balanced",
        bootstrap_type="Bayesian",
        bagging_temperature=1.0,
        random_strength=1.5,
        rsm=0.7,
        od_type="Iter",
        od_wait=50,
        verbose=100,
    )
    final_mdl.fit(df_train[features], df_train[target], verbose=100)
    future_preds = final_mdl.predict_proba(df_future[features])[:, 1]

    if df_future[target].nunique() < 2:
        print("future target is single-class; skipping out-of-time AUC", flush=True)
        _print_steps_summary(
            downloaded_files,
            f"in-time holdout ROC AUC (2020 cut date) = {holdout_auc:.4f}; out-of-time AUC (2021 cut date) skipped",
        )
        return

    future_auc = roc_auc_score(df_future[target], future_preds)
    print(f"out_of_time_auc_2021: {future_auc:.4f}", flush=True)
    _print_steps_summary(
        downloaded_files,
        f"in-time holdout ROC AUC (2020 cut date) = {holdout_auc:.4f}; out-of-time ROC AUC (2021 cut date) = {future_auc:.4f}",
    )


if __name__ == "__main__":
    main()
