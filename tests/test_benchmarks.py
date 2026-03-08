#!/usr/bin/env python

import os
import pytest

RUN_BENCHMARKS = os.getenv("RUN_RELBENCH_BENCHMARKS") == "1"
if not RUN_BENCHMARKS:
    pytest.skip(
        "RelBench benchmarks are disabled. Set RUN_RELBENCH_BENCHMARKS=1 to run this module.",
        allow_module_level=True,
    )

import datetime
from pathlib import Path
from urllib.request import urlretrieve

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from graphreduce.enum import ComputeLayerEnum, PeriodUnit, SQLOpType
from graphreduce.graph_reduce import GraphReduce
from graphreduce.models import sqlop
from graphreduce.node import DuckdbNode
from graphreduce.stypes import infer_df_stype


RELBENCH_BASE_URL = "https://open-relbench.s3.us-east-1.amazonaws.com/rel-stack"
RELBENCH_TABLES = [
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


@pytest.fixture(scope="session")
def relbench_data_dir():
    # Persistent, repo-relative cache path so benchmark data is reused
    # across local runs and can be cached in CI.
    data_dir = Path("tests/data/relbench/rel-stack")
    data_dir.mkdir(parents=True, exist_ok=True)
    for table in RELBENCH_TABLES:
        out_path = data_dir / table
        if out_path.exists():
            continue
        try:
            urlretrieve(f"{RELBENCH_BASE_URL}/{table}", out_path)
        except Exception as exc:
            pytest.skip(f"Could not download relbench data: {exc}")
    return data_dir


def _train_catboost(df: pd.DataFrame, label_cols: list[str], id_excludes: list[str]) -> float:
    if not label_cols:
        raise ValueError("No label columns found.")
    target = label_cols[0]
    df = df.copy()
    df[target] = (df[target].fillna(0) > 0).astype("int8")

    if df[target].nunique() < 2:
        pytest.skip("Label is single-class after filtering; skipping model training.")

    CatBoostClassifier = pytest.importorskip("catboost").CatBoostClassifier

    stypes = infer_df_stype(df)
    features = [
        k
        for k, v in stypes.items()
        if str(v) == "numerical"
        and k not in id_excludes
        and "label" not in k
        and "had_engagement" not in k
    ]
    features = [c for c in features if c in df.columns]
    if not features:
        pytest.skip("No numerical features selected.")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        df[features],
        df[target],
        test_size=0.20,
        stratify=df[target],
        random_state=42,
    )

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    test_preds = np.zeros(len(X_test))

    for idx_tr, idx_va in skf.split(X_train_full, y_train_full):
        X_tr, X_va = X_train_full.iloc[idx_tr], X_train_full.iloc[idx_va]
        y_tr, y_va = y_train_full.iloc[idx_tr], y_train_full.iloc[idx_va]

        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=200,
            learning_rate=0.05,
            depth=6,
            auto_class_weights="Balanced",
            verbose=False,
        )
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True, verbose=False)
        test_preds += model.predict_proba(X_test)[:, 1] / 2.0

    auc = roc_auc_score(y_test, test_preds)
    return float(auc)


def _train_catboost_regressor(df: pd.DataFrame, label_cols: list[str], id_excludes: list[str]) -> float:
    if not label_cols:
        raise ValueError("No label columns found.")
    target = label_cols[0]
    df = df.copy()
    df[target] = df[target].fillna(0).astype("float64")

    CatBoostRegressor = pytest.importorskip("catboost").CatBoostRegressor

    stypes = infer_df_stype(df)
    features = [
        k
        for k, v in stypes.items()
        if str(v) == "numerical"
        and k not in id_excludes
        and "label" not in k
        and "had_engagement" not in k
    ]
    features = [c for c in features if c in df.columns]
    if not features:
        pytest.skip("No numerical features selected.")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        df[features],
        df[target],
        test_size=0.20,
        random_state=42,
    )

    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    test_preds = np.zeros(len(X_test))

    for idx_tr, idx_va in kf.split(X_train_full):
        X_tr, X_va = X_train_full.iloc[idx_tr], X_train_full.iloc[idx_va]
        y_tr, y_va = y_train_full.iloc[idx_tr], y_train_full.iloc[idx_va]

        model = CatBoostRegressor(
            loss_function="MAE",
            eval_metric="MAE",
            iterations=200,
            learning_rate=0.05,
            depth=6,
            verbose=False,
        )
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True, verbose=False)
        test_preds += model.predict(X_test) / 2.0

    mae = mean_absolute_error(y_test, test_preds)
    return float(mae)


@pytest.mark.skipif(not RUN_BENCHMARKS, reason="Set RUN_RELBENCH_BENCHMARKS=1 to run RelBench benchmarks.")
def test_relbench_user_badges_duckdb(relbench_data_dir):
    cut_date = datetime.datetime(2020, 1, 1)
    con = duckdb.connect()

    user = DuckdbNode(
        fpath=_duck_path(relbench_data_dir / "Users.csv"),
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
        fpath=_duck_path(relbench_data_dir / "Posts.csv"),
        prefix="post",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "OwnerUserId", "PostTypeId", "AcceptedAnswerId", "ParentId", "Title", "Tags", "Body", "CreationDate"],
        table_name="posts",
    )
    badge = DuckdbNode(
        fpath=_duck_path(relbench_data_dir / "Badges.csv"),
        prefix="bad",
        pk="Id",
        date_key="Date",
        columns=["Id", "UserId", "Class", "Name", "Date"],
        table_name="badges",
    )
    post_history = DuckdbNode(
        fpath=_duck_path(relbench_data_dir / "PostHistory.csv"),
        prefix="ph",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "PostHistoryTypeId", "PostId", "RevisionGUID", "CreationDate", "UserId", "Text", "Comment", "ContentLicense"],
        table_name="post_history",
    )
    post_links = DuckdbNode(
        fpath=_duck_path(relbench_data_dir / "PostLinks.csv"),
        prefix="plink",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "CreationDate", "PostId", "RelatedPostId", "LinkTypeId"],
        table_name="post_links",
    )
    vote = DuckdbNode(
        fpath=_duck_path(relbench_data_dir / "Votes.csv"),
        prefix="vote",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "PostId", "VoteTypeId", "UserId", "CreationDate", "BountyAmount"],
        table_name="votes",
    )
    comment = DuckdbNode(
        fpath=_duck_path(relbench_data_dir / "Comments.csv"),
        prefix="comm",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "PostId", "Score", "Text", "CreationDate", "UserId", "ContentLicense"],
        table_name="comments",
    )
    tag = DuckdbNode(
        fpath=_duck_path(relbench_data_dir / "Tags.csv"),
        prefix="tag",
        pk="Id",
        date_key=None,
        columns=["Id", "TagName", "Count", "ExcerptPostId", "WikiPostId"],
        table_name="tags",
    )

    gr = GraphReduce(
        name="relbench-user-badges-test",
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

    gr.add_entity_edge(parent_node=user, relation_node=post, parent_key="Id", relation_key="OwnerUserId", reduce=True)
    gr.add_entity_edge(parent_node=user, relation_node=vote, parent_key="Id", relation_key="UserId", reduce=True)
    gr.add_entity_edge(parent_node=user, relation_node=comment, parent_key="Id", relation_key="UserId", reduce=True)
    gr.add_entity_edge(parent_node=user, relation_node=badge, parent_key="Id", relation_key="UserId", reduce=True)
    gr.add_entity_edge(parent_node=post, relation_node=post_history, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(parent_node=post, relation_node=post_links, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(parent_node=post, relation_node=vote, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(parent_node=post, relation_node=comment, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(parent_node=post, relation_node=tag, parent_key="Id", relation_key="ExcerptPostId", reduce=True)

    gr.do_transformations_sql()
    out_df = con.sql(f"select * from {gr.parent_node._cur_data_ref}").to_df()
    con.close()

    assert len(out_df) > 0
    label_cols = [c for c in out_df.columns if c.startswith("bad_") and "label" in c.lower()]
    assert label_cols
    auc = _train_catboost(out_df, label_cols, id_excludes=["user_Id", "user_AccountId"])
    assert 0.0 <= auc <= 1.0


@pytest.mark.skipif(not RUN_BENCHMARKS, reason="Set RUN_RELBENCH_BENCHMARKS=1 to run RelBench benchmarks.")
def test_relbench_post_votes_duckdb(relbench_data_dir):
    cut_date = datetime.datetime(2020, 1, 1)
    con = duckdb.connect()

    post = DuckdbNode(
        fpath=_duck_path(relbench_data_dir / "Posts.csv"),
        prefix="post",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "OwnerUserId", "PostTypeId", "AcceptedAnswerId", "ParentId", "Title", "Tags", "Body", "CreationDate"],
        table_name="posts",
        do_filters_ops=[
            sqlop(
                optype=SQLOpType.where,
                opval=f"post_CreationDate <= '{cut_date.date()}'",
            )
        ],
    )
    vote = DuckdbNode(
        fpath=_duck_path(relbench_data_dir / "Votes.csv"),
        prefix="vote",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "PostId", "VoteTypeId", "UserId", "CreationDate", "BountyAmount"],
        table_name="votes",
    )
    comment = DuckdbNode(
        fpath=_duck_path(relbench_data_dir / "Comments.csv"),
        prefix="comm",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "PostId", "Score", "Text", "CreationDate", "UserId", "ContentLicense"],
        table_name="comments",
    )
    post_history = DuckdbNode(
        fpath=_duck_path(relbench_data_dir / "PostHistory.csv"),
        prefix="ph",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "PostHistoryTypeId", "PostId", "RevisionGUID", "CreationDate", "UserId", "Text", "Comment", "ContentLicense"],
        table_name="post_history",
    )
    post_links = DuckdbNode(
        fpath=_duck_path(relbench_data_dir / "PostLinks.csv"),
        prefix="plink",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "CreationDate", "PostId", "RelatedPostId", "LinkTypeId"],
        table_name="post_links",
    )
    tag = DuckdbNode(
        fpath=_duck_path(relbench_data_dir / "Tags.csv"),
        prefix="tag",
        pk="Id",
        date_key=None,
        columns=["Id", "TagName", "Count", "ExcerptPostId", "WikiPostId"],
        table_name="tags",
    )
    user = DuckdbNode(
        fpath=_duck_path(relbench_data_dir / "Users.csv"),
        prefix="user",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "DisplayName", "Location", "ProfileImageUrl", "WebsiteUrl", "AboutMe", "CreationDate"],
        table_name="users",
    )
    badge = DuckdbNode(
        fpath=_duck_path(relbench_data_dir / "Badges.csv"),
        prefix="bad",
        pk="Id",
        date_key="Date",
        columns=["Id", "UserId", "Class", "Name", "Date"],
        table_name="badges",
    )

    gr = GraphReduce(
        name="relbench-post-votes-test",
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

    gr.add_entity_edge(parent_node=post, relation_node=vote, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(parent_node=post, relation_node=comment, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(parent_node=post, relation_node=post_history, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(parent_node=post, relation_node=post_links, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(parent_node=post, relation_node=tag, parent_key="Id", relation_key="ExcerptPostId", reduce=True)
    gr.add_entity_edge(parent_node=post, relation_node=user, parent_key="OwnerUserId", relation_key="Id", reduce=True)
    gr.add_entity_edge(parent_node=user, relation_node=badge, parent_key="Id", relation_key="UserId", reduce=True)

    gr.do_transformations_sql()
    out_df = con.sql(f"select * from {gr.parent_node._cur_data_ref}").to_df()
    con.close()

    assert len(out_df) > 0
    label_cols = [c for c in out_df.columns if c.startswith("vote_") and "label" in c.lower()]
    assert label_cols
    mae = _train_catboost_regressor(out_df, label_cols, id_excludes=["post_Id", "post_OwnerUserId"])
    assert np.isfinite(mae)
