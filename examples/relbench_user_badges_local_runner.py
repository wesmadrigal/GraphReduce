#!/usr/bin/env python
"""Run rel-stack user-badges example aligned to the RelBench task definition."""

from __future__ import annotations

import datetime
from pathlib import Path

import duckdb
import pandas as pd
from catboost import CatBoostClassifier
from relbench.metrics import accuracy, average_precision, f1, roc_auc
from relbench_dataset_utils import materialize_relbench_dataset

from graphreduce.enum import ComputeLayerEnum, PeriodUnit, SQLOpType
from graphreduce.graph_reduce import GraphReduce
from graphreduce.models import sqlop
from graphreduce.node import DuckdbNode
from graphreduce.stypes import infer_df_stype

TABLE_NAME_TO_FILENAME = {
    "users": "Users.csv",
    "posts": "Posts.csv",
    "badges": "Badges.csv",
    "postHistory": "PostHistory.csv",
    "postLinks": "PostLinks.csv",
    "votes": "Votes.csv",
    "comments": "Comments.csv",
}

VALIDATION_CUT_DATE = datetime.datetime(2020, 10, 1)
TEST_CUT_DATE = datetime.datetime(2021, 1, 1)
LABEL_PERIOD_DAYS = 365 // 4
TRAIN_CUT_DATE = VALIDATION_CUT_DATE - datetime.timedelta(days=LABEL_PERIOD_DAYS)


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
) -> tuple[pd.DataFrame, str]:
    _prepare_view(con, "users_src", data_dir / "Users.csv")
    _prepare_view(con, "posts_src", data_dir / "Posts.csv")
    _prepare_view(con, "badges_src", data_dir / "Badges.csv")
    _prepare_view(con, "post_history_src", data_dir / "PostHistory.csv")
    _prepare_view(con, "post_links_src", data_dir / "PostLinks.csv")
    _prepare_view(con, "votes_src", data_dir / "Votes.csv")
    _prepare_view(con, "comments_src", data_dir / "Comments.csv")

    user = DuckdbNode(
        fpath="users_src",
        prefix="user",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "DisplayName", "Location", "ProfileImageUrl", "WebsiteUrl", "AboutMe", "CreationDate"],
        do_filters_ops=[
            sqlop(optype=SQLOpType.where, opval=f"user_CreationDate <= '{cut_date}'"),
            sqlop(optype=SQLOpType.where, opval="user_Id is not null"),
        ],
    )
    post = DuckdbNode(
        fpath="posts_src",
        prefix="post",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "OwnerUserId", "PostTypeId", "ParentId", "Title", "Tags", "Body", "CreationDate"],
    )
    badge = DuckdbNode(
        fpath="badges_src",
        prefix="bad",
        pk="Id",
        date_key="Date",
        columns=["Id", "UserId", "Class", "Name", "Date"],
        do_filters_ops=[sqlop(optype=SQLOpType.where, opval="bad_UserId is not null")],
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
    vote_user = DuckdbNode(
        fpath="votes_src",
        prefix="voteu",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "PostId", "VoteTypeId", "UserId", "CreationDate"],
    )
    comment_user = DuckdbNode(
        fpath="comments_src",
        prefix="commu",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "PostId", "Text", "CreationDate", "UserId", "ContentLicense"],
    )
    vote_post = DuckdbNode(
        fpath="votes_src",
        prefix="votep",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "PostId", "VoteTypeId", "UserId", "CreationDate"],
    )
    comment_post = DuckdbNode(
        fpath="comments_src",
        prefix="commp",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "PostId", "Text", "CreationDate", "UserId", "ContentLicense"],
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
        auto_labels=False,
        date_filters_on_agg=True,
        auto_feature_hops_back=4,
        auto_feature_hops_front=0,
    )

    for node in [user, post, badge, post_history, post_links, vote_user, comment_user, vote_post, comment_post]:
        gr.add_node(node)

    gr.add_entity_edge(parent_node=user, relation_node=post, parent_key="Id", relation_key="OwnerUserId", reduce=True)
    gr.add_entity_edge(parent_node=user, relation_node=vote_user, parent_key="Id", relation_key="UserId", reduce=True)
    gr.add_entity_edge(parent_node=user, relation_node=comment_user, parent_key="Id", relation_key="UserId", reduce=True)
    gr.add_entity_edge(parent_node=user, relation_node=badge, parent_key="Id", relation_key="UserId", reduce=True)
    gr.add_entity_edge(parent_node=post, relation_node=post_history, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(parent_node=post, relation_node=post_links, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(parent_node=post, relation_node=vote_post, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(parent_node=post, relation_node=comment_post, parent_key="Id", relation_key="PostId", reduce=True)

    gr.do_transformations_sql()
    features = con.sql(f"select * from {gr.parent_node._cur_data_ref}").to_df().copy()

    labels = con.sql(
        f"""
        WITH timestamp_df AS (
            SELECT TIMESTAMP '{cut_date}' AS timestamp
        )
        SELECT
            t.timestamp,
            u.Id AS UserId,
            CASE WHEN COUNT(b.Id) >= 1 THEN 1 ELSE 0 END AS WillGetBadge
        FROM
            timestamp_df t
        LEFT JOIN users_src u
            ON u.CreationDate <= t.timestamp
        LEFT JOIN badges_src b
            ON u.Id = b.UserID
            AND b.Date > t.timestamp
            AND b.Date <= t.timestamp + INTERVAL '{LABEL_PERIOD_DAYS} days'
        GROUP BY
            t.timestamp,
            u.Id
        """
    ).to_df()

    labels = labels.dropna(subset=["UserId"]).copy()
    labels["UserId"] = labels["UserId"].astype("int64")
    frame = features.merge(labels[["UserId", "WillGetBadge"]], left_on="user_Id", right_on="UserId", how="inner")
    frame = frame.drop(columns=["UserId"])
    frame["WillGetBadge"] = frame["WillGetBadge"].astype("int8")
    return frame, "WillGetBadge"


def main() -> None:
    data_dir = Path("tests/data/relbench/rel-stack")
    materialized_files = materialize_relbench_dataset("rel-stack", data_dir, TABLE_NAME_TO_FILENAME)

    con = duckdb.connect()
    print("Running rel-stack user-badges task...", flush=True)

    df_train, target = _build_badges_frame(con, data_dir, TRAIN_CUT_DATE)
    df_val, target_val = _build_badges_frame(con, data_dir, VALIDATION_CUT_DATE)
    df_test, target_test = _build_badges_frame(con, data_dir, TEST_CUT_DATE)

    if target != target_val or target != target_test:
        raise ValueError(f"Target mismatch across splits: train={target}, val={target_val}, test={target_test}")

    stypes = infer_df_stype(df_train)
    features = [
        k
        for k, v in stypes.items()
        if str(v) == "numerical"
        and k not in ["user_Id", "user_AccountId"]
        and "label" not in k.lower()
        and k != target
        and k in df_val.columns
        and k in df_test.columns
    ]

    print("materialized_files:", materialized_files, flush=True)
    print("train_cut_date:", TRAIN_CUT_DATE.date(), flush=True)
    print("validation_cut_date:", VALIDATION_CUT_DATE.date(), flush=True)
    print("test_cut_date:", TEST_CUT_DATE.date(), flush=True)
    print("label_period_days:", LABEL_PERIOD_DAYS, flush=True)
    print("target:", target, flush=True)
    print("train_rows:", len(df_train), flush=True)
    print("validation_rows:", len(df_val), flush=True)
    print("test_rows:", len(df_test), flush=True)
    print("feature_count:", len(features), flush=True)

    if not features or df_train[target].nunique() < 2:
        print("insufficient features or single-class training target; skipping model fit", flush=True)
        return

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
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
    model.fit(
        df_train[features].fillna(0),
        df_train[target],
        eval_set=(df_val[features].fillna(0), df_val[target]),
        use_best_model=True,
        verbose=100,
    )

    val_pred = model.predict_proba(df_val[features].fillna(0))[:, 1]
    test_pred = model.predict_proba(df_test[features].fillna(0))[:, 1]

    val_metrics = {
        "average_precision": float(average_precision(df_val[target].to_numpy(), val_pred)),
        "accuracy": float(accuracy(df_val[target].to_numpy(), val_pred)),
        "f1": float(f1(df_val[target].to_numpy(), val_pred)),
        "roc_auc": float(roc_auc(df_val[target].to_numpy(), val_pred)),
    }
    test_metrics = {
        "average_precision": float(average_precision(df_test[target].to_numpy(), test_pred)),
        "accuracy": float(accuracy(df_test[target].to_numpy(), test_pred)),
        "f1": float(f1(df_test[target].to_numpy(), test_pred)),
        "roc_auc": float(roc_auc(df_test[target].to_numpy(), test_pred)),
    }

    print("validation_metrics:", val_metrics, flush=True)
    print("test_metrics:", test_metrics, flush=True)


if __name__ == "__main__":
    main()
