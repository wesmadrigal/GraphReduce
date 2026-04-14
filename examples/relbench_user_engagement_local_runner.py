#!/usr/bin/env python
"""Run rel-stack user-engagement example aligned to the RelBench task definition."""

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


def _build_user_engagement_frame(
    con: duckdb.DuckDBPyConnection,
    cut_date: datetime.datetime,
) -> tuple[pd.DataFrame, str]:
    user = DuckdbNode(
        fpath="users_src",
        prefix="user",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "DisplayName", "Location", "ProfileImageUrl", "WebsiteUrl", "AboutMe", "CreationDate"],
        do_filters_ops=[
            sqlop(
                optype=SQLOpType.where,
                opval=f"""(
                    user_Id != -1
                    AND user_CreationDate <= '{cut_date}'
                    AND (
                        EXISTS (
                            SELECT 1
                            FROM posts_src p
                            WHERE p.OwnerUserId = user_Id
                              AND p.CreationDate <= '{cut_date}'
                        )
                        OR EXISTS (
                            SELECT 1
                            FROM votes_src v
                            WHERE v.UserId = user_Id
                              AND v.CreationDate <= '{cut_date}'
                        )
                        OR EXISTS (
                            SELECT 1
                            FROM comments_src c
                            WHERE c.UserId = user_Id
                              AND c.CreationDate <= '{cut_date}'
                        )
                    )
                )""",
            )
        ],
    )
    post = DuckdbNode(
        fpath="posts_src",
        prefix="post",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "OwnerUserId", "PostTypeId", "ParentId", "Title", "Tags", "Body", "CreationDate"],
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
    post_vote = DuckdbNode(
        fpath="votes_src",
        prefix="pvote",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "PostId", "VoteTypeId", "UserId", "CreationDate"],
    )
    post_comment = DuckdbNode(
        fpath="comments_src",
        prefix="pcomm",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "PostId", "Text", "CreationDate", "UserId", "ContentLicense"],
    )
    post_comment_user = DuckdbNode(
        fpath="users_src",
        prefix="pcu",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "DisplayName", "Location", "ProfileImageUrl", "WebsiteUrl", "AboutMe", "CreationDate"],
    )
    post_comment_badge = DuckdbNode(
        fpath="badges_src",
        prefix="pcbad",
        pk="Id",
        date_key="Date",
        columns=["Id", "UserId", "Class", "Name", "Date"],
    )

    gr = GraphReduce(
        name=f"relbench-user-engagement-{cut_date.date()}",
        parent_node=user,
        compute_layer=ComputeLayerEnum.duckdb,
        sql_client=con,
        cut_date=cut_date,
        compute_period_val=3650,
        compute_period_unit=PeriodUnit.day,
        auto_features=True,
        auto_labels=False,
        date_filters_on_agg=True,
        auto_feature_hops_back=3,
        auto_feature_hops_front=0,
    )

    for node in [user, post, vote, comment, post_vote, post_comment, post_comment_user, post_comment_badge]:
        gr.add_node(node)

    gr.add_entity_edge(user, post, parent_key="Id", relation_key="OwnerUserId", reduce=True)
    gr.add_entity_edge(user, vote, parent_key="Id", relation_key="UserId", reduce=True)
    gr.add_entity_edge(user, comment, parent_key="Id", relation_key="UserId", reduce=True)
    gr.add_entity_edge(post, post_vote, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(post, post_comment, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(post_comment, post_comment_user, parent_key="UserId", relation_key="Id", reduce=True)
    gr.add_entity_edge(post_comment_user, post_comment_badge, parent_key="Id", relation_key="UserId", reduce=True)

    gr.do_transformations_sql()
    features = con.sql(f"select * from {gr.parent_node._cur_data_ref}").to_df().copy()

    labels = con.sql(
        f"""
        WITH
        timestamp_df AS (
            SELECT TIMESTAMP '{cut_date}' AS timestamp
        ),
        all_engagement AS (
            SELECT p.id, p.owneruserid AS userid, p.creationdate
            FROM posts_src p
            UNION
            SELECT v.id, v.userid, v.creationdate
            FROM votes_src v
            UNION
            SELECT c.id, c.userid, c.creationdate
            FROM comments_src c
        ),
        active_users AS (
            SELECT
                t.timestamp,
                u.id,
                COUNT(DISTINCT a.id) AS n_engagement
            FROM timestamp_df t
            CROSS JOIN users_src u
            LEFT JOIN all_engagement a
                ON u.id = a.userid
                AND a.creationdate <= t.timestamp
            WHERE u.id != -1
            GROUP BY t.timestamp, u.id
        )
        SELECT
            u.timestamp,
            u.id AS OwnerUserId,
            IF(COUNT(DISTINCT a.id) >= 1, 1, 0) AS contribution
        FROM active_users u
        LEFT JOIN all_engagement a
            ON u.id = a.userid
            AND a.creationdate > u.timestamp
            AND a.creationdate <= u.timestamp + INTERVAL '{LABEL_PERIOD_DAYS} days'
        WHERE u.n_engagement >= 1
        GROUP BY u.timestamp, u.id
        """
    ).to_df()

    labels["OwnerUserId"] = labels["OwnerUserId"].astype("int64")
    frame = features.merge(labels[["OwnerUserId", "contribution"]], left_on="user_Id", right_on="OwnerUserId", how="inner")
    frame = frame.drop(columns=["OwnerUserId"])
    frame["contribution"] = frame["contribution"].astype("int8")
    return frame, "contribution"


def main() -> None:
    data_dir = Path("tests/data/relbench/rel-stack")
    materialized_files = materialize_relbench_dataset("rel-stack", data_dir, TABLE_NAME_TO_FILENAME)

    con = duckdb.connect()
    _prepare_view(con, "users_src", data_dir / "Users.csv")
    _prepare_view(con, "posts_src", data_dir / "Posts.csv")
    _prepare_view(con, "badges_src", data_dir / "Badges.csv")
    _prepare_view(con, "votes_src", data_dir / "Votes.csv")
    _prepare_view(con, "comments_src", data_dir / "Comments.csv")

    print("Running rel-stack user-engagement task...", flush=True)

    df_train, target = _build_user_engagement_frame(con, TRAIN_CUT_DATE)
    df_val, target_val = _build_user_engagement_frame(con, VALIDATION_CUT_DATE)
    df_test, target_test = _build_user_engagement_frame(con, TEST_CUT_DATE)

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
        depth=6,
        auto_class_weights="Balanced",
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
