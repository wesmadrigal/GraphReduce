#!/usr/bin/env python
"""Shared helpers for rel-stack task-table examples."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Callable

import duckdb
import pandas as pd
from relbench.base import Table

from graphreduce.enum import ComputeLayerEnum, PeriodUnit, SQLOpType
from graphreduce.graph_reduce import GraphReduce
from graphreduce.models import sqlop
from graphreduce.node import DuckdbNode

from relbench_dataset_utils import (
    RelBenchFrameStore,
    get_relbench_dataset_db,
    get_relbench_task,
    get_relbench_split_task_table,
    get_single_timestamp_task_table,
    iter_training_frames,
    register_relbench_db_views,
)


STACK_TABLE_NAME_TO_FILENAME = {
    "users": "Users.csv",
    "posts": "Posts.csv",
    "badges": "Badges.csv",
    "postHistory": "PostHistory.csv",
    "postLinks": "PostLinks.csv",
    "votes": "Votes.csv",
    "comments": "Comments.csv",
}

def materialize_rel_stack(data_dir: Path | None = None) -> list[str]:
    return []


def prepare_stack_views(con: duckdb.DuckDBPyConnection, data_dir: Path | None = None) -> None:
    _, db = get_relbench_dataset_db("rel-stack", download=True, upto_test_timestamp=False)
    register_relbench_db_views(
        con,
        db,
        {
            "users": "users_src",
            "posts": "posts_src",
            "badges": "badges_src",
            "postHistory": "post_history_src",
            "postLinks": "post_links_src",
            "votes": "votes_src",
            "comments": "comments_src",
        },
    )


def _user_node(cut_date: datetime.datetime) -> DuckdbNode:
    return DuckdbNode(
        fpath="users_src",
        prefix="user",
        pk="Id",
        date_key="CreationDate",
        columns=[
            "Id",
            "DisplayName",
            "Location",
            "ProfileImageUrl",
            "WebsiteUrl",
            "AboutMe",
            "CreationDate",
        ],
        do_filters_ops=[
            sqlop(optype=SQLOpType.where, opval=f"user_CreationDate <= '{cut_date}'"),
            sqlop(optype=SQLOpType.where, opval="user_Id is not null"),
        ],
    )


def _post_node(cut_date: datetime.datetime | None = None) -> DuckdbNode:
    filters = []
    if cut_date is not None:
        filters.extend(
            [
                sqlop(
                    optype=SQLOpType.where,
                    opval=f"post_CreationDate <= '{cut_date}'",
                ),
                sqlop(optype=SQLOpType.where, opval="post_PostTypeId = 1"),
                sqlop(optype=SQLOpType.where, opval="post_OwnerUserId is not null"),
                sqlop(optype=SQLOpType.where, opval="post_OwnerUserId != -1"),
            ]
        )
    return DuckdbNode(
        fpath="posts_src",
        prefix="post",
        pk="Id",
        date_key="CreationDate",
        columns=[
            "Id",
            "OwnerUserId",
            "PostTypeId",
            "ParentId",
            "Title",
            "Tags",
            "Body",
            "CreationDate",
        ],
        do_filters_ops=filters or None,
    )


def _badge_node(prefix: str = "bad") -> DuckdbNode:
    return DuckdbNode(
        fpath="badges_src",
        prefix=prefix,
        pk="Id",
        date_key="Date",
        columns=["Id", "UserId", "Class", "Name", "Date"],
    )


def _vote_node(prefix: str = "vote") -> DuckdbNode:
    return DuckdbNode(
        fpath="votes_src",
        prefix=prefix,
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "PostId", "VoteTypeId", "UserId", "CreationDate"],
    )


def _comment_node(prefix: str = "comm") -> DuckdbNode:
    return DuckdbNode(
        fpath="comments_src",
        prefix=prefix,
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "PostId", "Text", "CreationDate", "UserId", "ContentLicense"],
    )


def _post_history_node() -> DuckdbNode:
    return DuckdbNode(
        fpath="post_history_src",
        prefix="ph",
        pk="Id",
        date_key="CreationDate",
        columns=[
            "Id",
            "PostHistoryTypeId",
            "PostId",
            "RevisionGUID",
            "CreationDate",
            "UserId",
            "Text",
            "Comment",
            "ContentLicense",
        ],
    )


def _post_links_node() -> DuckdbNode:
    return DuckdbNode(
        fpath="post_links_src",
        prefix="plink",
        pk="Id",
        date_key="CreationDate",
        columns=["Id", "CreationDate", "PostId", "RelatedPostId", "LinkTypeId"],
    )


def build_user_badge_features(
    con: duckdb.DuckDBPyConnection,
    cut_date: datetime.datetime,
) -> pd.DataFrame:
    user = _user_node(cut_date)
    post = _post_node()
    badge = _badge_node()
    post_history = _post_history_node()
    post_links = _post_links_node()
    vote_user = _vote_node("voteu")
    comment_user = _comment_node("commu")
    vote_post = _vote_node("votep")
    comment_post = _comment_node("commp")

    gr = GraphReduce(
        name=f"relbench-user-badge-{cut_date.date()}",
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
    for node in [
        user,
        post,
        badge,
        post_history,
        post_links,
        vote_user,
        comment_user,
        vote_post,
        comment_post,
    ]:
        gr.add_node(node)

    gr.add_entity_edge(user, post, parent_key="Id", relation_key="OwnerUserId", reduce=True)
    gr.add_entity_edge(user, vote_user, parent_key="Id", relation_key="UserId", reduce=True)
    gr.add_entity_edge(user, comment_user, parent_key="Id", relation_key="UserId", reduce=True)
    gr.add_entity_edge(user, badge, parent_key="Id", relation_key="UserId", reduce=True)
    gr.add_entity_edge(post, post_history, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(post, post_links, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(post, vote_post, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(post, comment_post, parent_key="Id", relation_key="PostId", reduce=True)

    gr.do_transformations_sql()
    frame = con.sql(f"select * from {gr.parent_node._cur_data_ref}").to_df().copy()
    gr._clean_refs()
    return frame


def build_user_engagement_features(
    con: duckdb.DuckDBPyConnection,
    cut_date: datetime.datetime,
) -> pd.DataFrame:
    user = DuckdbNode(
        fpath="users_src",
        prefix="user",
        pk="Id",
        date_key="CreationDate",
        columns=[
            "Id",
            "DisplayName",
            "Location",
            "ProfileImageUrl",
            "WebsiteUrl",
            "AboutMe",
            "CreationDate",
        ],
        do_filters_ops=[
            sqlop(
                optype=SQLOpType.where,
                opval=f"""(
                    user_Id != -1
                    AND (
                        EXISTS (
                            SELECT 1 FROM posts_src p
                            WHERE p.OwnerUserId = user_Id
                              AND p.CreationDate <= '{cut_date}'
                        )
                        OR EXISTS (
                            SELECT 1 FROM votes_src v
                            WHERE v.UserId = user_Id
                              AND v.CreationDate <= '{cut_date}'
                        )
                        OR EXISTS (
                            SELECT 1 FROM comments_src c
                            WHERE c.UserId = user_Id
                              AND c.CreationDate <= '{cut_date}'
                        )
                    )
                )""",
            )
        ],
    )
    post = _post_node()
    vote = _vote_node("vote")
    comment = _comment_node("comm")
    post_vote = _vote_node("pvote")
    post_comment = _comment_node("pcomm")
    post_comment_user = DuckdbNode(
        fpath="users_src",
        prefix="pcu",
        pk="Id",
        date_key="CreationDate",
        columns=[
            "Id",
            "DisplayName",
            "Location",
            "ProfileImageUrl",
            "WebsiteUrl",
            "AboutMe",
            "CreationDate",
        ],
    )
    post_comment_badge = _badge_node("pcbad")

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
    nodes = [
        user,
        post,
        vote,
        comment,
        post_vote,
        post_comment,
        post_comment_user,
        post_comment_badge,
    ]
    for node in nodes:
        gr.add_node(node)

    gr.add_entity_edge(user, post, parent_key="Id", relation_key="OwnerUserId", reduce=True)
    gr.add_entity_edge(user, vote, parent_key="Id", relation_key="UserId", reduce=True)
    gr.add_entity_edge(user, comment, parent_key="Id", relation_key="UserId", reduce=True)
    gr.add_entity_edge(post, post_vote, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(post, post_comment, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(post_comment, post_comment_user, parent_key="UserId", relation_key="Id", reduce=True)
    gr.add_entity_edge(post_comment_user, post_comment_badge, parent_key="Id", relation_key="UserId", reduce=True)

    gr.do_transformations_sql()
    frame = con.sql(f"select * from {gr.parent_node._cur_data_ref}").to_df().copy()
    gr._clean_refs()
    return frame


def build_post_votes_features(
    con: duckdb.DuckDBPyConnection,
    cut_date: datetime.datetime,
) -> pd.DataFrame:
    post = _post_node(cut_date)
    vote = _vote_node("vote")
    comment = _comment_node("comm")
    post_history = _post_history_node()
    post_links = _post_links_node()
    user = DuckdbNode(
        fpath="users_src",
        prefix="user",
        pk="Id",
        date_key="CreationDate",
        columns=[
            "Id",
            "DisplayName",
            "Location",
            "ProfileImageUrl",
            "WebsiteUrl",
            "AboutMe",
            "CreationDate",
        ],
    )
    badge = _badge_node()

    gr = GraphReduce(
        name=f"relbench-post-votes-{cut_date.date()}",
        parent_node=post,
        compute_layer=ComputeLayerEnum.duckdb,
        sql_client=con,
        cut_date=cut_date,
        compute_period_val=3650,
        compute_period_unit=PeriodUnit.day,
        date_filters_on_agg=True,
        auto_features=True,
        auto_labels=False,
        auto_feature_hops_back=4,
        auto_feature_hops_front=0,
    )
    nodes = [post, vote, comment, post_history, post_links, user, badge]
    for node in nodes:
        gr.add_node(node)

    gr.add_entity_edge(post, vote, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(post, comment, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(post, post_history, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(post, post_links, parent_key="Id", relation_key="PostId", reduce=True)
    gr.add_entity_edge(post, user, parent_key="OwnerUserId", relation_key="Id", reduce=True)
    gr.add_entity_edge(user, badge, parent_key="Id", relation_key="UserId", reduce=True)

    gr.do_transformations_sql()
    frame = con.sql(f"select * from {gr.parent_node._cur_data_ref}").to_df().copy()
    gr._clean_refs()
    return frame


def build_task_split_frame(
    con: duckdb.DuckDBPyConnection,
    task_name: str,
    split: str,
    feature_builder: Callable[[duckdb.DuckDBPyConnection, datetime.datetime], pd.DataFrame],
    feature_entity_col: str,
    use_all_timestamps: bool = False,
    max_timestamps: int | None = None,
) -> tuple[object, Table, pd.DataFrame, pd.Timestamp]:
    if use_all_timestamps:
        task, task_table, cut_timestamps = get_relbench_split_task_table(
            "rel-stack", task_name, split, download=True
        )
        if max_timestamps is not None and len(cut_timestamps) > max_timestamps:
            raise ValueError(
                "Subsampling RelBench timestamps is disabled; use the complete "
                "official task schedule"
            )
        cut_timestamp = pd.Timestamp(cut_timestamps[-1])
    else:
        task, task_table, cut_timestamp = get_single_timestamp_task_table(
            "rel-stack", task_name, split, download=True
        )
        cut_timestamps = [pd.Timestamp(cut_timestamp)]

    frame_store = RelBenchFrameStore(
        f"rel-stack-{task_name}-{split}", persist_each_frame=True
    )
    def build_frame(frame_con: duckdb.DuckDBPyConnection, timestamp: pd.Timestamp) -> pd.DataFrame:
        features = feature_builder(frame_con, timestamp.to_pydatetime())
        task_df = task_table.df.copy()
        task_df = task_df[
            pd.to_datetime(task_df[task.time_col]) == pd.Timestamp(timestamp)
        ]
        task_df[task.entity_col] = pd.to_numeric(task_df[task.entity_col], errors="coerce")
        features[feature_entity_col] = pd.to_numeric(
            features[feature_entity_col], errors="coerce"
        )
        task_df = task_df.dropna(subset=[task.entity_col])
        features = features.dropna(subset=[feature_entity_col])
        task_df[task.entity_col] = task_df[task.entity_col].astype("int64")
        features[feature_entity_col] = features[feature_entity_col].astype("int64")

        return features.merge(
            task_df[[task.time_col, task.entity_col, task.target_col]],
            left_on=feature_entity_col,
            right_on=task.entity_col,
            how="right",
            validate="one_to_one",
        )

    frame_workers = None if split == "train" else 1
    for frame in iter_training_frames(con, cut_timestamps, build_frame, workers=frame_workers):
        frame_store.append(frame)

    return task, task_table, frame_store, cut_timestamp


def target_table_from_frame(task, frame: pd.DataFrame) -> Table:
    return Table(
        df=frame[[task.time_col, task.entity_col, task.target_col]].copy(),
        fkey_col_to_pkey_table={task.entity_col: task.entity_table},
        pkey_col=None,
        time_col=task.time_col,
    )


def select_shared_numeric_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    excluded_cols: set[str],
) -> list[str]:
    return [
        col
        for col in train_df.select_dtypes(include=["number", "bool"]).columns
        if col not in excluded_cols
        and col != target_col
        and "label" not in col.lower()
        and col in val_df.columns
        and col in test_df.columns
    ]
