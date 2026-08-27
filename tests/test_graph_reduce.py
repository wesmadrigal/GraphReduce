#!/usr/bin/env python

import os
import typing
import sqlite3
import datetime
from decimal import Decimal

import pytest
import pandas as pd
from icecream import ic
import duckdb

from graphreduce.node import GraphReduceNode, DynamicNode, SQLNode, DuckdbNode, AthenaNode, RedshiftNode
from graphreduce.graph_reduce import GraphReduce
from graphreduce.enum import ComputeLayerEnum, PeriodUnit, StorageFormatEnum, ProviderEnum, SQLOpType
from graphreduce.models import sqlop


data_path = '/'.join(os.path.abspath(__file__).split('/')[0:-1]) + '/data/cust_data'
print(data_path)


def test_is_ts_data_returns_false_for_empty_sql_relation():
    con = duckdb.connect()
    con.sql(
        """
        CREATE TABLE empty_events (
            evt_id INTEGER,
            evt_parent_id INTEGER,
            evt_created_at TIMESTAMP
        )
        """
    )
    node = DuckdbNode(
        fpath="empty_events",
        prefix="evt",
        pk="id",
        date_key="created_at",
        compute_layer=ComputeLayerEnum.duckdb,
        client=con,
        columns=["id", "parent_id", "created_at"],
    )
    node._cur_data_ref = "empty_events"

    assert node.is_ts_data("parent_id") is False


def test_sql_auto_features_skips_numeric_aggs_for_string_backed_numerical_stype(monkeypatch):
    sample = pd.DataFrame(
        {
            "tran_order_id": [1, 1, 2],
            "tran_id": [10, 11, 12],
            "tran_amount": [Decimal("10.50"), Decimal("2.25"), Decimal("4.00")],
            "tran_source_name": [
                "subscription_contract_checkout_one",
                "web",
                "pos",
            ],
        }
    )
    monkeypatch.setattr(
        "graphreduce.node.infer_df_stype",
        lambda _df: {
            "tran_order_id": "categorical",
            "tran_id": "categorical",
            "tran_amount": "numerical",
            "tran_source_name": "numerical",
        },
    )
    node = SQLNode(
        fpath="transaction",
        pk="id",
        prefix="tran",
        compute_layer=ComputeLayerEnum.redshift,
    )

    ops = node.sql_auto_features(
        table_df_sample=sample,
        reduce_key="order_id",
        type_func_map={
            "numerical": ["median", "mean", "sum", "min", "max"],
            "categorical": ["count"],
        },
    )
    agg_sql = [op.opval for op in ops if op.optype == SQLOpType.aggfunc]

    assert "sum(tran_amount) as tran_amount_sum" in agg_sql
    assert not any("tran_source_name" in op for op in agg_sql)


def test_sql_auto_features_treats_zip_like_numerical_stype_as_categorical(monkeypatch):
    sample = pd.DataFrame(
        {
            "evt_user_id": [1, 1, 2, 2],
            "evt_zip": [94110, 90015, 94110, 85281],
            "evt_lat": [37.7, 34.0, 37.8, 33.4],
        }
    )
    monkeypatch.setattr(
        "graphreduce.node.infer_df_stype",
        lambda _df: {
            "evt_user_id": "categorical",
            "evt_zip": "numerical",
            "evt_lat": "numerical",
        },
    )
    node = SQLNode(
        fpath="events",
        pk="id",
        prefix="evt",
        compute_layer=ComputeLayerEnum.sqlite,
    )

    ops = node.sql_auto_features(
        table_df_sample=sample,
        reduce_key="user_id",
        type_func_map={
            "numerical": ["mean", "sum", "min", "max"],
            "categorical": ["count", "nunique"],
        },
    )
    agg_sql = [op.opval for op in ops if op.optype == SQLOpType.aggfunc]

    assert "COUNT(DISTINCT evt_zip) as evt_zip_nunique" in agg_sql
    assert any("evt_zip_94110_count" in op for op in agg_sql)
    assert any("evt_zip_94110_share" in op for op in agg_sql)
    assert not any("sum(evt_zip)" in op for op in agg_sql)
    assert not any("avg(evt_zip)" in op for op in agg_sql)
    assert "sum(evt_lat) as evt_lat_sum" in agg_sql


def test_sql_auto_features_generates_propagating_categorical_and_text_ops():
    sample = pd.DataFrame(
        {
            "evt_user_id": [1, 1, 2, 2],
            "evt_status": ["paid", "failed", "paid", "pending"],
            "evt_comment_text": [
                "Customer paid online and asked for a receipt.",
                "Payment failed at https://example.com/order/1!",
                "Customer asked whether order 42 shipped?",
                "",
            ],
        }
    )
    node = SQLNode(
        fpath="events",
        pk="id",
        prefix="evt",
        compute_layer=ComputeLayerEnum.sqlite,
    )

    ops = node.sql_auto_features(
        table_df_sample=sample,
        reduce_key="user_id",
        type_func_map={"categorical": ["count", "nunique"]},
    )
    agg_sql = [op.opval for op in ops if op.optype == SQLOpType.aggfunc]

    assert "COUNT(DISTINCT evt_status) as evt_status_nunique" in agg_sql
    assert any("evt_status_paid_count" in op for op in agg_sql)
    assert any("evt_status_paid_share" in op for op in agg_sql)
    assert any("evt_status_paid_any" in op for op in agg_sql)
    assert any("evt_comment_text_length_avg" in op for op in agg_sql)
    assert any("evt_comment_text_empty_share" in op for op in agg_sql)
    assert any("evt_comment_text_url_count" in op for op in agg_sql)
    assert any("evt_comment_text_number_share" in op for op in agg_sql)


def test_sql_auto_features_categorical_text_sql_executes_on_sqlite():
    conn = sqlite3.connect(":memory:")
    rows = pd.DataFrame(
        {
            "evt_user_id": [1, 1, 2, 2],
            "evt_status": ["paid", "failed", "paid", "pending"],
            "evt_comment_text": [
                "Customer paid online and asked for a receipt.",
                "Payment failed at https://example.com/order/1!",
                "Customer asked whether order 42 shipped?",
                "",
            ],
        }
    )
    rows.to_sql("events", conn, index=False)
    node = SQLNode(
        fpath="events",
        pk="id",
        prefix="evt",
        client=conn,
        compute_layer=ComputeLayerEnum.sqlite,
    )

    ops = node.sql_auto_features(
        table_df_sample=rows,
        reduce_key="user_id",
        type_func_map={"categorical": ["count", "nunique"]},
    )
    result = pd.read_sql_query(node.build_query(ops), conn)

    user_1 = result[result["evt_user_id"] == 1].iloc[0]
    assert user_1["evt_status_paid_count"] == 1
    assert user_1["evt_status_paid_any"] == 1
    assert user_1["evt_comment_text_url_count"] == 1
    assert user_1["evt_comment_text_number_count"] == 1
    assert result["evt_comment_text_length_avg"].notna().all()
    conn.close()


def test_sql_auto_annotate_creates_generic_categorical_text_and_gated_numeric_ops():
    sample = pd.DataFrame(
        {
            "evt_user_id": [1, 1, 2, 2],
            "evt_status": ["yes", "invited", "yes", "ignored"],
            "evt_outcome_type": ["Primary", "Secondary", "Primary", "Secondary"],
            "evt_count": [3, 2, 5, 7],
            "evt_p_value": [0.04, 0.2, 0.8, 0.1],
            "evt_notes": [
                "Primary outcome improved by 42 percent.",
                "Invitation sent by email.",
                "See https://example.com/result",
                "",
            ],
        }
    )
    node = SQLNode(
        fpath="events",
        pk="id",
        prefix="evt",
        compute_layer=ComputeLayerEnum.sqlite,
        auto_annotate_features=True,
    )

    ops = node.sql_auto_annotate(sample)
    select_sql = [op.opval for op in ops if op.optype == SQLOpType.select]

    assert (
        "CASE WHEN evt_status = 'yes' THEN 1 ELSE 0 END as evt_status__gr_is_yes"
        in select_sql
    )
    assert (
        "CASE WHEN evt_outcome_type = 'Primary' THEN 1 ELSE 0 END as evt_outcome_type__gr_is_primary"
        in select_sql
    )
    assert (
        "CASE WHEN evt_outcome_type = 'Primary' THEN evt_p_value END as evt_p_value__gr_when_evt_outcome_type_primary"
        in select_sql
    )
    assert "LENGTH(COALESCE(evt_notes, '')) as evt_notes__gr_length" in select_sql
    assert any("evt_notes__gr_has_url" in op for op in select_sql)


def test_graph_parent_date_key_auto_annotates_age_days():
    conn = sqlite3.connect(":memory:")
    rows = pd.DataFrame(
        {
            "user_id": [1, 2],
            "created_at": ["2020-01-01", "2020-01-06"],
        }
    )
    rows.to_sql("users", conn, index=False)
    parent = SQLNode(
        fpath="users",
        pk="id",
        prefix="user",
        date_key="created_at",
        columns=["user_id", "created_at"],
        client=conn,
        compute_layer=ComputeLayerEnum.sqlite,
    )
    graph = GraphReduce(
        parent_node=parent,
        cut_date=datetime.datetime(2020, 1, 11),
        compute_layer=ComputeLayerEnum.sqlite,
        sql_client=conn,
    )
    graph.add_node(parent)
    graph.hydrate_graph_attrs()

    loaded = pd.read_sql_query(parent.build_query(parent.do_data()), conn)
    loaded.to_sql("prefixed_users", conn, index=False)
    parent._cur_data_ref = "prefixed_users"
    annotated = pd.read_sql_query(parent.build_query(parent.do_annotate()), conn)

    assert annotated["user__gr_parent_age_days"].tolist() == [10, 5]
    conn.close()


def test_graph_compute_horizon_is_added_to_each_nodes_ts_periods():
    parent_periods = [7, 30, 365]
    child_periods = [30, 365, 730]
    parent = SQLNode(
        fpath="users",
        pk="id",
        prefix="user",
        ts_periods=parent_periods,
        compute_layer=ComputeLayerEnum.sqlite,
    )
    child = SQLNode(
        fpath="events",
        pk="id",
        prefix="evt",
        ts_periods=child_periods,
        compute_layer=ComputeLayerEnum.sqlite,
    )
    graph = GraphReduce(
        parent_node=parent,
        compute_period_val=520,
        compute_period_unit=PeriodUnit.week,
        compute_layer=ComputeLayerEnum.sqlite,
    )
    graph.add_node(parent)
    graph.add_node(child)
    graph.hydrate_graph_attrs()

    assert parent.ts_periods == [7, 30, 365, 3640]
    assert child.ts_periods == [30, 365, 730, 3640]
    assert parent_periods == [7, 30, 365]
    assert child_periods == [30, 365, 730]


def test_graph_compute_horizon_at_most_one_year_does_not_expand_ts_periods():
    parent = SQLNode(
        fpath="users",
        pk="id",
        prefix="user",
        ts_periods=[7, 30, 365],
        compute_layer=ComputeLayerEnum.sqlite,
    )
    graph = GraphReduce(
        parent_node=parent,
        compute_period_val=365,
        compute_period_unit=PeriodUnit.day,
        compute_layer=ComputeLayerEnum.sqlite,
    )
    graph.add_node(parent)
    graph.hydrate_graph_attrs()

    assert parent.ts_periods == [7, 30, 365]


def test_graph_feature_settings_override_every_node_when_provided():
    parent = SQLNode(
        fpath="users",
        pk="id",
        prefix="user",
        feature_families=("base",),
        feature_family_max_columns=2,
        ts_periods=[7],
        categorical_cardinality_threshold=3,
        categorical_top_k=1,
        auto_text_features=True,
        auto_annotate_features=False,
        auto_annotate_max_categorical_columns=2,
        auto_annotate_max_gated_numeric_cols=1,
        auto_annotate_gated_numeric_top_k=1,
        compute_layer=ComputeLayerEnum.sqlite,
    )
    child = SQLNode(
        fpath="events",
        pk="id",
        prefix="evt",
        feature_families=("episode",),
        feature_family_max_columns=3,
        ts_periods=[30],
        categorical_top_k=2,
        compute_layer=ComputeLayerEnum.sqlite,
    )
    graph = GraphReduce(
        parent_node=parent,
        compute_layer=ComputeLayerEnum.sqlite,
        feature_families=("temporal", "conditional", "temporal"),
        feature_family_max_columns=8,
        ts_periods=(1, 30, 90),
        categorical_cardinality_threshold=12,
        categorical_top_k=5,
        auto_text_features=False,
        auto_annotate_features=True,
        auto_annotate_max_categorical_columns=7,
        auto_annotate_max_gated_numeric_cols=4,
        auto_annotate_gated_numeric_top_k=3,
    )
    graph.add_node(parent)
    graph.add_node(child)

    graph.hydrate_graph_attrs()

    for node in (parent, child):
        assert node.feature_families == ("temporal", "conditional")
        assert node.feature_family_max_columns == 8
        assert node.ts_periods == [1, 30, 90]
        assert node.categorical_cardinality_threshold == 12
        assert node.categorical_top_k == 5
        assert node.auto_text_features is False
        assert node.auto_annotate_features is True
        assert node.auto_annotate_max_categorical_columns == 7
        assert node.auto_annotate_max_gated_numeric_cols == 4
        assert node.auto_annotate_gated_numeric_top_k == 3

    assert parent.ts_periods is not child.ts_periods


def test_graph_omitted_feature_settings_preserve_node_configuration():
    parent = SQLNode(
        fpath="users",
        pk="id",
        prefix="user",
        feature_families=("base", "episode"),
        feature_family_max_columns=2,
        ts_periods=[7],
        categorical_top_k=1,
        auto_text_features=False,
        compute_layer=ComputeLayerEnum.sqlite,
    )
    child = SQLNode(
        fpath="events",
        pk="id",
        prefix="evt",
        feature_families=("base", "temporal"),
        feature_family_max_columns=5,
        ts_periods=[30, 90],
        categorical_top_k=4,
        auto_text_features=True,
        compute_layer=ComputeLayerEnum.sqlite,
    )
    graph = GraphReduce(
        parent_node=parent,
        compute_layer=ComputeLayerEnum.sqlite,
    )
    graph.add_node(parent)
    graph.add_node(child)

    graph.hydrate_graph_attrs()

    assert parent.feature_families == ("base", "episode")
    assert parent.feature_family_max_columns == 2
    assert parent.ts_periods == [7]
    assert parent.categorical_top_k == 1
    assert parent.auto_text_features is False
    assert child.feature_families == ("base", "temporal")
    assert child.feature_family_max_columns == 5
    assert child.ts_periods == [30, 90]
    assert child.categorical_top_k == 4
    assert child.auto_text_features is True


def test_graph_feature_settings_accept_explicit_zero_and_empty_overrides():
    node = SQLNode(
        fpath="events",
        pk="id",
        prefix="evt",
        feature_family_max_columns=5,
        ts_periods=[7, 30],
        categorical_top_k=4,
        compute_layer=ComputeLayerEnum.sqlite,
    )
    graph = GraphReduce(
        parent_node=node,
        compute_layer=ComputeLayerEnum.sqlite,
        feature_family_max_columns=0,
        ts_periods=(),
        categorical_top_k=0,
    )
    graph.add_node(node)

    graph.hydrate_graph_attrs()

    assert node.feature_family_max_columns == 0
    assert node.ts_periods == []
    assert node.categorical_top_k == 0


def test_graph_rejects_unknown_feature_family_override():
    with pytest.raises(ValueError, match="Unknown feature families"):
        GraphReduce(feature_families=("base", "unknown"))


def test_sql_auto_annotate_outputs_feed_existing_sql_auto_features():
    conn = sqlite3.connect(":memory:")
    rows = pd.DataFrame(
        {
            "evt_user_id": [1, 1, 2, 2],
            "evt_status": ["yes", "invited", "yes", "ignored"],
            "evt_outcome_type": ["Primary", "Secondary", "Primary", "Secondary"],
            "evt_count": [3, 2, 5, 7],
            "evt_p_value": [0.04, 0.2, 0.8, 0.1],
            "evt_notes": [
                "Primary outcome improved by 42 percent.",
                "Invitation sent by email.",
                "See https://example.com/result",
                "",
            ],
        }
    )
    rows.to_sql("events", conn, index=False)
    node = SQLNode(
        fpath="events",
        pk="id",
        prefix="evt",
        client=conn,
        compute_layer=ComputeLayerEnum.sqlite,
        auto_annotate_features=True,
    )

    annotate_ops = node.sql_auto_annotate(rows)
    annotated = pd.read_sql_query(node.build_query(annotate_ops), conn)
    annotated.to_sql("annotated_events", conn, index=False)
    feature_ops = node.sql_auto_features(
        annotated,
        reduce_key="user_id",
        type_func_map={
            "numerical": ["mean", "sum", "min", "max"],
            "categorical": ["count", "nunique"],
        },
    )
    feature_sql = [op.opval for op in feature_ops if op.optype == SQLOpType.aggfunc]

    assert "sum(evt_status__gr_is_yes) as evt_status__gr_is_yes_sum" in feature_sql
    assert "avg(evt_status__gr_is_yes) as evt_status__gr_is_yes_avg" in feature_sql
    assert (
        "min(evt_p_value__gr_when_evt_outcome_type_primary) as evt_p_value__gr_when_evt_outcome_type_primary_min"
        in feature_sql
    )
    assert "sum(evt_notes__gr_has_url) as evt_notes__gr_has_url_sum" in feature_sql

    result = pd.read_sql_query(
        node.build_query(feature_ops, data_ref="annotated_events"), conn
    )
    user_1 = result[result["evt_user_id"] == 1].iloc[0]
    assert user_1["evt_status__gr_is_yes_sum"] == 1
    assert user_1["evt_outcome_type__gr_is_primary_sum"] == 1
    assert user_1["evt_p_value__gr_when_evt_outcome_type_primary_min"] == 0.04
    conn.close()


def test_sql_auto_feature_families_generate_temporal_conditionals_and_episodes():
    conn = sqlite3.connect(":memory:")
    rows = pd.DataFrame(
        {
            "evt_id": [1, 2, 3, 4],
            "evt_user_id": [1, 1, 1, 2],
            "evt_status": ["invited", "yes", "invited", "no"],
            "evt_revision_guid": [
                "revision-1",
                "revision-2",
                "revision-3",
                "revision-4",
            ],
            "evt_notes": [
                "A sufficiently long note that should be handled as text.",
                "Another sufficiently long note that should remain text.",
                "A third sufficiently long note that should remain text.",
                "A fourth sufficiently long note that should remain text.",
            ],
            "evt_value": [1.0, 2.0, 3.0, 4.0],
            "evt_ts": [
                "2024-01-01",
                "2024-01-05",
                "2024-01-09",
                "2024-01-08",
            ],
        }
    )
    rows.to_sql("events", conn, index=False)
    node = SQLNode(
        fpath="events",
        pk="id",
        prefix="evt",
        date_key="ts",
        client=conn,
        compute_layer=ComputeLayerEnum.sqlite,
        cut_date=datetime.datetime(2024, 1, 10),
        feature_families=("base", "conditional", "temporal", "episode"),
        annotation_expressions={"is_invited": "{status} = 'invited'"},
        auto_annotate_features=True,
        annotation_expressions_only=True,
        ts_periods=[1, 7, 30],
    )
    node._cur_data_ref = "events"

    annotated = pd.read_sql_query(
        node.build_query(node.sql_auto_annotate(rows)), conn
    )
    annotated.to_sql("annotated_events", conn, index=False)
    feature_ops = node.sql_auto_features(
        annotated,
        reduce_key="user_id",
        type_func_map={
            "categorical": ["count", "nunique"],
            "numerical": ["sum", "mean", "min", "max"],
        },
    )
    feature_sql = [op.opval for op in feature_ops if op.optype == SQLOpType.aggfunc]

    assert any("evt_gr_is_invited_count_7d" in op for op in feature_sql)
    assert any("evt_gr_is_invited_share_7d" in op for op in feature_sql)
    assert any("evt_num_episodes_7d" in op for op in feature_sql)
    assert any("evt_num_unique_episodes_7d" in op for op in feature_sql)
    assert not any("revision_guid" in op for op in feature_sql)
    assert not any("notes_" in op and "_count_7d" in op for op in feature_sql)

    result = pd.read_sql_query(
        node.build_query(feature_ops, data_ref="annotated_events"), conn
    )
    user_1 = result[result["evt_user_id"] == 1].iloc[0]
    assert user_1["evt_gr_is_invited_count_7d"] == 1
    assert user_1["evt_gr_is_invited_share_7d"] == 0.5
    assert user_1["evt_num_episodes_7d"] == 2
    conn.close()


def test_sql_auto_annotate_can_run_semantic_expressions_only():
    sample = pd.DataFrame(
        {
            "res_statusId": [1, 11],
            "res_position": [1, 12],
        }
    )
    node = SQLNode(
        fpath="results",
        pk="id",
        prefix="res",
        compute_layer=ComputeLayerEnum.sqlite,
        auto_annotate_features=True,
        annotation_expressions={
            "did_not_finish": "{statusId} <> 1",
            "is_top3": "{position} <= 3",
        },
        annotation_expressions_only=True,
    )

    select_sql = [op.opval for op in node.sql_auto_annotate(sample) if op.optype == SQLOpType.select]
    assert "CASE WHEN res_statusId <> 1 THEN 1 ELSE 0 END as res__gr_did_not_finish" in select_sql
    assert "CASE WHEN res_position <= 3 THEN 1 ELSE 0 END as res__gr_is_top3" in select_sql
    assert not any("res_statusId__gr_is_" in op for op in select_sql)


def test_sql_auto_feature_families_keep_value_annotations_out_of_predicates():
    sample = pd.DataFrame(
        {
            "evt_id": [1, 2],
            "evt_user_id": [1, 1],
            "evt_status": ["serious", "other"],
            "evt_subjects_affected": [4, 7],
            "evt_date": ["2024-01-01", "2024-01-02"],
        }
    )
    conn = sqlite3.connect(":memory:")
    sample.to_sql("events", conn, index=False)
    node = SQLNode(
        fpath="events",
        pk="id",
        prefix="evt",
        date_key="date",
        client=conn,
        compute_layer=ComputeLayerEnum.sqlite,
        cut_date=datetime.datetime(2024, 1, 3),
        feature_families=("conditional", "temporal"),
        annotation_expressions={
            "is_serious": "{status} = 'serious'",
            "serious_subjects_affected": (
                "value",
                "CASE WHEN {status} = 'serious' THEN {subjects_affected} ELSE 0 END",
            ),
        },
        auto_annotate_features=True,
        annotation_expressions_only=True,
        ts_periods=[7],
    )
    annotated = pd.read_sql_query(
        node.build_query(node.sql_auto_annotate(sample), data_ref="events"),
        conn,
    )
    annotated.to_sql("annotated_events", conn, index=False)
    node._cur_data_ref = "annotated_events"
    # The assertion below is planner-level: value annotations are temporal
    # inputs, while only predicate annotations receive conditional features.
    ops = node.sql_auto_features(
        annotated,
        reduce_key="user_id",
        type_func_map={"numerical": ["sum"]},
    )
    feature_sql = [op.opval for op in ops if op.optype == SQLOpType.aggfunc]
    assert any(
        "evt_gr_value_serious_subjects_affected_sum_7d" in op
        for op in feature_sql
    )
    assert not any(
        "value_serious_subjects_affected_count_7d" in op for op in feature_sql
    )
    conn.close()


def test_sql_auto_semantic_family_compiles_configured_annotations():
    sample = pd.DataFrame(
        {
            "res_resultId": [1, 2, 3],
            "res_position": [1, 4, 2],
        }
    )
    node = SQLNode(
        fpath="results",
        pk="resultId",
        prefix="res",
        date_key="date",
        compute_layer=ComputeLayerEnum.sqlite,
        feature_families=("semantic",),
        annotation_expressions={"is_top3": "{position} <= 3"},
    )

    select_sql = [
        op.opval for op in node.sql_auto_annotate(sample) if op.optype == SQLOpType.select
    ]

    assert "CASE WHEN res_position <= 3 THEN 1 ELSE 0 END as res__gr_is_top3" in select_sql
    assert not any("res_position_top3" in op for op in select_sql)


def test_sql_auto_sequence_family_generates_rates_and_activity_span():
    conn = sqlite3.connect(":memory:")
    sample = pd.DataFrame(
        {
            "evt_id": [1, 2, 3],
            "evt_user_id": [1, 1, 1],
            "evt_ts": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-09"]),
            "evt_value": [1.0, 2.0, 3.0],
        }
    )
    sample.to_sql("events", conn, index=False)
    node = SQLNode(
        fpath="events",
        pk="id",
        prefix="evt",
        date_key="ts",
        client=conn,
        compute_layer=ComputeLayerEnum.sqlite,
        cut_date=datetime.datetime(2024, 1, 10),
        feature_families=("sequence",),
        ts_periods=[1, 7, 30],
    )
    node._cur_data_ref = "events"

    feature_ops = node.sql_auto_features(
        sample,
        reduce_key="user_id",
        type_func_map={"numerical": ["sum", "mean", "min", "max"]},
    )
    feature_sql = [op.opval for op in feature_ops if op.optype == SQLOpType.aggfunc]
    assert any("evt_activity_rate_7d" in op for op in feature_sql)
    assert any("evt_activity_share_7d" in op for op in feature_sql)
    assert any("evt_activity_burst_1v7" in op for op in feature_sql)
    assert any("evt_active_span_seconds" in op for op in feature_sql)

    result = pd.read_sql_query(node.build_query(feature_ops), conn)
    assert result.loc[0, "evt_activity_rate_7d"] > 0
    assert result.loc[0, "evt_active_span_seconds"] > 0
    conn.close()


def test_sql_auto_context_family_preserves_peer_relative_numeric_signal():
    conn = sqlite3.connect(":memory:")
    sample = pd.DataFrame(
        {
            "res_resultId": [1, 2, 3, 4],
            "res_driverId": [10, 10, 11, 11],
            "res_raceId": [100, 100, 100, 101],
            "res_position": [1, 4, 2, 8],
            "res_date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-05"]
            ),
        }
    )
    sample.to_sql("results", conn, index=False)
    node = SQLNode(
        fpath="results",
        pk="resultId",
        prefix="res",
        client=conn,
        compute_layer=ComputeLayerEnum.sqlite,
        feature_families=("context",),
        context_keys=("raceId",),
    )

    annotated = pd.read_sql_query(
        node.build_query(node.sql_auto_annotate(sample), data_ref="results"), conn
    )
    assert "res_raceId__gr_context_size" in annotated.columns
    assert "res_position__gr_context_res_raceid_delta" in annotated.columns
    assert annotated.loc[0, "res_raceId__gr_context_size"] == 3
    assert annotated.loc[0, "res_position__gr_context_res_raceid_delta"] == pytest.approx(-4 / 3)
    conn.close()


def test_sql_auto_context_family_does_not_infer_peer_keys():
    sample = pd.DataFrame(
        {
            "res_id": [1, 2],
            "res_group_id": [10, 10],
            "res_value": [1.0, 3.0],
        }
    )
    node = SQLNode(
        fpath="results",
        pk="id",
        prefix="res",
        compute_layer=ComputeLayerEnum.sqlite,
        feature_families=("context",),
    )

    assert node.sql_auto_annotate(sample) == []


def test_custom_node_definition():
    class CustNode(GraphReduceNode):
        def do_filters(self):
            pass
        def do_annotate(self):
            pass
        def do_normalize(self):
            pass
        def do_reduce(self, reduce_key):
            pass
        def do_labels(self, reduce_key):
            pass
        def do_post_join_annotate(self):
            pass
    cust = CustNode(
            fpath=os.path.join(data_path, 'cust.csv'),
            fmt = 'csv',
            prefix='cust',
            compute_layer=ComputeLayerEnum.pandas,
            date_key=None
            )
    assert isinstance(cust, CustNode)


def test_custom_node_graph():
    class CustNode(GraphReduceNode):
        def do_filters(self):
            pass
        def do_annotate(self):
            pass
        def do_normalize(self):
            pass
        def do_reduce(self, reduce_key):
            pass
        def do_labels(self, reduce_key):
            pass
        def do_post_join_annotate(self):
            pass
    cust = CustNode(
            fpath=os.path.join(data_path, 'cust.csv'),
            fmt = 'csv',
            prefix='cust',
            compute_layer=ComputeLayerEnum.pandas,
            date_key=None
            )
    class OrderNode(GraphReduceNode):
        def do_filters(self):
            pass
        def do_annotate(self):
            pass
        def do_normalize(self):
            pass
        def do_reduce(self, reduce_key):
            return self.prep_for_features().groupby(self.colabbr(reduce_key)).agg(**{
                self.colabbr("num_orders"): pd.NamedAgg(column=self.colabbr(self.pk), aggfunc="count")
                }).reset_index()
        def do_labels(self, reduce_key):
            pass
        def do_post_join_annotate(self):
            pass
    order = OrderNode(
            fpath=os.path.join(data_path, 'orders.csv'),
            pk='id',
            fmt = 'csv',
            prefix='ord',
            compute_layer=ComputeLayerEnum.pandas,
            date_key='ts'
            )
    gr = GraphReduce(
            name='test_two_node_graph',
            parent_node=cust,
            compute_layer=ComputeLayerEnum.pandas,
            cut_date=datetime.datetime(2023, 6, 1)
            )
    ic(gr.params)
    gr.add_entity_edge(
            parent_node=cust,
            relation_node=order,
            parent_key='id',
            relation_key='customer_id',
            relation_type='parent_child',
            reduce=True,
            )
    gr.do_transformations()
    ic(gr.parent_node.df.head())
    assert len(gr.parent_node.df) == 4




def test_dynamic_node_instance():
    node = DynamicNode(
            fpath=os.path.join(data_path, 'cust.csv'),
            fmt='csv',
            prefix='cust',
            compute_layer=ComputeLayerEnum.pandas,
            date_key=None
            )
    assert isinstance(node, DynamicNode)


def test_get_data():
    node = DynamicNode(
            fpath=os.path.join(data_path, 'cust.csv'),
            fmt='csv',
            prefix='cust',
            compute_layer=ComputeLayerEnum.pandas,
            date_key=None
            )
    node.do_data()
    print(node.df)
    assert len(node.df) == 4


def test_filter_data():
    node = DynamicNode(
            fpath=os.path.join(data_path, 'cust.csv'),
            fmt='csv',
            prefix='cust',
            compute_layer=ComputeLayerEnum.pandas,
            date_key=None
            )
    node.do_data()
    node.do_filters()
    print(node.df)
    assert len(node.df) == 4


def test_multi_node_customer():

    cust_node = DynamicNode(
            fpath=os.path.join(data_path, 'cust.csv'),
            fmt='csv',
            prefix='cust',
            date_key=None,
            pk='id',
            )

    orders_node = DynamicNode(
            fpath=os.path.join(data_path, 'orders.csv'),
            fmt='csv',
            prefix='ord',
            date_key='ts',
            pk='id',
            )

    nots_node = DynamicNode(
            fpath=os.path.join(data_path, 'notifications.csv'),
            fmt='csv',
            prefix='not',
            date_key='ts',
            pk='id',
            )

    gr = GraphReduce(
            parent_node=cust_node,
            fmt='csv',
            compute_layer=ComputeLayerEnum.pandas,
            auto_features=True,
            auto_labels=True,
            cut_date=datetime.datetime(2023, 6, 30),
        # Feature parameters.
        compute_period_unit=PeriodUnit.day,
        compute_period_val=365,
        # Label parameters.
        label_node=orders_node,
        label_field='id',
        label_operation='count',
        label_period_unit=PeriodUnit.day,
        label_period_val=30,
        auto_feature_hops_back=3,
        auto_feature_hops_front=0
            )
    gr.add_node(cust_node)
    gr.add_node(orders_node)
    gr.add_node(nots_node)

    assert len(gr) == 3

    gr.add_entity_edge(
            parent_node=cust_node,
            relation_node=orders_node,
            parent_key='id',
            relation_key='customer_id',
            relation_type='parent_child',
            reduce=True
            )
    gr.add_entity_edge(
            parent_node=cust_node,
            relation_node=nots_node,
            parent_key='id',
            relation_key='customer_id',
            relation_type='parent_child',
            reduce=True
            )

    gr.do_transformations()
    assert len(gr.parent_node.df) == 4



def test_multi_node():

    cust_node = DynamicNode(
            fpath=os.path.join(data_path, 'cust.csv'),
            fmt='csv',
            prefix='cust',
            date_key=None,
            pk='customer_id',
            )
    ord_node = DynamicNode(
            fpath=os.path.join(data_path, 'orders.csv'),
            fmt='csv',
            prefix='ord',
            date_key='ts',
            pk='id',
            )
    not_node = DynamicNode(
            fpath=os.path.join(data_path, 'notifications.csv'),
            fmt='csv',
            prefix='not',
            date_key='ts',
            pk='id'
            )
    ni_node = DynamicNode(
            fpath=os.path.join(data_path, 'notification_interactions.csv'),
            fmt='csv',
            prefix='ni',
            date_key='ts',
            pk='id'
            )

    gr = GraphReduce(
            parent_node=cust_node,
            fmt='csv',
            compute_layer=ComputeLayerEnum.pandas,
            auto_features=True,
            auto_labels=True,
            cut_date=datetime.datetime(2023, 6, 30),
        # Feature parameters.
        compute_period_unit=PeriodUnit.day,
        compute_period_val=365,
        # Label parameters.
        label_node=ord_node,
        label_field='id',
        label_operation='count',
        label_period_unit=PeriodUnit.day,
        label_period_val=30,
        auto_feature_hops_back=3,
        auto_feature_hops_front=1
            )
    gr.add_node(cust_node)
    gr.add_node(ord_node)
    gr.add_node(not_node)
    gr.add_node(ni_node)

    assert len(gr) == 4

    gr.add_entity_edge(
            parent_node=cust_node,
            relation_node=ord_node,
            parent_key='id',
            relation_key='customer_id',
            relation_type='parent_child',
            reduce=True
            )
    gr.add_entity_edge(
            parent_node=cust_node,
            relation_node=not_node,
            parent_key='id',
            relation_key='customer_id',
            relation_type='parent_child',
            reduce=True
            )
    gr.add_entity_edge(
            parent_node=not_node,
            relation_node=ni_node,
            parent_key='id',
            relation_key='notification_id',
            relation_type='parent_child',
            reduce=True
            )

    gr.do_transformations()
    assert len(gr.parent_node.df) == 4



def _setup_sqlite():
    conn = sqlite3.connect(":memory:")
    files = [x for x in os.listdir(data_path) if x.endswith('.csv')]
    for f in files:
        df = pd.read_csv(f"{data_path}/{f}")
        name = f.split('.')[0]
        df.to_sql(name, conn, if_exists='replace', index=False)
    return conn

def _teardown_sqlite(conn):
    try:
        conn.close()
    except Exception as e:
        ic(e)


class CustNode(SQLNode):
    def do_annotate(self) -> typing.Union[sqlop, typing.List[sqlop]]:
        return [
            sqlop(optype=SQLOpType.select, opval=f"*, LENGTH({self.colabbr('name')}) as {self.colabbr('name_length')}")
        ]

    def do_filters(self) -> typing.Union[sqlop, typing.List[sqlop]]:
        return [
            sqlop(optype=SQLOpType.where, opval=f"{self.colabbr('id')} < 3")
        ]

    def do_normalize(self):
        pass


    def do_reduce(self, reduce_key):
        pass

    def do_post_join_annotate(self):
        pass

    def do_post_join_filters(self):
        pass


class NotificationNode(SQLNode):
    def do_annotate(self) -> typing.List[sqlop]:
        return [
            sqlop(optype=SQLOpType.select, opval=f"*, strftime('%m', {self.colabbr('ts')})")
        ]

    def do_filters(self) -> typing.List[sqlop]:
        return [
            sqlop(optype=SQLOpType.where, opval=f"{self.colabbr('ts')} > '2022-06-01'")
        ]

    def do_normalize(self):
        pass

    def do_reduce(self, reduce_key):
        return [
            # Shouldn't this just be a select?
            sqlop(optype=SQLOpType.aggfunc, opval=f"count(*) as {self.colabbr('num_notifications')}"),
            sqlop(optype=SQLOpType.agg, opval=f"{self.colabbr(reduce_key)}"),
        ]




def test_sql_node_definition():
    conn = _setup_sqlite()
    cust = CustNode(fpath='cust',
                prefix='cust',
                client=conn,
                compute_layer=ComputeLayerEnum.sqlite,
                columns=['id','name'])
    notif = NotificationNode(fpath='notifications',
                prefix='not',
                client=conn,
                compute_layer=ComputeLayerEnum.sqlite,
                columns=['id', 'customer_id', 'ts'],
                date_key='ts'
        )
    _teardown_sqlite(conn)
    assert isinstance(cust, CustNode)



def test_sql_graph_transform():
    conn = _setup_sqlite()
    cust = CustNode(fpath='cust',
                prefix='cust',
                client=conn,
                compute_layer=ComputeLayerEnum.sqlite,
                columns=['id','name'])
    notif = NotificationNode(fpath='notifications',
                prefix='not',
                client=conn,
                compute_layer=ComputeLayerEnum.sqlite,
                columns=['id', 'customer_id', 'ts'],
                date_key='ts'
        )
    gr = GraphReduce(
            name='sql_dialect_example',
            parent_node=cust,
            compute_layer=ComputeLayerEnum.sqlite,
            use_temp_tables=True,
            lazy_execution=False,
            sql_client=conn,
            )
    gr.add_node(cust)
    gr.add_node(notif)
    gr.add_entity_edge(cust, notif, parent_key='id', relation_key='customer_id', reduce=True)
    gr.do_transformations_sql()
    _teardown_sqlite(conn)
    assert len(gr) == 2



def test_sql_graph_auto_fe():
    conn = _setup_sqlite()
    cust = SQLNode(fpath='cust',
                pk='id',
                prefix='cust',
                client=conn,
                compute_layer=ComputeLayerEnum.sqlite,
                columns=['id','name'])

    notif = SQLNode(fpath='notifications',
                    prefix='not',
                    pk='id',
                    client=conn,
                    compute_layer=ComputeLayerEnum.sqlite,
                    columns=['id','customer_id','ts'],
                    date_key='ts')

    ni = SQLNode(fpath='notification_interactions',
                    prefix='ni',
                    pk='id',
                    client=conn,
                    compute_layer=ComputeLayerEnum.sqlite,
                    columns=['id','notification_id','interaction_type_id','ts'],
                    date_key='ts')

    order = SQLNode(fpath='orders',
                   pk='id',
                   prefix='ord',
                   client=conn,
                   compute_layer=ComputeLayerEnum.sqlite,
                   columns=['id','customer_id','ts','amount','type','is_online','is_store'],
                    date_key='ts')

    gr = GraphReduce(
        name='sql_autofe',
        parent_node=cust,
        # Cut date for filtering.
        cut_date=datetime.datetime(2023, 6, 30),
        # Feature parameters.
        compute_period_unit=PeriodUnit.day,
        compute_period_val=730,
        # Label parameters.
        label_node=order,
        label_field='id',
        label_operation='bool',
        label_period_unit=PeriodUnit.day,
        label_period_val=90,
        compute_layer=ComputeLayerEnum.sqlite,
        use_temp_tables=True,
        lazy_execution=False,
        # Auto feature engineering params.
        auto_features=True,
        auto_feature_hops_back=3,
        auto_feature_hops_front=1,
        sql_client=conn,
        dry=False
    )
    gr.add_node(cust)
    gr.add_node(order)
    gr.add_node(notif)
    gr.add_node(ni)

    gr.add_entity_edge(
        cust,
        notif,
        parent_key='id',
        relation_key='customer_id',
        reduce=True
    )

    gr.add_entity_edge(
        notif,
        ni,
        parent_key='id',
        relation_key='notification_id',
        reduce=True
    )

    gr.add_entity_edge(
        cust,
        order,
        parent_key='id',
        relation_key='customer_id',
        reduce=True
    )
    gr.plot_graph('cust_graph.html')
    gr.do_transformations_sql()
    for node in gr.nodes():
        print(node._temp_refs)
    d = pd.read_sql_query(f"select * from {gr.parent_node._cur_data_ref}", conn)
    d.to_csv('sql_df_out.csv')
    ic(d.columns)
    ic(d)
    _teardown_sqlite(conn)
    assert len(d) == 4


def test_train_false_skips_custom_labels_pandas():
    class ScoreCustNode(GraphReduceNode):
        def do_filters(self):
            pass
        def do_annotate(self):
            pass
        def do_normalize(self):
            pass
        def do_reduce(self, reduce_key):
            pass
        def do_labels(self, reduce_key):
            pass
        def do_post_join_annotate(self):
            pass

    class ScoreOrderNode(GraphReduceNode):
        def do_filters(self):
            pass
        def do_annotate(self):
            pass
        def do_normalize(self):
            pass
        def do_reduce(self, reduce_key):
            return self.prep_for_features().groupby(self.colabbr(reduce_key)).agg(**{
                self.colabbr("num_orders"): pd.NamedAgg(column=self.colabbr(self.pk), aggfunc="count")
            }).reset_index()
        def do_labels(self, reduce_key):
            raise AssertionError("do_labels should not run when train=False")
        def do_post_join_annotate(self):
            pass

    cust = ScoreCustNode(
        fpath=os.path.join(data_path, 'cust.csv'),
        fmt='csv',
        pk='id',
        prefix='cust',
        compute_layer=ComputeLayerEnum.pandas,
        date_key=None,
    )
    orders = ScoreOrderNode(
        fpath=os.path.join(data_path, 'orders.csv'),
        fmt='csv',
        pk='id',
        prefix='ord',
        compute_layer=ComputeLayerEnum.pandas,
        date_key='ts',
    )

    gr = GraphReduce(
        name='score_without_labels_pandas',
        parent_node=cust,
        compute_layer=ComputeLayerEnum.pandas,
        cut_date=datetime.datetime(2023, 6, 30),
        label_node=orders,
        label_field='id',
        label_operation='count',
        label_period_unit=PeriodUnit.day,
        label_period_val=30,
        train=False,
    )
    gr.add_node(cust)
    gr.add_node(orders)
    gr.add_entity_edge(
        parent_node=cust,
        relation_node=orders,
        parent_key='id',
        relation_key='customer_id',
        relation_type='parent_child',
        reduce=True,
    )

    gr.do_transformations()

    assert len(gr.parent_node.df) == 4
    assert not any("label" in col for col in gr.parent_node.df.columns)


def test_sql_op_execution_log_by_method():
    conn = _setup_sqlite()
    order = SQLNode(
        fpath='orders',
        pk='id',
        prefix='ord',
        client=conn,
        compute_layer=ComputeLayerEnum.sqlite,
        columns=['id', 'customer_id', 'ts', 'amount'],
        date_key='ts',
    )

    cust = SQLNode(
        fpath='cust',
        pk='id',
        prefix='cust',
        client=conn,
        compute_layer=ComputeLayerEnum.sqlite,
        columns=['id', 'name'],
        do_post_join_annotate_ops=[
            sqlop(optype=SQLOpType.select, opval="*"),
            sqlop(optype=SQLOpType.select, opval="case when ord_id_label > 0 then 1 else 0 end as has_order_label"),
        ],
        do_post_join_filters_ops=[
            sqlop(optype=SQLOpType.where, opval="cust_id >= 1"),
        ],
        do_post_join_annotate_requires=[order],
    )

    notif = SQLNode(
        fpath='notifications',
        prefix='not',
        pk='id',
        client=conn,
        compute_layer=ComputeLayerEnum.sqlite,
        columns=['id', 'customer_id', 'ts'],
        date_key='ts',
        do_reduce_ops=[
            sqlop(optype=SQLOpType.aggfunc, opval="count(*) as not_num_notifications"),
            sqlop(optype=SQLOpType.agg, opval="not_customer_id"),
        ],
    )

    gr = GraphReduce(
        name='sql_op_execution_log',
        parent_node=cust,
        cut_date=datetime.datetime(2023, 6, 30),
        compute_period_unit=PeriodUnit.day,
        compute_period_val=730,
        label_node=order,
        label_field='id',
        label_operation='bool',
        label_period_unit=PeriodUnit.day,
        label_period_val=90,
        compute_layer=ComputeLayerEnum.sqlite,
        use_temp_tables=True,
        lazy_execution=False,
        auto_features=True,
        auto_feature_hops_back=3,
        auto_feature_hops_front=0,
        sql_client=conn,
        date_filters_on_agg=True,
    )

    gr.add_node(cust)
    gr.add_node(order)
    gr.add_node(notif)

    gr.add_entity_edge(
        cust,
        notif,
        parent_key='id',
        relation_key='customer_id',
        reduce=True
    )
    gr.add_entity_edge(
        cust,
        order,
        parent_key='id',
        relation_key='customer_id',
        reduce=True
    )

    gr.do_transformations_sql()

    sql_ops_by_method = gr.get_executed_sqlops_by_method(exclude_date_filters=True)
    ic({
        method: [f"{op.optype.value}:{op.opval}" for op in ops]
        for method, ops in sql_ops_by_method.items()
    })

    assert "do_data" in sql_ops_by_method
    assert "do_reduce" in sql_ops_by_method
    assert "do_labels" in sql_ops_by_method
    assert "do_post_join_annotate" in sql_ops_by_method
    assert "do_post_join_filters" in sql_ops_by_method
    assert all(
        isinstance(op, sqlop)
        for ops in sql_ops_by_method.values()
        for op in ops
    )
    assert not any(
        op.optype == SQLOpType.where and "2023-06-30" in op.opval
        for op in sql_ops_by_method["do_reduce"]
    )

    _teardown_sqlite(conn)


def test_train_false_skips_custom_labels_sql():
    conn = _setup_sqlite()

    class ScoreOrderNode(SQLNode):
        def do_labels(self, reduce_key):
            raise AssertionError("do_labels should not run when train=False")

    cust = SQLNode(
        fpath='cust',
        pk='id',
        prefix='cust',
        client=conn,
        compute_layer=ComputeLayerEnum.sqlite,
        columns=['id', 'name'],
    )
    orders = ScoreOrderNode(
        fpath='orders',
        pk='id',
        prefix='ord',
        client=conn,
        compute_layer=ComputeLayerEnum.sqlite,
        columns=['id', 'customer_id', 'ts', 'amount'],
        date_key='ts',
    )

    gr = GraphReduce(
        name='score_without_labels_sql',
        parent_node=cust,
        cut_date=datetime.datetime(2023, 6, 30),
        compute_period_unit=PeriodUnit.day,
        compute_period_val=730,
        label_node=orders,
        label_field='id',
        label_operation='bool',
        label_period_unit=PeriodUnit.day,
        label_period_val=90,
        compute_layer=ComputeLayerEnum.sqlite,
        use_temp_tables=True,
        lazy_execution=False,
        sql_client=conn,
        train=False,
    )
    gr.add_node(cust)
    gr.add_node(orders)
    gr.add_entity_edge(cust, orders, parent_key='id', relation_key='customer_id', reduce=True)

    gr.do_transformations_sql()

    sql_ops_by_method = gr.get_executed_sqlops_by_method()
    assert "do_labels" not in sql_ops_by_method

    score_df = pd.read_sql_query(
        f"select * from {gr.parent_node._cur_data_ref}",
        conn,
    )
    assert len(score_df) == 10
    assert not any("label" in col for col in score_df.columns)

    _teardown_sqlite(conn)


def test_athena_execute_query_raises_on_failed_state():
    class FailedAthenaClient:
        def start_query_execution(self, QueryString, ResultConfiguration):
            return {"QueryExecutionId": "bad-query-id"}

        def get_query_execution(self, QueryExecutionId):
            return {
                "QueryExecution": {
                    "Status": {
                        "State": "FAILED",
                        "StateChangeReason": "missing column",
                    }
                }
            }

    node = AthenaNode(
        fpath="some_table",
        prefix="ath",
        pk="id",
        client=FailedAthenaClient(),
        s3_output_location="s3://bucket/results/",
        columns=["id"],
    )

    with pytest.raises(Exception, match="Query select missing_column FAILED: missing column"):
        node.execute_query("select missing_column")


def test_redshift_execute_query_raises_cursor_errors():
    class FailedCursor:
        def execute(self, qry):
            raise RuntimeError("redshift syntax error")

    class FailedRedshiftClient:
        def cursor(self):
            return FailedCursor()

    node = RedshiftNode(
        fpath="some_table",
        prefix="rs",
        pk="id",
        client=FailedRedshiftClient(),
        columns=["id"],
    )

    with pytest.raises(RuntimeError, match="redshift syntax error"):
        node.execute_query("select missing_column", ret_df=False, commit=True)


def test_apply_frozen_execution_plan():
    original_train_conn = _setup_sqlite()
    train_cut_date = datetime.datetime(2023, 6, 30)

    original_train_order = SQLNode(
        fpath='orders',
        pk='id',
        prefix='ord',
        client=original_train_conn,
        compute_layer=ComputeLayerEnum.sqlite,
        columns=['id', 'customer_id', 'ts', 'amount'],
        date_key='ts',
    )

    original_train_cust = SQLNode(
        fpath='cust',
        pk='id',
        prefix='cust',
        client=original_train_conn,
        compute_layer=ComputeLayerEnum.sqlite,
        columns=['id', 'name'],
        do_post_join_annotate_ops=[
            sqlop(optype=SQLOpType.select, opval="*"),
            sqlop(optype=SQLOpType.select, opval="case when ord_id_label > 0 then 1 else 0 end as has_order_label"),
        ],
        do_post_join_annotate_requires=[original_train_order],
    )
    original_train_notif = SQLNode(
        fpath='notifications',
        prefix='not',
        pk='id',
        client=original_train_conn,
        compute_layer=ComputeLayerEnum.sqlite,
        columns=['id', 'customer_id', 'ts'],
        date_key='ts',
    )

    original_train_gr = GraphReduce(
        name='sql_freeze_original_train',
        parent_node=original_train_cust,
        cut_date=train_cut_date,
        compute_period_unit=PeriodUnit.day,
        compute_period_val=730,
        label_node=original_train_order,
        label_field='id',
        label_operation='bool',
        label_period_unit=PeriodUnit.day,
        label_period_val=90,
        compute_layer=ComputeLayerEnum.sqlite,
        use_temp_tables=True,
        lazy_execution=False,
        auto_features=True,
        auto_feature_hops_back=3,
        auto_feature_hops_front=0,
        sql_client=original_train_conn,
        date_filters_on_agg=True,
        train=True,
    )

    original_train_gr.add_node(original_train_cust)
    original_train_gr.add_node(original_train_order)
    original_train_gr.add_node(original_train_notif)
    original_train_gr.add_entity_edge(original_train_cust, original_train_notif, parent_key='id', relation_key='customer_id', reduce=True)
    original_train_gr.add_entity_edge(original_train_cust, original_train_order, parent_key='id', relation_key='customer_id', reduce=True)
    original_train_gr.do_transformations_sql()
    original_train_df = pd.read_sql_query(
        f"select * from {original_train_gr.parent_node._cur_data_ref}",
        original_train_conn,
    ).sort_values("cust_id").reset_index(drop=True)
    frozen_plan = original_train_gr.freeze_execution_plan()
    ic({
        method: [f"{op.optype.value}:{op.opval}" for op in ops]
        for method, ops in frozen_plan["ops_by_method"].items()
    })
    _teardown_sqlite(original_train_conn)

    replay_train_conn = _setup_sqlite()
    replay_train_order = SQLNode(
        fpath='orders',
        pk='id',
        prefix='ord',
        client=replay_train_conn,
        compute_layer=ComputeLayerEnum.sqlite,
        columns=['id', 'customer_id', 'ts', 'amount'],
        date_key='ts',
    )
    replay_train_cust = SQLNode(
        fpath='cust',
        pk='id',
        prefix='cust',
        client=replay_train_conn,
        compute_layer=ComputeLayerEnum.sqlite,
        columns=['id', 'name'],
        do_post_join_annotate_requires=[replay_train_order],
    )
    replay_train_notif = SQLNode(
        fpath='notifications',
        prefix='not',
        pk='id',
        client=replay_train_conn,
        compute_layer=ComputeLayerEnum.sqlite,
        columns=['id', 'customer_id', 'ts'],
        date_key='ts',
    )

    replay_train_gr = GraphReduce(
        name='sql_freeze_replay_train',
        parent_node=replay_train_cust,
        cut_date=train_cut_date,
        compute_period_unit=PeriodUnit.day,
        compute_period_val=730,
        label_node=replay_train_order,
        label_field='id',
        label_operation='bool',
        label_period_unit=PeriodUnit.day,
        label_period_val=90,
        compute_layer=ComputeLayerEnum.sqlite,
        use_temp_tables=True,
        lazy_execution=False,
        auto_features=True,
        auto_feature_hops_back=3,
        auto_feature_hops_front=0,
        sql_client=replay_train_conn,
        date_filters_on_agg=True,
        train=True,
    )
    replay_train_gr.add_node(replay_train_cust)
    replay_train_gr.add_node(replay_train_order)
    replay_train_gr.add_node(replay_train_notif)
    replay_train_gr.add_entity_edge(replay_train_cust, replay_train_notif, parent_key='id', relation_key='customer_id', reduce=True)
    replay_train_gr.add_entity_edge(replay_train_cust, replay_train_order, parent_key='id', relation_key='customer_id', reduce=True)
    replay_train_gr.apply_execution_plan(frozen_plan)
    replay_train_gr.do_transformations_sql()

    replay_train_df = pd.read_sql_query(
        f"select * from {replay_train_gr.parent_node._cur_data_ref}",
        replay_train_conn,
    ).sort_values("cust_id").reset_index(drop=True)
    replay_train_sql_ops_by_method = replay_train_gr.get_executed_sqlops_by_method()
    ic({
        method: [f"{op.optype.value}:{op.opval}" for op in ops]
        for method, ops in replay_train_sql_ops_by_method.items()
    })

    pd.testing.assert_frame_equal(
        original_train_df.sort_index(axis=1),
        replay_train_df.sort_index(axis=1),
        check_dtype=False,
    )
    assert "do_labels" in replay_train_sql_ops_by_method

    _teardown_sqlite(replay_train_conn)

    score_conn = _setup_sqlite()
    score_cust = SQLNode(
        fpath='cust',
        pk='id',
        prefix='cust',
        client=score_conn,
        compute_layer=ComputeLayerEnum.sqlite,
        columns=['id', 'name'],
    )
    score_order = SQLNode(
        fpath='orders',
        pk='id',
        prefix='ord',
        client=score_conn,
        compute_layer=ComputeLayerEnum.sqlite,
        columns=['id', 'customer_id', 'ts', 'amount'],
        date_key='ts',
    )
    score_notif = SQLNode(
        fpath='notifications',
        prefix='not',
        pk='id',
        client=score_conn,
        compute_layer=ComputeLayerEnum.sqlite,
        columns=['id', 'customer_id', 'ts'],
        date_key='ts',
    )

    score_gr = GraphReduce(
        name='sql_freeze_score',
        parent_node=score_cust,
        compute_period_unit=PeriodUnit.day,
        compute_period_val=730,
        label_node=score_order,
        label_field='id',
        label_operation='bool',
        label_period_unit=PeriodUnit.day,
        label_period_val=90,
        compute_layer=ComputeLayerEnum.sqlite,
        use_temp_tables=True,
        lazy_execution=False,
        auto_features=True,
        auto_feature_hops_back=3,
        auto_feature_hops_front=0,
        sql_client=score_conn,
        date_filters_on_agg=True,
        train=False,
    )

    score_gr.add_node(score_cust)
    score_gr.add_node(score_order)
    score_gr.add_node(score_notif)
    score_gr.add_entity_edge(score_cust, score_notif, parent_key='id', relation_key='customer_id', reduce=True)
    score_gr.add_entity_edge(score_cust, score_order, parent_key='id', relation_key='customer_id', reduce=True)
    score_gr.apply_execution_plan(frozen_plan)
    score_gr.do_transformations_sql()

    score_sql_ops_by_method = score_gr.get_executed_sqlops_by_method()
    ic({
        method: [f"{op.optype.value}:{op.opval}" for op in ops]
        for method, ops in score_sql_ops_by_method.items()
    })

    assert "do_reduce" in score_sql_ops_by_method
    assert "do_labels" not in score_sql_ops_by_method
    assert "do_post_join_annotate" not in score_sql_ops_by_method
    score_df = pd.read_sql_query(
        f"select * from {score_gr.parent_node._cur_data_ref}",
        score_conn,
    ).sort_values("cust_id").reset_index(drop=True)
    assert len(score_df) == 4
    assert not any(
        "2023-06-30" in op.opval
        for ops in score_sql_ops_by_method.values()
        for op in ops
    )

    _teardown_sqlite(score_conn)


@pytest.mark.skip(reason="Not implemented yet")
def test_daft_graph():
    cust_node = DynamicNode(
            fpath=os.path.join(data_path, 'cust.csv'),
            fmt='csv',
            prefix='cust',
            date_key=None,
            pk='id',
            )

    order_node = DynamicNode(
            fpath=os.path.join(data_path, 'orders.csv'),
            fmt='csv',
            prefix='ord',
            date_key='ts',
            pk='id',
            )

    gr = GraphReduce(
            parent_node=cust_node,
            fmt='csv',
            compute_layer=ComputeLayerEnum.daft,
            auto_features=True,
            compute_period_val=730
            )
    gr.add_node(cust_node)
    gr.add_node(order_node)

    assert len(gr) == 2

    gr.add_entity_edge(
            parent_node=cust_node,
            relation_node=order_node,
            parent_key='id',
            relation_key='customer_id',
            relation_type='parent_child',
            reduce=True
            )

    gr.do_transformations()
    print(gr.parent_node.df.show(10))
    assert gr.parent_node.df.count_rows() == 4


def test_duckdb_node():
    import duckdb
    from graphreduce.node import DuckdbNode
    from graphreduce.enum import ComputeLayerEnum

    con = duckdb.connect()

    node = DuckdbNode(
            #fpath=os.path.join(data_path, 'cust.csv'),
            fpath=f"'{os.path.join(data_path, 'cust.csv')}'",
            pk='id',
            prefix='cust',
            compute_layer=ComputeLayerEnum.duckdb,
            client=con,
            columns=['id', 'name'],
            # Needed for filesystem tables.
            table_name='customer'
            )
    print(node.build_query(node.do_data()))
    con.close()
    assert node.do_data() != None
    assert len(node.columns) == 2


def test_duckdb_graph_noreduce():
    con = duckdb.connect()
    cust = DuckdbNode(
            fpath=f"'{os.path.join(data_path, 'cust.csv')}'",
            prefix='cust',
            pk='id',
            columns=['id','name'],
            table_name='customer'
            )
    orders = DuckdbNode(
            fpath=f"'{os.path.join(data_path, 'orders.csv')}'",
            prefix='ord',
            pk='id',
            date_key='ts',
            columns=['id','customer_id','ts'],
            table_name='orders'
            )
    gr = GraphReduce(
        name='duckdb test',
        parent_node=cust,
        compute_period_val=365,
        compute_period_unit=PeriodUnit.day,
        auto_features=True,
        auto_labels=False,
        label_node=None,
        label_field=None,
        label_op=None,
        compute_layer=ComputeLayerEnum.duckdb,
        sql_client=con
        )
    gr.add_node(cust)
    gr.add_node(orders)
    gr.add_entity_edge(parent_node=cust,relation_node=orders,parent_key='id',relation_key='customer_id',reduce=False)
    gr.do_transformations_sql()
    res = con.sql(f"select * from {gr.parent_node._cur_data_ref}").to_df()
    ic(res)
    ic(res.shape)
    assert res.shape[0] > 4
    con.close()


def test_duckdb_graph_reduce():
    con = duckdb.connect()
    cust = DuckdbNode(
            fpath=f"'{os.path.join(data_path, 'cust.csv')}'",
            prefix='cust',
            pk='id',
            columns=['id','name'],
            table_name='customer'
            )
    orders = DuckdbNode(
            fpath=f"'{os.path.join(data_path, 'orders.csv')}'",
            prefix='ord',
            pk='id',
            date_key='ts',
            columns=['id','customer_id','ts', 'amount'],
            table_name='orders'
            )
    gr = GraphReduce(
        name='duckdb test',
        parent_node=cust,
        compute_period_val=365,
        compute_period_unit=PeriodUnit.day,
        cut_date=datetime.datetime(2023, 5, 1),
        auto_features=True,
        auto_labels=True,
        label_node=orders,
        label_field='id',
        label_operation='count',
        label_period_val=90,
        label_period_unit=PeriodUnit.day,
        compute_layer=ComputeLayerEnum.duckdb,
        sql_client=con
        )
    gr.add_node(cust)
    gr.add_node(orders)
    gr.add_entity_edge(parent_node=cust,relation_node=orders,parent_key='id',relation_key='customer_id',reduce=True)
    gr.do_transformations_sql()
    res = con.sql(f"select * from {gr.parent_node._cur_data_ref}").to_df()

    ic(res)
    ic(res.columns)
    ic(res.shape)
    assert res.shape[0] == 4
    con.close()


def test_date_node_propagates_through_undated_intermediate_sql_node():
    con = duckdb.connect()
    cut_date = datetime.datetime(2023, 5, 1)

    cust = DuckdbNode(
        fpath=f"'{os.path.join(data_path, 'cust.csv')}'",
        prefix="cust",
        pk="id",
        columns=["id", "name"],
        table_name="customer",
    )
    notification = DuckdbNode(
        fpath=f"'{os.path.join(data_path, 'notifications.csv')}'",
        prefix="notif",
        pk="id",
        date_key=None,
        columns=["id", "customer_id", "ts"],
        table_name="notifications",
    )
    interaction = DuckdbNode(
        fpath=f"'{os.path.join(data_path, 'notification_interactions.csv')}'",
        prefix="ni",
        pk="id",
        date_key="ts",
        columns=["id", "notification_id", "interaction_type_id", "ts"],
        table_name="notification_interactions",
    )
    date_node = DuckdbNode(
        fpath=f"'{os.path.join(data_path, 'orders.csv')}'",
        prefix="cd",
        pk="customer_id",
        date_key="first_order_date",
        table_name="customer_cut_date",
        do_data_ops=sqlop(
            optype=SQLOpType.custom,
            opval=f"""
                SELECT
                    customer_id AS cd_customer_id,
                    min(ts) AS cd_first_order_date
                FROM '{os.path.join(data_path, 'orders.csv')}'
                GROUP BY customer_id
            """,
        ),
        client=con,
    )

    gr = GraphReduce(
        name="date_node_undated_intermediate",
        parent_node=cust,
        compute_period_val=365,
        compute_period_unit=PeriodUnit.day,
        cut_date=cut_date,
        auto_features=True,
        auto_labels=False,
        label_node=None,
        label_field=None,
        compute_layer=ComputeLayerEnum.duckdb,
        sql_client=con,
        date_node=date_node,
    )
    gr.add_node(cust)
    gr.add_node(notification)
    gr.add_node(interaction)
    gr.add_entity_edge(
        parent_node=cust,
        relation_node=notification,
        parent_key="id",
        relation_key="customer_id",
        reduce=True,
    )
    gr.add_entity_edge(
        parent_node=notification,
        relation_node=interaction,
        parent_key="id",
        relation_key="notification_id",
        reduce=True,
    )
    gr.add_entity_edge(
        parent_node=cust,
        relation_node=date_node,
        parent_key="id",
        relation_key="customer_id",
    )

    gr.do_transformations_sql()

    assert getattr(notification, "date_node", None) is not None
    assert getattr(interaction, "date_node", None) is not None
    assert any("cd_first_order_date" in sql for sql in gr.sql_ops if sql)
    con.close()


def test_duckdb_graph_raises_on_erroneous_sqlop():
    con = duckdb.connect()
    cust = DuckdbNode(
            fpath=f"'{os.path.join(data_path, 'cust.csv')}'",
            prefix='cust',
            pk='id',
            columns=['id','name'],
            table_name='customer',
            do_annotate_ops=[
                sqlop(optype=SQLOpType.select, opval="*"),
                sqlop(optype=SQLOpType.select, opval="missing_column as bad_col"),
                ]
            )
    orders = DuckdbNode(
            fpath=f"'{os.path.join(data_path, 'orders.csv')}'",
            prefix='ord',
            pk='id',
            date_key='ts',
            columns=['id','customer_id','ts', 'amount'],
            table_name='orders'
            )
    gr = GraphReduce(
        name='duckdb bad sqlop test',
        parent_node=cust,
        compute_period_val=365,
        compute_period_unit=PeriodUnit.day,
        cut_date=datetime.datetime(2023, 5, 1),
        auto_features=True,
        auto_labels=True,
        label_node=orders,
        label_field='id',
        label_operation='count',
        label_period_val=90,
        label_period_unit=PeriodUnit.day,
        compute_layer=ComputeLayerEnum.duckdb,
        sql_client=con
        )
    gr.add_node(cust)
    gr.add_node(orders)
    gr.add_entity_edge(parent_node=cust,relation_node=orders,parent_key='id',relation_key='customer_id',reduce=True)

    try:
        with pytest.raises(Exception, match="missing_column"):
            gr.do_transformations_sql()
    finally:
        con.close()


def test_duckdb_inference_sample_prefers_and_backfills_populated_rows():
    con = duckdb.connect()
    try:
        con.sql("""
            CREATE TABLE sparse_features (
                id INTEGER,
                feature_a VARCHAR,
                feature_b INTEGER
            )
            """)
        con.sql("""
            INSERT INTO sparse_features VALUES
                (1, NULL, NULL),
                (2, 'populated', NULL),
                (3, NULL, 10)
            """)
        node = DuckdbNode(
            fpath="sparse_features",
            prefix="sp",
            pk="id",
            compute_layer=ComputeLayerEnum.duckdb,
            client=con,
            columns=["id", "feature_a", "feature_b"],
        )

        naive_sample = node.get_sample(n=1)
        inference_sample = node.get_inference_sample(n=1, backfill_per_column=1)

        assert naive_sample["feature_a"].notna().sum() == 0
        assert naive_sample["feature_b"].notna().sum() == 0
        assert inference_sample["feature_a"].notna().sum() > 0
        assert inference_sample["feature_b"].notna().sum() > 0
    finally:
        con.close()


def test_duckdb_join_deps():
    con = duckdb.connect()
    orders = DuckdbNode(
            fpath=f"'{os.path.join(data_path, 'orders.csv')}'",
            prefix='ord',
            pk='id',
            date_key='ts',
            columns=['id','customer_id','ts', 'amount'],
            table_name='orders',
            do_reduce_ops=[
                sqlop(optype=SQLOpType.agg, opval="ord_customer_id"),
                sqlop(optype=SQLOpType.aggfunc, opval="count(ord_id) as ord_num_orders")
                ]
            )
    notif = DuckdbNode(
            fpath=f"'{os.path.join(data_path, 'notifications.csv')}'",
            prefix='notif',
            pk='id',
            date_key='ts',
            columns=['id','customer_id','ts'],
            table_name='notifications',
            do_reduce_ops=[
                sqlop(optype=SQLOpType.agg, opval="notif_customer_id"),
                sqlop(optype=SQLOpType.aggfunc, opval="count(notif_id) as notif_num_notifications")
                ]
            )
    cust = DuckdbNode(
            fpath=f"'{os.path.join(data_path, 'cust.csv')}'",
            prefix='cust',
            pk='id',
            columns=['id','name'],
            table_name='customer',
            do_post_join_annotate_ops=[
                sqlop(optype=SQLOpType.select, opval="*"),
                sqlop(optype=SQLOpType.select, opval="notif_num_notifications / ord_num_orders as notifs_per_order")
                ],
            do_post_join_filters_ops=[
                sqlop(optype=SQLOpType.where, opval="notifs_per_order >= 2")
                ],
            do_post_join_annotate_requires=[orders,notif],
            do_post_join_filters_requires=[orders, notif]
            )

    gr = GraphReduce(
        name='duckdb test',
        parent_node=cust,
        compute_period_val=365,
        compute_period_unit=PeriodUnit.day,
        cut_date=datetime.datetime(2023, 5, 1),
        auto_features=True,
        auto_labels=True,
        label_node=orders,
        label_field='id',
        label_operation='count',
        label_period_val=90,
        label_period_unit=PeriodUnit.day,
        compute_layer=ComputeLayerEnum.duckdb,
        sql_client=con
        )
    gr.add_node(cust)
    gr.add_node(orders)
    gr.add_node(notif)
    gr.add_entity_edge(parent_node=cust,relation_node=orders,parent_key='id',relation_key='customer_id',reduce=True)
    gr.add_entity_edge(parent_node=cust,relation_node=notif,parent_key='id',relation_key='customer_id',reduce=True)
    gr.do_transformations_sql()
    res = con.sql(f"select * from {gr.parent_node._cur_data_ref}").to_df()
    ic(res)
    ic(res.columns)
    ic(res.shape)
    assert res.shape[0] == 2


def test_duckdb_join_deps2():
    con = duckdb.connect()
    orders = DuckdbNode(
            fpath=f"'{os.path.join(data_path, 'orders.csv')}'",
            prefix='ord',
            pk='id',
            date_key='ts',
            columns=['id','customer_id','ts', 'amount'],
            table_name='orders',
            do_reduce_ops=[
                sqlop(optype=SQLOpType.agg, opval="ord_customer_id"),
                sqlop(optype=SQLOpType.aggfunc, opval="count(ord_id) as ord_num_orders")
                ]
            )
    notif = DuckdbNode(
            fpath=f"'{os.path.join(data_path, 'notifications.csv')}'",
            prefix='notif',
            pk='id',
            date_key='ts',
            columns=['id','customer_id','ts'],
            table_name='notifications',
            do_reduce_ops=[
                sqlop(optype=SQLOpType.agg, opval="notif_customer_id"),
                sqlop(optype=SQLOpType.aggfunc, opval="count(notif_id) as notif_num_notifications")
                ]
            )
    cust = DuckdbNode(
            fpath=f"'{os.path.join(data_path, 'cust.csv')}'",
            prefix='cust',
            pk='id',
            columns=['id','name'],
            table_name='customer',
            do_post_join_annotate_ops=[
                sqlop(optype=SQLOpType.select, opval="*"),
                sqlop(optype=SQLOpType.select, opval="case when ord_num_orders > 2 then 1 else 0 end as had_prior_orders")
                ],
            #do_post_join_filters_ops=[
            #    sqlop(optype=SQLOpType.where, opval="notifs_per_order >= 2")
            #    ],
            do_post_join_annotate_requires=[orders],#,notif],
            #do_post_join_filters_requires=[orders, notif]
            )

    gr = GraphReduce(
        name='duckdb test',
        parent_node=cust,
        compute_period_val=365,
        compute_period_unit=PeriodUnit.day,
        cut_date=datetime.datetime(2023, 5, 1),
        auto_features=True,
        auto_labels=True,
        label_node=orders,
        label_field='id',
        label_operation='count',
        label_period_val=90,
        label_period_unit=PeriodUnit.day,
        compute_layer=ComputeLayerEnum.duckdb,
        sql_client=con
        )
    gr.add_node(cust)
    gr.add_node(orders)
    gr.add_node(notif)
    gr.add_entity_edge(parent_node=cust,relation_node=orders,parent_key='id',relation_key='customer_id',reduce=True)
    gr.add_entity_edge(parent_node=cust,relation_node=notif,parent_key='id',relation_key='customer_id',reduce=True)
    gr.do_transformations_sql()
    res = con.sql(f"select * from {gr.parent_node._cur_data_ref}").to_df()
    ic(res)
    ic(res.columns)
    ic(res.shape)
    assert res.shape == (4,8)



def test_date_filters_on_agg():
    con = duckdb.connect()
    cut_date=datetime.datetime(2023, 7, 1)
    orders = DuckdbNode(
            fpath=f"'{os.path.join(data_path, 'orders.csv')}'",
            prefix='ord',
            pk='id',
            date_key='ts',
            columns=['id','customer_id','ts', 'amount'],
            table_name='orders',
            do_reduce_ops=[
                sqlop(optype=SQLOpType.agg, opval="ord_customer_id"),
                sqlop(optype=SQLOpType.aggfunc, opval="count(ord_id) as ord_num_orders"),
                sqlop(optype=SQLOpType.aggfunc, opval="MAX(ord_ts) as ord_max_ts")
                ]
            )
    notif = DuckdbNode(
            fpath=f"'{os.path.join(data_path, 'notifications.csv')}'",
            prefix='notif',
            pk='id',
            date_key='ts',
            columns=['id','customer_id','ts'],
            table_name='notifications',
            do_reduce_ops=[
                sqlop(optype=SQLOpType.agg, opval="notif_customer_id"),
                sqlop(optype=SQLOpType.aggfunc, opval="count(notif_id) as notif_num_notifications"),
                sqlop(optype=SQLOpType.aggfunc, opval="max(notif_ts) as notif_max_ts")
                ]
            )
    cust = DuckdbNode(
            fpath=f"'{os.path.join(data_path, 'cust.csv')}'",
            prefix='cust',
            pk='id',
            columns=['id','name'],
            table_name='customer',
            do_post_join_annotate_ops=[
                sqlop(optype=SQLOpType.select, opval="*"),
                sqlop(optype=SQLOpType.select, opval="case when ord_num_orders > 2 then 1 else 0 end as had_prior_orders"),
                sqlop(optype=SQLOpType.select, opval=f"DATE_DIFF('DAY', ord_max_ts, '{str(cut_date)}') as time_since_order"),
                sqlop(optype=SQLOpType.select, opval=f"DATE_DIFF('DAY', notif_max_ts, '{str(cut_date)}') as time_since_notif")
                ],
            #do_post_join_filters_ops=[
            #    sqlop(optype=SQLOpType.where, opval="notifs_per_order >= 2")
            #    ],
            do_post_join_annotate_requires=[orders,notif]
            #do_post_join_annotate_requires=[notif],#,notif],
            #do_post_join_filters_requires=[orders, notif]
            )
    gr = GraphReduce(
        name='duckdb test',
        parent_node=cust,
        compute_period_val=365,
        compute_period_unit=PeriodUnit.day,
        cut_date=datetime.datetime(2023, 7, 1),
        auto_features=True,
        auto_labels=True,
        label_node=orders,
        label_field='id',
        label_operation='count',
        label_period_val=90,
        label_period_unit=PeriodUnit.day,
        compute_layer=ComputeLayerEnum.duckdb,
        sql_client=con,
        date_filters_on_agg=True
        )
    ic(gr.params)
    gr.add_node(cust)
    gr.add_node(orders)
    gr.add_node(notif)
    gr.add_entity_edge(parent_node=cust,relation_node=orders,parent_key='id',relation_key='customer_id',reduce=True)
    gr.add_entity_edge(parent_node=cust,relation_node=notif,parent_key='id',relation_key='customer_id',reduce=True)
    gr.do_transformations_sql()
    res = con.sql(f"select * from {gr.parent_node._cur_data_ref}").to_df()
    ic(res)
    assert res[res['cust_id'] == 1].time_since_order.values[0] == 30
    assert res[res['cust_id'] == 2].time_since_order.values[0] == 181
