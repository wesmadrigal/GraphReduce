"""Execution-state restoration and data parity for wide DuckDB reductions."""
import duckdb
import pandas as pd
import pytest

from graphreduce.node import DuckdbNode


def node_for(connection):
    node = object.__new__(DuckdbNode)
    node.get_client = lambda: connection
    return node


def wide_query(destination="reduced", missing=False):
    column = "missing" if missing else "i"
    aggregates = ", ".join(f"count(distinct {column}%{n}) as v{n}" for n in range(17, 34))
    return f"CREATE TEMP TABLE {destination} AS SELECT i%5 AS k,{aggregates} FROM range(1000) t(i) GROUP BY i%5"


def test_wide_reduction_preserves_values_and_connection_threads():
    con = duckdb.connect()
    try:
        con.sql("SET threads=8")
        con.sql(wide_query("reference"))
        node_for(con).execute_query(wide_query(), ret_df=False)
        assert con.sql("SELECT current_setting('threads')").fetchone()[0] == 8
        pd.testing.assert_frame_equal(con.sql("SELECT * FROM reference ORDER BY k").to_df(),
            con.sql("SELECT * FROM reduced ORDER BY k").to_df(), check_exact=True)
    finally:
        con.close()


def test_thread_budget_restores_after_native_query_error():
    con = duckdb.connect()
    try:
        con.sql("SET threads=8")
        with pytest.raises(duckdb.BinderException):
            node_for(con).execute_query(wide_query(missing=True), ret_df=False)
        assert con.sql("SELECT current_setting('threads')").fetchone()[0] == 8
    finally:
        con.close()


class RecordingConnection:
    def __init__(self, threads=8, fail=False):
        self.threads, self.fail, self.calls = threads, fail, []

    def sql(self, query):
        self.calls.append((query, self.threads))
        if query == "SELECT current_setting('threads')":
            return self
        if query.startswith("SET threads="):
            self.threads = int(query.partition("=")[2])
        return self

    def fetchone(self):
        return (self.threads,)

    def to_df(self):
        return pd.DataFrame({"threads": [self.threads]})


def test_only_eager_wide_aggregations_receive_the_budget():
    connection = RecordingConnection()
    node = node_for(connection)
    query = wide_query()
    node.execute_query(query, ret_df=False)
    assert (query, 1) in connection.calls
    assert connection.threads == 8
    # Sampling, annotations, joins, and narrow reductions keep the user's budget.
    for query in ["SELECT * FROM source LIMIT 1000",
                  "CREATE TEMP TABLE annotated AS SELECT *,1 AS value FROM source",
                  "CREATE TEMP TABLE narrow AS SELECT key,count(distinct x) FROM source GROUP BY key"]:
        connection.calls.clear()
        node.execute_query(query, ret_df=False)
        assert connection.calls == [(query, 8)]


def test_single_worker_connections_remain_single_worker():
    connection = RecordingConnection(threads=1)
    node_for(connection).execute_query(wide_query(), ret_df=False)
    assert connection.threads == 1
    assert not any(query.startswith("SET threads=") for query, _ in connection.calls)
