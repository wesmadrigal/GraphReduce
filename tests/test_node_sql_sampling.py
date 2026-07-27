import duckdb

from graphreduce.node import _balanced_sql_add


def test_balanced_sql_add_handles_wide_non_null_score():
    expressions = [
        f"CASE WHEN {index} IS NOT NULL THEN 1 ELSE 0 END"
        for index in range(1500)
    ]
    score = _balanced_sql_add(expressions)

    con = duckdb.connect()
    result = con.sql(f"SELECT {score} AS score").fetchone()[0]

    assert result == 1500
