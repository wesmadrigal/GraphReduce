#!/usr/bin/env python
from __future__ import annotations

# std lib
import abc
import datetime
import hashlib
import numbers
import re
import typing
import time

# third party
import pandas as pd
from dask import dataframe as dd
from structlog import get_logger

try:
    import pyspark
    from pyspark.sql import functions as F, types as T
except ImportError:  # pragma: no cover - optional dependency
    pyspark = None
    F = None
    T = None

try:
    import daft
except ImportError:  # pragma: no cover - optional dependency
    daft = None

# from daft.unity_catalog import UnityCatalog
try:
    from pyiceberg.catalog.rest import RestCatalog
except ImportError:  # pragma: no cover - optional dependency
    RestCatalog = None

try:
    import duckdb
except ImportError:  # pragma: no cover - optional dependency
    duckdb = None

try:
    import trino
except ImportError:  # pragma: no cover - optional dependency
    trino = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None


# internal
from graphreduce.enum import ComputeLayerEnum, PeriodUnit, SQLOpType
from graphreduce.storage import StorageClient
from graphreduce.models import sqlop
from graphreduce.common import (
    clean_datetime_pandas,
)
from graphreduce.constants import FUNCTION_COMBOS
from graphreduce.stypes import infer_df_stype


logger = get_logger("Node")

NUMERIC_VALUE_AUTO_AGGS = {"avg", "max", "mean", "median", "min", "sum"}
TEXT_STYPES = {"text", "text_tokenized", "text_embedded"}
TEXT_COLUMN_NAME_HINTS = {
    "body",
    "comment",
    "description",
    "message",
    "note",
    "notes",
    "review",
    "summary",
    "text",
    "title",
}
CATEGORICAL_IDENTIFIER_NAME_HINTS = {
    "code",
    "ean",
    "isbn",
    "phone",
    "postal",
    "postalcode",
    "postcode",
    "sku",
    "telephone",
    "upc",
    "zip",
    "zipcode",
}
AUTO_ANNOTATED_MARKER = "__gr_"
AUTO_ANNOTATED_VALUE_MARKER = "__gr_value_"
FEATURE_FAMILY_NAMES = {
    "base",
    "conditional",
    "temporal",
    "episode",
    "semantic",
    "sequence",
    "context",
}

KeySpec = typing.Union[str, typing.Sequence[str]]


def key_parts(key: KeySpec, name: str = "key") -> typing.Tuple[str, ...]:
    """Return an ordered, validated tuple of columns for a key specification."""

    if isinstance(key, str):
        parts = (key,)
    elif isinstance(key, (list, tuple)):
        parts = tuple(key)
    else:
        raise TypeError(f"{name} must be a string, list, or tuple")

    if not parts:
        raise ValueError(f"{name} must contain at least one column")
    if any(not isinstance(part, str) or not part for part in parts):
        raise ValueError(f"{name} must contain non-empty string column names")
    if len(set(parts)) != len(parts):
        raise ValueError(f"{name} contains duplicate columns: {parts}")
    return parts


def normalize_key(key: KeySpec, name: str = "key") -> KeySpec:
    """Canonicalize one-column keys to strings and composite keys to tuples."""

    parts = key_parts(key, name=name)
    return parts[0] if len(parts) == 1 else parts

def _sample_is_numeric_object_series(series: pd.Series) -> bool:
    """
    Pandas represents Decimal-backed SQL numerics as object dtype. Treat those
    as numeric, but do not trust varchar/string samples for SQL numeric aggs.
    """
    non_null = series.dropna().head(100)
    if len(non_null) == 0:
        return False
    return all(isinstance(value, numbers.Number) for value in non_null)


def _should_skip_numeric_sql_agg(
    series: pd.Series,
    semantic_type: str,
    func: str,
) -> bool:
    if semantic_type != "numerical" or func not in NUMERIC_VALUE_AUTO_AGGS:
        return False
    if pd.api.types.is_numeric_dtype(series):
        return False
    return not _sample_is_numeric_object_series(series)


def _is_collection_series(series: pd.Series) -> bool:
    sample_vals = series.dropna().head(20)
    collection_types = (list, dict, tuple)
    if np is not None:
        collection_types = collection_types + (np.ndarray,)
    return sample_vals.map(lambda v: isinstance(v, collection_types)).any()


def _sql_literal(value: typing.Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, numbers.Number):
        return str(value)
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def _safe_sql_alias_part(value: typing.Any, max_len: int = 40) -> str:
    alias = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).lower()).strip("_")
    if not alias:
        alias = "missing"
    if len(alias) > max_len:
        digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()[:8]
        alias = f"{alias[:max_len]}_{digest}"
    return alias


def _balanced_sql_add(expressions: typing.Sequence[str]) -> str:
    """Build a shallow SQL addition tree for wide-row scoring expressions."""

    if not expressions:
        return "0"
    if len(expressions) == 1:
        return expressions[0]
    midpoint = len(expressions) // 2
    return (
        f"({_balanced_sql_add(expressions[:midpoint])} + "
        f"{_balanced_sql_add(expressions[midpoint:])})"
    )


def _series_looks_like_text(col: str, series: pd.Series, semantic_type: str) -> bool:
    if (
        semantic_type != "categorical"
        and str(series.dtype) not in ["object", "string"]
    ):
        return False
    non_null = series.dropna().head(100)
    if len(non_null) == 0:
        return False
    if not non_null.map(lambda v: isinstance(v, str)).all():
        return False
    if semantic_type in TEXT_STYPES:
        return True

    lengths = non_null.map(len)
    avg_len = lengths.mean()
    max_len = lengths.max()
    space_share = non_null.map(lambda v: " " in v.strip()).mean()
    name_parts = set(re.split(r"[^0-9a-zA-Z]+", col.lower()))
    has_name_hint = bool(name_parts & TEXT_COLUMN_NAME_HINTS)

    return (
        avg_len >= 40
        or max_len >= 120
        or (avg_len >= 24 and space_share >= 0.5)
        or (has_name_hint and avg_len >= 12)
    )


def _sql_context_annotation_ops(
    table_df_sample: pd.DataFrame,
    stypes: typing.Dict[str, typing.Any],
    context_keys: typing.Sequence[str],
    pk: typing.Optional[KeySpec],
    max_numeric_columns: int,
    column_prefix: str = "",
) -> typing.List[sqlop]:
    """Create configured row-level peer/context features before reduction.

    ``context_keys`` is intentionally caller-supplied. The library can build
    the window features, but it cannot infer which relationships define a
    meaningful peer group for an arbitrary relational schema.
    """

    columns = list(table_df_sample.columns)
    column_lower = {column.lower(): column for column in columns}

    def resolve_column(name: str) -> typing.Optional[str]:
        candidates = [str(name)]
        if column_prefix:
            candidates.append(f"{column_prefix}_{name}")
        for candidate in candidates:
            if candidate in columns:
                return candidate
            resolved = column_lower.get(candidate.lower())
            if resolved is not None:
                return resolved
        suffix = f"_{str(name).lower()}"
        matches = [column for column in columns if column.lower().endswith(suffix)]
        return matches[0] if len(matches) == 1 else None

    pk_aliases = {
        alias
        for alias in (
            resolve_column(pk_part) for pk_part in key_parts(pk, name="pk")
        )
        if alias is not None
    } if pk else set()
    context_columns = []
    for context_key in context_keys:
        context_column = resolve_column(context_key)
        if (
            context_column
            and context_column not in pk_aliases
            and context_column not in context_columns
        ):
            context_columns.append(context_column)
    if not context_columns:
        return []

    numeric_columns = []
    for col, stype in stypes.items():
        if col in context_columns or col in pk_aliases:
            continue
        if _column_name_looks_like_identifier(col.rsplit("_", 1)[-1]):
            continue
        if col.lower().endswith("_date") or col.lower().endswith("_timestamp"):
            continue
        if _is_auto_annotated_feature_col(col):
            continue
        if str(stype) != "numerical" and not pd.api.types.is_numeric_dtype(table_df_sample[col]):
            continue
        numeric_columns.append(col)
    numeric_columns = numeric_columns[:max(0, int(max_numeric_columns))]

    ops: typing.List[sqlop] = []
    for context_col in context_columns:
        context_alias = _safe_sql_alias_part(context_col)
        ops.append(
            sqlop(
                optype=SQLOpType.select,
                opval=(
                    f"COUNT(*) OVER (PARTITION BY {context_col}) as "
                    f"{context_col}{AUTO_ANNOTATED_MARKER}context_size"
                ),
            )
        )
        for value_col in numeric_columns:
            ops.append(
                sqlop(
                    optype=SQLOpType.select,
                    opval=(
                        f"{value_col} - AVG({value_col}) OVER "
                        f"(PARTITION BY {context_col}) as "
                        f"{value_col}{AUTO_ANNOTATED_MARKER}context_{context_alias}_delta"
                    ),
                )
            )
    return ops


def _column_name_looks_like_identifier(col: str) -> bool:
    col_lower = col.lower()
    return (
        col_lower == "id"
        or col_lower.split("_")[-1].endswith("id")
        or col_lower == "uuid"
        or col_lower == "guid"
        or col_lower == "identifier"
        or col_lower.endswith("key")
    )


def _column_name_looks_like_categorical_identifier(col: str) -> bool:
    name_parts = set(re.split(r"[^0-9a-zA-Z]+", col.lower()))
    compact = re.sub(r"[^0-9a-zA-Z]+", "", col.lower())
    return (
        bool(name_parts & CATEGORICAL_IDENTIFIER_NAME_HINTS)
        or compact.endswith("zipcode")
        or compact.endswith("postalcode")
        or compact.endswith("postcode")
    )


def _sql_bool_aggregate_ops(condition: str, alias_prefix: str) -> typing.List[sqlop]:
    indicator = f"CASE WHEN {condition} THEN 1 ELSE 0 END"
    share_indicator = f"CASE WHEN {condition} THEN 1.0 ELSE 0.0 END"
    return [
        sqlop(
            optype=SQLOpType.aggfunc,
            opval=f"SUM({indicator}) as {alias_prefix}_count",
        ),
        sqlop(
            optype=SQLOpType.aggfunc,
            opval=f"AVG({share_indicator}) as {alias_prefix}_share",
        ),
        sqlop(
            optype=SQLOpType.aggfunc,
            opval=f"MAX({indicator}) as {alias_prefix}_any",
        ),
    ]


def _sql_text_aggregate_ops(col: str) -> typing.List[sqlop]:
    alias_col = _safe_sql_alias_part(col)
    coalesced = f"COALESCE({col}, '')"
    trimmed = f"TRIM({coalesced})"
    length_expr = f"LENGTH({coalesced})"
    empty_condition = f"{col} IS NULL OR LENGTH({trimmed}) = 0"
    word_count_expr = (
        f"CASE WHEN {empty_condition} THEN 0 "
        f"ELSE LENGTH({trimmed}) - LENGTH(REPLACE({trimmed}, ' ', '')) + 1 END"
    )
    number_condition = " OR ".join(
        [f"{col} LIKE '%{digit}%'" for digit in range(10)]
    )
    url_condition = (
        f"LOWER({coalesced}) LIKE '%http://%' "
        f"OR LOWER({coalesced}) LIKE '%https://%' "
        f"OR LOWER({coalesced}) LIKE '%www.%'"
    )
    pattern_ops = []
    for name, condition in [
        ("empty", empty_condition),
        ("url", url_condition),
        ("number", number_condition),
        ("question", f"{col} LIKE '%?%'"),
        ("exclamation", f"{col} LIKE '%!%'"),
    ]:
        pattern_ops.extend(
            _sql_bool_aggregate_ops(condition, f"{alias_col}_{name}")
        )

    return [
        sqlop(
            optype=SQLOpType.aggfunc,
            opval=f"AVG({length_expr}) as {alias_col}_length_avg",
        ),
        sqlop(
            optype=SQLOpType.aggfunc,
            opval=f"MAX({length_expr}) as {alias_col}_length_max",
        ),
        sqlop(
            optype=SQLOpType.aggfunc,
            opval=f"SUM({length_expr}) as {alias_col}_length_sum",
        ),
        sqlop(
            optype=SQLOpType.aggfunc,
            opval=f"AVG({word_count_expr}) as {alias_col}_word_count_avg",
        ),
        sqlop(
            optype=SQLOpType.aggfunc,
            opval=f"MAX({word_count_expr}) as {alias_col}_word_count_max",
        ),
        *pattern_ops,
    ]


def _sql_categorical_aggregate_ops(
    col: str,
    series: pd.Series,
    cardinality_threshold: int,
    top_k: int,
) -> typing.List[sqlop]:
    non_null = series.dropna()
    if len(non_null) == 0 or _is_collection_series(non_null):
        return []

    alias_col = _safe_sql_alias_part(col)
    ops = [
        sqlop(
            optype=SQLOpType.aggfunc,
            opval=f"COUNT(DISTINCT {col}) as {alias_col}_nunique",
        )
    ]

    value_counts = non_null.value_counts()
    if len(value_counts) <= cardinality_threshold:
        category_values = list(value_counts.index)
    else:
        category_values = list(value_counts.head(top_k).index)

    used_aliases = set()
    for value in category_values:
        if pd.isna(value):
            continue
        category_alias = _safe_sql_alias_part(value)
        alias_prefix = f"{alias_col}_{category_alias}"
        if alias_prefix in used_aliases:
            digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()[:8]
            alias_prefix = f"{alias_prefix}_{digest}"
        used_aliases.add(alias_prefix)
        ops.extend(
            _sql_bool_aggregate_ops(f"{col} = {_sql_literal(value)}", alias_prefix)
        )

    if (
        len(value_counts) > cardinality_threshold
        and top_k > 0
        and len(category_values)
    ):
        literals = ", ".join([_sql_literal(value) for value in category_values])
        ops.extend(
            _sql_bool_aggregate_ops(
                f"{col} NOT IN ({literals})", f"{alias_col}_other"
            )
        )

    return ops


def _is_auto_annotated_feature_col(col: str) -> bool:
    return AUTO_ANNOTATED_MARKER in col


def _is_auto_predicate_feature_col(col: str) -> bool:
    return (
        _is_auto_annotated_feature_col(col)
        and AUTO_ANNOTATED_VALUE_MARKER not in col
    )


def _sql_auto_annotate_ops(
    table_df_sample: pd.DataFrame,
    stypes: typing.Dict[str, typing.Any],
    cardinality_threshold: int,
    top_k: int,
    max_categorical_columns: int,
    max_gated_numeric_cols: int,
    gated_numeric_top_k: int,
    auto_text_features: bool,
    annotation_expressions: typing.Optional[
        typing.Dict[str, typing.Union[str, typing.Tuple[str, str]]]
    ] = None,
    column_prefix: str = "",
    annotation_expressions_only: bool = False,
    context_features: bool = False,
    context_keys: typing.Sequence[str] = (),
    context_pk: typing.Optional[str] = None,
    context_max_numeric_columns: int = 4,
) -> typing.List[sqlop]:
    ops: typing.List[sqlop] = [sqlop(optype=SQLOpType.select, opval="*")]
    aliases: typing.Set[str] = set(table_df_sample.columns)

    def add_select(expr: str, alias: str) -> None:
        if alias in aliases:
            return
        aliases.add(alias)
        ops.append(sqlop(optype=SQLOpType.select, opval=f"{expr} as {alias}"))

    # Annotation expressions are deliberately compiled before generic type
    # inference so callers can expose domain predicates as numeric indicators
    # that the normal SQL feature planner can aggregate.
    sample_columns = set(table_df_sample.columns)
    for name, spec in (annotation_expressions or {}).items():
        expression_mode = "predicate"
        if isinstance(spec, (tuple, list)):
            expression_mode, expression = spec
        else:
            expression = spec
        placeholders = set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", expression))
        resolved_expression = expression
        valid = True
        for placeholder in placeholders:
            candidate = placeholder if placeholder in sample_columns else None
            if candidate is None:
                prefixed = f"{column_prefix}_{placeholder}"
                if prefixed in sample_columns:
                    candidate = prefixed
            if candidate is None:
                # SQL expressions may use literal braces, but an unresolved
                # column placeholder is always a configuration error. Skip it
                # here so one optional annotation cannot break a whole graph.
                valid = False
                break
            resolved_expression = resolved_expression.replace(
                f"{{{placeholder}}}", candidate
            )
        if not valid:
            continue
        if expression_mode == "value":
            alias = (
                f"{column_prefix}{AUTO_ANNOTATED_VALUE_MARKER}"
                f"{_safe_sql_alias_part(name)}"
            )
            add_select(resolved_expression, alias)
        else:
            alias = (
                f"{column_prefix}{AUTO_ANNOTATED_MARKER}"
                f"{_safe_sql_alias_part(name)}"
            )
            add_select(f"CASE WHEN {resolved_expression} THEN 1 ELSE 0 END", alias)

    if context_features:
        for context_op in _sql_context_annotation_ops(
            table_df_sample,
            stypes,
            context_keys=context_keys,
            pk=context_pk,
            max_numeric_columns=context_max_numeric_columns,
            column_prefix=column_prefix,
        ):
            expression = context_op.opval
            alias = expression.rsplit(" as ", 1)[-1]
            if alias not in aliases:
                aliases.add(alias)
                ops.append(context_op)

    if annotation_expressions_only:
        return ops if len(ops) > 1 else []

    numeric_cols = []
    for col, stype in stypes.items():
        series = table_df_sample[col]
        semantic_type = str(stype)
        if (
            semantic_type == "numerical"
            and _column_name_looks_like_categorical_identifier(col)
        ):
            semantic_type = "categorical"
        if (
            semantic_type == "numerical"
            and not _column_name_looks_like_identifier(col)
            and not _is_collection_series(series)
            and (
                pd.api.types.is_numeric_dtype(series)
                or _sample_is_numeric_object_series(series)
            )
        ):
            numeric_cols.append(col)
    numeric_cols = numeric_cols[:max_gated_numeric_cols]

    categorical_specs = []
    for col, stype in stypes.items():
        series = table_df_sample[col]
        semantic_type = str(stype)
        if (
            semantic_type == "numerical"
            and _column_name_looks_like_categorical_identifier(col)
        ):
            semantic_type = "categorical"
        if _is_collection_series(series):
            continue

        if auto_text_features and _series_looks_like_text(col, series, semantic_type):
            alias_col = _safe_sql_alias_part(col)
            coalesced = f"COALESCE({col}, '')"
            trimmed = f"TRIM({coalesced})"
            empty_condition = f"{col} IS NULL OR LENGTH({trimmed}) = 0"
            word_count_expr = (
                f"CASE WHEN {empty_condition} THEN 0 "
                f"ELSE LENGTH({trimmed}) - LENGTH(REPLACE({trimmed}, ' ', '')) + 1 END"
            )
            number_condition = " OR ".join(
                [f"{col} LIKE '%{digit}%'" for digit in range(10)]
            )
            url_condition = (
                f"LOWER({coalesced}) LIKE '%http://%' "
                f"OR LOWER({coalesced}) LIKE '%https://%' "
                f"OR LOWER({coalesced}) LIKE '%www.%'"
            )
            add_select(f"LENGTH({coalesced})", f"{alias_col}{AUTO_ANNOTATED_MARKER}length")
            add_select(word_count_expr, f"{alias_col}{AUTO_ANNOTATED_MARKER}word_count")
            add_select(
                f"CASE WHEN {number_condition} THEN 1 ELSE 0 END",
                f"{alias_col}{AUTO_ANNOTATED_MARKER}has_number",
            )
            add_select(
                f"CASE WHEN {url_condition} THEN 1 ELSE 0 END",
                f"{alias_col}{AUTO_ANNOTATED_MARKER}has_url",
            )
            add_select(
                f"CASE WHEN {empty_condition} THEN 1 ELSE 0 END",
                f"{alias_col}{AUTO_ANNOTATED_MARKER}is_empty",
            )
            continue

        non_null = series.dropna()
        if len(non_null) == 0:
            continue
        value_counts = non_null.value_counts()
        cardinality = len(value_counts)
        is_categorical = semantic_type == "categorical" or str(series.dtype) in [
            "object",
            "string",
        ]
        is_low_cardinality_numeric_hint = (
            semantic_type == "numerical"
            and cardinality <= cardinality_threshold
            and not _column_name_looks_like_identifier(col)
        )
        if not is_categorical and not is_low_cardinality_numeric_hint:
            continue
        if cardinality > cardinality_threshold and top_k <= 0:
            continue

        category_values = (
            list(value_counts.index)
            if cardinality <= cardinality_threshold
            else list(value_counts.head(top_k).index)
        )
        categorical_specs.append((col, category_values, cardinality))
        if len(categorical_specs) >= max_categorical_columns:
            break

    for col, category_values, cardinality in categorical_specs:
        alias_col = _safe_sql_alias_part(col)
        used_aliases: typing.Set[str] = set()
        for value in category_values:
            if pd.isna(value):
                continue
            category_alias = _safe_sql_alias_part(value)
            alias = f"{alias_col}{AUTO_ANNOTATED_MARKER}is_{category_alias}"
            if alias in used_aliases:
                digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()[:8]
                alias = f"{alias}_{digest}"
            used_aliases.add(alias)
            add_select(
                f"CASE WHEN {col} = {_sql_literal(value)} THEN 1 ELSE 0 END",
                alias,
            )

        if len(category_values) and cardinality > cardinality_threshold:
            literals = ", ".join([_sql_literal(value) for value in category_values])
            add_select(
                f"CASE WHEN {col} NOT IN ({literals}) THEN 1 ELSE 0 END",
                f"{alias_col}{AUTO_ANNOTATED_MARKER}is_other",
            )

    for cat_col, category_values, _ in categorical_specs:
        cat_alias_col = _safe_sql_alias_part(cat_col)
        for value in category_values[:gated_numeric_top_k]:
            if pd.isna(value):
                continue
            category_alias = _safe_sql_alias_part(value)
            condition = f"{cat_col} = {_sql_literal(value)}"
            for numeric_col in numeric_cols:
                if numeric_col == cat_col:
                    continue
                numeric_alias = _safe_sql_alias_part(numeric_col)
                alias = (
                    f"{numeric_alias}{AUTO_ANNOTATED_MARKER}"
                    f"when_{cat_alias_col}_{category_alias}"
                )
                add_select(f"CASE WHEN {condition} THEN {numeric_col} END", alias)

    return ops if len(ops) > 1 else []


SPARK_DF_TYPES = tuple()
if pyspark is not None:  # pragma: no branch
    SPARK_DF_TYPES = tuple(
        t
        for t in [
            getattr(getattr(pyspark, "sql", None), "DataFrame", None),
            getattr(
                getattr(getattr(pyspark, "sql", None), "dataframe", None),
                "DataFrame",
                None,
            ),
            getattr(
                getattr(
                    getattr(getattr(pyspark, "sql", None), "connect", None),
                    "dataframe",
                    None,
                ),
                "DataFrame",
                None,
            ),
        ]
        if t is not None
    )

DAFT_DF_TYPES = tuple()
if daft is not None:  # pragma: no branch
    daft_df = getattr(
        getattr(getattr(daft, "dataframe", None), "dataframe", None), "DataFrame", None
    )
    DAFT_DF_TYPES = (daft_df,) if daft_df is not None else tuple()


def _require_backend(module: typing.Any, backend: str, extra: str) -> None:
    if module is None:
        raise ImportError(
            f'{backend} backend is not installed. Install with `pip install "graphreduce[{extra}]"`.'
        )


class GraphReduceNode(metaclass=abc.ABCMeta):
    """
    Base node class, which can be used directly
    or subclassed for further customization.

    Many helpful methods are implemented and can
    be used as is, but for different engines
    and dialects (e.g., SQL vs. python) it can
    be necessary to implement an engine-specific
    methods (e.g., `do_data` to get data from Snowflake)

    The methods `do_annotate`, `do_filters`,
    `do_normalize`, `do_reduce`, `do_labels`,
    `do_post_join_annotate`, and `do_post_join_filters`
    are abstractmethods which must be defined.
    """

    fpath: str
    fmt: str
    pk: KeySpec
    prefix: str
    date_key: str
    compute_layer: ComputeLayerEnum
    cut_date: typing.Optional[datetime.datetime]
    compute_period_val: typing.Union[int, float]
    compute_period_unit: PeriodUnit
    reduce: bool
    label_period_val: typing.Optional[typing.Union[int, float]]
    label_period_unit: typing.Optional[PeriodUnit]
    label_field: typing.Optional[str]
    spark_sqlctx: typing.Optional[pyspark.sql.SQLContext]
    columns: typing.List
    storage_client: typing.Optional[StorageClient]
    # Only for SQL dialects at the moment.
    lazy_execution: bool

    def __init__(
        self,
        # IF is SQL dialect this should be a table name.
        fpath: str = "",
        # If is SQL dialect "sql" is fine here.
        fmt: str = "",
        pk: typing.Optional[KeySpec] = None,
        prefix: str = None,
        date_key: str = None,
        compute_layer: ComputeLayerEnum = None,
        # 'python' or 'sql'
        dialect: str = "python",
        cut_date: datetime.datetime = datetime.datetime.now(),
        compute_period_val: typing.Union[int, float] = 365,
        compute_period_unit: PeriodUnit = PeriodUnit.day,
        reduce: bool = True,
        label_period_val: typing.Optional[typing.Union[int, float]] = None,
        label_period_unit: typing.Optional[PeriodUnit] = None,
        label_field: typing.Optional[str] = None,
        label_operation: typing.Optional[typing.Union[str, callable]] = None,
        spark_sqlctx: pyspark.sql.SQLContext = None,
        columns: list = [],
        storage_client: typing.Optional[StorageClient] = None,
        checkpoints: list = [],
        # Only for SQL dialects at the moment.
        lazy_execution: bool = False,
        # Read encoding.
        delimiter: str = None,
        encoding: str = None,
        catalog_client: typing.Any = None,
        # The time-series period in days to use.
        ts_periods: list = [1, 3, 4, 7, 14, 30, 60, 90, 180, 365, 730],
        type_func_map: dict = {},
        categorical_cardinality_threshold: int = 20,
        categorical_top_k: int = 20,
        auto_text_features: bool = True,
        auto_annotate_features: bool = False,
        auto_annotate_max_categorical_columns: int = 20,
        auto_annotate_max_gated_numeric_cols: int = 8,
        auto_annotate_gated_numeric_top_k: int = 5,
        feature_families: typing.Optional[typing.Sequence[str]] = None,
        annotation_expressions: typing.Optional[
            typing.Dict[str, typing.Union[str, typing.Tuple[str, str]]]
        ] = None,
        annotation_expressions_only: bool = False,
        feature_family_max_columns: int = 16,
        is_date_node: bool = False,
        context_keys: typing.Optional[typing.Sequence[str]] = None,
        execution_namespace: typing.Optional[str] = None,
    ):
        """
        Constructor
        """
        # For when this is already set on the class definition.
        configured_pk = self.pk if hasattr(self, "pk") else pk
        self.pk = (
            normalize_key(configured_pk, name="pk")
            if configured_pk is not None
            else None
        )
        # For when this is already set on the class definition.
        if not hasattr(self, "prefix"):
            self.prefix = prefix

        if not self.prefix:
            raise Exception(f"{self.__class__} instances must have a prefix")
        # For when this is already set on the class definition.
        if not hasattr(self, "date_key"):
            self.date_key = date_key
        self.fpath = fpath
        self.fmt = fmt
        self.compute_layer = compute_layer
        self.cut_date = cut_date
        self.compute_period_val = compute_period_val
        self.compute_period_unit = compute_period_unit
        self.reduce = reduce
        self.label_period_val = label_period_val
        self.label_period_unit = label_period_unit
        self.label_field = label_field
        self.label_operation = label_operation
        self.spark_sqlctx = spark_sqlctx
        self.columns = columns

        # Read options
        self.delimiter = delimiter if delimiter else ","
        self.encoding = encoding

        # Lazy execution for the SQL nodes.
        self._lazy_execution = lazy_execution
        self._storage_client = storage_client
        self.execution_namespace = execution_namespace
        # List of merged neighbor classes.
        self._merged = []
        # List of checkpoints.

        # Logical types of the original columns from `woodwork`.
        self._logical_types = {}

        self._stypes = {}

        if not self.date_key:
            logger.warning(f"no `date_key` set for {self}")

        self._catalog_client = catalog_client
        self.ts_periods = ts_periods
        self.type_func_map = type_func_map
        self.categorical_cardinality_threshold = categorical_cardinality_threshold
        self.categorical_top_k = categorical_top_k
        self.auto_text_features = auto_text_features
        self.auto_annotate_features = auto_annotate_features
        self.auto_annotate_max_categorical_columns = auto_annotate_max_categorical_columns
        self.auto_annotate_max_gated_numeric_cols = auto_annotate_max_gated_numeric_cols
        self.auto_annotate_gated_numeric_top_k = auto_annotate_gated_numeric_top_k
        if feature_families is None:
            feature_families = ("base",)
        elif isinstance(feature_families, str):
            feature_families = (feature_families,)
        unknown_families = set(feature_families) - FEATURE_FAMILY_NAMES
        if unknown_families:
            raise ValueError(
                f"Unknown feature families {sorted(unknown_families)}; "
                f"expected one of {sorted(FEATURE_FAMILY_NAMES)}"
            )
        self.feature_families = tuple(dict.fromkeys(feature_families))
        self.annotation_expressions = annotation_expressions or {}
        self.annotation_expressions_only = annotation_expressions_only
        self.context_keys = tuple(context_keys or ())
        self.feature_family_max_columns = max(0, int(feature_family_max_columns))

        self.is_date_node = is_date_node

        if self.is_date_node and not self.date_key:
            raise Exception(f"Date nodes must have `date_key` set got {self.date_key}")

    def __repr__(self):
        """
        Instance representation
        """
        return (
            f"<GraphReduceNode: fpath={self.fpath} fmt={self.fmt} prefix={self.prefix}>"
        )

    def __str__(self):
        """
        Instances string
        """
        return (
            f"<GraphReduceNode: fpath={self.fpath} fmt={self.fmt} prefix={self.prefix}>"
        )

    def _is_identifier(self, col: str) -> bool:
        """
        Check if a column is an identifier.
        """
        return _column_name_looks_like_identifier(col)

    def _is_bool(
        self,
        col: str,
    ) -> bool:
        pass

    def is_ts_data(
        self,
        reduce_key: typing.Optional[KeySpec] = None,
    ) -> bool:
        """
        Determines if the data is timeseries.
        """
        if self.date_key:
            reduce_columns = self.colabbrs(reduce_key)
            if (
                self.compute_layer == ComputeLayerEnum.pandas
                or self.compute_layer == ComputeLayerEnum.dask
            ):
                grouped = self.df.groupby(list(reduce_columns)).size()
                if len(self.df) and len(grouped) / len(self.df) < 0.9:
                    return True
            elif self.compute_layer == ComputeLayerEnum.spark:
                _require_backend(pyspark, "spark", "spark")
                grouped = (
                    self.df.groupBy(*reduce_columns)
                    .agg(F.count(F.lit(1)))
                    .count()
                )
                n = self.df.count()
                if n and float(grouped) / float(n) < 0.9:
                    return True
            # TODO(wes): define the SQL logic.
            elif self.compute_layer in [
                ComputeLayerEnum.sqlite,
                ComputeLayerEnum.snowflake,
                ComputeLayerEnum.databricks,
                ComputeLayerEnum.athena,
                ComputeLayerEnum.redshift,
                ComputeLayerEnum.trino,
                ComputeLayerEnum.duckdb,
            ]:
                # run a group by and get the value.
                reduce_key_sql = ", ".join(reduce_columns)
                grp_qry = f"""
                select count(*) as grouped_rows
                from (
                    select {reduce_key_sql}, count(*)
                    FROM {self._cur_data_ref}
                    group by {reduce_key_sql}
                ) t
                """
                row_qry = f"""
                select count(*) as row_count from {self._cur_data_ref}
                """
                grp_df = self.execute_query(grp_qry, ret_df=True)
                grp_df.columns = [c.lower() for c in grp_df.columns]
                row_df = self.execute_query(row_qry, ret_df=True)
                row_df.columns = [c.lower() for c in row_df.columns]
                grp_count = grp_df["grouped_rows"].values[0]
                row_count = row_df["row_count"].values[0]
                if row_count and float(grp_count) / float(row_count) < 0.9:
                    return True
            elif self.compute_layer == ComputeLayerEnum.daft:
                _require_backend(daft, "daft", "daft")
                count_column = self.colabbrs(self.pk)[0]
                grouped = (
                    self.df.groupby(*reduce_columns)
                    .agg(self.df[count_column].count())
                    .count_rows()
                )
                n = self.df.count_rows()
                if n and float(grouped) / float(n) < 0.9:
                    return True
        return False

    def reload(self):
        """
        Refresh the node.
        """
        self._merged = []
        self._checkpoints = []
        self.df = None
        self._logical_types = {}

    def do_data(
        self,
    ) -> typing.Union[pd.DataFrame, dd.DataFrame, pyspark.sql.dataframe.DataFrame]:
        """
        Get some data
        """

        if self.compute_layer.value == "pandas":
            if not hasattr(self, "df") or (
                hasattr(self, "df") and not isinstance(self.df, pd.DataFrame)
            ):
                if self.encoding and self.delimiter:
                    self.df = getattr(pd, f"read_{self.fmt}")(
                        self.fpath, encoding=self.encoding, delimiter=self.delimiter
                    )
                else:
                    self.df = getattr(pd, f"read_{self.fmt}")(self.fpath)
                # Initialize woodwork.
                # self.df.ww.init()
                # self._logical_types = self.df.ww.logical_types

                # Rename columns with prefixes.
                if len(self.columns):
                    self.df = self.df[[c for c in self.columns]]
                self.columns = list(self.df.columns)
                self.df.columns = [f"{self.prefix}_{c}" for c in self.df.columns]
                # Infer the semantic type with `torch_frame`.
                self._stypes = infer_df_stype(self.df.head(100))
        elif self.compute_layer.value == "dask":
            if not hasattr(self, "df") or (
                hasattr(self, "df") and not isinstance(self.df, dd.DataFrame)
            ):
                self.df = getattr(dd, f"read_{self.fmt}")(self.fpath)
                # Initialize woodwork.
                # self.df.ww.init()
                # self._logical_types = self.df.ww.logical_types

                # Rename columns with prefixes.
                if len(self.columns):
                    self.df = self.df[[c for c in self.columns]]
                self.columns = list(self.df.columns)
                self.df.columns = [f"{self.prefix}_{c}" for c in self.df.columns]
                # Infer the semantic type with `torch_frame`.
                self._stypes = infer_df_stype(self.df.head())
        elif self.compute_layer.value == "spark":
            _require_backend(pyspark, "spark", "spark")
            if not hasattr(self, "df") or (
                hasattr(self, "df") and not isinstance(self.df, SPARK_DF_TYPES)
            ):
                if self.fmt != "sql":
                    self.df = getattr(self.spark_sqlctx.read, f"{self.fmt}")(self.fpath)
                elif self.fmt == "sql":
                    self.df = self.spark_sqlctx.sql(f"select * from {self.fpath}")

                if self.columns:
                    self.df = self.df.select(self.columns)
                for c in self.df.columns:
                    self.df = self.df.withColumnRenamed(c, f"{self.prefix}_{c}")

                # Infer the semantic type with `torch_frame`.
                self._stypes = infer_df_stype(self.df.sample(0.5).limit(10).toPandas())
        elif self.compute_layer.value == "daft":
            _require_backend(daft, "daft", "daft")
            if not hasattr(self, "df") or (
                hasattr(self, "df") and not isinstance(self.df, DAFT_DF_TYPES)
            ):
                # Iceberg.
                if self._catalog_client:
                    if RestCatalog is not None and isinstance(
                        self._catalog_client, RestCatalog
                    ):
                        tbl = self._catalog_client.load_table(self.fpath)
                        self.df = daft.read_iceberg(tbl)
                    elif self._catalog_client.__class__.__name__ == "UnityCatalog":
                        # elif isinstance(self._catalog_client, UnityCatalog):
                        tbl = self._catalog_client.load_table(self.fpath)
                        # TODO(wes): support more than just deltalake.
                        self.df = daft.read_deltalake(tbl)
                else:
                    self.df = getattr(daft, f"read_{self.fmt}")(self.fpath)
                self.columns = [c.name() for c in self.df.columns]
                for col in self.df.columns:
                    self.df = self.df.with_column(f"{self.prefix}_{col.name()}", col)
                newcols = [
                    c for c in self.df.columns if c.name().startswith(self.prefix)
                ]
                self.df = self.df.select(*newcols)

                # Infer the semantic type with `torch_frame`.
                n = self.df.count_rows()
                m = 100
                frac = float(m / n) if m / n < 1 else 1.0
                self._stypes = infer_df_stype(self.df.sample(frac).to_pandas())
        # at this point of connectors we may want to try integrating
        # with something like fugue: https://github.com/fugue-project/fugue
        elif self.compute_layer.value == "ray":
            pass

        elif self.compute_layer.value == "snowflake":
            pass

        elif self.compute_layer.value == "postgres":
            pass

        elif self.compute_layer.value == "redshift":
            pass

        elif self.compute_layer.value == "trino":
            pass

    @abc.abstractmethod
    def do_filters(self):
        """
        do some filters on the data
        """
        pass

    @abc.abstractmethod
    def do_annotate(self):
        """
        Implement custom annotation functionality
        for annotating this particular data
        """
        pass

    @abc.abstractmethod
    def do_post_join_annotate(self):
        """
        Implement custom annotation functionality
        for annotating data after joining with
        child data
        """
        pass

    @abc.abstractmethod
    def do_normalize(self):
        pass

    def do_post_join_filters(self):
        """
        Filter operations that require some
        additional relational data to perform.
        """
        pass

    def do_post_join_reduce(self, reduce_key: KeySpec):
        """
        Implementation for reduce operations
        after a join.
        """
        pass

    def auto_features(
        self,
        reduce_key: KeySpec,
        type_func_map: dict = {},
        compute_layer: ComputeLayerEnum = ComputeLayerEnum.pandas,
    ):
        """
        If we're doing automatic features
        this function will run a series of
        automatic aggregations.  The top-level
        `GraphReduce` object will handle joining
        the results together.
        """
        if compute_layer == ComputeLayerEnum.pandas:
            return self.pandas_auto_features(
                reduce_key=reduce_key, type_func_map=type_func_map
            )
        elif compute_layer == ComputeLayerEnum.dask:
            return self.dask_auto_features(
                reduce_key=reduce_key, type_func_map=type_func_map
            )
        elif compute_layer == ComputeLayerEnum.spark:
            return self.spark_auto_features(
                reduce_key=reduce_key, type_func_map=type_func_map
            )
        elif self.compute_layer in [
            ComputeLayerEnum.snowflake,
            ComputeLayerEnum.sqlite,
            ComputeLayerEnum.mysql,
            ComputeLayerEnum.postgres,
            ComputeLayerEnum.redshift,
            ComputeLayerEnum.databricks,
            ComputeLayerEnum.trino,
            ComputeLayerEnum.duckdb,
        ]:
            sample_df = self.get_inference_sample()
            return self.sql_auto_features(
                sample_df, reduce_key=reduce_key, type_func_map=type_func_map
            )
        elif self.compute_layer == ComputeLayerEnum.daft:
            return self.daft_auto_features(
                reduce_key=reduce_key, type_func_map=type_func_map
            )

    def auto_labels(
        self,
        reduce_key: KeySpec,
        type_func_map: dict = {},
        compute_layer: ComputeLayerEnum = ComputeLayerEnum.pandas,
    ):
        """
        If we're doing automatic features
        this function will run a series of
        automatic aggregations.  The top-level
        `GraphReduce` object will handle joining
        the results together.
        """
        if compute_layer == ComputeLayerEnum.pandas:
            return self.pandas_auto_labels(
                reduce_key=reduce_key, type_func_map=type_func_map
            )
        elif compute_layer == ComputeLayerEnum.dask:
            return self.dask_auto_labels(
                reduce_key=reduce_key, type_func_map=type_func_map
            )
        elif compute_layer == ComputeLayerEnum.spark:
            return self.spark_auto_labels(
                reduce_key=reduce_key, type_func_map=type_func_map
            )
        elif compute_layer == ComputeLayerEnum.daft:
            return self.daft_auto_labels(
                reduce_key=reduce_key, type_func_map=type_func_map
            )

    def pandas_auto_features(
        self, reduce_key: KeySpec, type_func_map: dict = {}
    ) -> pd.DataFrame:
        """
        Pandas implementation of dynamic propagation of features.
        This is basically automated feature engineering but suffixed
        with `_propagation` to indicate that we are propagating data
        upward through the graph from child nodes with no feature
        definitions.
        """
        agg_funcs = {}
        reduce_columns = self.colabbrs(reduce_key)
        reduce_column_names = set(reduce_columns) | set(key_parts(reduce_key))

        ts_data = self.is_ts_data(reduce_key)
        if ts_data:
            # Make sure the dates are cleaned.
            self.df = clean_datetime_pandas(self.df, self.colabbr(self.date_key))
            # First sort the data by dates.
            self.df = self.df.sort_values(self.colabbr(self.date_key), ascending=True)
            self.df[f"prev_{self.colabbr(self.date_key)}"] = self.df.groupby(
                list(reduce_columns)
            )[self.colabbr(self.date_key)].shift(1)
            # Get the time between the two different records.
            self.df[self.colabbr("time_between_records")] = self.df.apply(
                lambda x: (
                    x[self.colabbr(self.date_key)]
                    - x[f"prev_{self.colabbr(self.date_key)}"]
                ).total_seconds(),
                axis=1,
            )

        # Make sure `self._stypes` is up to date.
        self._stypes = infer_df_stype(self.df.sample(min(1000, len(self.df))))
        for col, stype in self._stypes.items():
            _type = str(stype)
            if col in reduce_column_names:
                continue
            if self._is_identifier(col):
                # We only perform counts for identifiers.
                agg_funcs[f"{col}_count"] = pd.NamedAgg(column=col, aggfunc="count")
            elif type_func_map.get(_type):
                for func in type_func_map[_type]:
                    if (
                        (_type == "numerical" or "timestamp")
                        and dict(self.df.dtypes)[col].__str__() == "object"
                        and func in ["min", "max", "median", "mean"]
                    ):
                        logger.info(
                            f"skipped aggregation on {col} because semantic numerical but physical object"
                        )
                        continue
                    col_new = f"{col}_{func}"
                    agg_funcs[col_new] = pd.NamedAgg(column=col, aggfunc=func)
        if not len(agg_funcs):
            logger.info(f"No aggregations for {self}")
            return self.df

        grouped = (
            self.prep_for_features()
            .groupby(list(reduce_columns))
            .agg(**agg_funcs)
            .reset_index()
        )
        if not len(grouped):
            return None
        # If we have time-series data take the time
        # since the last event and the cut date.
        if ts_data:
            logger.info(f"computed post-aggregation features for {self}")

            def is_tz_aware(series):
                return series.dt.tz is not None

            if is_tz_aware(grouped[f"{self.colabbr(self.date_key)}_max"]):
                grouped[f"{self.colabbr(self.date_key)}_max"] = grouped[
                    f"{self.colabbr(self.date_key)}_max"
                ].dt.tz_localize(None)

            grouped[self.colabbr("time_since_last_event")] = grouped.apply(
                lambda x: (
                    self.cut_date - x[f"{self.colabbr(self.date_key)}_max"]
                ).total_seconds(),
                axis=1,
            )

            # Number of events in last strata of time
            for d in self.ts_periods:
                if d > self.compute_period_val:
                    continue
                feat_prepped = self.prep_for_features()
                if is_tz_aware(feat_prepped[self.colabbr(self.date_key)]):
                    feat_prepped[self.colabbr(self.date_key)] = feat_prepped[
                        self.colabbr(self.date_key)
                    ].dt.tz_localize(None)

                feat_prepped[self.colabbr("time_since_cut")] = feat_prepped.apply(
                    lambda x: (
                        (self.cut_date - x[self.colabbr(self.date_key)]).total_seconds()
                        / 86400
                    ),
                    axis=1,
                )
                sub = feat_prepped[
                    (feat_prepped[self.colabbr("time_since_cut")] >= 0)
                    & (feat_prepped[self.colabbr("time_since_cut")] <= d)
                ]
                days_group = (
                    sub.groupby(list(reduce_columns))
                    .size()
                    .rename(self.colabbr(f"{d}d_num_events"))
                    .reset_index()
                )
                # join this back to the main dataset.
                grouped = grouped.merge(
                    days_group, on=list(reduce_columns), how="left"
                )
            logger.info(f"merged all ts groupings to {self}")
        return grouped

    def daft_auto_features(
        self, reduce_key: KeySpec, type_func_map: dict = {}
    ) -> pd.DataFrame:
        """
        Daft implementation of dynamic propagation of features.
        This is basically automated feature engineering but suffixed
        with `_propagation` to indicate that we are propagating data
        upward through the graph from child nodes with no feature
        definitions.
        """

        # Temporary hack until a `daft` implementation
        # of window functions is available.  This will
        # also, unfortunately, limit us to single machine
        # data sizes with daft until then.
        _require_backend(daft, "daft", "daft")
        original_df = self.df
        self.compute_layer = ComputeLayerEnum.pandas
        self.df = self.df.to_pandas()
        grouped = self.pandas_auto_features(reduce_key, type_func_map=type_func_map)
        grouped = daft.from_pandas(grouped)
        self.df = original_df
        self.compute_layer = ComputeLayerEnum.daft
        return grouped

    def dask_auto_features(
        self,
        reduce_key: KeySpec,
        type_func_map: dict = {},
    ) -> dd.DataFrame:
        """
        Dask implementation of dynamic propagation of features.
        This is basically automated feature engineering but suffixed
        with `_propagation` to indicate that we are propagating data
        upward through the graph from child nodes with no feature
        definitions.
        """
        agg_funcs = {}
        reduce_column_names = set(self.colabbrs(reduce_key)) | set(
            key_parts(reduce_key)
        )
        for col, stype in self._stypes.items():
            if col in reduce_column_names:
                continue
            _type = str(stype)
            if type_func_map.get(_type):
                for func in type_func_map[_type]:
                    col_new = f"{col}_{func}"
                    agg_funcs[col_new] = pd.NamedAgg(column=col, aggfunc=func)
        return (
            self.prep_for_features()
            .groupby(list(self.colabbrs(reduce_key)))
            .agg(**agg_funcs)
            .reset_index()
        )

    def spark_auto_features(
        self,
        reduce_key: KeySpec,
        type_func_map: dict = {},
    ) -> pyspark.sql.DataFrame:
        """
        Spark implementation of dynamic propagation of features.
        This is basically automated feature engineering but suffixed
        with `_propagation` to indicate that we are propagating data
        upward through the graph from child nodes with no feature
        definitions.
        """
        _require_backend(pyspark, "spark", "spark")

        self._stypes = infer_df_stype(self.df.sample(0.5).limit(10).toPandas())
        agg_funcs = []
        reduce_columns = self.colabbrs(reduce_key)
        reduce_column_names = set(reduce_columns) | set(key_parts(reduce_key))
        ts_data = self.is_ts_data(reduce_key)
        if ts_data:
            logger.info(f"{self} is time-series data")
        for col, stype in self._stypes.items():
            _type = str(stype)

            if col in reduce_column_names:
                continue
            if self._is_identifier(col):
                func = "count"
                col_new = f"{col}_{func}"
                agg_funcs.append(F.count(F.col(col)).alias(col_new))
            elif type_func_map.get(_type):
                for func in type_func_map[_type]:
                    if func == "nunique":
                        func = "count_distinct"
                    col_new = f"{col}_{func}"
                    agg_funcs.append(getattr(F, func)(F.col(col)).alias(col_new))
        grouped = self.prep_for_features().groupby(*reduce_columns).agg(*agg_funcs)
        # If we have time-series data take the time
        # since the last event and the cut date.
        if ts_data:
            # convert the date key to a timestamp
            date_key_field = [
                x
                for x in self.df.schema.fields
                if x.name == self.colabbr(self.date_key)
            ][0]
            if date_key_field.dataType not in [T.TimestampType(), T.DateType()]:
                logger.info(
                    f"{self} date key was {date_key_field.dataType} - converting to Timestamp"
                )
                self.df = self.df.withColumn(
                    self.colabbr(self.date_key),
                    F.to_timestamp(F.col(self.colabbr(self.date_key))),
                )
            logger.info(f"computed post-aggregation features for {self}")
            spark_datetime = self.spark_sqlctx.sql(
                f"SELECT TO_DATE('{self.cut_date.strftime('%Y-%m-%d')}') as cut_date"
            )
            if "cut_date" not in grouped.columns:
                grouped = grouped.crossJoin(spark_datetime)

            grouped = grouped.withColumn(
                self.colabbr("time_since_last_event"),
                F.unix_timestamp(F.col("cut_date"))
                - F.unix_timestamp(F.col(f"{self.colabbr(self.date_key)}_max")),
            ).drop(F.col("cut_date"))
            if "cut_date" not in self.df.columns:
                self.df = self.df.crossJoin(spark_datetime)

            # Number of events in last strata of time
            for d in self.ts_periods:
                if d > self.compute_period_val:
                    continue
                feat_prepped = self.prep_for_features()
                feat_prepped = feat_prepped.withColumn(
                    self.colabbr("time_since_cut"),
                    F.unix_timestamp(F.col("cut_date"))
                    - F.unix_timestamp(F.col(self.colabbr(self.date_key))),
                ).drop(F.col("cut_date"))
                sub = feat_prepped.filter(
                    (feat_prepped[self.colabbr("time_since_cut")] >= 0)
                    & (feat_prepped[self.colabbr("time_since_cut")] <= (d * 86400))
                )
                days_group = sub.groupBy(*reduce_columns).agg(
                    F.count(F.lit(1)).alias(
                        self.colabbr(f"{d}d_num_events")
                    )
                )
                # join this back to the main dataset.
                grouped = grouped.join(
                    days_group, on=list(reduce_columns), how="left"
                )
            logger.info(f"merged all ts groupings to {self}")
        if "cut_date" in grouped.columns:
            grouped = grouped.drop(F.col("cut_date"))
        return grouped

    def _date_subtract_days(self, col: str, days: int) -> str:
        """Return SQL for `col - N days` in the current compute layer's dialect."""
        if self.__class__.__name__ in ["DuckdbNode", "PostgresNode", "RedshiftNode"]:
            return f"{col} - INTERVAL '{days} days'"

        elif self.__class__.__name__ in ["SnowflakeNode", "MySQLNode"]:
            return f"DATEADD(DAY, -{days}, {col})"

        elif self.__class__.__name__ == "DatabricksNode":
            return f"{col} - INTERVAL {days} DAYS"

        elif self.__class__.__name__ == "AthenaNode":
            return f"{col} - INTERVAL '{days}' DAY"

        elif self.__class__.__name__ == "SQLNode":  # SQLite
            return f"DATE({col}, '-{days} days')"

        else:
            raise NotImplementedError(
                f"_date_subtract_days not implemented for {self.__class__.__name__}"
            )

    def sql_auto_features(
        self,
        table_df_sample: typing.Union[pd.DataFrame, dd.DataFrame],
        reduce_key: KeySpec,
        type_func_map: dict = {},
        feature_families: typing.Optional[typing.Sequence[str]] = None,
    ) -> typing.List[sqlop]:
        """
        SQL dialect implementation of automated
        feature engineering.

        At the moment we're just using `pandas` inferred
        data types for these operations.  This is an
        area that can benefit from `woodwork` and other
        data type inference libraries.

        1) Loop over columns
          - if it's an identifier just call `count`
          - if a function map is defined for the semantic or physical type, apply it
            - if it's a categorical and there are only 2 unique values and they are digits call `sum` (booleans)
            - when this is a combinatorial function `col_func_func` make sure the current function is
              in the list of available functions for combinatorials (e.g., `count` cannot be applied after `sum`)
        2) check if we're dealing with time-series data
          - get time since last event (dialect-dependent implementation)
          - loop through the historical periods in `self.ts_periods` (typically 30 - 365 days)
            and get the number of rows within all of the historical periods (e.g., `num_events_30d`, `num_events_60d`)
          - loop over the periods again and compute the change between them (e.g., `num_events_30d / num_events_60d as change_30dv60d`)
            this computes slope / directional changes
        3) finally, add the point-in-time correctness where clauses based on the top-level cut date parameter
           and compute period parameter

        ``feature_families`` extends the legacy ``base`` aggregates with:
          - ``semantic``: caller-provided domain predicates and value
            annotations compiled through ``annotation_expressions``.
          - ``conditional``: point-in-time counts, shares, presence, and changes
            for categorical values and boolean annotations.
          - ``temporal``: windowed numeric sum/average/min/max aggregates.
          - ``sequence``: normalized activity rates, activity shares, burst
            ratios, and active-span features over configured periods.
          - ``episode``: row and distinct-primary-key counts, including windows.
          - ``context``: row-level peer-group sizes and numeric deltas for
            caller-configured ``context_keys`` before child rows are reduced.
        """
        agg_funcs = []
        selected_families = (
            self.feature_families
            if feature_families is None
            else feature_families
        )
        if isinstance(selected_families, str):
            selected_families = (selected_families,)
        families = set(selected_families)
        unknown_families = families - FEATURE_FAMILY_NAMES
        if unknown_families:
            raise ValueError(
                f"Unknown feature families {sorted(unknown_families)}; "
                f"expected one of {sorted(FEATURE_FAMILY_NAMES)}"
            )
        conditional_specs: list[tuple[str, str]] = []
        semantic_conditional_specs: list[tuple[str, str]] = []
        generic_conditional_specs: list[tuple[str, str]] = []
        temporal_numeric_cols: list[str] = []
        semantic_temporal_cols: list[str] = []
        generic_temporal_cols: list[str] = []
        reduce_columns = self.colabbrs(reduce_key)
        reduce_column_names = set(reduce_columns) | set(key_parts(reduce_key))
        # Always need to update this
        # because we never know if
        # the original columns comprise all
        # of the columns currently in the df.
        self._stypes = infer_df_stype(table_df_sample)

        # Physical types.
        ptypes = {col: str(t) for col, t in table_df_sample.dtypes.to_dict().items()}

        ts_data = self.is_ts_data(reduce_key)
        if "temporal" in families and ts_data:
            for col, stype in self._stypes.items():
                if col in reduce_column_names or self._is_identifier(col):
                    continue
                if col == self.colabbr(self.date_key):
                    continue
                if _is_auto_annotated_feature_col(col) or (
                    str(stype) == "numerical"
                    and (
                        pd.api.types.is_numeric_dtype(table_df_sample[col])
                        or _sample_is_numeric_object_series(table_df_sample[col])
                    )
                ):
                    if _is_auto_annotated_feature_col(col):
                        semantic_temporal_cols.append(col)
                    else:
                        generic_temporal_cols.append(col)
            temporal_numeric_cols = semantic_temporal_cols[
                : self.feature_family_max_columns
            ]
            remaining_temporal = max(
                0, self.feature_family_max_columns - len(temporal_numeric_cols)
            )
            temporal_numeric_cols += generic_temporal_cols[:remaining_temporal]

        if "conditional" in families:
            for col, stype in self._stypes.items():
                if (
                    col in reduce_column_names
                    or self._is_identifier(col)
                    or _is_collection_series(table_df_sample[col])
                ):
                    continue
                if self.date_key and col == self.colabbr(self.date_key):
                    continue
                if _is_auto_predicate_feature_col(col) and (
                    pd.api.types.is_numeric_dtype(table_df_sample[col])
                    or _sample_is_numeric_object_series(table_df_sample[col])
                ):
                    semantic_conditional_specs.append((
                        _safe_sql_alias_part(col),
                        f"{col} = 1",
                    ))
                    continue
                semantic_type = str(stype)
                if (
                    semantic_type == "numerical"
                    and _column_name_looks_like_categorical_identifier(col)
                ):
                    semantic_type = "categorical"
                if semantic_type != "categorical" and str(
                    table_df_sample[col].dtype
                ) not in {"object", "string"}:
                    continue
                if _series_looks_like_text(
                    col, table_df_sample[col], semantic_type
                ):
                    continue
                non_null = table_df_sample[col].dropna()
                if len(non_null) == 0:
                    continue
                values = list(non_null.value_counts().head(self.categorical_top_k).index)
                for value in values:
                    if pd.isna(value):
                        continue
                    alias = _safe_sql_alias_part(f"{col}_{value}")
                    generic_conditional_specs.append(
                        (alias, f"{col} = {_sql_literal(value)}")
                    )
            conditional_specs = semantic_conditional_specs[
                : self.feature_family_max_columns
            ]
            remaining_conditional = max(
                0, self.feature_family_max_columns - len(conditional_specs)
            )
            conditional_specs += generic_conditional_specs[:remaining_conditional]
        # Add only 1 count column.
        counted = False
        for col, stype in self._stypes.items():
            # Get the last function applied (if any)
            if col.lower().split("_")[-1] in ["avg", "sum", "count", "min", "max"]:
                last_function = col.lower().split("_")[-1]
            else:
                last_function = None
            # Check if it is a label first.
            if "_label" in col:
                label_func_map = {
                    "count": "sum",
                    "sum": "sum",
                    "min": "min",
                    "max": "max",
                }
            _type = str(stype)
            if (
                _type == "numerical"
                and _column_name_looks_like_categorical_identifier(col)
            ):
                _type = "categorical"
            if col in reduce_column_names:
                continue
            if self._is_identifier(col):
                # We only perform counts for identifiers.
                func = "count"
                col_new = f"{col}_{func}"
                if not counted:
                    agg_funcs.append(
                        sqlop(
                            optype=SQLOpType.aggfunc,
                            opval=f"{func}" + f"({col}) as {col_new}",
                        )
                    )
                    counted = True
            elif type_func_map.get(_type) or _type in TEXT_STYPES:
                if _is_auto_annotated_feature_col(col) and (
                    pd.api.types.is_numeric_dtype(table_df_sample[col])
                    or _sample_is_numeric_object_series(table_df_sample[col])
                ):
                    for func in ["sum", "avg", "min", "max"]:
                        col_new = f"{col}_{func}"
                        op = sqlop(
                            optype=SQLOpType.aggfunc,
                            opval=f"{func}({col}) as {col_new}",
                        )
                        if op not in agg_funcs:
                            agg_funcs.append(op)
                    continue

                # If the physical data type is
                # a boolean override functionality
                # that might have been applied and
                # just call `sum`.
                if ptypes[col] == "bool":
                    col_new = f"{col}_sum"
                    op = sqlop(
                        optype=SQLOpType.aggfunc,
                        opval=f"sum(case when {col} then 1 else 0 end) as {col_new}",
                    )
                    if op not in agg_funcs:
                        agg_funcs.append(op)
                    continue

                if self.auto_text_features and _series_looks_like_text(
                    col, table_df_sample[col], _type
                ):
                    for op in _sql_text_aggregate_ops(col):
                        if op not in agg_funcs:
                            agg_funcs.append(op)
                    continue

                if _type == "categorical":
                    for op in _sql_categorical_aggregate_ops(
                        col=col,
                        series=table_df_sample[col],
                        cardinality_threshold=self.categorical_cardinality_threshold,
                        top_k=self.categorical_top_k,
                    ):
                        if op not in agg_funcs:
                            agg_funcs.append(op)
                    continue

                for func in type_func_map.get(_type, []):
                    # There should be a better top-level mapping
                    # but for now this will do.  SQL engines typically
                    # don't have 'median' and 'mean'.  'mean' is typically
                    # just called 'avg'.
                    if _should_skip_numeric_sql_agg(table_df_sample[col], _type, func):
                        logger.info(
                            f"skipped numeric aggregation {func} on {col} because "
                            "semantic numerical but physical values are not numeric"
                        )
                        continue
                    elif func in self.FUNCTION_MAPPING:
                        func = self.FUNCTION_MAPPING.get(func)

                    elif not func or func == "nunique":
                        continue

                    # If it's a categorical with only 2 scalar values
                    # and it's a digit type then sum it.
                    non_null_vals = table_df_sample[~table_df_sample[col].isnull()][col]
                    sample_vals = non_null_vals.head(20)
                    collection_types = (list, dict, tuple)
                    if np is not None:
                        collection_types = collection_types + (np.ndarray,)
                    is_collection_col = sample_vals.map(
                        lambda v: isinstance(v, collection_types)
                    ).any()

                    # digit categoricals that are actually
                    # booleans
                    if (
                        _type == "categorical"
                        and not is_collection_col
                        and len(non_null_vals.unique()) <= 2
                        and len(non_null_vals) > 0
                        and str(sample_vals.values[0]).isdigit()
                    ):
                        func = "sum"


                    if func:
                        if func == "count" and counted:
                            continue

                        # Check if there was a last function
                        # applied and, if so, if the recommended
                        # function is in it's available combinations.
                        if last_function:
                            # If the function is not in the function
                            # combos it means we've selected the wrong
                            # function.  Let's loop through the appropriate
                            # function combos and append them.
                            if func in FUNCTION_COMBOS[last_function]:
                                col_new = f"{col}_{func}"
                                op = sqlop(
                                    optype=SQLOpType.aggfunc,
                                    opval=f"{func}" + f"({col}) as {col_new}",
                                )
                                if op not in agg_funcs:
                                    agg_funcs.append(op)

                            else:
                                col_new = f"{col}_avg"
                                op = sqlop(
                                    optype=SQLOpType.aggfunc,
                                    opval=f"{func}" + f"({col}) as {col_new}",
                                )
                                if op not in agg_funcs:
                                    agg_funcs.append(op)
                        else:
                            col_new = f"{col}_{func}"
                            op = sqlop(
                                optype=SQLOpType.aggfunc,
                                opval=f"{func}" + f"({col}) as {col_new}",
                            )
                            if op not in agg_funcs:
                                agg_funcs.append(op)

                        if func == "count":
                            counted = True

        # If we have time-series data we want to
        # do historical counts over the last periods.
        if ts_data:
            logger.info(f"had time-series aggregations for {self}")
            if hasattr(self, "date_node") and self.date_node:
                # Use the date_node column as the reference point
                ref_col = f"MAX({self.date_node.prefix}_{self.date_node.date_key})"
                max_col = f"MAX({self.prefix}_{self.date_key})"
            else:
                # Fallback to fixed cut_date (original behavior)
                ref_col = f"TIMESTAMP '{str(self.cut_date)}'"
                max_col = f"MAX({self.prefix}_{self.date_key})"

            # Now define the aggfunc per dialect
            if self.__class__.__name__ == "SQLNode":  # SQLite
                ref_col = f"'{str(self.cut_date)}'"
                aggfunc = sqlop(
                    optype=SQLOpType.aggfunc,
                    opval=f"(julianday({ref_col}) - julianday({max_col})) * 86400 AS {self.prefix}_seconds_since_last",
                )

            elif self.__class__.__name__ == "DuckdbNode":
                aggfunc = sqlop(
                    optype=SQLOpType.aggfunc,
                    opval=f"date_diff('second', {max_col}, {ref_col}) AS {self.prefix}_seconds_since_last",
                )

            elif (
                self.__class__.__name__ == "PostgresNode"
                or self.__class__.__name__ == "RedshiftNode"
            ):
                aggfunc = sqlop(
                    optype=SQLOpType.aggfunc,
                    opval=f"EXTRACT(EPOCH FROM ({ref_col} - {max_col})) AS {self.prefix}_seconds_since_last",
                )

            elif self.__class__.__name__ == "SnowflakeNode":
                aggfunc = sqlop(
                    optype=SQLOpType.aggfunc,
                    opval=f"TIMESTAMPDIFF(SECOND, {max_col}, {ref_col}) AS {self.prefix}_seconds_since_last",
                )

            elif self.__class__.__name__ == "DatabricksNode":
                aggfunc = sqlop(
                    optype=SQLOpType.aggfunc,
                    opval=f"timestampdiff(SECOND, {max_col}, {ref_col}) AS {self.prefix}_seconds_since_last",
                )

            elif self.__class__.__name__ == "AthenaNode":  # Presto/Trino
                aggfunc = sqlop(
                    optype=SQLOpType.aggfunc,
                    opval=f"date_diff('second', {max_col}, {ref_col}) AS {self.prefix}_seconds_since_last",
                )

            elif self.__class__.__name__ == "MySQLNode":
                aggfunc = sqlop(
                    optype=SQLOpType.aggfunc,
                    opval=f"TIMESTAMPDIFF(SECOND, {max_col}, {ref_col}) AS {self.prefix}_seconds_since_last",
                )

            elif self.__class__.__name__ == "TrinoNode":
                aggfunc = sqlop(
                        optype=SQLOpType.aggfunc,
                        opval=f"date_diff('second', TRY_CAST({max_col} AS TIMESTAMP), {ref_col}) AS {self.prefix}_seconds_since_last"
                        )

            else:
                raise NotImplementedError(
                    f"seconds_since_last not implemented for {self.__class__.__name__}"
                )

            # Append it to the agg_funcs.
            agg_funcs.append(aggfunc)

            # Determine the reference timestamp (either fixed cut_date or date_node column)
            if hasattr(self, "date_node") and self.date_node:
                # Use the date_node column as the reference point (dynamic per row)
                ref_ts = f"{self.date_node.prefix}_{self.date_node.date_key}"
                use_dynamic_ref = True
            else:
                # Fallback to fixed cut_date (original behavior)
                ref_ts = f"'{str(self.cut_date)}'"
                use_dynamic_ref = False

            # === 1. Rolling Event Counts (num_events_Xd) ===
            for period in self.ts_periods:
                if use_dynamic_ref:
                    # Relative to date_node column
                    threshold = self._date_subtract_days(
                        ref_ts, period
                    )  # We'll define this helper below
                    case_expr = f"CASE WHEN {self.colabbr(self.date_key)} >= {threshold} THEN 1 ELSE 0 END"
                else:
                    # Fixed cut_date
                    delt = self.cut_date - datetime.timedelta(days=period)
                    case_expr = f"CASE WHEN {self.colabbr(self.date_key)} >= '{str(delt)}' THEN 1 ELSE 0 END"

                aggfunc = sqlop(
                    optype=SQLOpType.aggfunc,
                    opval=f"SUM({case_expr}) as {self.prefix}_num_events_{period}d",
                )
                agg_funcs.append(aggfunc)

            # === 2. Period Change Ratios (Xd vs Yd) ===
            for ix in range(len(self.ts_periods) - 1):
                period1 = self.ts_periods[ix]  # e.g. 1
                period2 = self.ts_periods[ix + 1]  # e.g. 3

                if use_dynamic_ref:
                    thresh1 = self._date_subtract_days(ref_ts, period1)
                    thresh2 = self._date_subtract_days(ref_ts, period2)
                    num_expr = f"SUM(CASE WHEN {self.colabbr(self.date_key)} >= {thresh1} THEN 1 ELSE 0 END)"
                    denom_expr = f"SUM(CASE WHEN {self.colabbr(self.date_key)} >= {thresh2} THEN 1 ELSE 0 END)"
                else:
                    delt1 = self.cut_date - datetime.timedelta(days=period1)
                    delt2 = self.cut_date - datetime.timedelta(days=period2)
                    num_expr = f"SUM(CASE WHEN {self.colabbr(self.date_key)} >= '{str(delt1)}' THEN 1 ELSE 0 END)"
                    denom_expr = f"SUM(CASE WHEN {self.colabbr(self.date_key)} >= '{str(delt2)}' THEN 1 ELSE 0 END)"

                # Choose safe division per dialect
                if self.__class__.__name__ == "DatabricksNode":
                    ratio_expr = f"try_divide({num_expr}, {denom_expr})"
                elif self.__class__.__name__ == "SnowflakeNode":
                    ratio_expr = f"DIV0({num_expr}, {denom_expr})"
                else:
                    ratio_expr = f"{num_expr} * 1.0 / NULLIF({denom_expr}, 0)"

                agg_funcs.append(
                    sqlop(
                        optype=SQLOpType.aggfunc,
                        opval=f"{ratio_expr} AS {self.prefix}_d{period1}v{period2}_change",
                    )
                )

        if "sequence" in families and ts_data:
            # Preserve trajectory information that lifetime counts and a
            # single recency value cannot express. These features remain
            # aggregate-safe and use the same configured temporal periods.
            period_counts: dict[int, str] = {}
            lifetime_count = "COUNT(*)"
            for period in self.ts_periods:
                if hasattr(self, "date_node") and self.date_node:
                    threshold = self._date_subtract_days(ref_col, period)
                else:
                    threshold = f"'{self.cut_date - datetime.timedelta(days=period)}'"
                count_expr = (
                    f"SUM(CASE WHEN {self.colabbr(self.date_key)} >= {threshold} "
                    "THEN 1 ELSE 0 END)"
                )
                period_counts[period] = count_expr
                agg_funcs.append(
                    sqlop(
                        optype=SQLOpType.aggfunc,
                        opval=(
                            f"{count_expr} * 1.0 / NULLIF({period}, 0) as "
                            f"{self.prefix}_activity_rate_{period}d"
                        ),
                    )
                )
                agg_funcs.append(
                    sqlop(
                        optype=SQLOpType.aggfunc,
                        opval=(
                            f"{count_expr} * 1.0 / NULLIF({lifetime_count}, 0) as "
                            f"{self.prefix}_activity_share_{period}d"
                        ),
                    )
                )

            for period1, period2 in zip(self.ts_periods, self.ts_periods[1:]):
                agg_funcs.append(
                    sqlop(
                        optype=SQLOpType.aggfunc,
                        opval=(
                            f"{period_counts[period1]} * 1.0 / "
                            f"NULLIF({period_counts[period2]}, 0) as "
                            f"{self.prefix}_activity_burst_{period1}v{period2}"
                        ),
                    )
                )

            date_col = self.colabbr(self.date_key)
            if self.__class__.__name__ in {"SQLNode", "SQLiteNode"}:
                span_seconds = (
                    f"(julianday(MAX({date_col})) - "
                    f"julianday(MIN({date_col}))) * 86400.0"
                )
            elif self.__class__.__name__ in {"DuckdbNode", "AthenaNode", "TrinoNode"}:
                span_seconds = (
                    f"date_diff('second', MIN({date_col}), MAX({date_col}))"
                )
            elif self.__class__.__name__ in {"PostgresNode", "RedshiftNode"}:
                span_seconds = (
                    f"EXTRACT(EPOCH FROM (MAX({date_col}) - MIN({date_col})))"
                )
            elif self.__class__.__name__ == "SnowflakeNode":
                span_seconds = (
                    f"TIMESTAMPDIFF(SECOND, MIN({date_col}), MAX({date_col}))"
                )
            elif self.__class__.__name__ == "DatabricksNode":
                span_seconds = (
                    f"timestampdiff(SECOND, MIN({date_col}), MAX({date_col}))"
                )
            elif self.__class__.__name__ == "MySQLNode":
                span_seconds = (
                    f"TIMESTAMPDIFF(SECOND, MIN({date_col}), MAX({date_col}))"
                )
            else:
                span_seconds = None

            if span_seconds is not None:
                agg_funcs.append(
                    sqlop(
                        optype=SQLOpType.aggfunc,
                        opval=(
                            f"{span_seconds} as {self.prefix}_active_span_seconds"
                        ),
                    )
                )
                agg_funcs.append(
                    sqlop(
                        optype=SQLOpType.aggfunc,
                        opval=(
                            f"{lifetime_count} * 1.0 / "
                            f"NULLIF(({span_seconds} / 86400.0) + 1, 0) as "
                            f"{self.prefix}_activities_per_active_day"
                        ),
                    )
                )

        # Feature families extend the conservative legacy aggregates with
        # point-in-time relational features. They are expressed as SQL
        # operations so every SQL backend gets the same planner output.
        if ts_data and ("conditional" in families or "temporal" in families):
            if hasattr(self, "date_node") and self.date_node:
                ref_ts = f"{self.date_node.prefix}_{self.date_node.date_key}"
                dynamic_ref = True
            else:
                ref_ts = f"'{str(self.cut_date)}'"
                dynamic_ref = False
            date_col = self.colabbr(self.date_key)

            def window_condition(period: int) -> str:
                if dynamic_ref:
                    threshold = self._date_subtract_days(ref_ts, period)
                else:
                    threshold = f"'{self.cut_date - datetime.timedelta(days=period)}'"
                return f"{date_col} >= {threshold}"

            if "conditional" in families:
                for alias, condition in conditional_specs:
                    window_counts: dict[int, str] = {}
                    for period in self.ts_periods:
                        window = window_condition(period)
                        count_expr = (
                            f"SUM(CASE WHEN ({window}) AND ({condition}) "
                            f"THEN 1 ELSE 0 END)"
                        )
                        denominator = (
                            f"SUM(CASE WHEN {window} THEN 1 ELSE 0 END)"
                        )
                        window_counts[period] = count_expr
                        for suffix, expression in [
                            ("count", count_expr),
                            (
                                "share",
                                f"{count_expr} * 1.0 / NULLIF({denominator}, 0)",
                            ),
                            (
                                "any",
                                f"MAX(CASE WHEN ({window}) AND ({condition}) "
                                "THEN 1 ELSE 0 END)",
                            ),
                        ]:
                            agg_funcs.append(
                                sqlop(
                                    optype=SQLOpType.aggfunc,
                                    opval=f"{expression} as {alias}_{suffix}_{period}d",
                                )
                            )
                    for period1, period2 in zip(self.ts_periods, self.ts_periods[1:]):
                        change = (
                            f"{window_counts[period1]} * 1.0 / "
                            f"NULLIF({window_counts[period2]}, 0)"
                        )
                        agg_funcs.append(
                            sqlop(
                                optype=SQLOpType.aggfunc,
                                opval=f"{change} as {alias}_d{period1}v{period2}_change",
                            )
                        )

            if "temporal" in families:
                for col in temporal_numeric_cols:
                    safe_col = _safe_sql_alias_part(col)
                    for period in self.ts_periods:
                        window = window_condition(period)
                        for suffix, function in [
                            ("sum", "SUM"),
                            ("avg", "AVG"),
                            ("min", "MIN"),
                            ("max", "MAX"),
                        ]:
                            expression = (
                                f"{function}(CASE WHEN {window} THEN {col} END)"
                            )
                            agg_funcs.append(
                                sqlop(
                                    optype=SQLOpType.aggfunc,
                                    opval=(
                                        f"{expression} as "
                                        f"{safe_col}_{suffix}_{period}d"
                                    ),
                                )
                            )

        if "episode" in families:
            # These denominators make conditional rates interpretable and are
            # useful for sparse entities. The PK distinct count avoids
            # over-weighting rows introduced by many-to-many joins.
            agg_funcs.append(
                sqlop(
                    optype=SQLOpType.aggfunc,
                    opval=f"COUNT(*) as {self.prefix}_num_episodes",
                )
            )
            pk_columns = self.colabbrs(self.pk) if self.pk else ()
            pk_col = pk_columns[0] if len(pk_columns) == 1 else None
            if len(pk_columns) > 1:
                logger.warning(
                    "skipping composite distinct-primary-key episode features",
                    node=str(self),
                    pk=pk_columns,
                )
            if pk_col and pk_col in table_df_sample.columns:
                agg_funcs.append(
                    sqlop(
                        optype=SQLOpType.aggfunc,
                        opval=(
                            f"COUNT(DISTINCT {pk_col}) as "
                            f"{self.prefix}_num_unique_episodes"
                        ),
                    )
                )
            if ts_data:
                if hasattr(self, "date_node") and self.date_node:
                    ref_ts = f"{self.date_node.prefix}_{self.date_node.date_key}"
                    dynamic_ref = True
                else:
                    ref_ts = f"'{str(self.cut_date)}'"
                    dynamic_ref = False
                date_col = self.colabbr(self.date_key)
                for period in self.ts_periods:
                    if dynamic_ref:
                        threshold = self._date_subtract_days(ref_ts, period)
                    else:
                        threshold = f"'{self.cut_date - datetime.timedelta(days=period)}'"
                    window = f"{date_col} >= {threshold}"
                    agg_funcs.append(
                        sqlop(
                            optype=SQLOpType.aggfunc,
                            opval=(
                                f"SUM(CASE WHEN {window} THEN 1 ELSE 0 END) as "
                                f"{self.prefix}_num_episodes_{period}d"
                            ),
                        )
                    )
                    if pk_col and pk_col in table_df_sample.columns:
                        agg_funcs.append(
                            sqlop(
                                optype=SQLOpType.aggfunc,
                                opval=(
                                    f"COUNT(DISTINCT CASE WHEN {window} THEN "
                                    f"{pk_col} END) as "
                                    f"{self.prefix}_num_unique_episodes_{period}d"
                                ),
                            )
                        )

        if not len(agg_funcs):
            logger.info(f"No aggregations for {self}")
            return None
        agg = sqlop(optype=SQLOpType.agg, opval=self.key_sql(reduce_key))
        # Need the aggregation and time-based filtering.
        tfilt = self.prep_for_features() if self.prep_for_features() else []

        return tfilt + agg_funcs + [agg]

    def sql_auto_labels(
        self,
        table_df_sample: typing.Union[pd.DataFrame, dd.DataFrame],
        reduce_key: KeySpec,
        type_func_map: dict = {},
    ) -> pd.DataFrame:
        """
        Pandas implementation of auto labeling based on
        provided columns.
        """
        agg_funcs = {}
        if not self._stypes:
            self._stypes = infer_df_stype(table_df_samp)
        for col, stype in self._stypes.items():
            if col.endswith("_label"):
                _type = str(stype)
                if type_func_map.get(_type):
                    for func in type_func_map[_type]:
                        col_new = f"{col}_{func}_label"
                        agg_funcs.append(
                            sqlop(
                                optype=SQLOpType.aggfunc,
                                opval=f"{func}" + f"({col}) as {col_new}",
                            )
                        )
        # Need the aggregation and time-based filtering.
        agg = sqlop(optype=SQLOpType.agg, opval=self.key_sql(reduce_key))
        tfilt = self.prep_for_labels() if self.prep_for_labels() else []
        return tfilt + agg_funcs + [agg]

    def pandas_auto_labels(
        self, reduce_key: KeySpec, type_func_map: dict = {}
    ) -> pd.DataFrame:
        """
        Pandas implementation of auto labeling based on
        provided columns.
        """
        agg_funcs = {}

        for col, stype in self._stypes.items():
            _type = str(stype)
            if (
                col.endswith("_label")
                or col == self.label_field
                or col == f"{self.colabbr(self.label_field)}"
            ):
                if type_func_map.get(_type):
                    for func in type_func_map[_type]:
                        col_new = f"{col}_{func}_label"
                        agg_funcs[col_new] = pd.NamedAgg(column=col, aggfunc=func)
        return (
            self.prep_for_labels()
            .groupby(list(self.colabbrs(reduce_key)))
            .agg(**agg_funcs)
            .reset_index()
        )

    def daft_auto_labels(
        self,
        reduce_key: KeySpec,
        type_func_map: dict = {},
    ) -> pd.DataFrame:
        """
        Daft implementation of auto labeling based
        on provided columns.
        """
        # Temporary hack until a `daft` implementation
        # of window functions is available.  This will
        # also, unfortunately, limit us to single machine
        # data sizes with daft until then.
        _require_backend(daft, "daft", "daft")
        original_df = self.df
        self.compute_layer = ComputeLayerEnum.pandas
        self.df = self.df.to_pandas()
        grouped = self.pandas_auto_labels(reduce_key, type_func_map=type_func_map)
        self.df = original_df
        self.compute_layer = ComputeLayerEnum.daft
        return grouped

    def dask_auto_labels(
        self,
        reduce_key: KeySpec,
        type_func_map: dict = {},
    ) -> dd.DataFrame:
        """
        Dask implementation of auto labeling based on
        provided columns.
        """
        agg_funcs = {}
        for col, stype in self._stypes.items():
            if col.endswith("_label"):
                _type = str(stype)
                if type_func_map.get(_type):
                    for func in type_func_map[_type]:
                        col_new = f"{col}_{func}_label"
                        agg_funcs[col_new] = pd.NamedAgg(column=col, aggfunc=func)
        return (
            self.prep_for_labels()
            .groupby(list(self.colabbrs(reduce_key)))
            .agg(**agg_funcs)
            .reset_index()
        )

    def spark_auto_labels(
        self,
        reduce_key: KeySpec,
        type_func_map: dict = {},
    ) -> pyspark.sql.DataFrame:
        """
        Spark implementation of auto labeling based on
        provided columns.
        """
        _require_backend(pyspark, "spark", "spark")
        agg_funcs = []
        for col, stype in self._stypes.items():
            _type = str(stype)
            if col.endswith("_label"):
                if type_func_map.get(_type):
                    for func in type_func_map[_type]:
                        if func == "nunique":
                            func = "count_distinct"
                        col_new = f"{col}_{func}_label"
                        agg_funcs.append(getattr(F, func)(F.col(col)).alias(col_new))
        return self.prep_for_labels().groupby(*self.colabbrs(reduce_key)).agg(
            *agg_funcs
        )

    @abc.abstractmethod
    def do_reduce(self, reduce_key):
        """
        Reduce operation or the node

        Args
            reduce_key : key to use to perform the reduce operation
            children : list of children nodes
        """
        pass

    @abc.abstractmethod
    def do_labels(
        self,
        reduce_key: typing.Optional[KeySpec] = None,
    ):
        """
        Generate labels
        """
        pass

    def colabbr(self, col: str) -> str:
        if not isinstance(col, str):
            raise TypeError(
                "colabbr() accepts one column name; use colabbrs() for a "
                "composite key"
            )
        prefix = f"{self.prefix}_"
        if col.startswith(prefix):
            return col
        return f"{prefix}{col}"

    def colabbrs(self, key: KeySpec) -> typing.Tuple[str, ...]:
        """Prefix every ordered component of a scalar or composite key."""

        return tuple(self.colabbr(part) for part in key_parts(key))

    def key_sql(self, key: KeySpec) -> str:
        """Render a scalar or composite key for SQL SELECT/GROUP BY clauses."""

        return ", ".join(self.colabbrs(key))

    def compute_period_minutes(
        self,
    ) -> int:
        """
        Convert the compute period to minutes
        """
        if self.compute_period_unit == PeriodUnit.second:
            return self.compute_period_val / 60
        elif self.compute_period_unit == PeriodUnit.minute:
            return self.compute_period_val
        elif self.compute_period_unit == PeriodUnit.hour:
            return self.compute_period_val * 60
        elif self.compute_period_unit == PeriodUnit.day:
            return self.compute_period_val * 1440
        elif self.compute_period_unit == PeriodUnit.week:
            return (self.compute_period_val * 7) * 1440
        elif self.compute_period_unit == PeriodUnit.month:
            return (self.compute_period_val * 30.417) * 1440

    def label_period_minutes(
        self,
    ) -> int:
        """
        Convert the label period to minutes
        """
        if self.label_period_unit == PeriodUnit.second:
            return self.label_period_val / 60
        elif self.label_period_unit == PeriodUnit.minute:
            return self.label_period_val
        elif self.label_period_unit == PeriodUnit.hour:
            return self.label_period_val * 60
        elif self.label_period_unit == PeriodUnit.day:
            return self.label_period_val * 1440
        elif self.label_period_unit == PeriodUnit.week:
            return (self.label_period_val * 7) * 1440
        elif self.label_period_unit == PeriodUnit.month:
            return (self.label_period_val * 30.417) * 1440

    def prep_for_features(
        self,
        allow_null: bool = False,
    ) -> typing.Union[
        pd.DataFrame, dd.DataFrame, pyspark.sql.dataframe.DataFrame, typing.List[sqlop]
    ]:
        """
        Prepare the dataset for feature aggregations / reduce
        """
        # if hasattr(self, 'date_node') and self.date_node:
        if self.date_key:
            # Date filters when we have a
            # date key.
            if (
                hasattr(self, "date_node")
                and self.date_node
                and self.compute_layer
                in [
                    ComputeLayerEnum.sqlite,
                    ComputeLayerEnum.postgres,
                    ComputeLayerEnum.snowflake,
                    ComputeLayerEnum.redshift,
                    ComputeLayerEnum.mysql,
                    ComputeLayerEnum.athena,
                    ComputeLayerEnum.databricks,
                    ComputeLayerEnum.trino,
                    ComputeLayerEnum.duckdb,
                ]
            ):
                logger.info(f"Got date column of {self.date_node.date_key}")
                date_col = self.colabbr(self.date_key)
                cutoff_col = f"{self.date_node.prefix}_{self.date_node.date_key}"
                days = (
                    self.compute_period_val
                )  # ← this is your lookback / compute period

                # Dialect-specific date subtraction
                if self.compute_layer in [
                    ComputeLayerEnum.duckdb,
                    ComputeLayerEnum.postgres,
                    ComputeLayerEnum.redshift,
                ]:
                    lower_bound = f"{cutoff_col} - INTERVAL '{days} days'"

                elif self.compute_layer in [
                    ComputeLayerEnum.snowflake,
                    ComputeLayerEnum.mysql,
                ]:
                    lower_bound = f"DATEADD(DAY, -{days}, {cutoff_col})"

                elif self.compute_layer == ComputeLayerEnum.databricks:
                    lower_bound = f"{cutoff_col} - INTERVAL {days} DAYS"  # no quotes around number

                elif self.compute_layer == ComputeLayerEnum.athena:
                    lower_bound = (
                        f"{cutoff_col} - INTERVAL '{days}' DAY"  # singular "DAY"
                    )

                elif self.compute_layer == ComputeLayerEnum.trino:
                    lower_bound = f"{cutoff_col} - INTERVAL '{days}' DAY"

                elif self.compute_layer == ComputeLayerEnum.sqlite:
                    lower_bound = f"DATE({cutoff_col}, '-{days} days')"

                else:
                    raise NotImplementedError(
                        f"Date subtraction not implemented for compute layer: {self.compute_layer}"
                    )

                return [
                    sqlop(optype=SQLOpType.where, opval=f"{date_col} < {cutoff_col}"),
                    sqlop(optype=SQLOpType.where, opval=f"{date_col} > {lower_bound}"),
                ]

            elif (
                self.cut_date
                and isinstance(self.cut_date, str)
                or isinstance(self.cut_date, datetime.datetime)
            ):
                # Using a SQL engine so need to return `sqlop` instances.
                if self.compute_layer in [
                    ComputeLayerEnum.sqlite,
                    ComputeLayerEnum.postgres,
                    ComputeLayerEnum.snowflake,
                    ComputeLayerEnum.redshift,
                    ComputeLayerEnum.mysql,
                    ComputeLayerEnum.athena,
                    ComputeLayerEnum.databricks,
                    ComputeLayerEnum.trino,
                    ComputeLayerEnum.duckdb,
                ]:
                    return [
                        sqlop(
                            optype=SQLOpType.where,
                            opval=f"{self.colabbr(self.date_key)} < '{str(self.cut_date)}'",
                        ),
                        sqlop(
                            optype=SQLOpType.where,
                            opval=f"{self.colabbr(self.date_key)} > '{str(self.cut_date - datetime.timedelta(minutes=self.compute_period_minutes()))}'",
                        ),
                    ]

                elif isinstance(self.df, pd.DataFrame) or isinstance(
                    self.df, dd.DataFrame
                ):
                    return self.df[
                        (
                            (self.df[self.colabbr(self.date_key)] < str(self.cut_date))
                            & (
                                self.df[self.colabbr(self.date_key)]
                                > str(
                                    self.cut_date
                                    - datetime.timedelta(
                                        minutes=self.compute_period_minutes()
                                    )
                                )
                            )
                        )
                        | (self.df[self.colabbr(self.date_key)].isnull())
                    ]
                elif isinstance(self.df, SPARK_DF_TYPES):
                    return self.df.filter(
                        (
                            (self.df[self.colabbr(self.date_key)] < self.cut_date)
                            & (
                                self.df[self.colabbr(self.date_key)]
                                > (
                                    self.cut_date
                                    - datetime.timedelta(
                                        minutes=self.compute_period_minutes()
                                    )
                                )
                            )
                        )
                        | (self.df[self.colabbr(self.date_key)].isNull())
                    )
                elif isinstance(self.df, DAFT_DF_TYPES):
                    return self.df.filter(
                        (
                            (self.df[self.colabbr(self.date_key)] < str(self.cut_date))
                            & (
                                self.df[self.colabbr(self.date_key)]
                                > str(
                                    (
                                        self.cut_date
                                        - datetime.timedelta(
                                            minutes=self.compute_period_minutes()
                                        )
                                    )
                                )
                            )
                        )
                        | (self.df[self.colabbr(self.date_key)].is_null())
                    )

            else:
                # Using a SQL engine so need to return `sqlop` instances.
                if self.compute_layer in [
                    ComputeLayerEnum.sqlite,
                    ComputeLayerEnum.postgres,
                    ComputeLayerEnum.snowflake,
                    ComputeLayerEnum.redshift,
                    ComputeLayerEnum.mysql,
                    ComputeLayerEnum.athena,
                    ComputeLayerEnum.databricks,
                    ComputeLayerEnum.duckdb,
                ]:
                    return [
                        sqlop(
                            optype=SQLOpType.where,
                            opval=f"{self.colabbr(self.date_key)} < '{str(datetime.datetime.now())}'",
                        ),
                        sqlop(
                            optype=SQLOpType.where,
                            opval=f"{self.colabbr(self.date_key)} > '{str(datetime.datetime.now() - datetime.timedelta(minutes=self.compute_period_minutes()))}'",
                        ),
                    ]

                elif isinstance(self.df, pd.DataFrame) or isinstance(
                    self.df, dd.DataFrame
                ):
                    return self.df[
                        (
                            (
                                self.df[self.colabbr(self.date_key)]
                                < datetime.datetime.now()
                            )
                            & (
                                self.df[self.colabbr(self.date_key)]
                                > (
                                    datetime.datetime.now()
                                    - datetime.timedelta(
                                        minutes=self.compute_period_minutes()
                                    )
                                )
                            )
                        )
                        | (self.df[self.colabbr(self.date_key)].isnull())
                    ]
                elif isinstance(self.df, SPARK_DF_TYPES):
                    return self.df.filter(
                        (
                            self.df[self.colabbr(self.date_key)]
                            > (
                                datetime.datetime.now()
                                - datetime.timedelta(
                                    minutes=self.compute_period_minutes()
                                )
                            )
                        )
                        | (self.df[self.colabbr(self.date_key)].isNull())
                    )
                elif isinstance(self.df, DAFT_DF_TYPES):
                    return self.df.filter(
                        (
                            self.df[self.colabbr(self.date_key)]
                            > (
                                datetime.datetime.now()
                                - datetime.timedelta(
                                    minutes=self.compute_period_minutes()
                                )
                            )
                        )
                        | (self.df[self.colabbr(self.date_key)].is_null())
                    )

        # SQL engine do nothing.
        elif not hasattr(self, "df"):
            return None
        # no-op
        return self.df

    def prep_for_labels(
        self,
    ) -> typing.Union[pd.DataFrame, dd.DataFrame, pyspark.sql.dataframe.DataFrame]:
        """
        Prepare the dataset for labels
        """
        if self.date_key:
            # Date filters when we have a
            # date node
            if (
                hasattr(self, "date_node")
                and self.date_node
                and self.compute_layer
                in [
                    ComputeLayerEnum.sqlite,
                    ComputeLayerEnum.postgres,
                    ComputeLayerEnum.snowflake,
                    ComputeLayerEnum.redshift,
                    ComputeLayerEnum.mysql,
                    ComputeLayerEnum.athena,
                    ComputeLayerEnum.databricks,
                    ComputeLayerEnum.trino,
                    ComputeLayerEnum.duckdb,
                ]
            ):
                date_col = f"{self.colabbr(self.date_key)}"
                cutoff_col = f"{self.date_node.prefix}_{self.date_node.date_key}"
                days = self.label_period_val

                if self.compute_layer in [
                    ComputeLayerEnum.duckdb,
                    ComputeLayerEnum.postgres,
                    ComputeLayerEnum.redshift,
                ]:
                    upper_bound = f"{cutoff_col} + INTERVAL '{days} days'"

                elif self.compute_layer in [
                    ComputeLayerEnum.snowflake,
                    ComputeLayerEnum.mysql,
                ]:
                    upper_bound = f"DATEADD(DAY, {days}, {cutoff_col})"

                elif self.compute_layer == ComputeLayerEnum.databricks:
                    upper_bound = f"{cutoff_col} + INTERVAL {days} DAYS"  # no quotes around number

                elif self.compute_layer in [ComputeLayerEnum.athena]:
                    upper_bound = (
                        f"{cutoff_col} + INTERVAL '{days}' DAY"  # singular "DAY"
                    )

                elif self.compute_layer == ComputeLayerEnum.trino:
                    upper_bound = f"{cutoff_col} + INTERVAL '{days}' DAY"

                elif self.compute_layer == ComputeLayerEnum.sqlite:
                    upper_bound = f"DATE({cutoff_col}, '+{days} days')"

                else:
                    # fallback or raise error
                    raise NotImplementedError(
                        f"Date arithmetic not implemented for {self.compute_layer}"
                    )

                return [
                    sqlop(optype=SQLOpType.where, opval=f"{date_col} > {cutoff_col}"),
                    sqlop(optype=SQLOpType.where, opval=f"{date_col} < {upper_bound}"),
                ]

            elif (
                self.cut_date
                and isinstance(self.cut_date, str)
                or isinstance(self.cut_date, datetime.datetime)
            ):
                # Using a SQL engine so need to return `sqlop` instances.
                if self.compute_layer in [
                    ComputeLayerEnum.sqlite,
                    ComputeLayerEnum.postgres,
                    ComputeLayerEnum.snowflake,
                    ComputeLayerEnum.redshift,
                    ComputeLayerEnum.mysql,
                    ComputeLayerEnum.athena,
                    ComputeLayerEnum.databricks,
                    ComputeLayerEnum.trino,
                    ComputeLayerEnum.duckdb,
                ]:
                    return [
                        sqlop(
                            optype=SQLOpType.where,
                            opval=f"{self.colabbr(self.date_key)} > '{str(self.cut_date)}'",
                        ),
                        sqlop(
                            optype=SQLOpType.where,
                            opval=f"{self.colabbr(self.date_key)} < '{str(self.cut_date + datetime.timedelta(minutes=self.label_period_minutes()))}'",
                        ),
                    ]

                elif isinstance(self.df, pd.DataFrame):
                    return self.df[
                        (self.df[self.colabbr(self.date_key)] > str(self.cut_date))
                        & (
                            self.df[self.colabbr(self.date_key)]
                            < str(
                                self.cut_date
                                + datetime.timedelta(
                                    minutes=self.label_period_minutes()
                                )
                            )
                        )
                    ]
                elif isinstance(self.df, SPARK_DF_TYPES):
                    return self.df.filter(
                        (self.df[self.colabbr(self.date_key)] > str(self.cut_date))
                        & (
                            self.df[self.colabbr(self.date_key)]
                            < str(
                                self.cut_date
                                + datetime.timedelta(
                                    minutes=self.label_period_minutes()
                                )
                            )
                        )
                    )
                elif isinstance(self.df, DAFT_DF_TYPES):
                    return self.df.filter(
                        (self.df[self.colabbr(self.date_key)] > str(self.cut_date))
                        & (
                            self.df[self.colabbr(self.date_key)]
                            < str(
                                self.cut_date
                                + datetime.timedelta(
                                    minutes=self.label_period_minutes()
                                )
                            )
                        )
                    )
            else:
                # Using a SQL engine so need to return `sqlop` instances.
                if self.compute_layer in [
                    ComputeLayerEnum.sqlite,
                    ComputeLayerEnum.postgres,
                    ComputeLayerEnum.snowflake,
                    ComputeLayerEnum.redshift,
                    ComputeLayerEnum.mysql,
                    ComputeLayerEnum.athena,
                    ComputeLayerEnum.databricks,
                    ComputeLayerEnum.trino,
                    ComputeLayerEnum.duckdb,
                ]:
                    return [
                        sqlop(
                            optype=SQLOpType.where,
                            opval=f"{self.colabbr(self.date_key)} > '{str(datetime.datetime.now() - datetime.timedelta(minutes=self.label_period_minutes()))}'",
                        )
                    ]

                elif isinstance(self.df, pd.DataFrame):
                    return self.df[
                        self.df[self.colabbr(self.date_key)]
                        > (
                            datetime.datetime.now()
                            - datetime.timedelta(minutes=self.label_period_minutes())
                        )
                    ]
                elif isinstance(self.df, SPARK_DF_TYPES):
                    return self.df.filter(
                        self.df[self.colabbr(self.date_key)]
                        > (
                            datetime.datetime.now()
                            - datetime.timedelta(minutes=self.label_period_minutes())
                        )
                    )
                elif isinstance(self.df, DAFT_DF_TYPES):
                    return self.df.filter(
                        self.df[self.colabbr(self.date_key)]
                        > (
                            datetime.datetime.now()
                            - datetime.timdelta(minutes=self.label_period_minutes())
                        )
                    )

        elif not hasattr(self, "df"):
            return None
        # no-op
        return self.df

    def default_label(
        self,
        op: typing.Union[str, callable],
        field: str,
        reduce_key: typing.Optional[KeySpec] = None,
    ) -> typing.Union[
        pd.DataFrame, dd.DataFrame, pyspark.sql.dataframe.DataFrame, typing.List[sqlop]
    ]:
        """
        Default label operation.

                Arguments
                ----------
                op: operation to call for label
                field: str label field to call operation on
                reduce: bool whether or not to reduce
        """
        if hasattr(self, "df"):
            if (
                self.compute_layer in [ComputeLayerEnum.pandas, ComputeLayerEnum.dask]
                and self.colabbr(field) in self.df.columns
            ):
                if self.reduce:
                    reduce_columns = list(self.colabbrs(reduce_key))
                    if callable(op):
                        return (
                            self.prep_for_labels()
                            .groupby(reduce_columns)
                            .agg(
                                **{
                                    self.colabbr(field + "_label"): pd.NamedAgg(
                                        column=self.colabbr(field), aggfunc=op
                                    )
                                }
                            )
                            .reset_index()
                        )
                    else:
                        if op == "bool":
                            grp = (
                                self.prep_for_labels()
                                .groupby(reduce_columns)
                                .agg(
                                    **{
                                        self.colabbr(field + "_label"): pd.NamedAgg(
                                            column=self.colabbr(field), aggfunc="count"
                                        )
                                    }
                                )
                                .reset_index()
                            )
                            grp[f"{self.colabbr(field)}_label"] = grp[
                                f"{self.colabbr(field)}_label"
                            ].apply(lambda x: 1 if x >= 1 else 0)
                            return grp
                        return (
                            self.prep_for_labels()
                            .groupby(reduce_columns)
                            .agg(
                                **{
                                    self.colabbr(field + "_label"): pd.NamedAgg(
                                        column=self.colabbr(field), aggfunc=op
                                    )
                                }
                            )
                            .reset_index()
                        )

                else:
                    label_df = self.prep_for_labels()
                    if callable(op):
                        label_df[self.colabbr(field) + "_label"] = label_df[
                            self.colabbr(field)
                        ].apply(op)
                    else:
                        label_df[self.colabbr(field) + "_label"] = label_df[
                            self.colabbr(field)
                        ].apply(lambda x: getattr(x, op)())
                    return label_df[
                        [
                            *self.colabbrs(self.pk),
                            self.colabbr(field) + "_label",
                        ]
                    ]

            elif (
                self.compute_layer == ComputeLayerEnum.spark
                and self.colabbr(field) in self.df.columns
            ):
                if self.reduce:
                    return (
                        self.prep_for_labels()
                        .groupBy(*self.colabbrs(reduce_key))
                        .agg(
                            getattr(F, op)(F.col(self.colabbr(field))).alias(
                                f"{self.colabbr(field)}_label"
                            )
                        )
                    )
                else:
                    pass
            elif (
                self.compute_layer == ComputeLayerEnum.daft
                and self.colabbr(field) in self.df.column_names
            ):
                if self.reduce:
                    aggcol = daft.col(self.colabbr(field))
                    return (
                        self.prep_for_labels()
                        .groupby(*self.colabbrs(reduce_key))
                        .agg(
                            getattr(aggcol, op)().alias(f"{self.colabbr(field)}_label")
                        )
                    )
                else:
                    pass
        elif self.compute_layer in [
            ComputeLayerEnum.snowflake,
            ComputeLayerEnum.sqlite,
            ComputeLayerEnum.mysql,
            ComputeLayerEnum.postgres,
            ComputeLayerEnum.redshift,
            ComputeLayerEnum.athena,
            ComputeLayerEnum.databricks,
            ComputeLayerEnum.trino,
            ComputeLayerEnum.duckdb,
        ]:
            if self.reduce:
                if op == "bool":
                    label_query = self.prep_for_labels() + [
                        sqlop(
                            optype=SQLOpType.agg,
                            opval=self.key_sql(reduce_key),
                        ),
                        sqlop(
                            optype=SQLOpType.aggfunc,
                            opval=f"CASE WHEN COUNT({self.colabbr(field)}) >= 1 THEN 1 ELSE 0 END as {self.colabbr(field)}_label",
                        ),
                    ]
                else:
                    label_query = self.prep_for_labels() + [
                        sqlop(
                            optype=SQLOpType.agg,
                            opval=self.key_sql(reduce_key),
                        ),
                        sqlop(
                            optype=SQLOpType.aggfunc,
                            opval=f"{op}"
                            + f"({self.colabbr(field)}) as {self.colabbr(field)}_label",
                        ),
                    ]
                logger.info(self.build_query(label_query))
                return label_query
            else:
                label_query = self.prep_for_labels() + [
                    sqlop(
                        optype=SQLOpType.select,
                        opval=f"{op}"
                        + f"({self.colabbr(field)}) as {self.colabbr(field)}_label",
                    )
                ]
                logger.info(self.build_query(label_query))
                return label_query
        else:
            pass

    def online_features(
        self,
    ):
        """
        Define online features.
        """
        pass

    def on_demand_features(
        self,
    ):
        """
        Define on demand features for this node.
        """
        pass


class DynamicNode(GraphReduceNode):
    """
    A dynamic architecture for entities with no logic
    needed in addition to the top-level GraphReduceNode
    parameters.  The required abstract methods:
        `do_annotate`
        `do_filters`
        `do_normalize`
        `do_post_join_filters`
        `do_post_join_annotate`

    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        Constructor
        """
        super().__init__(*args, **kwargs)

    def do_filters(self):
        pass

    def do_annotate(self):
        pass

    def do_post_join_annotate(self):
        pass

    def do_normalize(self):
        pass

    def do_post_join_filters(self):
        pass

    def do_reduce(self, reduce_key: KeySpec):
        pass

    def do_labels(self, reduce_key: KeySpec):
        pass


class GraphReduceQueryException(Exception):
    pass


class SQLNode(GraphReduceNode):
    """
    Base node for SQL engines.  Makes some common
    operations available.  This class should be
    extended for engines that do not conform
    to a single `client` interface, such as
    AWS Athena, which requires additional params.

    Subclasses should simply extend the `SQLNode` interface:
    """

    FUNCTION_MAPPING = {
        "mean": "avg",
        "median": None,
        "nunique": None,
    }

    PICK_ONE_VALUE_FUNCTION = {
        ComputeLayerEnum.snowflake: "any_value",
        ComputeLayerEnum.redshift: "any_value",
    }

    def __init__(
        self,
        *args,
        client: typing.Any = None,
        table_name: typing.Optional[str] = None,
        lazy_execution: bool = False,
        dry_run: bool = False,
        # For loading the data in.
        do_data_ops: typing.Optional[typing.List[sqlop]] = None,
        # For widening the data in terms of columns.
        do_annotate_ops: typing.Optional[typing.List[sqlop]] = None,
        # For shrinking the row count with filters.
        do_filters_ops: typing.Optional[typing.List[sqlop]] = None,
        # For reduction / compression via aggregation.
        do_reduce_ops: typing.Optional[typing.List[sqlop]] = None,
        # For computing machine learning labels.
        do_labels_ops: typing.Optional[typing.List[sqlop]] = None,
        # For adding data to widen the data after a join.
        do_post_join_annotate_ops: typing.Optional[typing.List[sqlop]] = None,
        # For applying filters after a join.
        do_post_join_filters_ops: typing.Optional[typing.List[sqlop]] = None,
        # Specification of the nodes that need to be joined prior
        # to executing `do_post_join_annotate`.
        do_post_join_annotate_requires: typing.Optional[
            typing.List[GraphReduceNode]
        ] = None,
        # Specification of the nodes that need to be joined prior
        # to executing `do_post_join_filters`.
        do_post_join_filters_requires: typing.Optional[
            typing.List[GraphReduceNode]
        ] = None,
        **kwargs,
    ):
        """
        Constructor for `SQLNode`.

                Arguments
                -------------
                client: a SQL engine client that can `.execute()` sql queries
                lazy_execution: bool whether or not to execute lazily
                dry_run: don't execute but log the SQL that would be executed
                do_annotate_ops: list of `sqlop` instances for `do_annotate`
                do_filters_ops: list of `sqlop` instances for `do_filters`
                do_reduce_ops: list of `sqlop` instances for `do_reduce`
                do_labels_ops: list of `sqlop` instances for `do_labels`
                do_post_join_annotate_ops: list of `sqlop` instances for `do_post_join_annotate`
                do_post_join_filters_ops: list of `sqlop` instances for `do_post_join_filters`
                do_post_join_annotate_requires: list of `SQLNode` instances that this method requires be joined prior to executing
                do_post_join_filters_requires: list of `SQLNode` instances that this method requires be joined priot to executing
        """
        self._sql_client = client
        self.table_name = table_name
        self.lazy_execution = lazy_execution
        self.dry_run = dry_run

        # The current data ref.
        self._cur_data_ref = None
        # A place to store temporary tables or views.
        self._temp_refs = {}
        self._all_refs = []
        self._removed_refs = []
        # A place to store the full SQL for creating temp references.
        # only ever store the current `_ref_sql`.
        self._ref_sql = None

        self.do_data_ops = do_data_ops
        self.do_annotate_ops = do_annotate_ops
        self.do_filters_ops = do_filters_ops
        self.do_reduce_ops = do_reduce_ops
        self.do_labels_ops = do_labels_ops
        self.do_post_join_annotate_ops = do_post_join_annotate_ops
        self.do_post_join_filters_ops = do_post_join_filters_ops
        self.do_post_join_annotate_requires = do_post_join_annotate_requires
        self.do_post_join_filters_requires = do_post_join_filters_requires

        # SQL operations for this node.
        self.sql_ops = []

        super().__init__(*args, **kwargs)

    def get_temp_refs(self):
        return self._temp_refs

    def get_pick_one_value_agg(self) -> str:
        return self.PICK_ONE_VALUE_FUNCTION.get(self.compute_layer, "first")

    def _clean_refs(self):
        """
        Cleanup tables created during execution.
        """
        for k, v in self._all_refs:
            if v not in self._removed_refs:
                sql = f"DROP VIEW {v}"
                self.execute_query(sql)
                self._removed_refs.append(v)
                logger.info(f"dropped {v}")

    def get_ref_name(
        self,
        fn: typing.Union[callable, str] = None,
        lookup: bool = False,
        schema: str = None,
    ) -> str:
        """
        Get a reference name for the function.
        """
        func_name = fn if isinstance(fn, str) else fn.__name__
        # If this ref name is already in
        # the _temp_refs dict create a new ref.

        # If the node has a `.table_name`
        # which is the case for `duckdb`
        # then use that.
        if hasattr(self, "table_name") and self.table_name:
            fpath = self.table_name
        # IF there is a schema in the fpath
        # we need to remove the schema.
        elif "." in self.fpath and "/" not in self.fpath:
            fpath = self.fpath.split(".")[-1]
        else:
            fpath = self.fpath

        namespace = self.execution_namespace
        namespace_suffix = f"_{namespace}" if namespace else ""
        if schema:
            ref_name = (
                f"{schema}.{fpath}_{self.prefix}_{func_name}"
                f"{namespace_suffix}_grtemp"
            )
        else:
            ref_name = f"{fpath}_{self.prefix}_{func_name}{namespace_suffix}_grtemp"
        if self._temp_refs.get(func_name):
            if lookup:
                return self._temp_refs[func_name]

            if ref_name in self._all_refs:
                i = 1
                while ref_name in self._all_refs:
                    ref_name = ref_name + str(i)
                    i += 1
            return ref_name

        else:
            if ref_name in self._all_refs:
                i = 1
                while ref_name in self._all_refs:
                    ref_name = ref_name + str(i)
                    i += 1
            return ref_name

    def create_ref(
        self,
        sql: str = "",
        fn: typing.Union[callable, str] = None,
        # Overwrite the
        overwrite: bool = False,
        schema: str = None,
        dry: bool = False,
    ) -> str:
        """
        Gets a temporary table or view name
        based on the method being called.
        """

        # No reference has been created for this method.
        fn = fn if isinstance(fn, str) else fn.__name__

        # If no SQL was provided use the current reference.
        if not sql:  # or self._temp_refs.get(fn):
            logger.info(
                f"no sql was provided for {fn} or {fn} has already been executed"
            )
            self._temp_refs[fn] = self._cur_data_ref
            return self._cur_data_ref

        if not self._temp_refs.get(fn) or overwrite:
            ref_name = self.get_ref_name(fn, schema=schema)
            self._all_refs.append(ref_name)
            created_ref = ref_name
            if not dry:
                created_ref = self.create_temp_view(sql, ref_name, dry=dry)
                if not created_ref:
                    return None
            self._temp_refs[fn] = created_ref
            return created_ref
        # Reference for this method already created
        # so we will just retrieve.
        else:
            return self._temp_refs[fn]

    def get_current_ref(self) -> str:
        """
        Returns the name of the current
        reference to the nodes data.
        """
        if not self._cur_data_ref:
            return self.fpath
        else:
            return self._cur_data_ref

    def create_date_table(
        self,
        # If not set will default to
        # the primary key of the table.
        lookup_key: typing.Optional[KeySpec] = None,
        # If not set will default
        # to instance attr.
        date_key: str = None,
    ) -> str:
        """
        Create a date table view
        id, date_key
        1,2022-01-01.
        """
        lookup_key = self.pk if lookup_key is None else lookup_key
        sql = f"""
        CREATE VIEW {view_name} AS
        SELECT {self.key_sql(lookup_key)}, {self.colabbr(self.date_key)}
        FROM {self._cur_data_ref}
        """
        pass

    def create_temp_view(
        self,
        qry: str,
        view_name: str,
        overwrite: bool = False,
        dry: bool = False,
    ) -> str:
        """
        Create a view with the results of
        the query.
        """
        self.execute_query(
            f"""
        DROP VIEW IF EXISTS {view_name}
        """,
            ret_df=False,
        )

        sql = f"""
        CREATE VIEW {view_name} AS
        {qry}
        """
        self._ref_sql = sql
        logger.info(sql)
        # Only execute when it is not a dry run
        # but always append the SQL.
        if not dry:
            self.execute_query(sql, ret_df=False)
        self._cur_data_ref = view_name
        return view_name

    # TODO(wes): optimize by storing previously
    # fetch samples.
    def get_sample(
        self,
        n: int = 1000,
        table: str = None,
    ) -> pd.DataFrame:
        """
        Gets a sample of rows for the current
        table or a parameterized table.
        """
        samp_query = """
            SELECT *
            FROM {table}
            LIMIT {n}
            """
        if not table:
            qry = samp_query.format(
                table=self._cur_data_ref if self._cur_data_ref else self.fpath, n=n
            )
        else:
            qry = samp_query.format(table=table, n=n)
        samp = self.execute_query(qry)
        logger.info(f"Got sample of {self._cur_data_ref} with columns: {samp.columns}")
        if not self._stypes:
            self._stypes = infer_df_stype(samp)
        return samp

    def _sample_identifier(self, identifier: str) -> str:
        """
        Format an identifier for SQL sampling queries.

        Subclasses can override this if a backend requires dialect-specific
        quoting for generated sampling expressions.
        """
        return identifier

    def _sample_table(self, table: str = None) -> str:
        return table if table else self.get_current_ref()

    def get_most_populated_sample(
        self,
        n: int = 1000,
        table: str = None,
        columns: typing.Optional[typing.List[str]] = None,
    ) -> pd.DataFrame:
        """
        Sample rows that have the most non-null values for semantic type inference.
        """
        table = self._sample_table(table)
        if columns is None:
            preview = self.get_sample(n=1, table=table)
            columns = list(preview.columns)
        if not columns:
            return self.get_sample(n=n, table=table)

        score = _balanced_sql_add(
            [
                f"CASE WHEN {self._sample_identifier(col)} IS NOT NULL THEN 1 ELSE 0 END"
                for col in columns
            ]
        )
        qry = f"""
            SELECT *
            FROM {table}
            ORDER BY ({score}) DESC
            LIMIT {n}
            """
        return self.execute_query(qry)

    def get_non_null_sample(
        self,
        column: str,
        n: int = 20,
        table: str = None,
    ) -> pd.DataFrame:
        """
        Sample rows where a specific column is populated.
        """
        table = self._sample_table(table)
        col = self._sample_identifier(column)
        qry = f"""
            SELECT *
            FROM {table}
            WHERE {col} IS NOT NULL
            LIMIT {n}
            """
        return self.execute_query(qry)

    def get_inference_sample(
        self,
        n: int = 1000,
        backfill_per_column: int = 20,
        table: str = None,
    ) -> pd.DataFrame:
        """
        Build a sample for semantic type inference with non-null examples.

        The base sample prefers rows with the most populated values. Any column
        still entirely null in that sample gets a small targeted backfill query.
        """
        table = self._sample_table(table)
        sample = self.get_most_populated_sample(n=n, table=table)
        missing_cols = [
            col for col in sample.columns if sample[col].notna().sum() == 0
        ]
        if not missing_cols:
            return sample

        backfills = [
            self.get_non_null_sample(col, n=backfill_per_column, table=table)
            for col in missing_cols
        ]
        populated_backfills = [df for df in backfills if df is not None and not df.empty]
        if not populated_backfills:
            return sample

        return pd.concat([sample, *populated_backfills], ignore_index=True)

    def build_query(
        self,
        ops: typing.Union[typing.List[sqlop], sqlop],
        data_ref: str = None,
    ) -> str:
        """
        Builds a SQL query given a list of `sqlop` instances.

             Parameters
             ----------
             ops: List of `sqlop` instances
             data_ref: (optional) str of the data reference to use
        """

        if not ops:
            return None

        # Custom ops are returned as is.
        if isinstance(ops, sqlop) and ops.optype == SQLOpType.custom:
            return ops.opval

        if isinstance(ops, list):
            pass
        elif isinstance(ops, sqlop):
            ops = [ops]

        select_anatomy = """
        SELECT {selects}
        FROM {from_}
        WHERE {wheres}
        """

        group_anatomy = """
        SELECT {selects},
        {aggfuncs}
        FROM {from_}
        WHERE {wheres}
        GROUP BY {group}
        """

        # If a data reference is passed as
        # a parameter we will use that.
        if data_ref:
            dr = data_ref
        else:
            dr = self._cur_data_ref if self._cur_data_ref else self.fpath

        # Table to select from.
        from_ = sqlop(optype=SQLOpType.from_, opval=dr)

        # Boolean if this is an aggregation function.
        if len([_x for _x in ops if _x.optype == SQLOpType.agg]):
            aggfuncs = [_x for _x in ops if _x.optype == SQLOpType.aggfunc]
            if not len(aggfuncs):
                raise GraphReduceQueryException(
                    "Aggregation queries must have at least 1 `sqlop` of type SQLOpType.aggfunc"
                )
            aggfuncs = ",".join([_x.opval for _x in aggfuncs])

            # Can only be one aggregation per query build.
            agg = [_x for _x in ops if _x.optype == SQLOpType.agg][0].opval

            wheres = [_x for _x in ops if _x.optype == SQLOpType.where]
            if not len(wheres):
                wheres = "true"
            else:
                wheres = " and ".join([_x.opval for _x in wheres])

            return group_anatomy.format(
                selects=agg,
                aggfuncs=aggfuncs,
                from_=from_.opval,
                group=agg,
                wheres=wheres,
            )

        # Otherwise go with standard select anatomy.
        else:
            selects = [_x for _x in ops if _x.optype == SQLOpType.select]
            # IF not select statements, select *.
            if not len(selects):
                selects = [sqlop(optype=SQLOpType.select, opval="*")]

            qry_selects = ",".join([_x.opval for _x in selects])
            # froms = [_x for _x in ops if _x.optype == SQLOpType.from_][0].opval
            wheres = [_x for _x in ops if _x.optype == SQLOpType.where]
            if len(wheres):
                qry_wheres = " and ".join([_x.opval for _x in wheres])
            else:
                qry_wheres = "true"

            return select_anatomy.format(
                selects=qry_selects, from_=from_.opval, wheres=qry_wheres
            )

    def get_client(self) -> typing.Any:
        return self._sql_client

    def execute_query(
        self,
        qry: str,
        ret_df: bool = True,
        commit: bool = False,
    ) -> typing.Optional[typing.Union[None, pd.DataFrame]]:
        """
        Execute a query and get back a dataframe.
        """

        client = self.get_client()
        if not ret_df:
            cur = client.cursor()
            cur.execute(qry)
            if commit:
                client.commit()
        else:
            return pd.read_sql_query(qry, client)

    def do_data(self) -> typing.Union[sqlop, typing.List[sqlop]]:
        """
        Load the data.
        """
        if self.do_data_ops:
            return self.do_data_ops
        col_renames = [f"{col} as {self.colabbr(col)}" for col in self.columns]
        sel = sqlop(optype=SQLOpType.select, opval=f"{','.join(col_renames)}")
        return [sel]

    def sql_auto_annotate(
        self,
        table_df_sample: pd.DataFrame,
    ) -> typing.List[sqlop]:
        self._stypes = infer_df_stype(table_df_sample)
        return _sql_auto_annotate_ops(
            table_df_sample=table_df_sample,
            stypes=self._stypes,
            cardinality_threshold=self.categorical_cardinality_threshold,
            top_k=self.categorical_top_k,
            max_categorical_columns=self.auto_annotate_max_categorical_columns,
            max_gated_numeric_cols=self.auto_annotate_max_gated_numeric_cols,
            gated_numeric_top_k=self.auto_annotate_gated_numeric_top_k,
            auto_text_features=self.auto_text_features,
            annotation_expressions=self.annotation_expressions,
            column_prefix=self.prefix,
            annotation_expressions_only=(
                self.annotation_expressions_only
                or (
                    bool({"semantic", "context"}.intersection(self.feature_families))
                    and not self.auto_annotate_features
                )
            ),
            context_features=(
                "context" in self.feature_families and bool(self.context_keys)
            ),
            context_keys=self.context_keys,
            context_pk=self.pk,
            context_max_numeric_columns=self.feature_family_max_columns,
        )

    def do_annotate(self) -> typing.Union[sqlop, typing.List[sqlop]]:
        """
        Should return a list of SQL statements
        casting columns as different types.
        """
        if self.do_annotate_ops:
            return self.do_annotate_ops
        semantic_requested = (
            "semantic" in self.feature_families and bool(self.annotation_expressions)
        )
        context_requested = "context" in self.feature_families and bool(self.context_keys)
        if self.auto_annotate_features or semantic_requested or context_requested:
            if self.dry_run:
                return None
            try:
                sample = self.get_inference_sample()
                return self.sql_auto_annotate(sample)
            except Exception as exc:
                logger.warning(f"skipped sql_auto_annotate for {self}: {exc}")
        return None

    def do_normalize(self) -> typing.Union[sqlop, typing.List[sqlop]]:
        return None

    def do_filters(self) -> typing.Union[sqlop, typing.List[sqlop]]:
        """
        Example:
        return [
            sqlop(optype=SQLOpType.where, opval=f"{self.colabbr('id')} < 1000"),
        ]
        """
        if self.do_filters_ops:
            return self.do_filters_ops
        return None

    # Returns aggregate functions
    # Returns aggregate
    def do_reduce(self, reduce_key) -> typing.Union[sqlop, typing.List[sqlop]]:
        """
        Example:
        return [
            sqlop(optype=SQLOpType.aggfunc, opval=f"count(*) as {self.colabbr('num_dupes')}"),
            sqlop(optype=SQLOpType.agg, opval=self.key_sql(reduce_key))
        ]
        """
        if self.do_reduce_ops:
            return self.do_reduce_ops
        return None

    def do_post_join_annotate(self) -> typing.Union[sqlop, typing.List[sqlop]]:
        if self.do_post_join_annotate_ops:
            if self.do_post_join_annotate_requires:
                # Check if all required nodes are merged.
                samp = self.get_sample(n=100)
                all_merged = True
                for node in self.do_post_join_annotate_requires:
                    if not len(
                        [c for c in samp.columns if c.startswith(f"{node.prefix}_")]
                    ):
                        all_merged = False
                        logger.debug("All dependencies not merged")
                if all_merged:
                    logger.debug("All dependencies merged")
                    return self.do_post_join_annotate_ops
                else:
                    return None
            else:
                return self.do_post_join_annotate_ops
        return None

    def do_labels(self, reduce_key: KeySpec) -> typing.Union[sqlop, typing.List[sqlop]]:
        if self.do_labels_ops:
            return self.do_labels_ops
        return None

    def do_post_join_filters(self) -> typing.Union[sqlop, typing.List[sqlop]]:
        if self.do_post_join_filters_ops:
            if self.do_post_join_filters_requires:
                # Check if all required nodes are merged.
                samp = self.get_sample(n=100)
                all_merged = True
                for node in self.do_post_join_filters_requires:
                    if not len(
                        [c for c in samp.columns if c.startswith(f"{node.prefix}_")]
                    ):
                        all_merged = False
                        logger.debug("All dependencies not merged")
                if all_merged:
                    logger.debug("All dependencies merged")
                    return self.do_post_join_filters_ops
                else:
                    return None
            else:
                return self.do_post_join_filters_ops

            return self.do_post_join_filters_ops
        return None

    def do_sql(self) -> str:
        """
        One function to compute this entire node.
        """
        pass


class AthenaNode(SQLNode):
    def __init__(
        self,
        *args,
        s3_output_location: str = None,
        **kwargs,
    ):
        """
        Constructor.
        """

        self.s3_output_location = s3_output_location
        super(AthenaNode, self).__init__(*args, **kwargs)

    def prep_for_features(self):
        if self.cut_date:
            if isinstance(self.cut_date, str):
                self.cut_date = parser.parse(self.cut_date)
        else:
            self.cut_date = datetime.datetime.now()

        if self.cut_date and self.date_key:
            return [
                sqlop(
                    optype=SQLOpType.where,
                    opval=f"{self.colabbr(self.date_key)} > timestamp '{str(self.cut_date - datetime.timedelta(minutes=self.compute_period_minutes()))}'",
                ),
                sqlop(
                    optype=SQLOpType.where,
                    opval=f"{self.colabbr(self.date_key)} < timestamp '{str(self.cut_date)}'",
                ),
            ]
        else:
            # do nothing
            return None

    def prep_for_labels(self):
        if self.cut_date:
            if isinstance(self.cut_date, str):
                self.cut_date = parser.parse(self.cut_date)
        else:
            self.cut_date = datetime.datetime.now()

        if self.cut_date and self.date_key:
            return [
                sqlop(
                    optype=SQLOpType.where,
                    opval=f"{self.colabbr(self.date_key)} < timestamp '{str(self.cut_date + datetime.timedelta(minutes=self.label_period_minutes()))}'",
                ),
                sqlop(
                    optype=SQLOpType.where,
                    opval=f"{self.colabbr(self.date_key)} > timestamp '{str(self.cut_date)}'",
                ),
            ]
        else:
            # do nothing
            return None

    def execute_query(
        self,
        qry: str,
        *args,
        **kwargs,
    ) -> typing.Optional[pd.DataFrame]:
        """
        Execute a query and get back a dataframe.
        """

        logger.info(f"attempting to execute: {qry}")
        client = self.get_client()
        resp = client.start_query_execution(
            QueryString=qry,
            ResultConfiguration={"OutputLocation": self.s3_output_location},
        )
        # While query is executing sleep
        qry_id = resp["QueryExecutionId"]

        qry_status = client.get_query_execution(QueryExecutionId=qry_id)
        if qry_status["QueryExecution"]["Status"]["State"] == "FAILED":
            reason = qry_status["QueryExecution"]["Status"].get(
                "StateChangeReason", "unknown reason"
            )
            raise Exception(f"Query {qry} FAILED: {reason}")

        else:
            while qry_status["QueryExecution"]["Status"]["State"] not in [
                "SUCCEEDED",
                "FAILED",
            ]:
                logger.info("sleeping and waiting for query to finish")
                time.sleep(1)
                qry_status = client.get_query_execution(QueryExecutionId=qry_id)

        if qry_status["QueryExecution"]["Status"]["State"] == "FAILED":
            reason = qry_status["QueryExecution"]["Status"].get(
                "StateChangeReason", "unknown reason"
            )
            raise Exception(f"Query {qry} FAILED: {reason}")

        results = client.get_query_results(QueryExecutionId=qry_id)
        colinfo = results["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]
        rows = results["ResultSet"]["Rows"]
        while results.get("NextToken"):
            results = client.get_query_results(
                QueryExecutionId=qry_id, NextToken=results["NextToken"]
            )
            for row in results["ResultSet"]["Rows"]:
                rows.append(row)
        # create a dataframe ready version of the data
        dfdata = []
        for row in rows[1:]:
            newrow = {}
            for i in range(len(row["Data"])):
                col = colinfo[i]["Name"]
                if row["Data"][i]:
                    val = row["Data"][i]["VarCharValue"]
                else:
                    val = None
                newrow.update({col: val})
            dfdata.append(newrow)
        return pd.DataFrame(dfdata)


class DatabricksNode(SQLNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Databricks temporary views
    # drop on their own after the
    # session ends.
    def _clean_refs(self):
        return

    def create_temp_view(
        self,
        qry: str,
        view_name: str,
        dry: bool = False,
    ) -> str:
        """
        Create a view with the results
        of the query.
        """
        if len(view_name.split(".")) > 1:
            view_name = view_name.split(".")[-1]

        sql = f"""
        CREATE OR REPLACE TEMPORARY VIEW {view_name} AS
        {qry}
        """
        self._ref_sql = sql
        if not dry:
            self.execute_query(sql, ret_df=False)
        self._cur_data_ref = view_name
        return view_name


class RedshiftNode(SQLNode):
    def __init__(self, *args, **kwargs):
        """
        Constructor.
        """
        super().__init__(*args, **kwargs)

    def _clean_refs(self):
        for k, v in self._temp_refs.items():
            if v not in self._removed_refs:
                sql = f"DROP TABLE IF EXISTS {v}"
                self.execute_query(sql, ret_df=False, commit=True)
                self._removed_refs.append(v)
                logger.info(f"dropped {v}")

    def create_temp_view(
        self,
        qry: str,
        view_name: str,
        dry: bool = False,
    ) -> str:
        """
        Create a view with the results
        of the query.
        """
        # try:
        self.execute_query(
            f"DROP TABLE IF EXISTS {view_name}", ret_df=False, commit=True
        )
        sql = f"""
            CREATE TABLE {view_name}
            AS {qry}
            """
        self._ref_sql = sql
        if not dry:
            self.execute_query(sql, ret_df=False, commit=True)
        self._cur_data_ref = view_name
        return view_name

    # To be deprecated.
    def _sql_auto_features(
        self,
        table_df_sample: typing.Union[pd.DataFrame, dd.DataFrame],
        reduce_key: KeySpec,
        type_func_map: dict = {},
    ) -> typing.List[sqlop]:
        """
        SQL dialect implementation of automated
        feature engineering.

        At the moment we're just using `pandas` inferred
        data types for these operations.  This is an
        area that can benefit from `woodwork` and other
        data type inference libraries.
        """
        agg_funcs = []

        # Always need to update this
        # because we never know if
        # the original columns comprise all
        # of the columns currently in the df.
        self._stypes = infer_df_stype(table_df_sample)
        reduce_columns = self.colabbrs(reduce_key)
        reduce_column_names = set(reduce_columns) | set(key_parts(reduce_key))

        # Physical types.
        ptypes = {col: str(t) for col, t in table_df_sample.dtypes.to_dict().items()}

        ts_data = self.is_ts_data(reduce_key)
        counted = False
        for col, stype in self._stypes.items():
            # Get the last function applied (if any)
            if col.lower().split("_")[-1] in ["avg", "sum", "count", "min", "max"]:
                last_function = col.lower().split("_")[-1]
            else:
                last_function = None

            _type = str(stype)
            if (
                _type == "numerical"
                and _column_name_looks_like_categorical_identifier(col)
            ):
                _type = "categorical"
            if col in reduce_column_names:
                continue
            if self._is_identifier(col):
                # We only perform counts for identifiers.
                func = "count"
                col_new = f"{col}_{func}"
                if not counted:
                    agg_funcs.append(
                        sqlop(
                            optype=SQLOpType.aggfunc,
                            opval=f"{func}" + f"({col}) as {col_new}",
                        )
                    )
                    counted = True

            elif type_func_map.get(_type):
                if ptypes[col] == "bool":
                    col_new = f"{col}_sum"
                    op = sqlop(
                        optype=SQLOpType.aggfunc,
                        opval=f"sum(case when {col} = 1 then 1 else 0 end) as {col_new}",
                    )
                    if op not in agg_funcs:
                        agg_funcs.append(op)
                    continue
                for func in type_func_map[_type]:
                    # There should be a better top-level mapping
                    # but for now this will do.  SQL engines typically
                    # don't have 'median' and 'mean'.  'mean' is typically
                    # just called 'avg'.
                    if _should_skip_numeric_sql_agg(table_df_sample[col], _type, func):
                        logger.info(
                            f"skipped numeric aggregation {func} on {col} because "
                            "semantic numerical but physical values are not numeric"
                        )
                        continue
                    elif func in self.FUNCTION_MAPPING:
                        func = self.FUNCTION_MAPPING.get(func)

                    elif not func or func == "nunique":
                        continue

                    # Redshift-specific
                    elif func == "first":
                        func = "any_value"

                    # For categorical scalar types check if it is
                    # a 0, 1 category and, if so, do a sum.
                    non_null_vals = table_df_sample[~table_df_sample[col].isnull()][col]
                    sample_vals = non_null_vals.head(20)
                    collection_types = (list, dict, tuple)
                    if np is not None:
                        collection_types = collection_types + (np.ndarray,)
                    is_collection_col = sample_vals.map(
                        lambda v: isinstance(v, collection_types)
                    ).any()
                    if (
                        _type == "categorical"
                        and not is_collection_col
                        and len(non_null_vals.unique()) <= 2
                        and len(non_null_vals) > 0
                        and str(sample_vals.values[0]).isdigit()
                    ):
                        func = "sum"

                    if func:
                        # Check if a count has been
                        # applied to this table yet.
                        if counted and func == "count":
                            continue

                        # Check if there was a last function
                        # applied and, if so, if the recommended
                        # function is in it's available combinations.
                        if last_function:
                            # If the function is not in the function
                            # combos it means we've selected the wrong
                            # function.  Let's loop through the appropriate
                            # function combos and append them.
                            if func in FUNCTION_COMBOS[last_function]:
                                col_new = f"{col}_{func}"
                                op = sqlop(
                                    optype=SQLOpType.aggfunc,
                                    opval=f"{func}" + f"({col}) as {col_new}",
                                )
                                if op not in agg_funcs:
                                    agg_funcs.append(op)
                            else:
                                col_new = f"{col}_avg"
                                op = sqlop(
                                    optype=SQLOpType.aggfunc,
                                    opval=f"{func}" + f"({col}) as {col_new}",
                                )
                                if op not in agg_funcs:
                                    agg_funcs.append(op)
                        else:
                            col_new = f"{col}_{func}"
                            op = sqlop(
                                optype=SQLOpType.aggfunc,
                                opval=f"{func}" + f"({col}) as {col_new}",
                            )
                            if op not in agg_funcs:
                                agg_funcs.append(op)

                        if func == "count":
                            counted = True

        # If we have time-series data we want to
        # do historical counts over the last periods.
        if ts_data:
            # Get the min and max dates.
            logger.info(f"had time-series aggregations for {self}")
            # Get the time since the last event.
            for period in self.ts_periods:
                # count the number of identifiers in this period.
                delt = self.cut_date - datetime.timedelta(days=period)
                aggfunc = sqlop(
                    optype=SQLOpType.aggfunc,
                    opval=f"SUM(CASE WHEN {self.colabbr(self.date_key)} >= '{str(delt)}' then 1 else 0 end) as {self.prefix}_num_events_{period}d",
                )
                agg_funcs.append(aggfunc)

        if not len(agg_funcs):
            logger.info(f"No aggregations for {self}")
            return None
        agg = sqlop(optype=SQLOpType.agg, opval=self.key_sql(reduce_key))
        # Need the aggregation and time-based filtering.
        tfilt = self.prep_for_features() if self.prep_for_features() else []

        return tfilt + agg_funcs + [agg]

    def sql_auto_labels(
        self,
        table_df_sample: typing.Union[pd.DataFrame, dd.DataFrame],
        reduce_key: KeySpec,
        type_func_map: dict = {},
    ) -> pd.DataFrame:
        """
        Pandas implementation of auto labeling based on
        provided columns.
        """
        agg_funcs = {}
        if not self._stypes:
            self._stypes = infer_df_stype(table_df_samp)
        for col, stype in self._stypes.items():
            if col.endswith("_label"):
                _type = str(stype)
                if type_func_map.get(_type):
                    for func in type_func_map[_type]:
                        if func == "first":
                            func = "any_value"
                        col_new = f"{col}_{func}_label"
                        agg_funcs.append(
                            sqlop(
                                optype=SQLOpType.aggfunc,
                                opval=f"{func}" + f"({col}) as {col_new}",
                            )
                        )
        # Need the aggregation and time-based filtering.
        agg = sqlop(optype=SQLOpType.agg, opval=self.key_sql(reduce_key))
        tfilt = self.prep_for_labels() if self.prep_for_labels() else []
        return tfilt + agg_funcs + [agg]


class SnowflakeNode(SQLNode):
    def __init__(self, *args, **kwargs):
        """
        Constructor.
        """
        super().__init__(*args, **kwargs)
        # Use an available database.

    def _clean_refs(self):
        # Get all views and find the ones
        # in temp refs that are still active.
        views = self.execute_query("show views")
        active_views = [row["name"] for ix, row in views.iterrows()]
        for k, v in self._temp_refs.items():
            if v not in self._removed_refs and v in active_views:
                sql = f"DROP TABLE {v}"
                self.execute_query(sql, ret_df=False)
                self._removed_refs.append(v)
                logger.info(f"dropped {v}")

    def use_db(
        self,
        db: str,
    ) -> bool:
        self.execute_query(f"use database {db}")
        return True

    def create_temp_view(
        self,
        qry: str,
        view_name: str,
        dry: bool = False,
    ) -> str:
        """
        Create a view with the results
        of the query.
        """
        sql = f"""
        CREATE OR REPLACE TEMPORARY TABLE {view_name}
        AS {qry}
        """
        logger.info(f"Creating temp table with {sql}")
        self._ref_sql = sql
        if not dry:
            self.execute_query(sql, ret_df=False)
        self._cur_data_ref = view_name
        return view_name


class TrinoNode(SQLNode):
    def __init__(self, *args, **kwargs):
        """
        Constructor.
        """
        _require_backend(trino, "trino", "trino")
        super().__init__(*args, **kwargs)

    def _create_temp_view(
        self,
        qry: str,
        view_name: str,
        dry: bool = False,
    ) -> str:
        sql = f"""
        CREATE OR REPLACE TEMPORARY VIEW {view_name}
        as {qry}
        """
        logger.info(f"Creating temp view with {sql}")
        self._ref_sql = sql
        if not dry:
            self.execute_query(sql, ret_df=False)
        self._cur_data_ref = view_name
        return view_name


class DuckdbNode(SQLNode):
    def __init__(self, *args, table_name: str = None, **kwargs):
        """
        ...SQLNode

                Arguments:
                ----------
                table_name: (optional) str of table name to use if `fpath` is filesystem
        """
        super().__init__(*args, **kwargs)
        _require_backend(duckdb, "duckdb", "duckdb")
        if "/" in self.fpath and not table_name:
            raise Exception("parameter 'table_name' must be set for duckdb files")
        elif "/" in self.fpath and table_name:
            self.table_name = table_name
        else:
            self.table_name = None

    def execute_query(
        self,
        qry: str,
        ret_df: bool = True,
        commit: bool = False,
    ) -> typing.Optional[typing.Union[None, pd.DataFrame]]:
        """
        Execute a query and get back a dataframe.
        """

        if not ret_df:
            self.get_client().sql(qry)
        else:
            return self.get_client().sql(qry).to_df()

    def _clean_refs(self):
        # Get all views and find the ones
        # in temp refs that are still active.
        temp_tables = self.get_client().sql("""
        SELECT * FROM temp.sqlite_master;
        """)
        active_tables = list(temp_tables.to_df()["name"])
        for k, v in self._temp_refs.items():
            if v not in self._removed_refs and v in active_tables:
                sql = f"DROP TABLE IF EXISTS {v}"
                self.execute_query(sql, ret_df=False)
                self._removed_refs.append(v)
                logger.info(f"dropped {v}")

    def create_temp_view(
        self,
        qry: str,
        view_name: str,
        dry: bool = False,
    ) -> str:
        """
        Create a view with the results
        of the query.
        """
        sql = f"""
        CREATE OR REPLACE TEMP TABLE {view_name}
        AS {qry}
        """
        self._ref_sql = sql
        logger.info(sql)
        if not dry:
            self.execute_query(sql, ret_df=False)
        self._cur_data_ref = view_name
        return view_name
