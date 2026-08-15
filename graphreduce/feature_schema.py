"""Training-sample feature schema profiling and typed manifests.

This module deliberately does not depend on a model backend.  It describes
which source columns GraphReduce may read, which columns are structural, and
which semantic types a downstream model adapter can consume.
"""

from __future__ import annotations

import enum
import numbers
import typing
from dataclasses import asdict, dataclass

import pandas as pd

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a core dependency
    np = None


NameSpec = typing.Optional[typing.Union[str, typing.Iterable[str]]]


class FeatureRole(str, enum.Enum):
    """The role one source column plays in a relational feature program."""

    PRIMARY_KEY = "primary_key"
    FOREIGN_KEY = "foreign_key"
    DATE_KEY = "date_key"
    TARGET = "target"
    NUMERICAL = "numerical"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    TIMESTAMP = "timestamp"
    TEXT = "text"
    EXCLUDED = "excluded"
    UNSAFE = "unsafe"


STRUCTURAL_ROLES = frozenset(
    {FeatureRole.PRIMARY_KEY, FeatureRole.FOREIGN_KEY, FeatureRole.DATE_KEY}
)
MODEL_FEATURE_ROLES = frozenset(
    {
        FeatureRole.NUMERICAL,
        FeatureRole.BOOLEAN,
        FeatureRole.CATEGORICAL,
        FeatureRole.TIMESTAMP,
        FeatureRole.TEXT,
    }
)


@dataclass(frozen=True)
class FeatureColumn:
    """Profile and selection decision for one source column."""

    name: str
    role: FeatureRole
    physical_dtype: str
    non_null_fraction: float
    cardinality: int
    selected: bool
    cutoff_safe: bool
    reason: str

    @property
    def structural(self) -> bool:
        return self.role in STRUCTURAL_ROLES

    @property
    def model_feature(self) -> bool:
        return self.selected and self.role in MODEL_FEATURE_ROLES

    @property
    def semantic_type(self) -> str:
        return self.role.value

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        result = asdict(self)
        result["role"] = self.role.value
        result["semantic_type"] = self.semantic_type
        return result


@dataclass(frozen=True)
class FeatureManifest:
    """Ordered, typed feature decisions fitted from a training sample."""

    columns: typing.Tuple[FeatureColumn, ...]
    source: typing.Optional[str] = None

    def __post_init__(self) -> None:
        names = [column.name for column in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("feature manifest contains duplicate column names")

    @property
    def by_name(self) -> typing.Dict[str, FeatureColumn]:
        return {column.name: column for column in self.columns}

    @property
    def graph_columns(self) -> typing.Tuple[str, ...]:
        """Columns needed for joins, time filtering, or feature generation."""

        return tuple(
            column.name
            for column in self.columns
            if column.structural or column.model_feature
        )

    @property
    def feature_columns(self) -> typing.Tuple[str, ...]:
        return tuple(
            column.name for column in self.columns if column.model_feature
        )

    @property
    def categorical_columns(self) -> typing.Tuple[str, ...]:
        return tuple(
            column.name
            for column in self.columns
            if column.model_feature and column.role == FeatureRole.CATEGORICAL
        )

    @property
    def text_columns(self) -> typing.Tuple[str, ...]:
        return tuple(
            column.name
            for column in self.columns
            if column.model_feature and column.role == FeatureRole.TEXT
        )

    @property
    def timestamp_columns(self) -> typing.Tuple[str, ...]:
        return tuple(
            column.name
            for column in self.columns
            if column.model_feature and column.role == FeatureRole.TIMESTAMP
        )

    def categorical_indices(
        self,
        columns: typing.Optional[typing.Sequence[str]] = None,
    ) -> typing.Tuple[int, ...]:
        ordered = list(self.feature_columns if columns is None else columns)
        categorical = set(self.categorical_columns)
        return tuple(
            index for index, name in enumerate(ordered) if name in categorical
        )

    def select_graph_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.graph_columns) - set(frame.columns)
        if missing:
            raise KeyError(f"frame is missing manifest columns: {sorted(missing)}")
        return frame.loc[:, list(self.graph_columns)].copy()

    def select_feature_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Select model-facing columns in the fitted training order."""

        missing = set(self.feature_columns) - set(frame.columns)
        if missing:
            raise KeyError(f"frame is missing manifest features: {sorted(missing)}")
        return frame.loc[:, list(self.feature_columns)].copy()

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        return {
            "source": self.source,
            "graph_columns": list(self.graph_columns),
            "feature_columns": list(self.feature_columns),
            "categorical_columns": list(self.categorical_columns),
            "text_columns": list(self.text_columns),
            "timestamp_columns": list(self.timestamp_columns),
            "columns": [column.to_dict() for column in self.columns],
        }


def _normalize_names(names: NameSpec) -> typing.Tuple[str, ...]:
    if names is None:
        return ()
    if isinstance(names, str):
        values = (names,)
    else:
        values = tuple(names)
    if any(not isinstance(name, str) or not name for name in values):
        raise ValueError("column names must be non-empty strings")
    return tuple(dict.fromkeys(values))


def _is_collection_series(series: pd.Series) -> bool:
    values = series.dropna().head(20)
    collection_types: typing.Tuple[type, ...] = (list, dict, tuple, set)
    if np is not None:
        collection_types = collection_types + (np.ndarray,)
    return bool(values.map(lambda value: isinstance(value, collection_types)).any())


def _is_numeric_object_series(series: pd.Series) -> bool:
    values = series.dropna().head(100)
    return bool(len(values)) and all(
        isinstance(value, numbers.Number) for value in values
    )


def _looks_like_text(
    name: str,
    series: pd.Series,
    *,
    minimum_average_length: float,
    minimum_unique_ratio: float,
) -> bool:
    values = series.dropna().astype(str).head(500)
    if len(values) == 0:
        return False
    normalized_name = name.lower().replace("-", "_")
    name_tokens = set(normalized_name.split("_"))
    name_hint = bool(
        name_tokens.intersection(
            {
                "body",
                "comment",
                "criteria",
                "description",
                "message",
                "note",
                "notes",
                "summary",
                "text",
                "title",
            }
        )
    )
    average_length = float(values.str.len().mean())
    unique_ratio = float(values.nunique(dropna=True) / len(values))
    contains_spaces = float(values.str.contains(r"\s", regex=True).mean())
    return name_hint or (
        average_length >= minimum_average_length
        and (unique_ratio >= minimum_unique_ratio or contains_spaces >= 0.25)
    )


def profile_feature_schema(
    frame: pd.DataFrame,
    *,
    primary_keys: NameSpec = None,
    foreign_keys: NameSpec = None,
    date_keys: NameSpec = None,
    target_columns: NameSpec = None,
    excluded_columns: NameSpec = None,
    unsafe_columns: NameSpec = None,
    source: typing.Optional[str] = None,
    min_non_null_fraction: float = 0.01,
    max_categorical_cardinality: int = 1000,
    text_minimum_average_length: float = 32.0,
    text_minimum_unique_ratio: float = 0.5,
) -> FeatureManifest:
    """Fit a model-agnostic feature manifest from a training sample.

    The caller remains responsible for declaring columns that were not
    available at the prediction cutoff.  GraphReduce cannot infer temporal
    availability from a physical dtype or a column name.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if not 0 <= min_non_null_fraction <= 1:
        raise ValueError("min_non_null_fraction must be between 0 and 1")
    if max_categorical_cardinality < 1:
        raise ValueError("max_categorical_cardinality must be positive")

    primary = set(_normalize_names(primary_keys))
    foreign = set(_normalize_names(foreign_keys))
    dates = set(_normalize_names(date_keys))
    targets = set(_normalize_names(target_columns))
    excluded = set(_normalize_names(excluded_columns))
    unsafe = set(_normalize_names(unsafe_columns))
    configured = primary | foreign | dates | targets | excluded | unsafe
    unknown = configured - set(frame.columns)
    if unknown:
        raise KeyError(f"configured columns are absent from sample: {sorted(unknown)}")

    profiles = []
    row_count = len(frame)
    for name in frame.columns:
        series = frame[name]
        non_null_count = int(series.notna().sum())
        non_null_fraction = (
            float(non_null_count / row_count) if row_count else 0.0
        )
        collection = _is_collection_series(series)
        if collection:
            cardinality = 0
        else:
            cardinality = int(series.nunique(dropna=True))

        if name in unsafe:
            role = FeatureRole.UNSAFE
            selected = False
            cutoff_safe = False
            reason = "declared unavailable at prediction cutoff"
        elif name in excluded:
            role = FeatureRole.EXCLUDED
            selected = False
            cutoff_safe = True
            reason = "explicitly excluded"
        elif name in targets:
            role = FeatureRole.TARGET
            selected = False
            cutoff_safe = False
            reason = "task target"
        elif name in primary:
            role = FeatureRole.PRIMARY_KEY
            selected = False
            cutoff_safe = True
            reason = "structural primary key"
        elif name in foreign:
            role = FeatureRole.FOREIGN_KEY
            selected = False
            cutoff_safe = True
            reason = "structural foreign key"
        elif name in dates:
            role = FeatureRole.DATE_KEY
            selected = False
            cutoff_safe = True
            reason = "structural event-time key"
        elif collection:
            role = FeatureRole.EXCLUDED
            selected = False
            cutoff_safe = True
            reason = "collection-valued columns require an explicit encoder"
        elif pd.api.types.is_bool_dtype(series):
            role = FeatureRole.BOOLEAN
            selected = True
            cutoff_safe = True
            reason = "boolean feature"
        elif pd.api.types.is_datetime64_any_dtype(series):
            role = FeatureRole.TIMESTAMP
            selected = True
            cutoff_safe = True
            reason = "timestamp feature"
        elif pd.api.types.is_numeric_dtype(series) or _is_numeric_object_series(series):
            role = FeatureRole.NUMERICAL
            selected = True
            cutoff_safe = True
            reason = "numeric feature"
        elif _looks_like_text(
            name,
            series,
            minimum_average_length=text_minimum_average_length,
            minimum_unique_ratio=text_minimum_unique_ratio,
        ):
            role = FeatureRole.TEXT
            selected = True
            cutoff_safe = True
            reason = "text feature"
        else:
            role = FeatureRole.CATEGORICAL
            selected = True
            cutoff_safe = True
            reason = "categorical feature"

        if selected and non_null_fraction < min_non_null_fraction:
            selected = False
            reason = "insufficient non-null coverage"
        elif selected and cardinality <= 1:
            selected = False
            reason = "constant or empty feature"
        elif (
            selected
            and role == FeatureRole.CATEGORICAL
            and cardinality > max_categorical_cardinality
        ):
            selected = False
            reason = "categorical cardinality exceeds configured limit"

        profiles.append(
            FeatureColumn(
                name=name,
                role=role,
                physical_dtype=str(series.dtype),
                non_null_fraction=non_null_fraction,
                cardinality=cardinality,
                selected=selected,
                cutoff_safe=cutoff_safe,
                reason=reason,
            )
        )

    return FeatureManifest(columns=tuple(profiles), source=source)
