"""Structured predicates for bounded automatic feature trajectories."""

from __future__ import annotations

import typing
import hashlib
import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EqualityPredicate:
    """A scalar equality condition selected from training data."""

    column: str
    value: typing.Any
    name: typing.Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.column, str) or not self.column:
            raise ValueError("predicate column must be a non-empty string")
        if self.name is not None and (
            not isinstance(self.name, str) or not self.name
        ):
            raise ValueError("predicate name must be a non-empty string")

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        return asdict(self)

    def feature_alias(self, column: typing.Optional[str] = None) -> str:
        """Return the stable SQL-safe stem used by generated features."""

        value = self.name or f"{column or self.column}_{self.value}"
        alias = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).lower()).strip("_")
        if not alias:
            return "missing"
        if len(alias) > 40:
            digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()[:8]
            return f"{alias[:40]}_{digest}"
        return alias

    @classmethod
    def from_dict(
        cls,
        value: typing.Mapping[str, typing.Any],
    ) -> "EqualityPredicate":
        return cls(
            column=value["column"],
            value=value["value"],
            name=value.get("name"),
        )
