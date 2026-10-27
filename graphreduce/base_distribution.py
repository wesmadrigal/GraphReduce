"""Bounded distribution summaries compiled into the existing SQL base family.

The caller supplies a point-in-time training sample. Numeric cutpoints are SQL
literals, so GraphReduce's ordinary frozen execution plan also freezes the CDF.
This is not a separately selectable feature family.
"""

import re

import numpy as np


BASE_CDF_MAX_COLUMNS = 16
BASE_CDF_MAX_FIT_ROWS = 20_000
BASE_CDF_QUANTILES = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
_CDF_COLUMN = re.compile(r"__cdf_(?:coverage|q\d+)_avg(?:_|$)")


def is_base_cdf_column(column: str) -> bool:
    """Recognize sketches even after they have propagated through more edges."""
    return bool(_CDF_COLUMN.search(column))


def base_cdf_expressions(series, column: str):
    """Return at most seven finite-value CDFs and coverage for one numeric input.

    Constant, binary and entirely unobserved inputs add no distribution features.
    Thresholds use an observed-value quantile, avoiding interpolation overflow and
    keeping ties meaningful on discrete measurements. Input order is canonical
    upstream; calibration is bounded independently of the source table size.
    """
    values = series.iloc[:BASE_CDF_MAX_FIT_ROWS].to_numpy(dtype=float, na_value=np.nan)
    values = values[np.isfinite(values)]
    if len(np.unique(values)) < 3:
        return []
    thresholds = np.unique(np.quantile(values, BASE_CDF_QUANTILES, method="lower"))
    thresholds = thresholds[thresholds < values.max()]
    # BETWEEN rejects both infinities and NaN (including DuckDB's ordered NaN).
    finite = f"{column} BETWEEN -1.7976931348623157e308 AND 1.7976931348623157e308"
    return [
        ("coverage", f"AVG(CASE WHEN {finite} THEN 1.0 ELSE 0.0 END)"),
        *[
            (
                f"q{index}",
                f"AVG(CASE WHEN {finite} THEN CASE WHEN {column} <= {float(value)!r} "
                "THEN 1.0 ELSE 0.0 END ELSE NULL END)",
            )
            for index, value in enumerate(thresholds)
        ],
    ]
