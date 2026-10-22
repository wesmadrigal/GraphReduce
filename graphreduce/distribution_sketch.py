"""Bounded, label-free sketches of numeric neighborhood distributions.

Ordinary means/extrema discard how mass is distributed between their endpoints.
Fit global cutpoints only on historical input rows, then emit per-root empirical
CDF values and nonmissing coverage. No target, schema names or neural dependency.
Point-in-time row eligibility is owned by the upstream neighborhood extractor.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DistributionSketch:
    quantiles: tuple = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
    max_fit_rows: int = 20000
    random_state: int = 0

    def fit(self, values):
        if values.empty or not values.columns.is_unique:
            raise ValueError("A nonempty numeric input with unique columns is required")
        if (
            self.max_fit_rows < 1
            or not self.quantiles
            or any(not 0 < q < 1 for q in self.quantiles)
        ):
            raise ValueError("Positive row budget and interior quantiles are required")
        if not all(pd.api.types.is_numeric_dtype(values[c]) for c in values):
            raise ValueError("Distribution sketches require numeric encoded payloads")
        sample = values
        if len(sample) > self.max_fit_rows:
            # Canonical input row order is supplied by the bounded extractor.
            # No validation statistics or targets participate in calibration.
            positions = np.random.default_rng(self.random_state).choice(
                len(sample), self.max_fit_rows, replace=False
            )
            sample = sample.iloc[np.sort(positions)]
        self.columns_ = tuple(values.columns)
        self.cutpoints_ = {}
        for column in self.columns_:
            x = sample[column].to_numpy(dtype=float, na_value=np.nan)
            x = x[np.isfinite(x)]
            if len(x) and np.ptp(x) > 0:
                points = np.unique(np.quantile(x, self.quantiles))
                self.cutpoints_[column] = points[points < x.max()]
            else:
                self.cutpoints_[column] = np.empty(0)
        self.audit_ = dict(
            fit_rows=len(sample),
            input_columns=len(self.columns_),
            cdf_columns=sum(len(v) for v in self.cutpoints_.values()),
            coverage_columns=len(self.columns_),
        )
        return self

    def transform(self, rows, values, roots):
        if not hasattr(self, "columns_"):
            raise ValueError("Fit a distribution sketch before inference")
        roots = pd.Index(roots)
        if not roots.is_unique or len(rows) != len(values):
            raise ValueError("Unique roots and aligned neighborhood rows are required")
        if tuple(values.columns) != self.columns_:
            raise ValueError("Distribution payload schema/order changed")
        if "__example_id" not in rows:
            raise ValueError("Neighborhood rows need example metadata")
        groups = roots.get_indexer(rows["__example_id"])
        keep = groups >= 0
        groups = groups[keep]
        count = np.bincount(groups, minlength=len(roots))
        output = {}
        for ordinal, column in enumerate(self.columns_):
            x = values[column].to_numpy(dtype=float, na_value=np.nan)[keep]
            valid = np.isfinite(x)
            observed = np.bincount(groups[valid], minlength=len(roots))
            coverage = np.divide(
                observed, count, out=np.full(len(roots), np.nan), where=count > 0
            )
            output[f"c{ordinal}__coverage"] = coverage
            for j, threshold in enumerate(self.cutpoints_[column]):
                below = np.bincount(
                    groups[valid & (x <= threshold)], minlength=len(roots)
                )
                output[f"c{ordinal}__cdf{j}"] = np.divide(
                    below, observed, out=np.full(len(roots), np.nan), where=observed > 0
                )
        return pd.DataFrame(output, index=roots, dtype=np.float32)
