"""Experimental bounded, target-free joint summaries before relational pooling.

Frozen numeric quantiles and categorical values generate ordinary aggregate SQL.
Matched marginal summaries use the same pairwise-complete observations as joint
summaries, so a controlled ablation can distinguish dependence from input gain.
The caller owns point-in-time extraction and exclusion of keys/labels.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


def _literal(value):
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float, np.number)):
        return repr(float(value))
    return "'" + str(value).replace("'", "''") + "'"


@dataclass
class JointDistributionSketch:
    """A bounded pair panel; no supervised selection or automatic defaults.

    ``expressions`` maps fitted input columns to caller-validated SQL expressions
    and returns separate marginal and joint operations. Each numeric pair emits
    a small joint CDF grid; category/numeric pairs emit conditional CDFs.
    """

    max_numeric: int = 4
    max_categorical: int = 2
    max_numeric_pairs: int = 2
    max_conditional_pairs: int = 2
    max_categories: int = 3
    max_fit_rows: int = 20_000
    quantiles: tuple = (0.25, 0.5, 0.75)

    def fit(self, values):
        if not values.columns.is_unique or values.empty:
            raise ValueError("Nonempty payloads with unique columns required")
        if min(self.max_numeric, self.max_categorical, self.max_categories,
               self.max_fit_rows) < 1 or min(self.max_numeric_pairs,
                                            self.max_conditional_pairs) < 0:
            raise ValueError("Invalid positive input / nonnegative pair budgets")
        if not self.quantiles or any(not 0 < q < 1 for q in self.quantiles):
            raise ValueError("Interior quantiles required")
        sample = values
        if len(sample) > self.max_fit_rows:
            # Value-based sampling is independent of example indexes/order.
            hashes = pd.util.hash_pandas_object(sample, index=False).to_numpy()
            order = np.argsort(hashes, kind="stable")[:self.max_fit_rows]
            sample = sample.iloc[order]
        numeric, categorical = [], []
        self.cutpoints_, self.categories_ = {}, {}
        for column in sorted(sample.columns):
            series = sample[column]
            observed = series.dropna()
            if observed.empty:
                continue
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                finite = series.to_numpy(dtype=float, na_value=np.nan)
                finite = finite[np.isfinite(finite)]
                if len(np.unique(finite)) >= 3:
                    points = np.unique(np.quantile(finite, self.quantiles, method="lower"))
                    points = points[points < finite.max()]
                    if len(points):
                        numeric.append((column, len(finite) / len(sample)))
                        self.cutpoints_[column] = points
            if observed.nunique() <= 32:
                counts = observed.value_counts()
                categories = sorted(counts.index, key=lambda v: (-int(counts[v]), str(v)))
                if len(categories) >= 2:
                    categorical.append((column, len(observed) / len(sample)))
                    self.categories_[column] = tuple(categories[:self.max_categories])
        numeric = [c for c, _ in sorted(numeric, key=lambda p: (-p[1], p[0]))[:self.max_numeric]]
        categorical = [c for c, _ in sorted(categorical, key=lambda p: (-p[1], p[0]))[:self.max_categorical]]
        self.numeric_pairs_ = self._balanced(list(combinations(numeric, 2)), self.max_numeric_pairs)
        self.conditional_pairs_ = self._balanced(
            [(c, n) for c in categorical for n in numeric if c != n],
            self.max_conditional_pairs,
        )
        self.columns_ = tuple(values.columns)
        self.audit_ = dict(
            fit_rows=len(sample), numeric_columns=numeric, categorical_columns=categorical,
            numeric_pairs=self.numeric_pairs_, conditional_pairs=self.conditional_pairs_,
            cutpoints={c: self.cutpoints_[c].tolist() for c in numeric},
            categories={c: list(map(str, self.categories_[c])) for c in categorical},
        )
        return self

    @staticmethod
    def _balanced(pairs, budget):
        counts, selected = {}, []
        pairs = list(pairs)
        while pairs and len(selected) < budget:
            pair = min(pairs, key=lambda p: (sum(counts.get(c, 0) for c in p), p))
            pairs.remove(pair)
            selected.append(pair)
            for column in pair:
                counts[column] = counts.get(column, 0) + 1
        return selected

    def expressions(self, columns):
        """Return (marginal, joint) lists of (safe alias, aggregate expression)."""
        if not hasattr(self, "columns_"):
            raise ValueError("Fit joint sketch before SQL generation")
        if set(columns) != set(self.columns_):
            raise ValueError("Payload schema changed")
        marginal, joint = [], []

        def count(predicate):
            return f"SUM(CASE WHEN {predicate} THEN 1 ELSE 0 END)"

        def fraction(predicate, denominator):
            return f"({count(predicate)} * 1.0 / NULLIF({count(denominator)}, 0))"

        def finite(column):
            return f"isfinite(TRY_CAST({columns[column]} AS DOUBLE))"

        for i, (a, b) in enumerate(self.numeric_pairs_):
            valid = f"({finite(a)} AND {finite(b)})"
            prefix = f"n{i}"
            marginal.extend([
                (prefix + "_support", count(valid)),
                (prefix + "_coverage", f"{count(valid)} * 1.0 / NULLIF(COUNT(*), 0)"),
            ])
            for side, column in enumerate((a, b)):
                for j, threshold in enumerate(self.cutpoints_[column]):
                    predicate = f"{valid} AND TRY_CAST({columns[column]} AS DOUBLE) <= {float(threshold)!r}"
                    marginal.append((f"{prefix}_m{side}_q{j}", fraction(predicate, valid)))
            for j, x in enumerate(self.cutpoints_[a]):
                for k, y in enumerate(self.cutpoints_[b]):
                    predicate = (
                        f"{valid} AND TRY_CAST({columns[a]} AS DOUBLE) <= {float(x)!r} "
                        f"AND TRY_CAST({columns[b]} AS DOUBLE) <= {float(y)!r}"
                    )
                    joint.append((f"{prefix}_joint_q{j}q{k}", fraction(predicate, valid)))
        for i, (category, column) in enumerate(self.conditional_pairs_):
            valid = f"({columns[category]} IS NOT NULL AND {finite(column)})"
            prefix = f"c{i}"
            marginal.append((prefix + "_support", count(valid)))
            for j, threshold in enumerate(self.cutpoints_[column]):
                predicate = f"{valid} AND TRY_CAST({columns[column]} AS DOUBLE) <= {float(threshold)!r}"
                marginal.append((f"{prefix}_value_q{j}", fraction(predicate, valid)))
            for k, value in enumerate(self.categories_[category]):
                condition = f"{valid} AND {columns[category]} = {_literal(value)}"
                marginal.extend([
                    (f"{prefix}_v{k}_support", count(condition)),
                    (f"{prefix}_v{k}_share", fraction(condition, valid)),
                ])
                for j, threshold in enumerate(self.cutpoints_[column]):
                    predicate = f"{condition} AND TRY_CAST({columns[column]} AS DOUBLE) <= {float(threshold)!r}"
                    joint.append((f"{prefix}_v{k}_conditional_q{j}", fraction(predicate, condition)))
        return marginal, joint
