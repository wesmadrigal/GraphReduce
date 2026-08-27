# Changelog

Notable changes to GraphReduce are documented here. The package build appends
this file to the README shown on PyPI, and the same notes should be copied into
the corresponding GitHub Release.

## [1.10.3] - 2026-08-27

### Added

- Added optional graph-wide SQL auto-feature configuration for feature
  families, family budgets, time-series periods, categorical cardinality and
  top-value limits, text features, and inferred annotation limits.

### Changed

- Graph-level feature settings now propagate to every node during graph
  hydration when explicitly supplied, while omitted settings preserve each
  node's existing configuration.

## [1.10.2] - 2026-08-17

### Added

- Added an automatic point-in-time age feature for graph parent nodes that
  define a date key.

### Changed

- Extended node time-series periods with the graph compute horizon, expressed
  in whole days, whenever that horizon exceeds one year.

## [1.10.1] - 2026-08-15

### Added

- Added training-sample schema profiling with typed, model-agnostic feature
  manifests for structural, numerical, boolean, categorical, timestamp, and
  text columns.
- Added explicit cutoff-safety declarations for unavailable or future-derived
  source columns.
- Added `GraphReduceNode.infer_feature_manifest()` to profile a node and
  optionally apply its safe source-column plan.

### Changed

- Exposed feature manifests and schema profiling through the top-level
  `graphreduce` package API.
- Documented how typed manifests compose with the existing automatic
  categorical indicators and category-gated numeric reductions.

## [1.10.0] - 2026-08-13

### Added

- Added ordered composite primary keys and relationship keys while preserving
  the existing scalar-key API.
- Added native multi-column joins and reductions for pandas, Dask, Spark,
  Daft, and SQL compute layers.
- Added composite-key support for automatic features, labels, time-series
  grouping, frozen execution plans, and dynamic date-node propagation.
- Added early validation for invalid key definitions, mismatched relationship
  key arity, and missing join columns.

### Changed

- Added `colabbrs()` and `key_sql()` helpers for custom node implementations
  that operate on scalar or composite keys.
- Composite primary-key episode features retain row-count metrics but skip
  dialect-dependent distinct-primary-key metrics with an explicit warning.

### Fixed

- Fixed reverse-edge joins so their parent and relation key mappings are
  correctly inverted.
- Prevented automatic feature generation from aggregating composite reduction
  key components as value columns.

## [1.9.17] - 2026-07-28

### Added

- Added opt-in SQL auto-feature families for conditional, temporal, episode,
  semantic, sequence, and peer-context features.
- Added automatic categorical, text, and gated-numeric SQL annotations with
  configurable cardinality and feature limits.
- Added a `relbench` optional dependency group for RelBench, CatBoost,
  PyArrow, and scikit-learn integrations.
- Added RelBench task selection and parallel training-frame generation
  utilities to the end-to-end examples.

### Changed

- Isolated temporary SQL tables with a per-execution namespace so concurrent
  GraphReduce runs do not collide.
- Improved generated SQL feature schemas across cut dates and bounded wide
  training frames to keep feature generation predictable.

### Fixed

- Fixed SQL inference sampling for wide or sparsely populated relations.
- Fixed date-node key propagation through undated intermediate SQL nodes.
- Fixed SQL auto-feature type handling for string-backed numeric semantic
  types, identifier-like columns, and collection-valued columns.
- Propagated Athena, Redshift, and DuckDB query failures instead of allowing
  invalid feature plans to continue.
- Handled constant regression targets in the RelBench training workflow.

For older versions, see the
[PyPI release history](https://pypi.org/project/graphreduce/#history).
