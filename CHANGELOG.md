# tsod Changelog

## [Unreleased]

### Added
- Data categories (`tsod.data`): `fit()` and `detect()` sort input into `SeriesData`, `SortedData` or `EvenlySpacedData`, and each detector declares the category it needs in the type hint of `_fit()`/`_detect()`.
- `UnevenSpacingWarning`: Diff, RollingStd, ConstantValue and Hampel detectors warn when the index steps differ by more than 1 second (time index) or are not equal (numeric index).

### Fixed

### Changed
- **Breaking:** every detector raises `UnsupportedIndexError` unless the index is a DatetimeIndex, TimedeltaIndex or numeric index.
- **Breaking:** every detector except `RangeDetector` raises `UnsortedIndexError` for an index not sorted in increasing order, e.g. after `pd.concat` without `ignore_index=True`.
- numba is imported defensively: tsod can be imported and used without numba installed, and only `HampelDetector.detect()` requires it (raises `ImportError` with an installation hint). numba remains a default dependency, so standard installs are unaffected.

## [0.3.0] - 2026-02-04

### Added
- Added support for anomaly detection on multicolumn dataframes
- Direction aware fit-methods for gradient and diff detectors

### Fixed
- Consistent gradient calculations with using dt > 1 
- Formatting, doc-strings and type hints

### Changed
- Enhanced constantValueDetector performance through algorithm optimization
- Documentation changed from Sphinx to Quarto
- Migrate to uv package manager with src/layout
- Updated notebooks exampels