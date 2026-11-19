# tsod Changelog

## [Unreleased]

### Added

### Fixed

### Changed
- `GradientDetector` and `ConstantGradientDetector` no longer require a DatetimeIndex: with a numeric index the gradient is per index unit (per step for a RangeIndex). A TimedeltaIndex works like a DatetimeIndex (per second).
- `str(GradientDetector)` shows `max_gradient` as stored instead of converting it to per hour.
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