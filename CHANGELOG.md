# tsod Changelog

## [Unreleased]

### Added
- `quantile` parameter for `DiffDetector`, `GradientDetector` and `RollingStandardDeviationDetector`: `fit()` sets the threshold at this quantile of the training values instead of the maximum (default 1.0 keeps the maximum), so outliers in the training data do not set the threshold.
- `ConstantGradientDetector(threshold=...)`: the threshold was fixed at 1e-7 and flagged slow, smooth signals.
- `HampelDetector(min_mad=...)`: lower limit for the scaled MAD, for flat or quantised data where the MAD can be 0.

### Fixed
- `HampelDetector` uses a symmetric window `[i - w, i + w]` (was `[i - w, i + w)`) and checks the first and last `w` points with a window cut off at the ends.
- `HampelDetector` accepts numpy integers for `window_size`.

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