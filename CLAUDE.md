# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`tsod` is a Python library for anomaly detection in time series data, specifically tailored for the water domain. It provides simple, consistent APIs for detecting anomalies using rule-based and statistical algorithms. The project prioritizes ease of use, computational speed for real-time detection, and operational deployment.

## Development Commands

This project uses [uv](https://docs.astral.sh/uv/) for Python package management, following the same setup as mikeio and modelskill.

### Quick Start
```bash
# Install dependencies and sync environment
uv sync

# Run tests
make test

# Run linting
make lint

# Run both linting and tests
make check
```

### Testing
```bash
# Run all tests
uv run pytest

# Run all tests with coverage
make coverage
# or: uv run pytest --cov=tsod tests

# Run a single test file
uv run pytest tests/test_detectors.py

# Run a specific test
uv run pytest tests/test_detectors.py::test_name
```

### Linting
```bash
# Run ruff linter (configured in pyproject.toml)
make lint
# or: uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
make format
# or: uv run ruff format .
```

### Installation
```bash
# Sync all dependencies including dev group (default)
# This includes test and docs dependencies
uv sync

# Sync only test dependencies (used in CI)
uv sync --group test --no-dev

# Build the package
make build
# or: uv build
```

### Documentation
```bash
# Build documentation (Sphinx)
make docs
# or: cd docs && uv run make html
```

### Available Make Targets
- `make check` - Run linting and tests
- `make test` - Run tests
- `make lint` - Run linting
- `make format` - Format code with ruff
- `make coverage` - Generate HTML coverage report
- `make build` - Build the package
- `make docs` - Build documentation
- `make clean` - Clean build artifacts

## Architecture

### Core Design Pattern

All anomaly detectors follow a scikit-learn-inspired API:

1. **Base class**: `Detector` (abstract base class in `src/tsod/base.py`)
   - `fit(data)`: Train detector on "normal" data (optional for some detectors)
   - `detect(data)`: Returns pd.Series of booleans (True = anomaly)
   - `validate(data)`: Ensures input is pd.Series or pd.DataFrame
   - `save(path)` / `load(path)`: Persistence using joblib

2. **Concrete detectors** (in `src/tsod/detectors.py`):
   - Must implement `_detect(data)` method
   - Optionally override `_fit(data)` for training
   - All work with pandas Series as primary data structure

### Two Types of Anomaly Detection

The library distinguishes between:
- **Outlier detection** (unsupervised): Training data may contain outliers; detector identifies them
- **Novelty detection** (semi-supervised): Training data is "clean"; new observations are classified as normal/anomaly

### Detector Categories

1. **Range-based**: `RangeDetector` - detects values outside min/max thresholds or quantiles
2. **Change-based**:
   - `DiffDetector` - sudden shifts (ignores time axis)
   - `GradientDetector` - rate of change per second (requires DatetimeIndex)
3. **Pattern-based**:
   - `ConstantValueDetector` - stuck sensors
   - `ConstantGradientDetector` - linear interpolation artifacts
4. **Statistical**:
   - `RollingStandardDeviationDetector` - high variation windows
   - `HampelDetector` (in `src/tsod/hampel.py`) - median absolute deviation, optimized with numba
5. **Combined**: `CombinedDetector` - combines multiple detectors with OR logic

### Key Implementation Details

- **Time handling**: Some detectors (e.g., `GradientDetector`) require `pd.DatetimeIndex` and calculate rates per second
- **Performance**: `HampelDetector` uses `@jit(nopython=True)` from numba for speed
- **Validation**: Input validation happens in base class before `_fit()` or `_detect()`
- **Missing data**: Most detectors handle NaN values via numpy's `nanquantile`, `nanmedian`, etc.

## Code Style

- Follow PEP8 (enforced by ruff with NumPy and Pandas rules enabled)
- Specific ignores in `pyproject.toml`: E501 (line length), E741, PD011
- Use NumPy-style docstrings (enabled checks are commented out in pyproject.toml but intended for future)
- All public methods should have docstrings with Parameters/Returns sections

## Testing

- Tests in `tests/` directory
- Fixtures defined in test files (e.g., `data_series`, `range_data`)
- Helper functions in `tests/data_generation.py` for creating test data
- CI runs on Ubuntu and Windows with Python 3.10 and 3.13

## Important Conventions

- **Index types**: Be aware that some detectors require DatetimeIndex (GradientDetector), others work with any index
- **Return values**: `detect()` always returns pd.Series of booleans, never modifies input
- **Direction parameter**: Several detectors support `direction="both|positive|negative"` for asymmetric detection
- **Window sizes**: Window-based detectors use integer window sizes (number of points, not time spans)
