# ADR-004: Support both pandas Series and DataFrame

**Status**: Accepted

**Date**: 2026-02

## Context
This decision extends [ADR-002](002-pandas-series-primary-data-structure.md).

Previously (ADR-002), the library only accepted pandas Series as input and returned pandas Series as output. However, users working with pandas often don't distinguish between a single-column DataFrame and a Series, and frequently need to apply anomaly detection to multiple time series simultaneously (e.g., data from multiple sensors, devices, or variables). 

## Decision

Extend detectors to accept both pandas Series and pandas DataFrame as input, maintaining type consistency between input and output. If `detect()` receives a Series, it returns a Series of boolean values. If `detect()` receives a DataFrame, it returns a DataFrame of boolean values with the same shape and column names. Internally, detectors work with DataFrames (Series inputs are converted to single-column DataFrames), and type hints use `@overload` decorators to ensure proper auto-completion and type checking. The detector rules (set or derived from running `fit()`) are applied column-wise to DataFrames.

## Alternatives Considered

**Only accept single-column dataframes**: Could convert to series internally and return series to user. Rejected because it would break input/output symmetry. 

## Consequences
Full backward compatibility when using series as input. 

**Positive**: More convenient for users working with multiple time series using idiomatic pandas patterns (apply operation to DataFrame columns). Input type matches output type with proper IDE auto-completion support through `@overload` decorators. 

**Negative**: Requires `pandas-stubs` for full type checking support with mypy and internal DataFrame conversion adds slight overhead for Series inputs.