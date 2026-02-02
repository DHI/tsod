# ADR-002: pandas Series as Primary Data Structure

**Status**: Superseded

**Date**: 2021-01

## Context

The library needed to choose a primary data structure for time series input and output. Water domain users work with various formats including DHI's dfs0 files (via mikeio), CSV files, databases, and numpy arrays. The library needed to balance accessibility, flexibility, and ease of use.

## Decision

Use pandas Series as the only supported input/output format for detectors, and leave all I/O operations to the user.

Detectors accept pandas Series as input and return pandas Series of boolean values (True = anomaly). The library does not provide file reading functionality or accept multiple input types. Users are responsible for converting their data format to pandas Series before detection.

## Alternatives Considered

**Support multiple input formats**: Could accept numpy arrays, pandas Series, and potentially dfs0 files directly. Rejected to keep the API simple and focused. Supporting multiple formats would complicate validation logic and potentially create dependencies on format-specific libraries.

## Consequences

**Positive**: Clear separation of concerns - library focuses solely on anomaly detection, not I/O. Simple, consistent API with single input/output type. No dependencies on format-specific libraries. Users retain flexibility to work with any data source by converting to pandas.

**Negative**: Users must handle data conversion themselves. Extra step for users working with numpy arrays or dfs0 files.
