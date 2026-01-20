# ADR-003: Time-Aware vs Time-Agnostic Detectors

**Status**: Accepted

**Date**: 2021-02

## Context

Some anomaly detection algorithms naturally work with sequential differences (e.g., detecting a jump of 10 units between consecutive points), while others need to account for time (e.g., detecting a rate of change of 5 units per second). Water domain applications require both types: detecting sudden level changes regardless of time spacing, and detecting flow rate anomalies that depend on the time dimension.

## Decision

Support both time-aware and time-agnostic detectors in the library.

Time-agnostic detectors (e.g., `DiffDetector`) work with point-to-point differences using `data.diff()`, independent of the time spacing between measurements. Time-aware detectors (e.g., `GradientDetector`) calculate rates per unit time and require a `DatetimeIndex` to convert time deltas to seconds via `.dt.total_seconds()`.

## Alternatives Considered

**Always require DatetimeIndex**: Would force users to provide temporal information even when not needed for the detection algorithm. Rejected because some valid use cases don't have meaningful time indices (e.g., sequential sample numbers).

**Convert all detectors to time-aware**: Would make all detectors use rates per second. Rejected because many water domain cases care about absolute changes regardless of time spacing (e.g., a sensor jump of 10 units is anomalous whether it happens over 1 second or 1 minute).

## Consequences

**Positive**: Flexibility to choose the right algorithm for the use case. Time-agnostic detectors work with any index type. Time-aware detectors properly handle irregular sampling rates common in operational monitoring.

**Negative**: Users must understand which detectors require DatetimeIndex. API inconsistency between detector types. Error messages appear at detection time rather than initialization.
