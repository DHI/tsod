# ADR-001: Scikit-learn-Inspired API Pattern

**Status**: Accepted

**Date**: 2024-01

## Context

The library needed an API pattern for anomaly detection that would support both stateless detectors (with explicit parameters) and stateful detectors (that learn from training data). The target users are water domain professionals who commonly work with scientific Python libraries like pandas, numpy, and scikit-learn. The pattern needed to be familiar, extensible, and support operational deployment where detectors are trained once and serialized for production use.

## Decision

Adopt a scikit-learn-inspired API pattern with `fit()` and `detect()` methods as the core interface for all anomaly detectors.

All detectors inherit from an abstract `Detector` base class that provides `fit()` (for training) and `detect()` (for inference). Concrete detectors implement protected methods `_fit()` and `_detect()`. The `fit()` method returns `self` to enable method chaining. The method name `detect()` was chosen over scikit-learn's `predict()` to better reflect the anomaly detection domain.

## Alternatives Considered

**Functional API**: A simpler function-based approach like `detect_range(data, min_value=0, max_value=100)` was considered but rejected because it cannot separate training from inference, doesn't support detector serialization, and lacks composability for combining multiple detectors.

**Direct scikit-learn compatibility**: Using scikit-learn's `BaseEstimator` and standard `predict()` method naming was considered but rejected to avoid heavy dependencies and to use domain-appropriate terminology (`detect` is clearer than `predict` for anomaly detection).

## Consequences

**Positive**: Familiar pattern for users who know scikit-learn. Extensible through base class inheritance (validation, type handling, serialization automatic for new detectors). Natural separation between training and inference phases for operational deployment.

**Negative**: This API pattern is now fundamental and cannot be changed without a major version bump. Not directly compatible with scikit-learn pipelines or grid search utilities.
