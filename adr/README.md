# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for the tsod project. ADRs help new developers (and future maintainers) understand why the codebase is structured the way it is.

## What is an ADR?

An Architecture Decision Record (ADR) documents a significant architectural decision made in the project, including the context, the decision itself, alternatives considered, and consequences.

## Format

Each ADR follows this structure:

- **Status**: Draft, Accepted, Superseded, or Deprecated
- **Date**: When the decision was made or drafted
- **Context**: The problem or requirement that prompted this decision
- **Decision**: What was chosen and why
- **Alternatives Considered**: Other options that were evaluated
- **Consequences**: Trade-offs, benefits, and implications

## Index

- [ADR-001](001-scikit-learn-api-pattern.md) - Scikit-learn-inspired API pattern
- [ADR-002](002-pandas-series-primary-data-structure.md) - pandas Series as primary data structure
- [ADR-003](003-time-aware-detectors.md) - Time-aware vs time-agnostic detectors

## Contributing

When making significant architectural changes, please:

1. Create a new ADR in Draft status
2. Discuss with the team
3. Update to Accepted status once implemented
4. Update this index with a link to the new ADR
