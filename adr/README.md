# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for the tsod project. ADRs help new developers (and future maintainers) understand why the codebase is structured the way it is.

## What is an ADR?

An Architecture Decision Record (ADR) documents a significant architectural decision made in the project, including the context, the decision itself, alternatives considered, and consequences.

## Format

Each ADR follows this structure:

- **Status**: Draft, Accepted, or Superseded
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

### Superseding an Existing ADR

When a new decision replaces an old one:

1. Create the new ADR following the normal process
2. In the new ADR, include a note in the Context section mentioning which ADR it supersedes (e.g., "This decision supersedes [ADR-001](001-previous-decision.md)")
3. Update the Status field of the old ADR from "Accepted" to "Superseded"
4. Do NOT modify the body of the old ADR (Context, Decision, Alternatives, Consequences) - it remains as an immutable historical record
5. Both ADRs remain in the repository to preserve the full decision history
