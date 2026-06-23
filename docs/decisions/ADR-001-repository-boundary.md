# ADR-001: SparrowML Repository Boundary

## Context

SparrowML needs to create deployment artifacts and evaluate external hardware targets while Sparrow-V owns its RTL and execution environment.

## Decision

SparrowML is separate from Sparrow-V. It consumes Sparrow-V through artifact and command contracts; no RTL is copied. Initial integration uses a local path and no Git submodule. A pinned Sparrow-V release or submodule may be considered only after the interface stabilizes.

## Consequences

Contracts are explicit and independently testable, but cross-repository compatibility must be validated by a future integration milestone.

## Alternatives rejected

Copying RTL creates divergent ownership; a submodule before contract stability creates unnecessary coupling; a cloud service adds scope without current need.
