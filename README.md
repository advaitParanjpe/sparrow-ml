# SparrowML

SparrowML is a local-first ML systems project for deterministic data processing, optimization, export, runtime tooling, and hardware-aware evaluation. It is separate from [Sparrow-V](docs/decisions/ADR-001-repository-boundary.md), which remains the owner of processor RTL, simulator, instructions, and hardware-specific execution.

**Current status: Phase 7 multi-layer Sparrow-V RTL/reference validation implemented.** The fixed `Linear(16,16) → ReLU → Linear(16,4)` flow executes its dense layers through five isolated `16→4` RTL workloads (four `fc1` partitions and one `fc2` run). ReLU and hidden requantization are exact host post-processing with explicit provenance; this is not monolithic optimized hardware execution or physical-hardware measurement.

The intended future flow is dataset → preprocessing → model → quantization/pruning → SparrowML IR → target lowering → packed artifacts → Sparrow-V runtime/RTL → metrics. See [architecture](docs/architecture.md) and the [roadmap](docs/build_roadmap.md).

## Layout

`src/` is the package; `configs/` defines project and target settings; `docs/` records workflow and decisions; `scripts/` provides offline checks; `data/` and `artifacts/` hold ignored local inputs/outputs; `experiments/` holds configurations and summaries.

## Setup and commands

Use Python 3.11+. Install locally with `python3 -m pip install -e '.[dev]'`.

`make run-multilayer-baseline` runs the Phase 6 flow. `make run-sparrowv-mlp-baseline` runs the Phase 7 RTL/reference workflow twice; `make test-phase7` and `make test-phase7-integration` run offline and real-RTL focused tests. Reported accuracy is fixture accuracy only: the classes are synthetic vibration-fault-style placeholders, not real-world claims.

## Milestone workflow

Read [AGENTS.md](AGENTS.md), [context](docs/codex_context.md), and the [current milestone](docs/current_milestone.md). Results are preserved in the tracked [result file](docs/codex_milestone_result.md).

## Data and artifacts

See [data contracts](docs/data_contracts.md), [data policy](data/README.md), and [artifact policy](artifacts/README.md). Do not commit raw downloads, checkpoints, generated deployments, secrets, or API keys unless a documented exception applies.
