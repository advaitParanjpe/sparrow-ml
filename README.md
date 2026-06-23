# SparrowML

SparrowML is a local-first ML systems project for deterministic data processing, optimization, export, runtime tooling, and hardware-aware evaluation. It is separate from [Sparrow-V](docs/decisions/ADR-001-repository-boundary.md), which remains the owner of processor RTL, simulator, instructions, and hardware-specific execution.

**Current status: Phase 2 INT8 post-training quantization implemented.** The deterministic synthetic four-class fixture and CPU FP32 `Linear(16, 4)` baseline now include train-only calibration, per-output-channel INT8 weights, INT32 biases, and inspectable integer reference inference. Pruning, compilation, deployment, and Sparrow-V execution are not implemented.

The intended future flow is dataset → preprocessing → model → quantization/pruning → SparrowML IR → target lowering → packed artifacts → Sparrow-V runtime/RTL → metrics. See [architecture](docs/architecture.md) and the [roadmap](docs/build_roadmap.md).

## Layout

`src/` is the package; `configs/` defines project and target settings; `docs/` records workflow and decisions; `scripts/` provides offline checks; `data/` and `artifacts/` hold ignored local inputs/outputs; `experiments/` holds configurations and summaries.

## Setup and commands

Use Python 3.11+. Install locally with `python3 -m pip install -e '.[dev]'`.

`make generate-fixture`, `make train-fp32`, `make evaluate-fp32`, and `make run-fp32-baseline` reproduce the Phase 1 workflow. `make calibrate-int8`, `make quantize-int8`, `make evaluate-int8`, and `make run-int8-baseline` reproduce Phase 2. Focused tests are `make test-phase1` and `make test-phase2`. Reported accuracy is fixture accuracy only: the classes are synthetic vibration-fault-style placeholders, not real-world claims. `make smoke`, `make check`, `make docs-check`, `make doctor`, and `make validate-contracts` remain offline commands.

## Milestone workflow

Read [AGENTS.md](AGENTS.md), [context](docs/codex_context.md), and the [current milestone](docs/current_milestone.md). Results are preserved in the tracked [result file](docs/codex_milestone_result.md).

## Data and artifacts

See [data contracts](docs/data_contracts.md), [data policy](data/README.md), and [artifact policy](artifacts/README.md). Do not commit raw downloads, checkpoints, generated deployments, secrets, or API keys unless a documented exception applies.
