# Source Manifest

| Area | Primary locations | Role |
| --- | --- | --- |
| Data ingestion | `src/sparrowml/data/wisdm.py`, `src/sparrowml/data/fixture.py` | WISDM preparation and deterministic synthetic fixture |
| Feature extraction and preprocessing | `src/sparrowml/data/`, `configs/experiments/` | window handling, 16 features, train-only transforms |
| Models and training | `src/sparrowml/models/`, `src/sparrowml/training/` | fixed linear and two-layer MLP training |
| Quantization | `src/sparrowml/quantization/` | calibration, INT8 tensors, explicit integer reference inference |
| Sparsity | `src/sparrowml/sparsity/` | controlled single-layer 2:4 experiments |
| Compiler and packages | `src/sparrowml/compiler/`, `src/sparrowml/artifacts/` | IR, lowering, deterministic package validation |
| Sparrow-V adapter | `src/sparrowml/targets/sparrow_v/`, `configs/targets/sparrow_v.yaml` | external target discovery, workload preparation, result validation |
| CLI and Make entry points | `src/sparrowml/cli.py`, `Makefile` | reproducible commands and help |
| Tests | `tests/` | phase-specific offline and optional integration coverage |
| Configurations | `configs/project.yaml`, `configs/experiments/`, `configs/targets/` | deterministic project, experiment, and target settings |
| Results and contracts | `docs/results/`, `docs/*contract*.md`, `docs/wisdm_evaluation_protocol.md` | measured results, interfaces, and protocol |

`data/` and `artifacts/` are local ignored roots. They contain no source-controlled raw WISDM data, processed windows, checkpoints, deployment packages, or simulator logs. Sparrow-V source and RTL are deliberately outside this repository.
