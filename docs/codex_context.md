# SparrowML Context

SparrowML is a local-first ML systems project that transforms models into Sparrow-V deployment artifacts and evaluates them through a local RTL simulator. Phase 1 provides a deterministic synthetic sensor fixture and CPU FP32 linear baseline. Phase 2 adds deterministic INT8 PTQ with train-only calibration and explicit integer reference inference. Phase 3 adds deterministic input-axis 2:4 pruning, fixed-mask fine-tuning, legal 3-bit sparse metadata, packing, and independent sparse integer reference inference. Phase 4 adds canonical IR and deterministic dense/sparse export packages. Phase 5 converts those packages to Sparrow-V's documented external manifest, runs dense/sparse RTL simulation twice, and validates exact accumulators.

Map: `src/sparrowml/` package; `configs/` repository and target settings; `docs/` architecture and milestone state; `scripts/` deterministic checks and launcher; `tests/` offline tests; `data/` ignored datasets; `artifacts/` ignored generated outputs; `experiments/` configurations and summaries. Sparrow-V is an external target via `targets/sparrow_v/`; no RTL belongs here.

Stable commands: `make smoke`, `make check`, `make docs-check`, `make doctor`, `make validate-contracts`, `make run-int8-baseline`, and `make milestone`. Target config resolves `SPARROWV_ROOT` or sibling `../sparrow-v`; bootstrap execution is disabled. Important files: `AGENTS.md`, `docs/current_milestone.md`, `configs/targets/sparrow_v.yaml`, and target contract schemas.

Decisions: local paths first, no submodule, repository-relative artifact paths, typed standard-library schemas, no ML frameworks yet. Do not read build outputs, ignored data/artifacts, or unrelated source modules unless the milestone requires them.
