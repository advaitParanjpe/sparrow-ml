# SparrowML Context

SparrowML is a local-first ML systems project that will transform models into Sparrow-V deployment artifacts and evaluate them. Phase 1 provides a deterministic synthetic sensor fixture and CPU FP32 linear baseline; it does not quantize, compile, or execute hardware.

Map: `src/sparrowml/` package; `configs/` repository and target settings; `docs/` architecture and milestone state; `scripts/` deterministic checks and launcher; `tests/` offline tests; `data/` ignored datasets; `artifacts/` ignored generated outputs; `experiments/` configurations and summaries. Sparrow-V is an external target via `targets/sparrow_v/`; no RTL belongs here.

Stable commands: `make smoke`, `make check`, `make docs-check`, `make doctor`, `make validate-contracts`, `make milestone`. Target config resolves `SPARROWV_ROOT` or sibling `../sparrow-v`; bootstrap execution is disabled. Important files: `AGENTS.md`, `docs/current_milestone.md`, `configs/targets/sparrow_v.yaml`, and target contract schemas.

Decisions: local paths first, no submodule, repository-relative artifact paths, typed standard-library schemas, no ML frameworks yet. Do not read build outputs, ignored data/artifacts, or unrelated source modules unless the milestone requires them.
