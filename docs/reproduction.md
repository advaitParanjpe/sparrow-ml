# Reproduction Guide

This guide separates fast repository verification from workflows that require local data or the external Sparrow-V checkout. All commands run from the repository root.

## Prerequisites

- Python 3.11 or newer;
- Python dependencies: PyTorch, NumPy, PyYAML, and pytest (`python3 -m pip install -e '.[dev]'`);
- Icarus Verilog and `vvp` for Sparrow-V RTL simulation;
- a compatible Sparrow-V sibling checkout;
- a local WISDM dataset copy for real-data preparation and training.

Configure local paths without adding them to tracked files:

```bash
export WISDM_ROOT=~/Datasets/WISDM/wisdm-dataset
export SPARROWV_ROOT=~/Desktop/projects/sparrow-v
```

`SPARROWV_ROOT` may be omitted when a compatible checkout exists at `../sparrow-v`. `WISDM_ROOT` is required for every `prepare-wisdm` or `run-wisdm-*` command that consumes data.

## Fast verification

These commands do not retrain WISDM or invoke RTL simulation. They check local configuration, source tests, package imports, documentation links, and repository policy.

```bash
make doctor
make check
make docs-check
```

Expected success indicators are `repository checks: passed` and passing pytest output. `make doctor` reports the resolved Sparrow-V location but does not require that it exist.

## Full real-data workflow

With `WISDM_ROOT` set, execute the stages in order:

```bash
make run-wisdm-phase8a  # prepares subject-held-out windows; requires WISDM
make run-wisdm-phase8b  # trains/evaluates/quantizes/exports; requires WISDM and may take longer
make run-wisdm-phase8c  # selected-sample RTL validation; requires WISDM, Sparrow-V, iverilog, vvp
```

`make run-wisdm-final` orchestrates the final bounded workflow after the prerequisites are available. It regenerates local ignored artifacts under `artifacts/phase8_wisdm/`; it is not needed for ordinary documentation checks and should not be used merely to inspect the tracked final metrics.

The expected Phase 8C outcome is exact agreement for 12 selected samples at `fc1`, hidden INT8, `fc2`, and prediction levels. Its reported cycle totals are partitioned simulation totals, not monolithic latency.

## RTL integration

Check target compatibility before running integration work:

```bash
make sparrowv-doctor
make prepare-sparrowv-mlp
make run-sparrowv-mlp
make validate-sparrowv-mlp
```

The first command checks the external checkout and required runner interface. The next commands generate an ignored workspace, execute five isolated workloads (four `fc1`, one `fc2`), and verify the result. `make run-sparrowv-mlp-baseline` repeats the workflow twice and checks deterministic semantic hashes. For the WISDM-specific integrated path, use `make run-wisdm-phase8c` after Phase 8B produces its package.

## Other reproducible checks

Synthetic and compiler-contract workflows do not require WISDM; Sparrow-V runtime workflows do require the target checkout.

```bash
make run-fp32-baseline
make run-int8-baseline
make run-sparse-baseline
make run-export-baseline
make test-phase1
make test-phase2
make test-phase3
make test-phase4
make test-phase5
make test-phase6
make test-phase7
make test-phase8a
make test-phase8b
make test-phase8c
```

Generated datasets, checkpoints, deployment packages, logs, and simulator workspaces are ignored. The final metrics and required command boundaries are documented in [final results](results/final_results.md).
