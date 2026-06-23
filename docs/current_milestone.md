# Repository Bootstrap

## Objective

Create the clean, local-first SparrowML scaffold and workflow infrastructure.

## Baseline

No SparrowML implementation exists. Sparrow-V remains a separate external repository.

## In scope

Repository layout, documentation, package skeleton, configuration, typed target contracts, CLI diagnostics, checks, tests, and milestone launcher.

## Out of scope

Training, dataset downloads, PyTorch, quantization, pruning, compiler implementation, hardware execution, RTL, commits, and pushes.

## Required files

All files listed in the bootstrap request, including workflow documents, launcher, checks, package modules, contract schemas, tests, and target configuration.

## Validation

`python3 -m compileall src scripts`; `python3 scripts/check_repo.py`; `python3 scripts/smoke_test.py`; `pytest`; `make smoke`; `make check`; `make docs-check`; `bash -n scripts/run_milestone.sh`; `git diff --check`.

## Acceptance criteria

The requested scaffold is coherent, offline-testable, contract-validated, accurately documented as scaffold-only, and does not copy or execute Sparrow-V.

## Stop conditions

Stop only for an ownership/licensing decision, missing required local tool, or a genuine architectural conflict.

## Result-file requirements

Update `docs/codex_milestone_result.md` during work. Finalize it as `COMPLETE`, `FAILED`, or `BLOCKED`; never leave it `IN_PROGRESS` on normal exit.
