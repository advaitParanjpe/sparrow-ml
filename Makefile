PYTHON ?= python3

.PHONY: help install test lint format-check check docs-check smoke doctor validate-contracts milestone clean generate-fixture train-fp32 evaluate-fp32 run-fp32-baseline test-phase1 calibrate-int8 quantize-int8 evaluate-int8 run-int8-baseline test-phase2 prune-2of4 finetune-sparse pack-sparse evaluate-sparse run-sparse-baseline test-phase3 lower-ir validate-ir export-sparrowv-dense export-sparrowv-sparse validate-export run-export-baseline test-phase4 sparrowv-doctor prepare-sparrowv-dense prepare-sparrowv-sparse run-sparrowv-dense run-sparrowv-sparse run-sparrowv-baseline test-phase5 test-phase5-integration train-mlp quantize-mlp evaluate-mlp-int8 export-mlp validate-mlp-export run-multilayer-baseline test-phase6 prepare-sparrowv-mlp run-sparrowv-mlp validate-sparrowv-mlp run-sparrowv-mlp-baseline test-phase7 test-phase7-integration

help:
	@echo "Targets: install test lint format-check check docs-check smoke doctor validate-contracts Phase 1-7 targets sparrowv-doctor prepare-sparrowv-{dense,sparse,mlp} run-sparrowv-{dense,sparse,baseline,mlp,mlp-baseline} test-phase5 test-phase5-integration test-phase7 test-phase7-integration milestone clean"

train-mlp quantize-mlp evaluate-mlp-int8 export-mlp run-multilayer-baseline:
	$(PYTHON) -m sparrowml.cli $@

validate-mlp-export:
	$(PYTHON) -m sparrowml.cli validate-mlp-export

test-phase6:
	$(PYTHON) -m pytest tests/test_phase6.py

install:
	$(PYTHON) -m pip install -e '.[dev]'

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) scripts/check_repo.py

format-check:
	$(PYTHON) -m compileall -q src scripts

docs-check:
	$(PYTHON) scripts/check_repo.py --docs-only

smoke:
	$(PYTHON) scripts/smoke_test.py

check: format-check lint test docs-check

doctor:
	$(PYTHON) -m sparrowml.cli doctor

validate-contracts:
	$(PYTHON) -m sparrowml.cli validate-contracts

generate-fixture:
	$(PYTHON) -m sparrowml.cli generate-fixture

train-fp32:
	$(PYTHON) -m sparrowml.cli train-fp32

evaluate-fp32:
	$(PYTHON) -m sparrowml.cli evaluate-fp32

run-fp32-baseline:
	$(PYTHON) -m sparrowml.cli run-fp32-baseline

test-phase1:
	$(PYTHON) -m pytest tests/test_phase1.py

calibrate-int8:
	$(PYTHON) -m sparrowml.cli calibrate-int8

quantize-int8:
	$(PYTHON) -m sparrowml.cli quantize-int8

evaluate-int8:
	$(PYTHON) -m sparrowml.cli evaluate-int8

run-int8-baseline:
	$(PYTHON) -m sparrowml.cli run-int8-baseline

test-phase2:
	$(PYTHON) -m pytest tests/test_phase2.py

prune-2of4 finetune-sparse pack-sparse evaluate-sparse run-sparse-baseline:
	$(PYTHON) -m sparrowml.cli $@

test-phase3:
	$(PYTHON) -m pytest tests/test_phase3.py

lower-ir:
	$(PYTHON) -m sparrowml.cli lower-ir --mode dense

validate-ir:
	$(PYTHON) -m sparrowml.cli lower-ir --mode dense --output artifacts/phase4_export/dense_ir.json
	$(PYTHON) -m sparrowml.cli validate-ir artifacts/phase4_export/dense_ir.json

export-sparrowv-dense:
	$(PYTHON) -m sparrowml.cli export-sparrowv --mode dense

export-sparrowv-sparse:
	$(PYTHON) -m sparrowml.cli export-sparrowv --mode sparse

validate-export:
	$(PYTHON) -m sparrowml.cli validate-export artifacts/phase4_export/dense

run-export-baseline:
	$(PYTHON) -m sparrowml.cli run-export-baseline

test-phase4:
	$(PYTHON) -m pytest tests/test_phase4.py

sparrowv-doctor:
	$(PYTHON) -m sparrowml.cli sparrowv-doctor

prepare-sparrowv-dense:
	$(PYTHON) -m sparrowml.cli prepare-sparrowv-run --mode dense

prepare-sparrowv-sparse:
	$(PYTHON) -m sparrowml.cli prepare-sparrowv-run --mode sparse

run-sparrowv-dense:
	$(PYTHON) -m sparrowml.cli run-sparrowv --mode dense

run-sparrowv-sparse:
	$(PYTHON) -m sparrowml.cli run-sparrowv --mode sparse

run-sparrowv-baseline:
	$(PYTHON) -m sparrowml.cli run-sparrowv-baseline

test-phase5:
	$(PYTHON) -m pytest tests/test_phase5.py

test-phase5-integration:
	SPARROWML_REQUIRE_SPARROWV=1 $(PYTHON) -m pytest tests/test_phase5_integration.py

prepare-sparrowv-mlp run-sparrowv-mlp run-sparrowv-mlp-baseline:
	$(PYTHON) -m sparrowml.cli $@

validate-sparrowv-mlp:
	$(PYTHON) -m sparrowml.cli validate-sparrowv-mlp artifacts/phase6_multilayer/export artifacts/phase7_multilayer_runtime/multilayer_result.json

test-phase7:
	$(PYTHON) -m pytest tests/test_phase7.py

test-phase7-integration:
	SPARROWML_REQUIRE_SPARROWV=1 $(PYTHON) -m pytest tests/test_phase7_integration.py

milestone:
	bash ./scripts/run_milestone.sh

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
