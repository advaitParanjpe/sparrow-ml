PYTHON ?= python3

.PHONY: help install test lint format-check check docs-check smoke doctor validate-contracts milestone clean generate-fixture train-fp32 evaluate-fp32 run-fp32-baseline test-phase1 calibrate-int8 quantize-int8 evaluate-int8 run-int8-baseline test-phase2

help:
	@echo "Targets: install test lint format-check check docs-check smoke doctor validate-contracts generate-fixture train-fp32 evaluate-fp32 run-fp32-baseline calibrate-int8 quantize-int8 evaluate-int8 run-int8-baseline test-phase1 test-phase2 milestone clean"

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

milestone:
	bash ./scripts/run_milestone.sh

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
