PYTHON ?= python3

.PHONY: help install test lint format-check check docs-check smoke doctor validate-contracts milestone clean

help:
	@echo "Targets: install test lint format-check check docs-check smoke doctor validate-contracts milestone clean"

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

milestone:
	bash ./scripts/run_milestone.sh

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
