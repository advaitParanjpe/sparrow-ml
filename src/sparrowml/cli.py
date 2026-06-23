"""Minimal CLI for local diagnostics and contract validation."""

from __future__ import annotations

import argparse
import shutil
import sys

from . import __version__
from .artifacts.schemas import DeploymentManifest, ModelMetadata
from .config import load_project_config
from .targets.sparrow_v.contracts import SparrowVResult


def _doctor() -> int:
    config = load_project_config()
    print(f"Python: {sys.version.split()[0]}")
    print(f"SparrowML import: {__version__}")
    print(f"Repository root: {config.root}")
    print(f"Sparrow-V root: {config.sparrow_v.root}")
    print(f"Sparrow-V exists: {config.sparrow_v.root.exists()}")
    print(f"python3 available: {shutil.which('python3') is not None}")
    print(f"git available: {shutil.which('git') is not None}")
    return 0


def _show_config() -> int:
    config = load_project_config()
    print("project:", config.name)
    print("artifacts_root:", config.artifacts_root)
    print("data_root:", config.data_root)
    print("sparrow_v.root:", config.sparrow_v.root)
    print("sparrow_v.enabled:", config.sparrow_v.enabled)
    return 0


def _validate_contracts() -> int:
    model = ModelMetadata("bootstrap-example", (1, 8), 2, "int8-planned")
    DeploymentManifest(model, "artifacts/input.bin", "artifacts/dense.bin", "artifacts/sparse.bin", "artifacts/sparse_meta.bin", "artifacts/biases.bin", "artifacts/expected.bin", "artifacts/program.bin", "artifacts/data.bin")
    SparrowVResult("complete", (4, -2), 0, 12, 8, 1, 1, 0, 64, 0)
    print("contracts: valid")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sparrowml")
    parser.add_argument("command", choices=("doctor", "show-config", "validate-contracts"))
    command = parser.parse_args(argv).command
    return {"doctor": _doctor, "show-config": _show_config, "validate-contracts": _validate_contracts}[command]()


if __name__ == "__main__":
    raise SystemExit(main())
