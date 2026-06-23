"""Audit the documented Sparrow-V external sensor workload interface."""
from __future__ import annotations
import shutil
import subprocess
from typing import Any
from .discovery import SparrowVCheckout

INTERFACE = "sparrowv_external_sensor_workload_v1"
RESULT_INTERFACE = "sparrowv_external_sensor_result_v1"

def _git(checkout: SparrowVCheckout) -> str | None:
    value = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout.root, text=True, capture_output=True, check=False)
    return value.stdout.strip() if value.returncode == 0 else None

def audit(checkout: SparrowVCheckout) -> dict[str, Any]:
    iverilog = shutil.which("iverilog")
    return {"format_version": "sparrowml_sparrowv_compatibility_v1", "checkout": {"identity": _git(checkout), "discovery_source": checkout.source, "repository_name": "sparrow-v"},
        "interface": {"workload_format": INTERFACE, "result_format": RESULT_INTERFACE,
            "dense_command": ["python3", "scripts/run_external_sensor_workload.py", "--manifest", "<workspace>/workload.json", "--workspace", "<workspace>"],
            "sparse_command": ["python3", "scripts/run_external_sensor_workload.py", "--manifest", "<workspace>/workload.json", "--workspace", "<workspace>"],
            "required_generated_files": ["workload.json"], "accepted_input_formats": ["JSON manifest"], "result_file": "result.json", "timeout_policy": "adapter-enforced host timeout"},
        "targets": {"dense_int8": True, "sparse_2of4_int8": True, "requires_rtl_changes": False},
        "tools": {"python3": shutil.which("python3") is not None, "iverilog": iverilog is not None, "verilator": shutil.which("verilator") is not None, "make": shutil.which("make") is not None},
        "counters": {"cycles": "measured", "retired_instructions": "measured", "vector_loads": "measured", "vector_stores": "measured", "dense_dot_products": "measured", "sparse_dot_products": "measured", "executed_int8_multiplications": "measured_sparse/unavailable_dense", "skipped_int8_multiplications": "measured_sparse/unavailable_dense", "dense_conceptual_int8_multiplications": "derived_dense/unavailable_sparse"},
        "compatible": iverilog is not None, "missing_requirements": [] if iverilog else ["iverilog"], "repository_modifications": "none"}
