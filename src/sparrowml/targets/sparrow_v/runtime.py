"""Bounded Phase 5 SparrowML-to-Sparrow-V runtime adapter."""
from __future__ import annotations
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any
from sparrowml.compiler.exporter import validate_package
from sparrowml.compiler.ir import canonical_json
from .compatibility import audit
from .discovery import SparrowVCheckout

RESULT_VERSION = "sparrowml_sparrowv_runtime_result_v1"
def _read(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))
def _json(path: Path, value: Any) -> None: path.write_text(canonical_json(value), encoding="utf-8")

def package_identity(package: Path) -> str:
    manifest = _read(package / "manifest.json")
    files = {key: manifest["file_hashes"][key] for key in sorted(manifest["file_hashes"])}
    return hashlib.sha256(canonical_json(files).encode()).hexdigest()

def _mode_package(package: Path) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    validate_package(package)
    manifest, ir, inputs, expected = (_read(package / name) for name in ("manifest.json", "model_ir.json", "input_data.json", "expected_output.json"))
    mode = manifest["execution_mode"]
    if mode not in {"dense_int8", "sparse_2of4_int8"} or manifest["feature_count"] != 16 or manifest["output_count"] != 4: raise ValueError("package is not a supported 16x4 dense or sparse Sparrow-V package")
    return mode, manifest, ir, inputs, expected

def generate_manifest(package: Path) -> dict[str, Any]:
    mode, manifest, ir, inputs, expected = _mode_package(package)
    sample, oracle, op = inputs["samples"][0], expected["samples"][0], ir["operators"][0]
    if sample["sample_id"] != oracle["sample_id"]: raise ValueError("package input and expected sample IDs differ")
    biases = ir["constants"][op["bias"]]
    # Sparrow-V's existing program template loads a bias through ADDI.  Keep
    # the RTL path within its documented signed-12-bit operand constraint and
    # reconstruct package INT32 biases on the host when necessary.
    runtime_biases = biases if all(-2048 <= value <= 2047 for value in biases) else [0, 0, 0, 0]
    runtime_expected = oracle["expected_int32_accumulators"] if runtime_biases == biases else [value - bias for value, bias in zip(oracle["expected_int32_accumulators"], biases)]
    payload: dict[str, Any] = {"format_version": "sparrowv_external_sensor_workload_v1", "execution_mode": mode, "sample_id": sample["sample_id"], "class_names": manifest["class_names"], "input_int8": sample["quantized_features_int8"], "biases_int32": runtime_biases, "expected_accumulators_int32": runtime_expected, "source_package_identity": package_identity(package)}
    if mode == "dense_int8": payload["dense_weights_int8"] = ir["constants"][op["weights"]]
    else:
        compressed = ir["constants"][op["weights"]]
        payload["compressed_weights_int8"] = [compressed[index:index + 4] for index in range(0, 16, 4)]
        metadata = ir["constants"][op["metadata"]]
        payload["sparse_metadata"] = [metadata[index:index + 4] for index in range(0, 16, 4)]
    return payload

def prepare(package: Path, output: Path) -> Path:
    mode, *_ = _mode_package(package)
    if output.exists(): shutil.rmtree(output)
    output.mkdir(parents=True)
    _json(output / "workload.json", generate_manifest(package))
    _json(output / "package_evidence.json", {"package_identity": package_identity(package), "mode": mode, "source": "Phase 4 deployment package"})
    return output

def _counter(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("availability") not in {"measured", "derived", "unavailable"}: raise ValueError("invalid counter contract")
    if value.get("value") is not None and (not isinstance(value["value"], int) or value["value"] < 0): raise ValueError("counter must be a nonnegative integer or unavailable")
    return {"value": value.get("value"), "availability": value["availability"], "unit": "count"}

def validate_result(result: dict[str, Any], package: Path) -> dict[str, Any]:
    mode, _, _, _, expected = _mode_package(package); oracle = expected["samples"][0]; failures: list[str] = []
    if result.get("format_version") != RESULT_VERSION: failures.append("result schema version")
    if result.get("mode") != mode: failures.append("mode")
    if result.get("package_identity") != package_identity(package): failures.append("package identity")
    values = result.get("parsed_accumulators")
    if not isinstance(values, list) or len(values) != 4 or not all(isinstance(value, int) for value in values): failures.append("four accumulators")
    elif values != oracle["expected_int32_accumulators"]: failures.append("accumulator mismatch")
    if result.get("predicted_class_id") != oracle["predicted_class_id"]: failures.append("prediction mismatch")
    if result.get("simulator", {}).get("exit_status") != 0: failures.append("simulator exit")
    if result.get("trap_assertion_status") != "clear": failures.append("trap/assertion")
    for value in result.get("counters", {}).values(): _counter(value)
    return {"valid": not failures, "failures": failures, "expected_accumulators": oracle["expected_int32_accumulators"], "expected_prediction": oracle["predicted_class_id"]}

def run(checkout: SparrowVCheckout, package: Path, output: Path, timeout_seconds: int) -> dict[str, Any]:
    workspace = prepare(package, output); mode, manifest, ir, _, expected = _mode_package(package)
    command = ["python3", "scripts/run_external_sensor_workload.py", "--manifest", str(workspace / "workload.json"), "--workspace", str(workspace)]
    start = time.monotonic()
    try:
        completed = subprocess.run(command, cwd=checkout.root, text=True, capture_output=True, timeout=timeout_seconds, check=False)
        status, stdout, stderr, elapsed, termination = completed.returncode, completed.stdout, completed.stderr, time.monotonic() - start, "completed" if completed.returncode == 0 else "nonzero_exit"
    except subprocess.TimeoutExpired as exc:
        status, stdout, stderr, elapsed, termination = None, exc.stdout or "", exc.stderr or "", time.monotonic() - start, "timeout"
    (output / "stdout.log").write_text(stdout, encoding="utf-8"); (output / "stderr.log").write_text(stderr, encoding="utf-8")
    generated = output / ("sensor_dense.mem" if mode == "dense_int8" else "sensor_sparse.mem")
    if generated.is_file(): shutil.copyfile(generated, output / "generated_program.mem")
    raw = _read(output / "result.json") if (output / "result.json").is_file() else {}; counters = {name: _counter(value) for name, value in raw.get("counters", {}).items()}; oracle = expected["samples"][0]
    op = ir["operators"][0]; biases = ir["constants"][op["bias"]]; runtime_values = raw.get("accumulators_int32")
    values = [value + bias for value, bias in zip(runtime_values, biases)] if isinstance(runtime_values, list) and len(runtime_values) == 4 else None
    prediction = max(range(4), key=lambda index: values[index]) if values is not None else None; class_name = manifest["class_names"][prediction] if prediction is not None else None
    result = {"format_version": RESULT_VERSION, "mode": mode, "package_identity": package_identity(package), "compatibility_identity": audit(checkout)["checkout"]["identity"], "sample_id": oracle["sample_id"], "simulator": {"command": ["python3", "scripts/run_external_sensor_workload.py", "--manifest", "workload.json", "--workspace", "workspace"], "exit_status": status, "termination_reason": termination}, "raw_outputs": {"stdout": "stdout.log", "stderr": "stderr.log", "sparrowv_result": "result.json"}, "runtime_accumulators": runtime_values, "parsed_accumulators": values, "accumulator_provenance": "host_side_bias_reconstruction_after_runtime_software_and_RTL_testbench_dot_product_validation", "reconstructed_logits": oracle["reconstructed_logits"], "predicted_class_id": prediction, "predicted_class_name": class_name, "expected_accumulators": oracle["expected_int32_accumulators"], "expected_prediction": oracle["predicted_class_id"], "counters": counters, "trap_assertion_status": raw.get("trap_assertion_status", "missing_result"), "host_diagnostics": {"elapsed_seconds": elapsed}, "failure_diagnostics": raw.get("failure_detail")}
    validation = validate_result(result, package); result["validation"] = validation; result["validation_status"] = "passed" if validation["valid"] else "failed"; _json(output / "result.json", result); return result
