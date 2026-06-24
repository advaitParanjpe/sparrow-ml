"""Fixed Phase 7 16->16->4 dense Sparrow-V runtime adapter."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from sparrowml.compiler.ir import canonical_json
from sparrowml.quantization.multilayer import validate_export, validate_ir
from .compatibility import audit
from .discovery import SparrowVCheckout

RESULT_VERSION = "sparrowml_sparrowv_multilayer_runtime_result_v1"
WORKLOAD_VERSION = "sparrowv_external_sensor_workload_v1"
PROVENANCE = {"rtl_produced", "runtime_software_produced", "host_reconstructed"}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_identity(package: Path) -> str:
    manifest = _read(package / "manifest.json")
    return hashlib.sha256(canonical_json({k: manifest["file_hashes"][k] for k in sorted(manifest["file_hashes"])}).encode()).hexdigest()


def _package(package: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = _read(package / "manifest.json")
    for name, digest in manifest.get("file_hashes", {}).items():
        if _sha(package / name) != digest:
            raise ValueError("package hash mismatch")
    validate_export(package)
    ir, inputs, traces = (_read(package / name) for name in ("model_ir.json", "input.json", "intermediate_reference.json"))
    if manifest.get("package_format_version") != "sparrowml_multilayer_package_v1":
        raise ValueError("unsupported Phase 6 package version")
    validate_ir(ir)
    if ir["operators"][0]["output_count"] != 16 or ir["operators"][3]["output_count"] != 4:
        raise ValueError("unsupported Phase 6 layer shapes")
    return manifest, ir, inputs, traces


def _sample(inputs: dict[str, Any], traces: dict[str, Any], sample_id: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    items = {item["sample_id"]: item for item in inputs["samples"]}; references = {item["sample_id"]: item for item in traces["samples"]}
    identifier = sample_id or inputs["samples"][0]["sample_id"]
    if identifier not in items or identifier not in references: raise ValueError("requested sample is absent from package")
    item, trace = items[identifier], references[identifier]
    if item["sample_id"] != trace["sample_id"] or item["input_int8"] != trace["input_int8"]:
        raise ValueError("Phase 6 input and intermediate trace differ")
    return item, trace


def _workload(sample_id: str, values: list[int], weights: list[list[int]], expected: list[int], identity: str, layer: str, channels: list[int]) -> dict[str, Any]:
    if len(values) != 16 or len(weights) != 4 or any(len(row) != 16 for row in weights):
        raise ValueError("external workload must be exactly 16x4")
    return {"format_version": WORKLOAD_VERSION, "execution_mode": "dense_int8", "sample_id": sample_id,
            "class_names": ["channel_0", "channel_1", "channel_2", "channel_3"], "input_int8": values,
            "dense_weights_int8": weights, "biases_int32": [0, 0, 0, 0],
            "expected_accumulators_int32": expected, "source_package_identity": identity,
            "sparrowml_layer_identifier": layer, "sparrowml_output_channels": channels}


def _counter(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("availability") not in {"measured", "derived", "unavailable"}:
        raise ValueError("invalid Sparrow-V counter")
    number = value.get("value")
    if number is not None and (not isinstance(number, int) or number < 0):
        raise ValueError("invalid Sparrow-V counter value")
    return {"value": number, "availability": value["availability"], "unit": "count"}


def _execute(checkout: SparrowVCheckout, workspace: Path, workload: dict[str, Any], timeout: int) -> dict[str, Any]:
    workspace.mkdir(parents=True, exist_ok=True)
    _json(workspace / "workload.json", workload)
    command = ["python3", "scripts/run_external_sensor_workload.py", "--manifest", str(workspace / "workload.json"), "--workspace", str(workspace)]
    start = time.monotonic()
    try:
        done = subprocess.run(command, cwd=checkout.root, text=True, capture_output=True, timeout=timeout, check=False)
        status, stdout, stderr, reason = done.returncode, done.stdout, done.stderr, "completed" if done.returncode == 0 else "nonzero_exit"
    except subprocess.TimeoutExpired as exc:
        status, stdout, stderr, reason = None, exc.stdout or "", exc.stderr or "", "timeout"
    (workspace / "stdout.log").write_text(stdout, encoding="utf-8")
    (workspace / "stderr.log").write_text(stderr, encoding="utf-8")
    raw = _read(workspace / "result.json") if (workspace / "result.json").is_file() else {}
    return {"command": ["python3", "scripts/run_external_sensor_workload.py", "--manifest", "workload.json", "--workspace", "workspace"],
            "exit_status": status, "termination_reason": reason, "raw_outputs": {"stdout": "stdout.log", "stderr": "stderr.log", "sparrowv_result": "result.json"},
            "accumulators": raw.get("accumulators_int32"), "counters": {k: _counter(v) for k, v in raw.get("counters", {}).items()},
            "trap_assertion_status": raw.get("trap_assertion_status", "missing_result"), "failure_detail": raw.get("failure_detail")}


def _aggregate(runs: list[dict[str, Any]], conceptual: int) -> dict[str, Any]:
    names = sorted({name for run in runs for name in run["counters"]})
    total: dict[str, Any] = {}
    for name in names:
        values = [run["counters"].get(name) for run in runs]
        if any(v is None or v["availability"] == "unavailable" or v["value"] is None for v in values):
            total[name] = {"value": None, "availability": "unavailable", "unit": "count"}
        else:
            availability = "derived" if all(v["availability"] == "derived" for v in values) else "measured"
            total[name] = {"value": sum(v["value"] for v in values), "availability": availability, "unit": "count"}
    total["dense_conceptual_int8_multiplications"] = {"value": conceptual, "availability": "derived", "unit": "count"}
    return {"per_run": [run["counters"] for run in runs], "aggregate": total,
            "simulator_invocations": len(runs), "cycle_total_label": "partitioned simulation cycle total" if len(runs) > 1 else "single simulation cycles"}


def _hidden(acc: list[int], ir: dict[str, Any]) -> dict[str, Any]:
    q, scales = ir["quantization"], ir["constants"]["fc1_scales"]
    pre = (np.asarray(acc, dtype=np.float64) * float(q["input_scale"]) * np.asarray(scales, dtype=np.float64)).tolist()
    relu = np.maximum(np.asarray(pre), 0.0).tolist()
    raw = np.rint(np.asarray(relu) / float(q["hidden_scale"])).astype(np.int64)
    hidden = np.clip(raw, 0, 127).astype(np.int8).astype(int).tolist()
    return {"source_accumulators": acc, "input_scale": q["input_scale"], "fc1_weight_scales": scales,
            "reconstructed_pre_relu_values": pre, "relu_values": relu, "hidden_scale": q["hidden_scale"],
            "rounding": "NumPy rint ties-to-even", "clamp_range": [0, 127], "hidden_int8": hidden,
            "provenance": "host_reconstructed"}


def prepare(package: Path, output: Path, sample_id: str | None = None) -> Path:
    _, ir, inputs, traces = _package(package); item, trace = _sample(inputs, traces, sample_id)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    identity = package_identity(package); constants = ir["constants"]
    for index in range(4):
        channels = list(range(index * 4, index * 4 + 4)); expected = [trace["fc1_acc_int32"][i] - constants["fc1_bias"][i] for i in channels]
        _json(output / "fc1" / f"partition_{index}" / "workload.json", _workload(item["sample_id"], item["input_int8"], constants["fc1_weights"][index * 4:index * 4 + 4], expected, identity, "fc1", channels))
    expected2 = [trace["fc2_acc_int32"][i] - constants["fc2_bias"][i] for i in range(4)]
    _json(output / "fc2" / "workload.json", _workload(item["sample_id"], trace["hidden_int8"], constants["fc2_weights"], expected2, identity, "fc2", [0, 1, 2, 3]))
    _json(output / "package_evidence.json", {"package_identity": identity, "package_format": "sparrowml_multilayer_package_v1", "sample_id": item["sample_id"], "graph_identifier": ir["model_name"], "fc1_partition_count": 4})
    return output


def run(checkout: SparrowVCheckout, package: Path, output: Path, timeout_seconds: int, sample_id: str | None = None) -> dict[str, Any]:
    manifest, ir, inputs, traces = _package(package); item, trace = _sample(inputs, traces, sample_id)
    prepare(package, output, sample_id); constants, identity = ir["constants"], package_identity(package)
    audit_data = audit(checkout); _json(output / "compatibility.json", {**audit_data, "multilayer": {"supported_input_count": 16, "external_output_count": 4, "direct_fc1_16_output_support": False, "partitioning_required": True, "bias_policy": "zero_bias_rtl_then_host_reconstruction"}})
    fc1_runs, raw_fc1 = [], []
    for index in range(4):
        workspace = output / "fc1" / f"partition_{index}"; workload = _read(workspace / "workload.json"); execution = _execute(checkout, workspace, workload, timeout_seconds)
        channels = workload["sparrowml_output_channels"]; values = execution["accumulators"]
        if not isinstance(values, list) or len(values) != 4: values = []
        raw_fc1.extend(values); fc1_runs.append({"partition_index": index, "output_channels": channels, "execution": execution, "raw_rtl_accumulators": values, "raw_accumulator_provenance": "rtl_produced", "biases_int32": constants["fc1_bias"][index * 4:index * 4 + 4], "post_bias_accumulators": [v + b for v, b in zip(values, constants["fc1_bias"][index * 4:index * 4 + 4])], "post_bias_provenance": "host_reconstructed"})
    fc1_post = [v for run in fc1_runs for v in run["post_bias_accumulators"]]
    hidden = _hidden(fc1_post, ir) if len(fc1_post) == 16 else {"hidden_int8": [], "provenance": "host_reconstructed"}
    _json(output / "intermediate" / "hidden_trace.json", hidden)
    fc2_workspace = output / "fc2"; workload = _read(fc2_workspace / "workload.json")
    if hidden.get("hidden_int8"):
        workload["input_int8"] = hidden["hidden_int8"]; workload["expected_accumulators_int32"] = [trace["fc2_acc_int32"][i] - constants["fc2_bias"][i] for i in range(4)]; _json(fc2_workspace / "workload.json", workload)
    fc2_exec = _execute(checkout, fc2_workspace, workload, timeout_seconds); raw_fc2 = fc2_exec["accumulators"] if isinstance(fc2_exec["accumulators"], list) else []
    fc2_post = [v + b for v, b in zip(raw_fc2, constants["fc2_bias"])]
    logits = (np.asarray(fc2_post, dtype=np.float64) * float(ir["quantization"]["hidden_scale"]) * np.asarray(constants["fc2_scales"], dtype=np.float64)).tolist() if len(fc2_post) == 4 else []
    prediction = int(np.argmax(logits)) if logits else None
    failures: list[str] = []
    if any(run["execution"]["exit_status"] != 0 or run["execution"]["trap_assertion_status"] != "clear" for run in fc1_runs + [{"execution": fc2_exec}]): failures.append("simulator failure or trap/assertion")
    if fc1_post != trace["fc1_acc_int32"]: failures.append("fc1 accumulator mismatch")
    if hidden.get("hidden_int8") != trace["hidden_int8"]: failures.append("hidden INT8 mismatch")
    if fc2_post != trace["fc2_acc_int32"]: failures.append("fc2 accumulator mismatch")
    if prediction != trace["predicted_class"]: failures.append("prediction mismatch")
    result = {"format_version": RESULT_VERSION, "package_format": manifest["package_format_version"], "package_identity": identity, "sparrow_v_commit": audit_data["checkout"]["identity"], "sample_id": item["sample_id"], "graph_identifier": ir["model_name"], "layer_execution_order": ["fc1", "ReLU", "RequantizeInt8", "fc2"], "fc1": {"partitions": fc1_runs, "raw_rtl_accumulators": raw_fc1, "post_bias_accumulators": fc1_post, "expected_accumulators": trace["fc1_acc_int32"], "exact_match": fc1_post == trace["fc1_acc_int32"], "bias_policy": "zero_bias_rtl_then_host_reconstruction", "counters": _aggregate([x["execution"] for x in fc1_runs], 256)}, "intermediate": {**hidden, "expected_hidden_int8": trace["hidden_int8"], "exact_match": hidden.get("hidden_int8") == trace["hidden_int8"]}, "fc2": {"execution": fc2_exec, "raw_rtl_accumulators": raw_fc2, "raw_accumulator_provenance": "rtl_produced", "biases_int32": constants["fc2_bias"], "post_bias_accumulators": fc2_post, "post_bias_provenance": "host_reconstructed", "expected_accumulators": trace["fc2_acc_int32"], "exact_match": fc2_post == trace["fc2_acc_int32"], "counters": _aggregate([fc2_exec], 64)}, "final_reconstructed_logits": logits, "final_prediction": prediction, "expected_prediction": trace["predicted_class"], "counter_summary": {"total_dense_int8_multiplications": {"value": 320, "availability": "derived", "unit": "count"}}, "validation": {"valid": not failures, "failures": failures}, "validation_status": "passed" if not failures else "failed"}
    _json(output / "counter_report.json", result["counter_summary"]); _json(output / "multilayer_result.json", result)
    return result


def validate_result(result: dict[str, Any], package: Path) -> dict[str, Any]:
    _, _, inputs, traces = _package(package); item, trace = _sample(inputs, traces, result.get("sample_id")); failures: list[str] = []
    if result.get("format_version") != RESULT_VERSION: failures.append("result schema")
    if result.get("package_identity") != package_identity(package): failures.append("package identity")
    if result.get("sample_id") != item["sample_id"]: failures.append("sample ID")
    if result.get("fc1", {}).get("post_bias_accumulators") != trace["fc1_acc_int32"]: failures.append("fc1")
    if result.get("intermediate", {}).get("hidden_int8") != trace["hidden_int8"]: failures.append("hidden")
    if result.get("fc2", {}).get("post_bias_accumulators") != trace["fc2_acc_int32"]: failures.append("fc2")
    if result.get("final_prediction") != trace["predicted_class"]: failures.append("prediction")
    for field in ("raw_accumulator_provenance", "post_bias_provenance"):
        if field in result.get("fc2", {}) and result["fc2"][field] not in PROVENANCE: failures.append("fc2 provenance")
    return {"valid": not failures, "failures": failures, "expected_sample_id": item["sample_id"]}


def semantic_view(result: dict[str, Any]) -> dict[str, Any]:
    return {"package_identity": result["package_identity"], "sample_id": result["sample_id"], "fc1": result["fc1"]["post_bias_accumulators"], "hidden": result["intermediate"]["hidden_int8"], "fc2": result["fc2"]["post_bias_accumulators"], "prediction": result["final_prediction"], "fc1_counters": result["fc1"]["counters"], "fc2_counters": result["fc2"]["counters"], "validation": result["validation"]}
