"""Deterministic Phase 4 deployment-package export and reload validation."""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sparrowml.data.fixture import load_fixture
from sparrowml.data.preprocessing import Standardization
from sparrowml.quantization.affine import quantize_int8
from sparrowml.quantization.integer_reference import infer_int8
from sparrowml.sparsity.integer_reference import infer_sparse_compressed
from sparrowml.sparsity.packing import pack_metadata, unpack_metadata
from .ir import canonical_json, ir_hash, parse_ir, serialize_ir
from .lowering import lower_artifact

PACKAGE_VERSION = "sparrowml_sparrowv_package_v1"


def _json(path: Path, value: Any) -> None: path.write_text(canonical_json(value), encoding="utf-8")
def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def _align(value: int, alignment: int) -> int: return (value + alignment - 1) // alignment * alignment
def _i8(values: Any) -> bytes: return np.asarray(values, dtype=np.int8).tobytes(order="C")
def _i32(values: Any) -> bytes: return np.asarray(values, dtype="<i4").tobytes(order="C")
def _f32(values: Any) -> bytes: return np.asarray(values, dtype="<f4").tobytes(order="C")


def _layout(ir: dict[str, Any], alignment: int, capacity: int) -> dict[str, Any]:
    op = ir["operators"][0]; mode = ir["execution_mode"]
    payloads = {"weights": _i8(ir["constants"][op["weights"]]), "bias": _i32(ir["constants"][op["bias"]]), "scales": _f32(ir["constants"][op["weight_scales"]])}
    if mode == "sparse_2of4_int8": payloads["metadata"] = pack_metadata(ir["constants"][op["metadata"]])
    names = ["input", "weights"] + (["metadata"] if mode == "sparse_2of4_int8" else []) + ["bias", "scales", "output"]
    lengths = {"input": 16, "output": 16, **{name: len(payload) for name, payload in payloads.items()}}
    types = {"input": "int8", "weights": "int8", "metadata": "uint8", "bias": "int32", "scales": "float32", "output": "int32"}
    shapes = {"input": [16], "weights": next(t["shape"] for t in ir["tensors"] if t["name"] == op["weights"]), "metadata": [len(ir["constants"].get("sparse_metadata", []))], "bias": [4], "scales": [4], "output": [4]}
    offset = 0; regions = []
    for name in names:
        offset = _align(offset, alignment); regions.append({"name": name, "byte_offset": offset, "byte_length": lengths[name], "alignment": alignment, "element_type": types[name], "logical_shape": shapes[name], "source_tensor": {"input": ir["input_tensor"], "output": ir["output_tensor"], "weights": op["weights"], "metadata": op.get("metadata"), "bias": op["bias"], "scales": op["weight_scales"]}[name]}); offset += lengths[name]
    total = _align(offset, alignment)
    if total > capacity: raise ValueError(f"memory layout total {total} exceeds target scratchpad {capacity}")
    return {"format_version": "sparrowml_memory_map_v1", "byte_order": "little-endian", "integer_encoding": "two's-complement", "float_encoding": "IEEE-754 float32", "padding_byte": 0, "regions": regions, "total_memory_bytes": total, "target_capacity_bytes": capacity}


def _sample(root: Path, ir: dict[str, Any], sample_ids: list[str]) -> tuple[dict[str, Any], bytes, list[dict[str, Any]]]:
    pre = json.loads((root / "artifacts/phase1_fp32/preprocessing.json").read_text())
    examples = {item.sample_id: item for item in load_fixture(root / "data/processed/sensor_fixture")}
    selected = [examples[item] for item in sample_ids]
    if len(selected) != len(sample_ids): raise ValueError("configured sample ID is missing")
    standardized = Standardization(np.asarray(pre["mean"]), np.asarray(pre["std"]), pre["version"]).transform(selected)
    values, _ = quantize_int8(standardized, float(ir["quantization"]["input_scale"]))
    records = [{"sample_id": item.sample_id, "expected_label": item.class_id, "class_name": item.class_name, "input_scale": ir["quantization"]["input_scale"], "input_zero_point": 0, "quantized_features_int8": values[index].astype(int).tolist()} for index, item in enumerate(selected)]
    return {"format_version": "sparrowml_input_data_v1", "samples": records}, _i8(values), records


def _outputs(ir: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    op = ir["operators"][0]; c = ir["constants"]; q = ir["quantization"]; outputs = []
    for record in records:
        inp = np.asarray(record["quantized_features_int8"], dtype=np.int8)
        if ir["execution_mode"] == "dense_int8": result = infer_int8(inp, np.asarray(c[op["weights"]], dtype=np.int8), np.asarray(c[op["bias"]], dtype=np.int32), q["input_scale"], np.asarray(q["weight_scales"]), tuple(ir["class_names"]))
        else: result = infer_sparse_compressed(inp, np.asarray(c[op["weights"]], dtype=np.int8), c[op["metadata"]], np.asarray(c[op["bias"]], dtype=np.int32), q["input_scale"], np.asarray(q["weight_scales"]), tuple(ir["class_names"]), 16)
        outputs.append({"sample_id": record["sample_id"], "expected_int32_accumulators": result.accumulators.astype(int).tolist(), "reconstructed_logits": result.logits.tolist(), "predicted_class_id": result.predicted_class, "predicted_class_name": ir["class_names"][result.predicted_class], "expected_label": record["expected_label"], "correct": result.predicted_class == record["expected_label"], "execution_mode": ir["execution_mode"]})
    return {"format_version": "sparrowml_expected_output_v1", "samples": outputs}


def export_package(root: Path, mode: str, output_root: Path | None = None, config_path: Path | None = None) -> Path:
    config = yaml.safe_load((config_path or root / "configs/experiments/sparrow_v_export.yaml").read_text())
    if mode not in {"dense", "sparse"}: raise ValueError("mode must be dense or sparse")
    source = root / config["source_artifacts"][mode]; target = yaml.safe_load((root / config["target_config"]).read_text())
    ir = lower_artifact(source, root); package = (output_root or root / config["output_directory"]) / mode
    if package.exists(): shutil.rmtree(package)
    package.mkdir(parents=True)
    alignment = int(config["alignment"]); memory = _layout(ir, alignment, int(target["scratchpad_bytes"]))
    _json(package / "model_ir.json", json.loads(serialize_ir(ir)))
    _json(package / "memory_map.json", memory)
    input_json, input_bytes, records = _sample(root, ir, list(config["selected_input_sample_ids"]))
    _json(package / "input_data.json", input_json); (package / "input_data.bin").write_bytes(input_bytes)
    image = bytearray(memory["total_memory_bytes"]); op = ir["operators"][0]
    payload = {"weights": _i8(ir["constants"][op["weights"]]), "bias": _i32(ir["constants"][op["bias"]]), "scales": _f32(ir["constants"][op["weight_scales"]])}
    if mode == "sparse": payload["metadata"] = pack_metadata(ir["constants"][op["metadata"]])
    for region in memory["regions"]:
        if region["name"] in payload: image[region["byte_offset"]:region["byte_offset"] + region["byte_length"]] = payload[region["name"]]
    (package / "model_data.bin").write_bytes(bytes(image))
    expected = _outputs(ir, records); _json(package / "expected_output.json", expected)
    commands = ["load input", f"load {('compressed weights and metadata' if mode == 'sparse' else 'dense weights')}", "load bias/scales", f"execute {('sparse 2:4' if mode == 'sparse' else 'dense')} dot products", "store outputs"]
    _json(package / "program.json", {"format_version": "sparrowml_symbolic_program_v1", "runtime_interface_version": "sparrowv_runtime_adapter_v1", "commands": commands})
    files = ["model_ir.json", "memory_map.json", "model_data.bin", "input_data.bin", "expected_output.json"]
    manifest = {"package_format_version": PACKAGE_VERSION, "target_name": target["name"], "target_architecture_version": target["architecture_version"], "model_name": ir["model_name"], "execution_mode": ir["execution_mode"], "source_artifact_hashes": {ir["source_artifact_identity"]: _sha(source)}, "model_ir_hash": ir_hash(ir), "feature_count": 16, "output_count": 4, "class_names": ir["class_names"], "preprocessing_version": ir["preprocessing_version"], "input_quantization": {"scale": ir["quantization"]["input_scale"], "zero_point": 0}, "weight_quantization": {"scales": ir["quantization"]["weight_scales"], "zero_points": [0]*4}, "accumulator_type": "int32", "memory_region_summary": memory["regions"], "binary_filenames": {"model": "model_data.bin", "input": "input_data.bin"}, "expected_output_filename": "expected_output.json", "export_validation_status": "pending", "required_future_runtime_interface_version": "sparrowv_runtime_adapter_v1", "file_hashes": {name: _sha(package / name) for name in files}}
    _json(package / "manifest.json", manifest)
    (package / "README.md").write_text(f"# SparrowML {mode} Sparrow-V package\n\nLogical little-endian deployment data only. Sparrow-V is not executed by this package.\n", encoding="utf-8")
    valid = validate_package(package)
    manifest["export_validation_status"] = "passed"; manifest["file_hashes"] = {name: _sha(package / name) for name in files}; _json(package / "manifest.json", manifest)
    _json(package / "export_report.json", {"format_version": "sparrowml_export_report_v1", "validation": valid, "file_hashes": {name: _sha(package / name) for name in files + ["manifest.json"]}})
    return package


def validate_package(package: Path) -> dict[str, Any]:
    manifest = json.loads((package / "manifest.json").read_text()); ir = parse_ir((package / "model_ir.json").read_text()); memory = json.loads((package / "memory_map.json").read_text()); expected = json.loads((package / "expected_output.json").read_text()); inputs = json.loads((package / "input_data.json").read_text())
    if manifest["model_ir_hash"] != ir_hash(ir): raise ValueError("manifest.model_ir_hash mismatch")
    data = (package / "model_data.bin").read_bytes()
    if len(data) != memory["total_memory_bytes"]: raise ValueError("model_data.bin length mismatch")
    previous = 0
    for region in memory["regions"]:
        if region["byte_offset"] < previous or region["byte_offset"] % region["alignment"]: raise ValueError("memory_map regions overlap or violate alignment")
        previous = region["byte_offset"] + region["byte_length"]
    op = ir["operators"][0]; region = {item["name"]: item for item in memory["regions"]}
    def read(name: str) -> bytes:
        item = region[name]; return data[item["byte_offset"]:item["byte_offset"] + item["byte_length"]]
    if read("weights") != _i8(ir["constants"][op["weights"]]) or read("bias") != _i32(ir["constants"][op["bias"]]) or read("scales") != _f32(ir["constants"][op["weight_scales"]]): raise ValueError("decoded model tensors differ from IR")
    if ir["execution_mode"] == "sparse_2of4_int8" and unpack_metadata(read("metadata"), 16) != ir["constants"][op["metadata"]]: raise ValueError("decoded sparse metadata differs from IR")
    raw_inputs = np.frombuffer((package / "input_data.bin").read_bytes(), dtype=np.int8).reshape(len(inputs["samples"]), 16)
    if [row.astype(int).tolist() for row in raw_inputs] != [row["quantized_features_int8"] for row in inputs["samples"]]: raise ValueError("input_data.bin differs from input_data.json")
    if _outputs(ir, inputs["samples"])["samples"] != expected["samples"]: raise ValueError("expected outputs do not reproduce reference inference")
    for name, digest in manifest.get("file_hashes", {}).items():
        if _sha(package / name) != digest: raise ValueError(f"hash mismatch for {name}")
    return {"passed": True, "samples": len(inputs["samples"]), "reference_equivalence": "exact"}


def run_baseline(root: Path) -> dict[str, Any]:
    output = root / "artifacts/phase4_export"; dense = export_package(root, "dense"); sparse = export_package(root, "sparse")
    summary = {"format_version": "sparrowml_phase4_export_summary_v1", "dense": validate_package(dense), "sparse": validate_package(sparse)}
    _json(output / "export_summary.json", summary); _json(output / "determinism.json", {mode: {name: _sha(path / name) for name in ["model_ir.json", "manifest.json", "memory_map.json", "model_data.bin", "input_data.bin", "expected_output.json"]} for mode, path in {"dense": dense, "sparse": sparse}.items()}); (output / "summary.md").write_text("# Phase 4 export\n\nDense and sparse packages validate against their exact integer references. Sparrow-V was not executed.\n", encoding="utf-8")
    return summary
