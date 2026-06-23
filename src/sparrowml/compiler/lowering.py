"""Deterministic lowering of existing Phase 2 and Phase 3 artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sparrowml.quantization.artifacts import validate_quantized_model
from sparrowml.sparsity.artifacts import validate_sparse_model
from .ir import IR_FORMAT_VERSION, relative_identity
from .validation import validate_ir


def _tensor(name: str, shape: list[int], element: str, role: str, order: str, quantization: dict[str, Any] | None = None) -> dict[str, Any]:
    size = {"int8": 1, "uint8": 1, "int32": 4, "float32": 4}[element]
    for dim in shape: size *= dim
    return {"name": name, "shape": shape, "element_type": element, "role": role, "logical_order": order, "storage_layout": "contiguous_row_major", "byte_size": size, "quantization": quantization}


def _source_identity(path: Path, root: Path) -> str:
    identity = path.resolve().relative_to(root.resolve()).as_posix()
    relative_identity(identity)
    return identity


def lower_artifact(path: Path, root: Path) -> dict[str, Any]:
    model = json.loads(path.read_text(encoding="utf-8"))
    mode = "dense_int8" if model.get("format_version") == "sparrowml_int8_linear_v1" else "sparse_2of4_int8"
    if mode == "dense_int8": validate_quantized_model(model)
    else: validate_sparse_model(model)
    scales = [float(v) for v in model["weight_scales"]]
    quantization = {"input_scale": float(model["input_scale"]), "input_zero_point": int(model["input_zero_point"]), "weight_scales": scales, "weight_zero_points": [0] * 4, "accumulator_type": "int32", "output_reconstruction_scales": [float(model["input_scale"]) * value for value in scales], "rounding_policy": "round-to-nearest ties-to-even", "clamping_policy": "[-128,127]"}
    tensors = [_tensor("input_int8", [16], "int8", "input", "feature_00..feature_15", {"scale": quantization["input_scale"], "zero_point": 0}), _tensor("output_accumulators", [4], "int32", "output", "class_names order"), _tensor("bias_int32", [4], "int32", "constant", "output_channel"), _tensor("weight_scales", [4], "float32", "constant", "output_channel")]
    constants: dict[str, Any] = {"bias_int32": [int(v) for v in model["bias_int32"]], "weight_scales": scales}
    if mode == "dense_int8":
        tensors.append(_tensor("weight_int8", [4, 16], "int8", "constant", "output_channel,input_feature", {"zero_point": 0}))
        constants["weight_int8"] = model["weight_int8"]
        op = {"op_type": "DenseLinearInt8", "input": "input_int8", "output": "output_accumulators", "weights": "weight_int8", "bias": "bias_int32", "weight_scales": "weight_scales", "feature_count": 16, "output_count": 4, "accumulation_type": "int32"}
    else:
        tensors += [_tensor("compressed_weight_int8", [16, 2], "int8", "constant", "output_channel,input_group,selected_lane", {"zero_point": 0}), _tensor("sparse_metadata", [16], "uint8", "constant", "output_channel,input_group")]
        constants.update({"compressed_weight_int8": model["compressed_weight_int8"], "sparse_metadata": [int(v) for v in model["metadata"]], "packed_metadata_hex": model["packed_metadata_hex"]})
        op = {"op_type": "SparseLinear2of4Int8", "input": "input_int8", "output": "output_accumulators", "weights": "compressed_weight_int8", "bias": "bias_int32", "weight_scales": "weight_scales", "metadata": "sparse_metadata", "feature_count": 16, "output_count": 4, "accumulation_type": "int32", "group_size": 4, "nonzero_count": 2, "grouping_axis": "input_feature", "group_traversal_order": "output_channel-major then ascending input group", "metadata_encoding_version": "sparrowml_2of4_meta3_v1", "compressed_weight_order": "lower selected lane then higher selected lane", "packed_metadata_byte_order": "LSB-first bit stream"}
    ir = {"format_version": IR_FORMAT_VERSION, "model_name": str(model["model_name"]), "execution_mode": mode, "input_tensor": "input_int8", "output_tensor": "output_accumulators", "tensors": tensors, "operators": [op], "constants": constants, "class_names": list(model["class_names"]), "preprocessing_version": str(model["preprocessing_version"]), "source_artifact_identity": _source_identity(path, root), "quantization": quantization}
    validate_ir(ir)
    return ir
