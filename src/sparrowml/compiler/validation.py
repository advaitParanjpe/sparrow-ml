"""Strict validation for the intentionally narrow SparrowML IR."""
from __future__ import annotations

import math
from pathlib import PurePosixPath
from typing import Any

from sparrowml.sparsity.metadata import decode_metadata
from .ir import ELEMENT_BYTES, IR_FORMAT_VERSION, relative_identity


def _fail(field: str, message: str) -> None:
    raise ValueError(f"{field}: {message}")


def validate_ir(ir: dict[str, Any]) -> None:
    required = {"format_version", "model_name", "execution_mode", "input_tensor", "output_tensor", "tensors", "operators", "constants", "class_names", "preprocessing_version", "source_artifact_identity", "quantization"}
    if set(ir) != required:
        _fail("IR", "unexpected or missing fields")
    if ir["format_version"] != IR_FORMAT_VERSION: _fail("format_version", "unsupported version")
    if ir["execution_mode"] not in {"dense_int8", "sparse_2of4_int8"}: _fail("execution_mode", "unsupported mode")
    if len(ir["class_names"]) != 4: _fail("class_names", "must contain four classes")
    if not ir["preprocessing_version"]: _fail("preprocessing_version", "is required")
    relative_identity(str(ir["source_artifact_identity"]))
    tensors = ir["tensors"]
    if not isinstance(tensors, list) or not tensors: _fail("tensors", "must be a non-empty list")
    names: set[str] = set()
    for tensor in tensors:
        needed = {"name", "shape", "element_type", "role", "logical_order", "storage_layout", "byte_size", "quantization"}
        if set(tensor) != needed: _fail("tensor", "unexpected or missing fields")
        name = str(tensor["name"])
        if not name or name in names: _fail("tensors.name", "must be unique and non-empty")
        names.add(name)
        shape = tensor["shape"]
        if not isinstance(shape, list) or not shape or any(not isinstance(v, int) or v <= 0 for v in shape): _fail(f"tensors.{name}.shape", "must be static positive dimensions")
        element = tensor["element_type"]
        if element not in ELEMENT_BYTES: _fail(f"tensors.{name}.element_type", "unsupported type")
        expected = ELEMENT_BYTES[element]
        for dim in shape: expected *= dim
        if tensor["byte_size"] != expected: _fail(f"tensors.{name}.byte_size", "does not match shape and element type")
    by_name = {item["name"]: item for item in tensors}
    if ir["input_tensor"] not in by_name or ir["output_tensor"] not in by_name: _fail("input_tensor/output_tensor", "must name tensors")
    if by_name[ir["input_tensor"]]["shape"] != [16] or by_name[ir["input_tensor"]]["element_type"] != "int8": _fail("input_tensor", "must be int8[16]")
    if by_name[ir["output_tensor"]]["shape"] != [4] or by_name[ir["output_tensor"]]["element_type"] != "int32": _fail("output_tensor", "must be int32[4]")
    q = ir["quantization"]
    if set(q) != {"input_scale", "input_zero_point", "weight_scales", "weight_zero_points", "accumulator_type", "output_reconstruction_scales", "rounding_policy", "clamping_policy"}: _fail("quantization", "unexpected or missing fields")
    if not math.isfinite(float(q["input_scale"])) or float(q["input_scale"]) <= 0: _fail("quantization.input_scale", "must be positive finite")
    if q["input_zero_point"] != 0: _fail("quantization.input_zero_point", "must be zero")
    if len(q["weight_scales"]) != 4 or any(not math.isfinite(float(v)) or float(v) <= 0 for v in q["weight_scales"]): _fail("quantization.weight_scales", "must be four positive finite values")
    if q["weight_zero_points"] != [0] * 4 or q["accumulator_type"] != "int32": _fail("quantization", "only symmetric INT8 and INT32 accumulation are supported")
    ops = ir["operators"]
    if len(ops) != 1: _fail("operators", "exactly one operator is supported")
    op = ops[0]; allowed = {"op_type", "input", "output", "weights", "bias", "weight_scales", "feature_count", "output_count", "accumulation_type"}
    if op.get("op_type") == "SparseLinear2of4Int8": allowed |= {"metadata", "group_size", "nonzero_count", "grouping_axis", "group_traversal_order", "metadata_encoding_version", "compressed_weight_order", "packed_metadata_byte_order"}
    if set(op) != allowed or op.get("op_type") not in {"DenseLinearInt8", "SparseLinear2of4Int8"}: _fail("operators[0]", "unsupported operator")
    if (ir["execution_mode"] == "dense_int8") != (op["op_type"] == "DenseLinearInt8"): _fail("execution_mode", "does not match operator")
    if op["feature_count"] != 16 or op["output_count"] != 4 or op["accumulation_type"] != "int32": _fail("operators[0]", "only 16-to-4 INT32 accumulation is supported")
    if any(op[key] not in by_name for key in ("input", "output", "weights", "bias", "weight_scales")): _fail("operators[0]", "references unknown tensor")
    if op["op_type"] == "SparseLinear2of4Int8":
        if (op["group_size"], op["nonzero_count"], op["grouping_axis"]) != (4, 2, "input_feature"): _fail("operators[0]", "requires input-axis 2:4 groups")
        if op["metadata"] not in by_name: _fail("operators[0].metadata", "references unknown tensor")
        values = ir["constants"].get(op["metadata"])
        if not isinstance(values, list) or len(values) != 16: _fail("constants.metadata", "requires 16 metadata values")
        try:
            for value in values: decode_metadata(int(value))
        except ValueError as exc: _fail("constants.metadata", str(exc))
    if not isinstance(ir["constants"], dict): _fail("constants", "must be an object")
