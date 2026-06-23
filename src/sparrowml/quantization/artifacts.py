"""Stable JSON serialization and validation for the small Phase 2 model."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .affine import INT8_MAX, INT8_MIN, INT32_MAX, INT32_MIN


def validate_quantized_model(model: dict[str, object]) -> None:
    required = {"format_version", "model_name", "source_fp32_checkpoint", "feature_count", "class_count", "class_names", "quantization", "input_scale", "input_zero_point", "weight_scales", "weight_zero_points", "weight_int8", "bias_int32", "tensor_shapes", "lane_order", "preprocessing_version", "calibration", "accumulator_type", "creation_configuration"}
    if model.get("format_version") != "sparrowml_int8_linear_v1" or required - model.keys():
        raise ValueError("unsupported or incomplete quantized model artifact")
    features, classes = int(model["feature_count"]), int(model["class_count"])
    weights = np.asarray(model["weight_int8"], dtype=np.int64)
    bias = np.asarray(model["bias_int32"], dtype=np.int64)
    scales = np.asarray(model["weight_scales"], dtype=np.float64)
    if features <= 0 or classes <= 0 or len(model["class_names"]) != classes or weights.shape != (classes, features) or bias.shape != (classes,) or scales.shape != (classes,):
        raise ValueError("quantized model tensor dimensions are invalid")
    if len(model["weight_zero_points"]) != classes or np.any(np.asarray(model["weight_zero_points"], dtype=np.int64) != 0) or np.any((weights < INT8_MIN) | (weights > INT8_MAX)) or np.any((bias < INT32_MIN) | (bias > INT32_MAX)) or float(model["input_scale"]) <= 0 or np.any(scales <= 0) or int(model["input_zero_point"]) != 0:
        raise ValueError("quantized model values are outside supported ranges")
    if Path(str(model["source_fp32_checkpoint"])).is_absolute():
        raise ValueError("artifact checkpoint identity must be repository-relative")


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
