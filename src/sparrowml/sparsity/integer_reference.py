"""Independent dense-form and compressed 2:4 integer reference inference."""

from __future__ import annotations

import numpy as np

from sparrowml.quantization.integer_reference import IntegerInferenceResult
from .metadata import decode_metadata


def infer_sparse_dense(inputs: np.ndarray, weights: np.ndarray, biases: np.ndarray, input_scale: float, scales: np.ndarray, class_names: tuple[str, ...]) -> IntegerInferenceResult:
    from sparrowml.quantization.integer_reference import infer_int8
    return infer_int8(inputs, weights, biases, input_scale, scales, class_names)


def infer_sparse_compressed(inputs: np.ndarray, compressed: np.ndarray, metadata: list[int], biases: np.ndarray, input_scale: float, scales: np.ndarray, class_names: tuple[str, ...], feature_count: int) -> IntegerInferenceResult:
    x = np.asarray(inputs, dtype=np.int8); packed = np.asarray(compressed, dtype=np.int8); bias = np.asarray(biases, dtype=np.int32); scale = np.asarray(scales, dtype=np.float64)
    groups = feature_count // 4
    if x.shape != (feature_count,) or packed.shape != (len(class_names) * groups, 2) or len(metadata) != packed.shape[0] or bias.shape != (len(class_names),) or scale.shape != bias.shape:
        raise ValueError("compressed sparse integer inference tensor dimensions are inconsistent")
    accumulators = bias.astype(np.int64)
    for index, meta in enumerate(metadata):
        lane0, lane1 = decode_metadata(meta); channel, group = divmod(index, groups); base = group * 4
        accumulators[channel] += int(x[base + lane0]) * int(packed[index, 0]) + int(x[base + lane1]) * int(packed[index, 1])
    if np.any(accumulators < -(2**31)) or np.any(accumulators > 2**31 - 1):
        raise ValueError("integer accumulator exceeds signed INT32 range")
    logits = accumulators.astype(np.float64) * input_scale * scale
    return IntegerInferenceResult(accumulators.astype(np.int32), logits, int(np.argmax(logits)), True)
