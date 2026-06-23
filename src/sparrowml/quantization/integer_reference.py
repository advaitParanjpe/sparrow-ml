"""Standalone exact integer affine inference; no framework linear operator is used."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .affine import INT32_MAX, INT32_MIN


@dataclass(frozen=True)
class IntegerInferenceResult:
    accumulators: np.ndarray
    logits: np.ndarray
    predicted_class: int
    accumulator_fits_int32: bool


def infer_int8(input_int8: np.ndarray, weight_int8: np.ndarray, bias_int32: np.ndarray, input_scale: float, weight_scales: np.ndarray, class_names: tuple[str, ...]) -> IntegerInferenceResult:
    inputs = np.asarray(input_int8, dtype=np.int8)
    weights = np.asarray(weight_int8, dtype=np.int8)
    biases = np.asarray(bias_int32, dtype=np.int32)
    scales = np.asarray(weight_scales, dtype=np.float64)
    if inputs.ndim != 1 or weights.ndim != 2 or weights.shape[1] != inputs.size or weights.shape[0] != biases.size or scales.shape != (weights.shape[0],) or len(class_names) != weights.shape[0]:
        raise ValueError("integer inference tensor dimensions are inconsistent")
    if input_scale <= 0 or np.any(scales <= 0):
        raise ValueError("integer inference scales must be positive")
    # Explicit int64 host arithmetic makes each INT8 product and accumulated sum inspectable.
    accumulators = biases.astype(np.int64) + np.sum(inputs.astype(np.int64)[None, :] * weights.astype(np.int64), axis=1, dtype=np.int64)
    fits = bool(np.all((accumulators >= INT32_MIN) & (accumulators <= INT32_MAX)))
    if not fits:
        raise ValueError("integer accumulator exceeds signed INT32 range")
    logits = accumulators.astype(np.float64) * input_scale * scales
    if not np.isfinite(logits).all():
        raise ValueError("reconstructed logits are not finite")
    return IntegerInferenceResult(accumulators.astype(np.int32), logits, int(np.argmax(logits)), fits)
