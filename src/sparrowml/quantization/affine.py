"""Explicit signed INT8 affine quantization with NumPy round-to-nearest-even."""

from __future__ import annotations

import numpy as np

INT8_MIN, INT8_MAX = -128, 127
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1


def symmetric_scale(values: np.ndarray) -> float:
    """Return max(abs(values))/127, using 1.0 for an all-zero tensor."""
    maximum = float(np.max(np.abs(values))) if values.size else 0.0
    return maximum / INT8_MAX if maximum else 1.0


def quantize_int8(values: np.ndarray, scale: float, zero_point: int = 0) -> tuple[np.ndarray, dict[str, int]]:
    if not np.isfinite(values).all() or not np.isfinite(scale) or scale <= 0 or zero_point != 0:
        raise ValueError("symmetric INT8 quantization requires finite values, positive scale, and zero point 0")
    rounded = np.rint(np.asarray(values, dtype=np.float64) / scale).astype(np.int64) + zero_point
    clipped = np.clip(rounded, INT8_MIN, INT8_MAX)
    return clipped.astype(np.int8), {
        "total_values": int(clipped.size),
        "values_at_negative_128": int((clipped == INT8_MIN).sum()),
        "values_at_127": int((clipped == INT8_MAX).sum()),
        "total_clipped_values": int((rounded != clipped).sum()),
    }


def dequantize_int8(values: np.ndarray, scale: float, zero_point: int = 0) -> np.ndarray:
    return (np.asarray(values, dtype=np.int8).astype(np.float64) - zero_point) * scale


def quantize_int32(values: np.ndarray, scales: np.ndarray) -> np.ndarray:
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("INT32 bias quantization requires finite values and positive scales")
    quantized = np.rint(np.asarray(values, dtype=np.float64) / np.asarray(scales, dtype=np.float64)).astype(np.int64)
    if np.any(quantized < INT32_MIN) or np.any(quantized > INT32_MAX):
        raise ValueError("quantized bias exceeds signed INT32 range")
    return quantized.astype(np.int32)
