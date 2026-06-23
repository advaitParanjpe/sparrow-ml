"""Training-split-only activation calibration."""

from __future__ import annotations

import numpy as np

from .affine import quantize_int8, symmetric_scale


def calibrate_symmetric_int8(features: np.ndarray, *, split: str = "train") -> dict[str, object]:
    if split != "train" or features.ndim != 2 or not features.size or not np.isfinite(features).all():
        raise ValueError("calibration requires a non-empty finite training feature matrix")
    scale = symmetric_scale(features)
    _, saturation = quantize_int8(features, scale)
    return {
        "format_version": "phase2_int8_calibration_v1",
        "scheme": "per_tensor_symmetric_int8",
        "rounding": "NumPy rint: round-to-nearest, ties-to-even",
        "clamping": "clamp to signed INT8 [-128, 127] after rounding",
        "calibration_split": split,
        "calibration_sample_count": int(features.shape[0]),
        "minimum_input_value": float(np.min(features)),
        "maximum_input_value": float(np.max(features)),
        "maximum_absolute_input_value": float(np.max(np.abs(features))),
        "input_scale": scale,
        "input_zero_point": 0,
        "input_saturation": {**saturation, "clipping_percentage": 100.0 * saturation["total_clipped_values"] / saturation["total_values"]},
    }
