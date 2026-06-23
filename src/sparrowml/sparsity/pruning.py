"""Inspectable deterministic 2:4 pruning and compression."""

from __future__ import annotations

import numpy as np

from .metadata import decode_metadata, encode_lanes


def _validate(weights: np.ndarray) -> np.ndarray:
    value = np.asarray(weights, dtype=np.int8)
    if value.ndim != 2 or value.shape[1] % 4:
        raise ValueError("2:4 weights must be a rank-two matrix with feature dimension divisible by four")
    return value


def prune_2of4(weights: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Keep largest absolute values, breaking ties by lower input-lane index."""
    dense = _validate(weights)
    sparse = np.zeros_like(dense)
    mask: list[dict[str, object]] = []
    for output in range(dense.shape[0]):
        for group in range(dense.shape[1] // 4):
            values = dense[output, group * 4 : group * 4 + 4]
            lanes = tuple(sorted(sorted(range(4), key=lambda lane: (-abs(int(values[lane])), lane))[:2]))
            sparse[output, group * 4 + lanes[0]] = values[lanes[0]]
            sparse[output, group * 4 + lanes[1]] = values[lanes[1]]
            mask.append({"output_channel": output, "group_index": group, "selected_lanes": list(lanes), "metadata": encode_lanes(lanes), "binary_mask": [int(lane in lanes) for lane in range(4)]})
    return sparse, mask


def compress_weights(sparse_weights: np.ndarray, mask: list[dict[str, object]] | None = None) -> tuple[np.ndarray, list[int]]:
    sparse = _validate(sparse_weights)
    if mask is None:
        _, mask = prune_2of4(sparse)
    if len(mask) != sparse.shape[0] * (sparse.shape[1] // 4):
        raise ValueError("2:4 mask group count does not match weights")
    compressed: list[list[int]] = []; metadata: list[int] = []
    for item in mask:
        output, group = int(item["output_channel"]), int(item["group_index"])
        lanes = tuple(int(lane) for lane in item["selected_lanes"])
        meta = encode_lanes(lanes)
        if int(item["metadata"]) != meta or item["binary_mask"] != [int(lane in lanes) for lane in range(4)]:
            raise ValueError("invalid 2:4 mask entry")
        group_values = sparse[output, group * 4 : group * 4 + 4]
        if any(int(group_values[lane]) != 0 for lane in range(4) if lane not in lanes):
            raise ValueError("sparse weights do not match 2:4 mask")
        compressed.append([int(group_values[lanes[0]]), int(group_values[lanes[1]])]); metadata.append(meta)
    return np.asarray(compressed, dtype=np.int8), metadata


def decompress_weights(compressed: np.ndarray, metadata: list[int], shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(compressed, dtype=np.int8)
    classes, features = shape
    if features % 4 or value.shape != (classes * (features // 4), 2) or len(metadata) != value.shape[0]:
        raise ValueError("compressed 2:4 tensors have inconsistent shapes")
    output = np.zeros(shape, dtype=np.int8)
    for index, meta in enumerate(metadata):
        lanes = decode_metadata(meta); channel, group = divmod(index, features // 4)
        output[channel, group * 4 + lanes[0]] = value[index, 0]
        output[channel, group * 4 + lanes[1]] = value[index, 1]
    return output
