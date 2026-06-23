"""Validation for the stable Phase 3 sparse JSON artifact."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from .metadata import decode_metadata
from .packing import unpack_metadata
from .pruning import decompress_weights


def validate_sparse_model(model: dict[str, object]) -> None:
    required = {"format_version", "model_name", "source_dense_int8_artifact", "feature_count", "class_count", "class_names", "grouping_axis", "group_size", "nonzero_count", "pruning_rule", "tie_breaking", "mask", "compressed_weight_int8", "metadata", "packed_metadata_hex", "weight_scales", "input_scale", "input_zero_point", "bias_int32", "tensor_shapes", "accumulator_type", "preprocessing_version", "fine_tuning", "storage_accounting"}
    if model.get("format_version") != "sparrowml_sparse_2of4_linear_v1" or required - model.keys():
        raise ValueError("unsupported or incomplete sparse model artifact")
    if Path(str(model["source_dense_int8_artifact"])).is_absolute() or model["grouping_axis"] != "input_feature" or int(model["group_size"]) != 4 or int(model["nonzero_count"]) != 2 or int(model["input_zero_point"]) != 0 or model["accumulator_type"] != "signed_int32":
        raise ValueError("sparse model contract fields are invalid")
    classes, features = int(model["class_count"]), int(model["feature_count"])
    metadata = [int(value) for value in model["metadata"]]; compressed = np.asarray(model["compressed_weight_int8"], dtype=np.int64)
    if len(model["class_names"]) != classes or features % 4 or compressed.shape != (classes * features // 4, 2) or np.any((compressed < -128) | (compressed > 127)) or len(metadata) != compressed.shape[0]:
        raise ValueError("sparse compressed tensor dimensions or values are invalid")
    for value in metadata: decode_metadata(value)
    payload = bytes.fromhex(str(model["packed_metadata_hex"]))
    if unpack_metadata(payload, len(metadata)) != metadata:
        raise ValueError("packed sparse metadata does not round-trip")
    mask = model["mask"]
    if len(mask) != len(metadata): raise ValueError("sparse mask group count is invalid")
    for item, value in zip(mask, metadata, strict=True):
        lanes = [int(lane) for lane in item["selected_lanes"]]
        if int(item["metadata"]) != value or lanes != list(decode_metadata(value)) or item["binary_mask"] != [int(index in lanes) for index in range(4)]:
            raise ValueError("sparse mask semantics are invalid")
    decompress_weights(compressed.astype(np.int8), metadata, (classes, features))
