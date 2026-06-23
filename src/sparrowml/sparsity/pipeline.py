"""Bounded end-to-end Phase 3 2:4 pruning, fine-tuning, packing, and evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import yaml

from sparrowml.data.fixture import CLASS_NAMES, load_fixture
from sparrowml.data.preprocessing import Standardization
from sparrowml.models.linear_classifier import LinearSensorClassifier
from sparrowml.quantization.affine import quantize_int8, quantize_int32, symmetric_scale
from sparrowml.quantization.artifacts import write_json
from sparrowml.quantization.integer_reference import infer_int8
from sparrowml.quantization.pipeline import run_baseline as run_int8_baseline
from sparrowml.training.seeds import set_deterministic_seeds
from sparrowml.training.trainer import split_examples
from .artifacts import validate_sparse_model
from .integer_reference import infer_sparse_compressed, infer_sparse_dense
from .packing import pack_metadata
from .pruning import compress_weights, decompress_weights, prune_2of4


def _load(config_path: str | Path | None) -> tuple[dict[str, object], Path, dict[str, list], Standardization]:
    root = Path(__file__).resolve().parents[3]; path = root / (config_path or "configs/experiments/sparse_2of4_baseline.yaml")
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8")); source, structure = config["source"], config["structure"]
        if structure != {**structure, "group_size": 4, "nonzero_count": 2, "grouping_axis": "input_feature", "pruning_rule": "largest_absolute_magnitude", "tie_breaking": "lower_lane_index"}:
            raise ValueError("only the documented deterministic 2:4 structure is supported")
        preprocessing = json.loads((root / source["preprocessing_path"]).read_text(encoding="utf-8")); splits = split_examples(load_fixture(root / source["fixture_directory"]))
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Phase 3 configuration {path}: {exc}") from exc
    return config, root, splits, Standardization(np.asarray(preprocessing["mean"]), np.asarray(preprocessing["std"]), preprocessing["version"])


def _dense_artifact(config: dict[str, object], root: Path) -> dict[str, object]:
    path = root / config["source"]["dense_int8_artifact"]
    if not path.exists(): run_int8_baseline()
    if not path.exists(): raise ValueError(f"missing Phase 2 artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _mask_tensor(mask: list[dict[str, object]], shape: tuple[int, int]) -> np.ndarray:
    value = np.zeros(shape, dtype=np.float32)
    for item in mask:
        for lane in item["selected_lanes"]: value[int(item["output_channel"]), int(item["group_index"]) * 4 + int(lane)] = 1.0
    return value


def prune(config_path: str | Path | None = None) -> dict[str, object]:
    config, root, _, _ = _load(config_path); dense = _dense_artifact(config, root)
    sparse, mask = prune_2of4(np.asarray(dense["weight_int8"], dtype=np.int8)); compressed, metadata = compress_weights(sparse, mask)
    output = root / config["output_directory"]; output.mkdir(parents=True, exist_ok=True)
    write_json(output / "pruning_mask.json", {"tensor_shape": list(sparse.shape), "grouping_axis": "input_feature", "group_traversal": "output_channel-major then ascending input group", "retained_weight_count": int(np.count_nonzero(_mask_tensor(mask, sparse.shape))), "pruned_weight_count": int(sparse.size - np.count_nonzero(_mask_tensor(mask, sparse.shape))), "mask": mask})
    write_json(output / "pruned_dense_int8.json", {"weight_int8": sparse.astype(int).tolist(), "metadata": metadata, "compressed_weight_int8": compressed.astype(int).tolist()})
    return {"sparse": sparse, "mask": mask, "compressed": compressed, "metadata": metadata, "dense": dense}


def finetune(config_path: str | Path | None = None) -> dict[str, object]:
    config, root, splits, standardization = _load(config_path); pruned = prune(config_path); source = config["source"]
    checkpoint = root / source["fp32_checkpoint"]
    if not checkpoint.exists(): raise ValueError(f"missing Phase 1 checkpoint: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True); model = LinearSensorClassifier(); model.load_state_dict(payload["state_dict"])
    mask = torch.from_numpy(_mask_tensor(pruned["mask"], (4, 16))); set_deterministic_seeds(int(config["seed"])); model.linear.weight.data.mul_(mask)
    matrices = {name: standardization.transform(items) for name, items in splits.items()}; labels = {name: torch.tensor([item.class_id for item in items], dtype=torch.long) for name, items in splits.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["fine_tuning"]["learning_rate"])); best_loss = float("inf"); best: dict[str, torch.Tensor] | None = None
    for _ in range(int(config["fine_tuning"]["epochs"])):
        model.train(); optimizer.zero_grad(); loss = torch.nn.functional.cross_entropy(model(torch.from_numpy(matrices["train"])), labels["train"]); loss.backward(); optimizer.step(); model.linear.weight.data.mul_(mask)
        model.eval()
        with torch.no_grad(): validation_loss = float(torch.nn.functional.cross_entropy(model(torch.from_numpy(matrices["validation"])), labels["validation"]))
        if validation_loss < best_loss: best_loss, best = validation_loss, {name: value.detach().clone() for name, value in model.state_dict().items()}
    assert best is not None; model.load_state_dict(best); model.linear.weight.data.mul_(mask)
    output = root / config["output_directory"]; torch.save({"state_dict": model.state_dict(), "mask": mask.numpy(), "best_validation_loss": best_loss, "seed": config["seed"]}, output / "sparse_fp32_checkpoint.pt")
    return {"model": model, "mask": pruned["mask"], "pruned": pruned, "best_validation_loss": best_loss}


def _quantize_sparse(model: LinearSensorClassifier, input_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weight = model.linear.weight.detach().cpu().numpy(); bias = model.linear.bias.detach().cpu().numpy(); scales = np.asarray([symmetric_scale(row) for row in weight], dtype=np.float64)
    return np.stack([quantize_int8(row, float(scale))[0] for row, scale in zip(weight, scales, strict=True)]), quantize_int32(bias, input_scale * scales), scales


def pack(config_path: str | Path | None = None) -> dict[str, object]:
    config, root, _, standardization = _load(config_path); tuned = finetune(config_path); dense = tuned["pruned"]["dense"]; input_scale = float(dense["input_scale"])
    weights, biases, scales = _quantize_sparse(tuned["model"], input_scale); weights *= _mask_tensor(tuned["mask"], weights.shape).astype(np.int8); compressed, metadata = compress_weights(weights, tuned["mask"]); restored = decompress_weights(compressed, metadata, weights.shape)
    if not np.array_equal(weights, restored): raise ValueError("compressed sparse weights do not exactly decompress")
    packed = pack_metadata(metadata); output = root / config["output_directory"]; (output / "packed_metadata.bin").write_bytes(packed)
    storage = {"dense_weight_bytes": int(weights.size), "compressed_weight_bytes": int(compressed.size), "metadata_bytes": len(packed), "sparse_weight_representation_bytes": int(compressed.size + len(packed)), "weight_storage_reduction_percent": 100.0 * (weights.size - compressed.size - len(packed)) / weights.size, "bias_bytes_separate": int(biases.size * 4), "weight_scale_bytes_separate": int(scales.size * 8)}
    artifact: dict[str, object] = {"format_version": "sparrowml_sparse_2of4_linear_v1", "model_name": "Linear(16, 4)", "source_dense_int8_artifact": str(config["source"]["dense_int8_artifact"]), "feature_count": 16, "class_count": 4, "class_names": list(CLASS_NAMES), "grouping_axis": "input_feature", "group_size": 4, "nonzero_count": 2, "pruning_rule": "largest absolute magnitude", "tie_breaking": "lower lane index", "group_traversal": "output_channel-major then ascending input group", "compressed_weight_order": "lower selected lane then higher selected lane", "mask": tuned["mask"], "compressed_weight_int8": compressed.astype(int).tolist(), "metadata": metadata, "packed_metadata_hex": packed.hex(), "metadata_packing": "three-bit values LSB-first in traversal order; final padding bits are zero", "weight_scales": scales.tolist(), "input_scale": input_scale, "input_zero_point": 0, "bias_int32": biases.astype(int).tolist(), "tensor_shapes": {"sparse_dense_weight_int8": [4, 16], "compressed_weight_int8": [16, 2], "bias_int32": [4]}, "accumulator_type": "signed_int32", "preprocessing_version": standardization.version, "fine_tuning": {**config["fine_tuning"], "mask_preserved": True, "checkpoint_selection": "minimum validation loss"}, "storage_accounting": storage}
    validate_sparse_model(artifact); write_json(output / "sparse_quantized_model.json", artifact); write_json(output / "storage_report.json", storage)
    return {"artifact": artifact, "weights": weights, "compressed": compressed, "metadata": metadata, "dense": dense, "tuned": tuned}


def _metrics(examples: list, standardization: Standardization, dense: dict[str, object], sparse_weights: np.ndarray, sparse_bias: np.ndarray, sparse_scales: np.ndarray, compressed: np.ndarray, metadata: list[int], fp_model: LinearSensorClassifier) -> tuple[dict[str, object], list[int]]:
    features = standardization.transform(examples); x, _ = quantize_int8(features, float(dense["input_scale"])); dense_w, dense_b, dense_s = np.asarray(dense["weight_int8"], dtype=np.int8), np.asarray(dense["bias_int32"], dtype=np.int32), np.asarray(dense["weight_scales"], dtype=np.float64)
    d = [infer_int8(row, dense_w, dense_b, float(dense["input_scale"]), dense_s, CLASS_NAMES) for row in x]; sd = [infer_sparse_dense(row, sparse_weights, sparse_bias, float(dense["input_scale"]), sparse_scales, CLASS_NAMES) for row in x]; sc = [infer_sparse_compressed(row, compressed, metadata, sparse_bias, float(dense["input_scale"]), sparse_scales, CLASS_NAMES, 16) for row in x]
    if any(not np.array_equal(a.accumulators, b.accumulators) for a, b in zip(sd, sc, strict=True)): raise ValueError("compressed sparse inference differs from dense-form sparse inference")
    sparse_logits = np.stack([item.logits for item in sc]); dense_logits = np.stack([item.logits for item in d]); pred = np.asarray([item.predicted_class for item in sc]); dense_pred = np.asarray([item.predicted_class for item in d]); target = np.asarray([item.class_id for item in examples])
    with torch.no_grad(): fp_pred = fp_model(torch.from_numpy(features)).numpy().argmax(axis=1)
    matrix = np.zeros((4, 4), dtype=np.int64)
    for truth, guess in zip(target, pred, strict=True): matrix[truth, guess] += 1
    errors = sparse_logits - dense_logits; acc = np.concatenate([item.accumulators for item in sc]).astype(int).tolist()
    return {"sample_count": len(examples), "fp32_dense_fixture_accuracy": float((fp_pred == target).mean()), "dense_int8_fixture_accuracy": float((dense_pred == target).mean()), "sparse_int8_fixture_accuracy": float((pred == target).mean()), "prediction_agreement_with_dense_int8": float((pred == dense_pred).mean()), "prediction_agreement_with_fp32": float((pred == fp_pred).mean()), "prediction_disagreements_with_dense_int8": int((pred != dense_pred).sum()), "confusion_matrix": matrix.tolist(), "logit_error_from_dense_int8": {"maximum_absolute": float(np.abs(errors).max()), "mean_absolute": float(np.abs(errors).mean()), "root_mean_square": float(np.sqrt(np.mean(errors ** 2)))}}, acc


def evaluate(config_path: str | Path | None = None) -> dict[str, object]:
    config, root, splits, standardization = _load(config_path); packed = pack(config_path); tuned = packed["tuned"]; pre = tuned["pruned"]
    dense = packed["dense"]; pre_weights = pre["sparse"]; pre_bias = np.asarray(dense["bias_int32"], dtype=np.int32); pre_scales = np.asarray(dense["weight_scales"], dtype=np.float64); pre_compressed, pre_meta = pre["compressed"], pre["metadata"]
    fp_payload = torch.load(root / config["source"]["fp32_checkpoint"], map_location="cpu", weights_only=True); fp = LinearSensorClassifier(); fp.load_state_dict(fp_payload["state_dict"]); fp.eval()
    reports: dict[str, object] = {}; accumulators: list[int] = []
    for name, examples in splits.items():
        before, _ = _metrics(examples, standardization, dense, pre_weights, pre_bias, pre_scales, pre_compressed, pre_meta, fp)
        after, observed = _metrics(examples, standardization, dense, packed["weights"], np.asarray(packed["artifact"]["bias_int32"], dtype=np.int32), np.asarray(packed["artifact"]["weight_scales"], dtype=np.float64), packed["compressed"], packed["metadata"], fp); reports[name] = {"before_fine_tuning": before, "after_fine_tuning": after}; accumulators.extend(observed)
    counts = {f"{value:03b}": packed["metadata"].count(value) for value in range(6)}; operations = {"dense_executed_multiplications_per_sample": 64, "sparse_executed_multiplications_per_sample": sum(2 for _ in packed["metadata"]), "sparse_skipped_multiplications_per_sample": sum(2 for _ in packed["metadata"]), "total_possible_dense_multiplications_per_sample": 64, "arithmetic_reduction_percent": 50.0}
    conservative = int(np.max(np.abs(np.asarray(packed["artifact"]["bias_int32"], dtype=np.int64))) + 8 * 128 * 128); test = reports["test"]["after_fine_tuning"]; threshold = config["acceptance_thresholds"]
    result = {"format_version": "phase3_sparse_evaluation_v1", "claim_scope": "All accuracy values are measured only on the deterministic synthetic fixture.", "evaluations": reports, "pattern_distribution": counts, "operation_accounting": operations, "accumulator": {"observed_minimum": min(accumulators), "observed_maximum": max(accumulators), "all_observed_fit_signed_int32": all(-(2**31) <= item <= 2**31 - 1 for item in accumulators), "theoretical_conservative_bound": conservative, "theoretical_bound_fits_signed_int32": conservative <= 2**31 - 1, "compressed_equals_dense_form": True}, "acceptance": {"passed": bool(test["sparse_int8_fixture_accuracy"] >= threshold["minimum_test_fixture_accuracy"] and test["dense_int8_fixture_accuracy"] - test["sparse_int8_fixture_accuracy"] <= threshold["maximum_test_accuracy_drop"] and test["prediction_agreement_with_dense_int8"] >= threshold["minimum_test_prediction_agreement"]), "thresholds": threshold}}
    if not result["acceptance"]["passed"]: raise ValueError("Phase 3 sparse acceptance gates failed")
    output = root / config["output_directory"]; write_json(output / "sparse_evaluation_metrics.json", result); write_json(output / "arithmetic_count_report.json", operations); write_json(output / "pattern_distribution.json", counts); write_json(output / "prediction_agreement.json", {name: reports[name]["after_fine_tuning"] for name in reports}); write_json(output / "sparse_confusion_matrix.json", reports["test"]["after_fine_tuning"]["confusion_matrix"])
    return result


def run_baseline(config_path: str | Path | None = None) -> dict[str, object]:
    config, root, _, _ = _load(config_path); result = evaluate(config_path); output = root / config["output_directory"]
    (output / "config_snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    artifact_bytes = (output / "sparse_quantized_model.json").read_bytes(); digest = hashlib.sha256(artifact_bytes).hexdigest(); write_json(output / "determinism.json", {"sparse_quantized_model_sha256": digest})
    test = result["evaluations"]["test"]; (output / "summary.md").write_text("# Phase 3 deterministic 2:4 structured sparsity\n\nMeasured fixture accuracy is only for the deterministic synthetic fixture. Groups traverse output channel then consecutive input-feature groups; metadata is LSB-first packed.\n\n" + f"Post-fine-tuning sparse test accuracy: {test['after_fine_tuning']['sparse_int8_fixture_accuracy']:.4%}; dense agreement: {test['after_fine_tuning']['prediction_agreement_with_dense_int8']:.4%}.\n", encoding="utf-8")
    return result
