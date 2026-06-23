"""Bounded Phase 2 calibration, quantization, evaluation, and artifact production."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
import yaml

from sparrowml.data.fixture import CLASS_NAMES, load_fixture
from sparrowml.data.preprocessing import Standardization
from sparrowml.models.linear_classifier import LinearSensorClassifier
from sparrowml.training.trainer import split_examples
from .affine import INT32_MAX, quantize_int8, quantize_int32, symmetric_scale
from .artifacts import validate_quantized_model, write_json
from .calibration import calibrate_symmetric_int8
from .integer_reference import infer_int8


def _load(config_path: str | Path | None) -> tuple[dict[str, object], Path, dict[str, list], Standardization]:
    root = Path(__file__).resolve().parents[3]
    path = root / (config_path or "configs/experiments/int8_ptq_baseline.yaml")
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        source = config["source"]
        output = config["output_directory"]
        if config["input_quantization"]["scheme"] != "per_tensor_symmetric_int8" or config["weight_quantization"]["scheme"] != "per_output_channel_symmetric_int8":
            raise ValueError("only documented symmetric INT8 schemes are supported")
        preprocessing = json.loads((root / source["preprocessing_path"]).read_text(encoding="utf-8"))
        examples = load_fixture(root / source["fixture_directory"])
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Phase 2 configuration {path}: {exc}") from exc
    return config, root, split_examples(examples), Standardization(np.asarray(preprocessing["mean"]), np.asarray(preprocessing["std"]), preprocessing["version"])


def calibrate(config_path: str | Path | None = None) -> dict[str, object]:
    config, root, splits, standardization = _load(config_path)
    report = calibrate_symmetric_int8(standardization.transform(splits["train"]))
    output = root / config["output_directory"]
    write_json(output / "calibration_report.json", report)
    (output / "config_snapshot.yaml").parent.mkdir(parents=True, exist_ok=True)
    (output / "config_snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return report


def quantize(config_path: str | Path | None = None) -> dict[str, object]:
    config, root, _, standardization = _load(config_path)
    output = root / config["output_directory"]
    calibration_path = output / "calibration_report.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.exists() else calibrate(config_path)
    checkpoint = root / config["source"]["checkpoint_path"]
    if not checkpoint.is_file():
        raise ValueError(f"missing Phase 1 checkpoint: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    weight = payload["state_dict"]["linear.weight"].detach().cpu().numpy()
    bias = payload["state_dict"]["linear.bias"].detach().cpu().numpy()
    scales = np.asarray([symmetric_scale(row) for row in weight], dtype=np.float64)
    rows = [quantize_int8(row, float(scale))[0] for row, scale in zip(weight, scales, strict=True)]
    weights = np.stack(rows)
    weight_saturation = [quantize_int8(row, float(scale))[1] for row, scale in zip(weight, scales, strict=True)]
    input_scale = float(calibration["input_scale"])
    biases = quantize_int32(bias, input_scale * scales)
    artifact: dict[str, object] = {
        "format_version": "sparrowml_int8_linear_v1", "model_name": "Linear(16, 4)",
        "source_fp32_checkpoint": str(config["source"]["checkpoint_path"]), "feature_count": 16, "class_count": 4, "class_names": list(CLASS_NAMES),
        "quantization": {"input_scheme": "per_tensor_symmetric_int8", "weight_scheme": "per_output_channel_symmetric_int8", "rounding": "NumPy rint: round-to-nearest, ties-to-even", "clamping": "[-128, 127]"},
        "input_scale": input_scale, "input_zero_point": 0, "weight_scales": scales.tolist(), "weight_zero_points": [0] * 4,
        "weight_int8": weights.astype(int).tolist(), "bias_int32": biases.astype(int).tolist(), "tensor_shapes": {"weight_int8": [4, 16], "bias_int32": [4]},
        "lane_order": "weight_int8[output_channel][input_feature]; feature order follows Phase 1 preprocessing", "preprocessing_version": standardization.version,
        "calibration": {"split": calibration["calibration_split"], "sample_count": calibration["calibration_sample_count"]}, "accumulator_type": "signed_int32", "creation_configuration": "configs/experiments/int8_ptq_baseline.yaml",
        "weight_saturation": {"total_values": 64, "values_at_negative_128": sum(item["values_at_negative_128"] for item in weight_saturation), "values_at_127": sum(item["values_at_127"] for item in weight_saturation), "total_clipped_values": sum(item["total_clipped_values"] for item in weight_saturation), "per_channel": weight_saturation},
        "bias_range": {"minimum": int(biases.min()), "maximum": int(biases.max())},
    }
    validate_quantized_model(artifact)
    write_json(output / "quantized_model.json", artifact)
    return artifact


def _model(checkpoint: Path) -> LinearSensorClassifier:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = LinearSensorClassifier(int(payload["input_features"]), int(payload["class_count"]))
    model.load_state_dict(payload["state_dict"]); model.eval()
    return model


def evaluate(config_path: str | Path | None = None) -> dict[str, object]:
    config, root, splits, standardization = _load(config_path)
    artifact = quantize(config_path)
    output = root / config["output_directory"]
    model = _model(root / config["source"]["checkpoint_path"])
    weights, biases = np.asarray(artifact["weight_int8"], dtype=np.int8), np.asarray(artifact["bias_int32"], dtype=np.int32)
    scales, input_scale = np.asarray(artifact["weight_scales"], dtype=np.float64), float(artifact["input_scale"])
    all_acc: list[int] = []; reports: dict[str, object] = {}
    for name, examples in splits.items():
        features = standardization.transform(examples); quantized, saturation = quantize_int8(features, input_scale)
        with torch.no_grad(): fp_logits = model(torch.from_numpy(features)).numpy().astype(np.float64)
        integer = [infer_int8(row, weights, biases, input_scale, scales, CLASS_NAMES) for row in quantized]
        approx = np.stack([item.logits for item in integer]); predicted = np.asarray([item.predicted_class for item in integer]); fp_pred = fp_logits.argmax(axis=1)
        targets = np.asarray([item.class_id for item in examples]); errors = approx - fp_logits; matrix = np.zeros((4, 4), dtype=np.int64)
        for target, pred in zip(targets, predicted, strict=True): matrix[target, pred] += 1
        all_acc.extend(np.concatenate([item.accumulators for item in integer]).astype(int).tolist())
        reports[name] = {"sample_count": len(examples), "fp32_fixture_accuracy": float((fp_pred == targets).mean()), "int8_fixture_accuracy": float((predicted == targets).mean()), "prediction_agreement": float((predicted == fp_pred).mean()), "prediction_disagreements": int((predicted != fp_pred).sum()), "confusion_matrix": matrix.tolist(), "logit_error": {"maximum_absolute": float(np.abs(errors).max()), "mean_absolute": float(np.abs(errors).mean()), "root_mean_square": float(np.sqrt(np.mean(errors**2))), "per_output_channel": [{"maximum_absolute": float(np.abs(errors[:, index]).max()), "mean_absolute": float(np.abs(errors[:, index]).mean()), "root_mean_square": float(np.sqrt(np.mean(errors[:, index] ** 2)))} for index in range(4)]}, "input_saturation": {**saturation, "clipping_percentage": 100.0 * saturation["total_clipped_values"] / saturation["total_values"]}}
    theoretical = int(np.max(np.abs(biases.astype(np.int64))) + 16 * 128 * 128)
    metrics = {"format_version": "phase2_int8_evaluation_v1", "claim_scope": "Fixture accuracy is measured on the synthetic deterministic fixture, not real-world accuracy.", "prediction_semantics": "argmax is computed over reconstructed per-channel real-valued logits, never directly over differently scaled accumulators.", "evaluations": reports, "accumulator": {"observed_minimum": min(all_acc), "observed_maximum": max(all_acc), "all_observed_fit_signed_int32": all(-(2**31) <= value <= INT32_MAX for value in all_acc), "theoretical_conservative_bound": theoretical, "theoretical_bound_fits_signed_int32": theoretical <= INT32_MAX}, "weight_saturation": artifact["weight_saturation"], "bias_range": artifact["bias_range"]}
    gate = config["acceptance_thresholds"]; test = reports["test"]
    metrics["acceptance"] = {"passed": bool(test["int8_fixture_accuracy"] >= float(gate["minimum_test_fixture_accuracy"]) and test["fp32_fixture_accuracy"] - test["int8_fixture_accuracy"] <= float(gate["maximum_test_accuracy_drop"]) and test["prediction_agreement"] >= float(gate["minimum_test_prediction_agreement"]) and metrics["accumulator"]["all_observed_fit_signed_int32"] and metrics["accumulator"]["theoretical_bound_fits_signed_int32"]), "thresholds": gate}
    if not metrics["acceptance"]["passed"]: raise ValueError("Phase 2 quantization acceptance gates failed")
    write_json(output / "integer_evaluation_metrics.json", metrics); write_json(output / "error_statistics.json", {name: reports[name]["logit_error"] for name in reports}); write_json(output / "confusion_matrix.json", reports["test"]["confusion_matrix"]); write_json(output / "prediction_agreement.json", {name: {key: reports[name][key] for key in ("prediction_agreement", "prediction_disagreements")} for name in reports})
    lines = ["# Phase 2 INT8 PTQ", "", "Measured fixture accuracy is for the deterministic synthetic fixture only, not real-world model quality.", "", "| Split | FP32 fixture accuracy | INT8 fixture accuracy | Agreement |", "| --- | ---: | ---: | ---: |"] + [f"| {name} | {reports[name]['fp32_fixture_accuracy']:.4%} | {reports[name]['int8_fixture_accuracy']:.4%} | {reports[name]['prediction_agreement']:.4%} |" for name in ("train", "validation", "test")] + ["", f"Input scale: `{input_scale}`; zero point: `0`.", f"Observed accumulator range: `{min(all_acc)}` to `{max(all_acc)}`; conservative bound: `{theoretical}`.", "", "INT8 arithmetic uses explicit signed products and integer accumulation; logits are reconstructed with input scale times each output-channel weight scale."]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics


def run_baseline(config_path: str | Path | None = None) -> dict[str, object]:
    calibrate(config_path); quantize(config_path); return evaluate(config_path)
