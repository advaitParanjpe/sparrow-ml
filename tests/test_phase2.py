import numpy as np
import pytest

from sparrowml.quantization.affine import dequantize_int8, quantize_int32, quantize_int8, symmetric_scale
from sparrowml.quantization.artifacts import validate_quantized_model
from sparrowml.quantization.calibration import calibrate_symmetric_int8
from sparrowml.quantization.integer_reference import infer_int8
from sparrowml.cli import main


def test_int8_primitives_round_clamp_zero_and_determinism():
    values = np.asarray([-200.0, -1.5, -0.5, 0.5, 1.5, 200.0])
    first, saturation = quantize_int8(values, 1.0)
    second, _ = quantize_int8(values, 1.0)
    assert first.tolist() == [-128, -2, 0, 0, 2, 127]
    assert np.array_equal(first, second)
    assert saturation["total_clipped_values"] == 2
    assert symmetric_scale(np.zeros((2, 2))) == 1.0
    assert dequantize_int8(np.asarray([-128, 127], dtype=np.int8), 0.5).tolist() == [-64.0, 63.5]


def test_calibration_is_train_only_and_stable():
    features = np.asarray([[-2.0, 0.0], [1.0, 3.0]], dtype=np.float32)
    report = calibrate_symmetric_int8(features)
    assert report["calibration_split"] == "train"
    assert report["calibration_sample_count"] == 2
    assert report["input_scale"] == 3.0 / 127.0
    with pytest.raises(ValueError):
        calibrate_symmetric_int8(features, split="test")


def test_bias_and_integer_reference_use_per_channel_scales():
    bias = quantize_int32(np.asarray([0.5, -0.5]), np.asarray([0.25, 0.5]))
    assert bias.tolist() == [2, -1]
    result = infer_int8(np.asarray([2, -3], dtype=np.int8), np.asarray([[4, -5], [-2, 6]], dtype=np.int8), bias, 0.5, np.asarray([0.5, 1.0]), ("a", "b"))
    assert result.accumulators.tolist() == [25, -23]
    assert result.logits.tolist() == [6.25, -11.5]
    assert result.predicted_class == 0
    with pytest.raises(ValueError):
        quantize_int32(np.asarray([1.0]), np.asarray([0.0]))


def test_artifact_schema_rejects_invalid_ranges():
    artifact = {"format_version": "sparrowml_int8_linear_v1", "model_name": "m", "source_fp32_checkpoint": "artifacts/a.pt", "feature_count": 1, "class_count": 1, "class_names": ["a"], "quantization": {}, "input_scale": 1.0, "input_zero_point": 0, "weight_scales": [1.0], "weight_zero_points": [0], "weight_int8": [[0]], "bias_int32": [0], "tensor_shapes": {}, "lane_order": "x", "preprocessing_version": "x", "calibration": {}, "accumulator_type": "signed_int32", "creation_configuration": "configs/x.yaml"}
    validate_quantized_model(artifact)
    artifact["weight_int8"] = [[128]]
    with pytest.raises(ValueError):
        validate_quantized_model(artifact)


def test_phase2_cli_commands_parse():
    with pytest.raises(SystemExit) as help_exit:
        main(["run-int8-baseline", "--help"])
    assert help_exit.value.code == 0
    with pytest.raises(SystemExit) as error:
        main(["calibrate-int8", "--config", "configs/experiments/missing.yaml"])
    assert error.value.code == 2
