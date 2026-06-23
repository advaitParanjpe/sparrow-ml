import numpy as np
import pytest
import torch

from sparrowml.models.mlp_classifier import MLPClassifier
from sparrowml.quantization.multilayer import build_ir, infer, validate_ir


def test_fixed_model_shapes_and_parameter_count():
    model = MLPClassifier()
    assert model.fc1.weight.shape == (16, 16)
    assert model(torch.zeros((3, 16))).shape == (3, 4)
    assert sum(p.numel() for p in model.parameters()) == 340


def test_integer_mlp_relu_requantization_and_prediction():
    q = {"input_scale": 1.0, "hidden_scale": 1.0, "layers": {
        "fc1": {"weight_int8": [[1] + [0] * 15] * 16, "bias_int32": [-2] + [0] * 15, "weight_scales": [1.0] * 16},
        "fc2": {"weight_int8": [[1] + [0] * 15, [-1] + [0] * 15, [0] * 16, [0] * 16], "bias_int32": [0] * 4, "weight_scales": [1.0] * 4},
    }}
    result = infer(np.asarray([1] + [0] * 15, dtype=np.int8), q)
    assert result["hidden_int8"][0] == 0
    assert result["fc2_acc_int32"][:2] == [0, 0]


def test_multilayer_ir_rejects_invalid_order():
    q = {"input_scale": 1.0, "hidden_scale": 1.0, "layers": {"fc1": {"weight_int8": [[0] * 16] * 16, "bias_int32": [0] * 16, "weight_scales": [1.0] * 16}, "fc2": {"weight_int8": [[0] * 16] * 4, "bias_int32": [0] * 4, "weight_scales": [1.0] * 4}}}
    ir = build_ir(q); validate_ir(ir)
    ir["operators"][1]["op_type"] = "DenseLinearInt8"
    with pytest.raises(ValueError, match="operator sequence"):
        validate_ir(ir)
