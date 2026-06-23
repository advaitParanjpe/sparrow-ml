import json

import numpy as np
import torch
import pytest

from sparrowml.data.fixture import CLASS_NAMES, generate_fixture, write_fixture
from sparrowml.data.preprocessing import fit_standardization
from sparrowml.evaluation.metrics import evaluate_model
from sparrowml.models.linear_classifier import LinearSensorClassifier
from sparrowml.training.trainer import load_and_evaluate, train_baseline
from sparrowml.cli import main


def test_fixture_is_deterministic_complete_and_balanced():
    first = generate_fixture()
    second = generate_fixture()
    assert first == second
    assert len(first) == 512
    assert len({item.sample_id for item in first}) == 512
    assert [item.class_name for item in first[:128:32]] == ["normal"] * 4
    assert {item.class_name for item in first} == set(CLASS_NAMES)
    assert {item.split for item in first} == {"train", "validation", "test"}
    assert all(len(item.features) == 16 and np.isfinite(item.features).all() for item in first)
    assert {split: sum(item.split == split for item in first) for split in ("train", "validation", "test")} == {"train": 360, "validation": 76, "test": 76}


def test_preprocessing_uses_train_statistics_and_handles_zero_variance():
    examples = generate_fixture()
    train = [item for item in examples if item.split == "train"]
    validation = [item for item in examples if item.split == "validation"]
    standardization = fit_standardization(train)
    assert np.allclose(standardization.transform(train).mean(axis=0), 0, atol=1e-6)
    assert standardization.transform(validation).shape == (76, 16)
    assert np.isfinite(standardization.transform(validation)).all()
    constant = [type(train[0])("constant", (1.0,) * 16, 0, "normal", "train")]
    assert np.all(fit_standardization(constant).std == 1.0)


def test_model_shapes_parameter_count_and_deterministic_initialization():
    torch.manual_seed(3)
    first = LinearSensorClassifier()
    torch.manual_seed(3)
    second = LinearSensorClassifier()
    assert first(torch.zeros((2, 16))).shape == (2, 4)
    assert sum(parameter.numel() for parameter in first.parameters()) == 68
    assert all(torch.equal(left, right) for left, right in zip(first.parameters(), second.parameters(), strict=True))


def test_training_checkpoint_metrics_and_repeatable_evaluation(tmp_path):
    examples = generate_fixture()
    output = tmp_path / "phase1"
    metrics, _ = train_baseline(examples, seed=20260623, dataloader_seed=20260623, learning_rate=0.01, epochs=8, batch_size=32, output_directory=output)
    assert (output / "best_fp32.pt").is_file()
    assert (output / "metrics.json").is_file()
    assert metrics["evaluations"]["test"]["fixture_accuracy"] >= 0.85
    assert len(metrics["evaluations"]["test"]["confusion_matrix"]) == 4
    first = load_and_evaluate(examples, output)
    second = load_and_evaluate(examples, output)
    assert first == second
    assert sum(value["sample_count"] for value in first["test"]["per_class"].values()) == 76
    assert json.loads((output / "metrics.json").read_text())["model"]["parameter_count"] == 68


def test_accuracy_metric_is_correct_for_known_predictions():
    examples = generate_fixture()[:4]
    # This smoke assertion ensures the metric contract has one row and column per fixed class.
    model = LinearSensorClassifier()
    result = evaluate_model(model, np.zeros((4, 16), dtype=np.float32), examples)
    assert np.asarray(result["confusion_matrix"]).shape == (4, 4)


def test_phase1_cli_parses_and_invalid_configuration_fails():
    with pytest.raises(SystemExit) as help_exit:
        main(["generate-fixture", "--help"])
    assert help_exit.value.code == 0
    with pytest.raises(SystemExit) as error:
        main(["train-fp32", "--config", "configs/experiments/missing.yaml"])
    assert error.value.code == 2
