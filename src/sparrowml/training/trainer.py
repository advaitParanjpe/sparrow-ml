"""Bounded CPU FP32 training for the Phase 1 linear baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sparrowml.data.fixture import CLASS_NAMES, SensorExample
from sparrowml.data.preprocessing import Standardization, fit_standardization
from sparrowml.evaluation.metrics import evaluate_model
from sparrowml.models.linear_classifier import LinearSensorClassifier
from sparrowml.training.seeds import set_deterministic_seeds


def split_examples(examples: list[SensorExample]) -> dict[str, list[SensorExample]]:
    return {split: [example for example in examples if example.split == split] for split in ("train", "validation", "test")}


def _targets(examples: list[SensorExample]) -> np.ndarray:
    return np.asarray([example.class_id for example in examples], dtype=np.int64)


def train_baseline(
    examples: list[SensorExample],
    *,
    seed: int,
    dataloader_seed: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    output_directory: Path,
) -> tuple[dict[str, object], Standardization]:
    """Train using validation loss only to select the persisted checkpoint."""
    set_deterministic_seeds(seed)
    splits = split_examples(examples)
    normalization = fit_standardization(splits["train"])
    matrices = {name: normalization.transform(items) for name, items in splits.items()}
    model = LinearSensorClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(dataloader_seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(matrices["train"]), torch.from_numpy(_targets(splits["train"]))),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / "best_fp32.pt"
    best_epoch, best_validation_loss = 0, float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        for features, targets in loader:
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(features), targets)
            loss.backward()
            optimizer.step()
        validation = evaluate_model(model, matrices["validation"], splits["validation"])
        validation_loss = float(validation["loss"])
        if validation_loss < best_validation_loss:
            best_epoch, best_validation_loss = epoch, validation_loss
            torch.save(
                {"state_dict": model.state_dict(), "input_features": 16, "class_count": 4, "seed": seed, "best_epoch": epoch},
                checkpoint_path,
            )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"])
    evaluations = {name: evaluate_model(model, matrices[name], splits[name]) for name in splits}
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metrics: dict[str, object] = {
        "experiment_version": "phase1_fp32_v1",
        "claim_scope": "All accuracy values are fixture accuracy measured on a synthetic deterministic fixture, not real-world accuracy.",
        "device": "cpu",
        "dtype": "float32",
        "class_order": list(CLASS_NAMES),
        "feature_count": 16,
        "class_count": 4,
        "split_sizes": {name: len(items) for name, items in splits.items()},
        "seeds": {"python_numpy_torch": seed, "dataloader_shuffle": dataloader_seed},
        "model": {"architecture": "Linear(16, 4)", "parameter_count": parameter_count},
        "best_epoch": best_epoch,
        "checkpoint": {"path": "best_fp32.pt", "size_bytes": checkpoint_path.stat().st_size},
        "evaluations": evaluations,
    }
    (output_directory / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_directory / "preprocessing.json").write_text(json.dumps(normalization.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_directory / "confusion_matrix.json").write_text(
        json.dumps(evaluations["test"]["confusion_matrix"], indent=2) + "\n", encoding="utf-8"
    )
    return metrics, normalization


def load_and_evaluate(examples: list[SensorExample], output_directory: Path) -> dict[str, object]:
    """Evaluate the saved checkpoint with the saved train-fit preprocessing statistics."""
    preprocessing = json.loads((output_directory / "preprocessing.json").read_text(encoding="utf-8"))
    normalization = Standardization(np.asarray(preprocessing["mean"]), np.asarray(preprocessing["std"]), preprocessing["version"])
    payload = torch.load(output_directory / "best_fp32.pt", map_location="cpu", weights_only=True)
    model = LinearSensorClassifier(int(payload["input_features"]), int(payload["class_count"]))
    model.load_state_dict(payload["state_dict"])
    splits = split_examples(examples)
    return {name: evaluate_model(model, normalization.transform(items), items) for name, items in splits.items()}
