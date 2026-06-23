"""Small direct metrics implementation for the fixed four-class fixture."""

from __future__ import annotations

import numpy as np
import torch

from sparrowml.data.fixture import CLASS_NAMES, SensorExample


def evaluate_model(model: torch.nn.Module, features: np.ndarray, examples: list[SensorExample]) -> dict[str, object]:
    model.eval()
    targets = np.asarray([example.class_id for example in examples], dtype=np.int64)
    with torch.no_grad():
        logits = model(torch.from_numpy(features)).cpu()
        loss = float(torch.nn.functional.cross_entropy(logits, torch.from_numpy(targets)).item())
        predictions = logits.argmax(dim=1).numpy()
    matrix = np.zeros((4, 4), dtype=np.int64)
    for target, prediction in zip(targets, predictions, strict=True):
        matrix[target, prediction] += 1
    per_class = {
        name: {"sample_count": int(matrix[index].sum()), "correct_count": int(matrix[index, index])}
        for index, name in enumerate(CLASS_NAMES)
    }
    return {
        "loss": loss,
        "fixture_accuracy": float((predictions == targets).mean()),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "predicted_class_distribution": {name: int((predictions == index).sum()) for index, name in enumerate(CLASS_NAMES)},
    }
