"""Train-only deterministic feature standardization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fixture import SensorExample


@dataclass(frozen=True)
class Standardization:
    mean: np.ndarray
    std: np.ndarray
    version: str = "standardize_train_v1"

    def transform(self, examples: list[SensorExample]) -> np.ndarray:
        values = feature_matrix(examples)
        return ((values - self.mean) / self.std).astype(np.float32)

    def as_dict(self) -> dict[str, object]:
        return {"version": self.version, "mean": self.mean.tolist(), "std": self.std.tolist()}


def feature_matrix(examples: list[SensorExample]) -> np.ndarray:
    values = np.asarray([example.features for example in examples], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 16 or not np.isfinite(values).all():
        raise ValueError("examples must contain finite 16-feature vectors")
    return values


def fit_standardization(train_examples: list[SensorExample], epsilon: float = 1.0e-8) -> Standardization:
    if not train_examples:
        raise ValueError("training split is empty")
    values = feature_matrix(train_examples)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < epsilon, 1.0, std)
    return Standardization(mean=mean, std=std)
