"""The Phase 1 FP32 model: a single logits-only linear layer."""

from __future__ import annotations

import torch


class LinearSensorClassifier(torch.nn.Module):
    def __init__(self, input_features: int = 16, class_count: int = 4) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_features, class_count)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)
