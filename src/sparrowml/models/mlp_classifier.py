"""Fixed, inspectable Phase 6 16-to-16-to-4 classifier."""
from __future__ import annotations

import torch


class MLPClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(16, 16)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16, 4)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(features)))

    def hidden(self, features: torch.Tensor) -> torch.Tensor:
        return self.relu(self.fc1(features))
