"""CPU-only reproducibility controls."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_deterministic_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
