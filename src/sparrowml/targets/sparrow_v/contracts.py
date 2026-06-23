"""Schemas shared between SparrowML-generated artifacts and Sparrow-V results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath


def _result_path(value: str | None) -> str | None:
    if value is None:
        return None
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError("simulator_log_path must be repository-relative")
    return value


@dataclass(frozen=True)
class SparrowVResult:
    completion_status: str
    output_logits: tuple[int, ...]
    predicted_class: int
    cycles: int
    retired_instructions: int
    vector_load_count: int
    vdot8_count: int
    vsdot8_count: int
    executed_multiplication_count: int
    skipped_multiplication_count: int
    simulator_log_path: str | None = None

    def __post_init__(self) -> None:
        if self.completion_status not in {"complete", "failed", "timeout"}:
            raise ValueError("invalid completion status")
        if not self.output_logits:
            raise ValueError("output_logits must not be empty")
        if not 0 <= self.predicted_class < len(self.output_logits):
            raise ValueError("predicted_class is outside output logits")
        counts = (
            self.cycles, self.retired_instructions, self.vector_load_count,
            self.vdot8_count, self.vsdot8_count, self.executed_multiplication_count,
            self.skipped_multiplication_count,
        )
        if any(value < 0 for value in counts):
            raise ValueError("hardware counters must be non-negative")
        _result_path(self.simulator_log_path)
