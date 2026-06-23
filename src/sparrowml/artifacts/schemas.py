"""Typed, validation-first schemas for future deployment artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath


def _relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact paths must be non-empty repository-relative paths")
    return value


@dataclass(frozen=True)
class ModelMetadata:
    name: str
    input_shape: tuple[int, ...]
    output_classes: int
    quantization: str

    def __post_init__(self) -> None:
        if not self.name or not self.input_shape or any(size <= 0 for size in self.input_shape):
            raise ValueError("model metadata requires a name and positive input shape")
        if self.output_classes <= 0:
            raise ValueError("output_classes must be positive")


@dataclass(frozen=True)
class DeploymentManifest:
    model: ModelMetadata
    input_sample_path: str
    dense_weights_path: str
    sparse_weights_path: str
    sparse_metadata_path: str
    biases_path: str
    expected_outputs_path: str
    program_image_path: str
    data_image_path: str

    def __post_init__(self) -> None:
        for value in self.__dict__.values():
            if isinstance(value, str):
                _relative_path(value)
