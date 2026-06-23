"""Deterministic synthetic vibration-fault-style sensor fixture."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

CLASS_NAMES = ("normal", "inner", "outer", "ball")
FEATURE_NAMES = tuple(f"feature_{index:02d}" for index in range(16))
DEFAULT_SEED = 20260623


@dataclass(frozen=True)
class SensorExample:
    sample_id: str
    features: tuple[float, ...]
    class_id: int
    class_name: str
    split: str


def _prototypes() -> np.ndarray:
    """Fixed, non-one-hot spectral-summary prototypes for the four placeholder classes."""
    axis = np.arange(16, dtype=np.float64)
    return np.stack(
        (
            0.30 * np.sin(axis / 2.4) + 0.10 * np.cos(axis / 4.0),
            0.30 * np.sin(axis / 2.4 + 0.55) + 0.10 * np.cos(axis / 3.3) + 0.42,
            0.30 * np.sin(axis / 2.4 - 0.65) + 0.13 * np.cos(axis / 2.1) - 0.38,
            0.24 * np.sin(axis / 1.8 + 1.1) + 0.16 * np.cos(axis / 3.8) + 0.08,
        )
    )


def generate_fixture(
    *,
    seed: int = DEFAULT_SEED,
    split_seed: int = DEFAULT_SEED,
    samples_per_class: int = 128,
    split_counts_per_class: dict[str, int] | None = None,
) -> list[SensorExample]:
    """Generate stable examples; IDs and split assignment do not depend on output paths."""
    split_counts = split_counts_per_class or {"train": 90, "validation": 19, "test": 19}
    if tuple(split_counts) != ("train", "validation", "test"):
        raise ValueError("split counts must be ordered as train, validation, test")
    if sum(split_counts.values()) != samples_per_class:
        raise ValueError("split counts must sum to samples_per_class")
    rng = np.random.default_rng(seed)
    split_rng = np.random.default_rng(split_seed)
    examples: list[SensorExample] = []
    # Shared low-frequency noise makes the task meaningful without encoding labels one-hot.
    basis = np.linspace(-0.22, 0.22, 16)
    for class_id, (class_name, prototype) in enumerate(zip(CLASS_NAMES, _prototypes(), strict=True)):
        ordering = split_rng.permutation(samples_per_class)
        membership: dict[int, str] = {}
        start = 0
        for split, count in split_counts.items():
            for index in ordering[start : start + count]:
                membership[int(index)] = split
            start += count
        for sample_index in range(samples_per_class):
            drift = rng.normal(0.0, 0.18) * basis
            independent = rng.normal(0.0, 0.10, size=16)
            features = prototype + drift + independent
            examples.append(
                SensorExample(
                    sample_id=f"sensor-{class_name}-{sample_index:03d}",
                    features=tuple(float(value) for value in features),
                    class_id=class_id,
                    class_name=class_name,
                    split=membership[sample_index],
                )
            )
    validate_examples(examples)
    return examples


def validate_examples(examples: list[SensorExample]) -> None:
    if len({example.sample_id for example in examples}) != len(examples):
        raise ValueError("sample IDs must be unique")
    for example in examples:
        if len(example.features) != 16 or not np.isfinite(example.features).all():
            raise ValueError(f"invalid features for {example.sample_id}")
        if example.class_name != CLASS_NAMES[example.class_id] or example.split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid labels for {example.sample_id}")


def fixture_metadata(examples: list[SensorExample], seed: int, split_seed: int) -> dict[str, object]:
    split_sizes = {split: sum(example.split == split for example in examples) for split in ("train", "validation", "test")}
    class_counts = {name: sum(example.class_name == name for example in examples) for name in CLASS_NAMES}
    return {
        "fixture_version": "sensor_fixture_v1",
        "provenance": "synthetic deterministic vibration-fault-style fixture; class names are placeholders, not a real-world dataset",
        "generation_seed": seed,
        "split_seed": split_seed,
        "feature_count": 16,
        "feature_order": list(FEATURE_NAMES),
        "class_order": list(CLASS_NAMES),
        "split_policy": "per-class deterministic permutation using split_seed",
        "split_sizes": split_sizes,
        "class_counts": class_counts,
    }


def write_fixture(directory: Path, examples: list[SensorExample], seed: int, split_seed: int) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "examples.jsonl").open("w", encoding="utf-8") as destination:
        for example in examples:
            destination.write(json.dumps(asdict(example), sort_keys=True, separators=(",", ":")) + "\n")
    metadata = fixture_metadata(examples, seed, split_seed)
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def load_fixture(directory: Path) -> list[SensorExample]:
    with (directory / "examples.jsonl").open(encoding="utf-8") as source:
        examples = [SensorExample(**{**item, "features": tuple(item["features"])}) for line in source if (item := json.loads(line))]
    validate_examples(examples)
    return examples
