from pathlib import Path

import numpy as np

from sparrowml.data.wisdm import CLASS_NAMES, Record, features, subject_splits


def _records() -> list[Record]:
    return [Record(str(subject), activity, activity[0], index, 1.0, 2.0, 3.0, f"data_{subject}.txt", index) for subject in range(10) for activity in CLASS_NAMES for index in range(100)]


def test_subject_splits_are_deterministic_and_disjoint():
    first, eligibility = subject_splits(_records())
    second, _ = subject_splits(_records())
    assert first == second
    assert not (set(first["train"]) & set(first["validation"]))
    assert eligibility["eligible_subject_count"] == 10


def test_features_have_exact_schema_and_zero_energy_is_finite():
    rows = [Record("1", "walking", "A", index, 0.0, 0.0, 0.0, "x", index) for index in range(80)]
    values = features(rows)
    assert len(values) == 16
    assert np.isfinite(values).all()
