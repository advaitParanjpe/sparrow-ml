# Experiment Policy

Use fixed seeds and separate train/dev/test splits; never tune on test data. Preserve raw configurations and record software plus hardware versions. Distinguish measured, simulated, estimated, and derived metrics. Do not cherry-pick results; retain failed experiments in summaries where relevant. Accuracy claims state dataset and split. Hardware claims state target, clock assumptions, and measurement scope.

For Phase 1, the configuration and output record generation/split, Python/NumPy/PyTorch, and DataLoader seeds. Best checkpoint selection uses validation loss only. All reported accuracy fields are named `fixture_accuracy` and are measured only on the deterministic synthetic fixture.
