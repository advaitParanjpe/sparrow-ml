# Experiment Policy

Use fixed seeds and separate train/dev/test splits; never tune on test data. Preserve raw configurations and record software plus hardware versions. Distinguish measured, simulated, estimated, and derived metrics. Do not cherry-pick results; retain failed experiments in summaries where relevant. Accuracy claims state dataset and split. Hardware claims state target, clock assumptions, and measurement scope.

For Phase 1, the configuration and output record generation/split, Python/NumPy/PyTorch, and DataLoader seeds. Best checkpoint selection uses validation loss only. All reported accuracy fields are named `fixture_accuracy` and are measured only on the deterministic synthetic fixture.

For Phase 2, activation calibration uses standardized training examples only. It uses `max(abs(x))/127`, zero point zero, NumPy `rint` (nearest with ties to even), and post-rounding clamp to `[-128, 127]`. Weights use the same scale policy independently per output row. Bias scale is input scale times that row's weight scale; bias and reference accumulation are signed INT32 after range validation. Per-channel accumulator values are reconstructed to real logits before argmax. Metrics remain fixture-scoped.

For Phase 3, pruning is deterministic: retain the two largest absolute INT8 weights in each consecutive input-feature group of four, resolving ties by lower lane index. Fine-tuning uses a fixed derived mask, reapplied after every CPU optimizer step, and chooses checkpoints only by validation loss. Test data is evaluated after selection. Pre- and post-fine-tuning sparse results are retained; they remain synthetic-fixture measurements, not hardware or general-model claims.

Phase 4 export is a deterministic transformation of existing Phase 2/3 artifacts and one fixed test sample. It records SHA-256 values and exact integer-reference equivalence, not model-quality or hardware measurements. Sparrow-V is neither invoked nor modified.

Phase 5 runs only Sparrow-V's documented external sensor-workload RTL interface. Counters are labelled measured, derived, or unavailable as supplied by that interface. Any host-side reconstruction, currently required for INT32 biases beyond the template's signed-12-bit immediate range, is explicitly identified and is not labelled an RTL value. Results are simulated architectural evidence, not FPGA, ASIC, timing, power, or general performance claims.
