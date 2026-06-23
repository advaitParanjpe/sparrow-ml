# Phase 1 FP32 baseline

The baseline is reproduced with `make run-fp32-baseline`. It generates 512 synthetic samples (128 per class) for the placeholder vibration-fault-style names `normal`, `inner`, `outer`, and `ball`. The fixed seed is `20260623`; each sample has 16 ordered numerical features. Deterministic per-class membership produces 360 train, 76 validation, and 76 test examples.

The model is FP32 CPU `Linear(16, 4)` with 68 parameters. Features are standardized with train-only mean and standard deviation; the saved preprocessing values are reused for validation and test. Adam trains for 50 epochs with batch size 32 and learning rate 0.01. Validation loss selects the best checkpoint; test data is only evaluated after selection.

Generated ignored artifacts are under `artifacts/phase1_fp32/`: configuration snapshot, fixture metadata, preprocessing statistics, checkpoint, metrics, confusion matrix, and generated summary. The bounded reference run measured 100% train, validation, and test fixture accuracy with test confusion matrix `[[19, 0, 0, 0], [0, 19, 0, 0], [0, 0, 19, 0], [0, 0, 0, 19]]`; this easy synthetic result is not a real-world accuracy or deployment/hardware result.

No quantization, pruning, compiler lowering, Sparrow-V execution, or hardware measurements are part of this phase.
