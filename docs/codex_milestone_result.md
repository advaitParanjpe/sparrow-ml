STATUS: COMPLETE
MILESTONE: Deterministic Sensor Dataset Fixture and FP32 Baseline

Implemented a deterministic synthetic vibration-fault-style fixture. It has 512 samples, 16 finite features per sample, fixed class order `normal`, `inner`, `outer`, `ball`, seed `20260623`, stable IDs, and balanced per-class splits: 360 train (90/class), 76 validation (19/class), 76 test (19/class). The names are placeholders, not a real dataset claim.

Implemented train-only standardization (`standardize_train_v1`) and a CPU FP32 `Linear(16, 4)` model with 68 parameters. Training uses Adam (learning rate 0.01, batch size 32, 50 epochs), fixed Python/NumPy/PyTorch/DataLoader seeds, and validation loss only for checkpoint selection. PyTorch deterministic algorithms are enabled; exact bitwise equivalence is limited to the tested software environment.

Measured reference run (`make run-fp32-baseline`): best epoch 50; checkpoint 2293 bytes; train loss 0.003876 / fixture accuracy 100%; validation loss 0.003156 / fixture accuracy 100%; test loss 0.005239 / fixture accuracy 100%. Test confusion matrix (rows true, columns predicted, class order above): `[[19, 0, 0, 0], [0, 19, 0, 0], [0, 0, 19, 0], [0, 0, 0, 19]]`. These are synthetic fixture-only measurements, not real-world accuracy claims.

Generated ignored artifacts in `artifacts/phase1_fp32/`: `config_snapshot.yaml`, `dataset_metadata.json`, `preprocessing.json`, `best_fp32.pt`, `metrics.json`, `confusion_matrix.json`, and `summary.md`. The fixture files are generated in `data/processed/sensor_fixture/`.

Validation passed:
- `python3 -m compileall src scripts`
- `pytest` (14 passed)
- `make test-phase1` (6 passed)
- `make smoke`
- `make check`
- `make docs-check`
- `git diff --check`
- `make run-fp32-baseline`

Changed implementation/configuration: Phase 1 fixture, preprocessing, model, training, metrics/reporting, CLI, Make targets, dependency declarations, focused tests, and `configs/experiments/fp32_sensor_baseline.yaml`. Updated README and Phase 1 architecture, roadmap, data, experiment-policy, context, and results documentation.

Remaining limitations: no quantization, pruning, compiler lowering, Sparrow-V execution, hardware metrics, or real-world dataset evaluation. The next recommended milestone is exact integer reference inference and INT8 post-training quantization. Sparrow-V was not modified. No commit or push occurred.
