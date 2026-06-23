# Milestone result: Exact INT8 Post-Training Quantization and Integer Reference Inference

STATUS: COMPLETE

Implemented deterministic Phase 2 PTQ: train-only per-tensor symmetric INT8 input calibration (`max_abs / 127`, zero point 0), per-output-channel symmetric INT8 weights, INT32 accumulator-domain bias, and explicit signed integer affine reference inference. Rounding is NumPy `rint` (nearest, ties-to-even); values are clamped after rounding to `[-128, 127]`. Predictions use reconstructed per-channel logits, not raw accumulators.

Measured synthetic-fixture results:

- Calibration: train split, 360 samples; input scale `0.020786371756726364`; no clipped train values (0/5760).
- Weight scales: `[0.006949088704867626, 0.00609635931300366, 0.006888654757672408, 0.006897269271490142]`; 0 clipped values (0/64), 2 values at 127, 0 at -128.
- Quantized bias range: `[-7702, 7926]`.
- Observed accumulator range: `[-96519, 80747]`; conservative bound `270070`; observed and bound both fit signed INT32.
- FP32 / INT8 fixture accuracy and agreement: train `100% / 100% / 100%`; validation `100% / 100% / 100%`; test `100% / 100% / 100%`.
- Test confusion matrix: `[[19, 0, 0, 0], [0, 19, 0, 0], [0, 0, 19, 0], [0, 0, 0, 19]]`.
- Test logit error: maximum absolute `0.0473749891252524`, mean absolute `0.012607279240496778`, RMS `0.015673325599854963`.

Generated ignored artifacts: `artifacts/phase2_int8/config_snapshot.yaml`, `calibration_report.json`, `quantized_model.json`, `integer_evaluation_metrics.json`, `error_statistics.json`, `confusion_matrix.json`, `prediction_agreement.json`, and `summary.md`. A repeated `make run-int8-baseline` produced the same SHA-256 for `quantized_model.json`.

Validation passed: `python3 -m compileall src scripts`; `pytest` (19 passed); `make test-phase1` (6 passed); `make test-phase2` (5 passed); `make smoke`; `make check`; `make docs-check`; `git diff --check`; and `make run-int8-baseline`.

Changed implementation includes `src/sparrowml/quantization/`, Phase 2 CLI commands, Make targets, configuration, focused tests, and Phase 2 documentation. Existing unrelated worktree edits in `.gitignore` and `scripts/check_repo.py` were preserved. No Sparrow-V files were modified. No commit or push occurred.

Limitations: results are only synthetic-fixture measurements; no pruning, sparse packing, compiler lowering, Sparrow-V execution, QAT, or hardware measurement is implemented. Next recommended milestone: deterministic 2:4 structured pruning, sparse fine-tuning, and weight packing.
