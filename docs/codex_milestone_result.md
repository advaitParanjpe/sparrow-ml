STATUS: COMPLETE

# Phase 6 multi-layer INT8 model and multi-operator export

Implemented the fixed CPU FP32 architecture `Linear(16,16) -> ReLU -> Linear(16,4)`. It has 340 parameters (`fc1`: 272, `fc2`: 68); the selected checkpoint is 3949 bytes. Training used seed 20260623, Adam learning rate 0.005, batch size 32, 50 epochs, deterministic CPU/DataLoader settings, and validation-loss checkpoint selection (best epoch 50). No test data was used for tuning.

Measured synthetic-fixture FP32 accuracy was 100% on train (360), validation (76), and test (76), with diagonal test confusion matrix `[[19,0,0,0],[0,19,0,0],[0,0,19,0],[0,0,0,19]]`. These are synthetic-fixture measurements, not general-model accuracy claims.

Input calibration used 360 training samples: scale 0.020786371756726364, zero point 0. Hidden ReLU calibration used the same training samples: min 0, max/max-abs 7.323272228240967, scale 0.05766356085229108, zero point 0, and zero clipped values. Both linear layers have signed per-output-channel INT8 weights and INT32 biases using incoming activation-scale times channel weight-scale. ReLU is reconstructed exactly from fc1 accumulator values, then hidden values use NumPy ties-to-even rounding and explicit clamp `[0,127]`.

Measured INT8 test accuracy and FP32/INT8 agreement were both 100%, with zero disagreements and zero hidden activation clipping. Test accumulator ranges were fc1 `[-77873, 78494]` and fc2 `[-34471, 24971]`, within signed INT32. Test final-logit error was max absolute 0.07607368794544023, mean absolute 0.020978947802905153, RMS 0.02604373826676825; test confusion matrix remained diagonal.

Added a backwards-compatible Phase 6 IR path with `DenseLinearInt8`, `ReLU`, `RequantizeInt8`, and `DenseLinearInt8`; it validates tensor producers/consumers, static shapes, order, and hidden range. The deterministic 528-byte package has four-byte alignment, an explicit 16-byte hidden buffer, and fits the 4096-byte configured scratchpad. It contains all required model, input, trace, symbolic-program, report, and manifest artifacts. Reload validation exactly reproduced decoded tensors, fc1/fc2 accumulators, hidden INT8 codes, and final predictions. Repeated export hashes matched byte-for-byte.

Validation passed:

- `python3 -m compileall src scripts`
- `pytest` (34 passed)
- `make test-phase1`, `test-phase2`, `test-phase3`, `test-phase4`, `test-phase5`, `test-phase6`
- `make smoke`, `make check`, `make docs-check`
- `git diff --check`
- `make run-multilayer-baseline`

Changed files: Phase 6 model, quantization/training/evaluation/export workflow, CLI, configuration, Make targets, focused tests, README, architecture/data/experiment/context/roadmap documentation, quantization contract, and Phase 6 results.

Remaining limitations: fixed `16->16->4` dense graph only; no sparse MLP, arbitrary graph compilation, or multi-layer Sparrow-V/RTL execution. Sparrow-V was not modified. No commit or push occurred. Recommended next milestone: multi-layer Sparrow-V runtime execution and intermediate RTL/reference validation.
