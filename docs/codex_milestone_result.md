STATUS: COMPLETE
MILESTONE: Complete WISDM Phase 8B Export and Phase 8C Sparrow-V Validation
PHASE_8A_STATUS: COMPLETE
PHASE_8B_STATUS: COMPLETE
PHASE_8C_STATUS: COMPLETE

Phase 8A preservation passed: parseable artifacts, 49 eligible subjects, disjoint 35/7/7 subject splits, 16 features, four classes, 25,768 windows, and `tests/test_phase8a.py` passed.

Original Phase 8B failure: `sparrowml.wisdm_pipeline.run` in `src/sparrowml/wisdm_pipeline.py` only invoked training and integer evaluation; it had no export/reload stage, so `run-wisdm-phase8b` completed without the mandated deployment package. The narrow reproduction completed with the preserved artifacts and no runtime exception; the failure was a missing implementation path rather than a surviving traceback. Fix: add Phase 8A gate, train-only calibration evidence, deterministic Phase 6-compatible package export, full package reload validation, trace/hash validation, and the Phase 8B command gate.

Measured quality preserved: FP32 accuracy 0.9259473531964131, macro-F1 0.9287458208759758, balanced accuracy 0.9296898801135173. INT8 accuracy 0.9175585768006942, macro-F1 0.9197794804065271, balanced accuracy 0.920638703760132; macro-F1 drop 0.008966340469448664 and agreement 0.9872722013306335. Calibration uses training data only. Test accumulator ranges are fc1 [-35610, 38794], fc2 [-21935, 19149], and hidden codes remain [0,127].

Phase 8B package: `artifacts/phase8_wisdm/phase8b/export`; manifest SHA-256 `fb3469601540eed281107ea8162a5eb31415846fe863c083bb4a4f5a693b3a41`, checkpoint SHA-256 `cfc427d0456d1884db0d8a151f43d76029663e3e9f3eb642e45afa411abc7f31`. Repeated export hashes are equal and package reload exactly reproduced inputs, weights, biases, scales, fc1/fc2 accumulators, hidden codes, and predictions. Absolute dataset paths are rejected by contract.

Phase 8C selected 12 held-out samples: eight lowest canonical correct examples (two per class) and four lowest canonical INT8 errors (one per true class). All 12 executed through Sparrow-V. Exact matches: fc1 12/12, hidden codes 12/12, fc2 12/12, predictions 12/12. Aggregate measured counters: fc1 partitions 23,232 cycles, 5,232 retired instructions, 1,536 vector loads, 768 dense dots; fc2 5,808 cycles, 1,308 retired instructions, 384 vector loads, 192 dense dots. Derived conceptual multiplications: 3,072 fc1 and 768 fc2. These are partitioned simulation totals, not optimized monolithic latency.

Validation passed: `python3 -m compileall src scripts`; `pytest` (44 passed, 1 optional integration skipped); `make test-phase1` through `make test-phase8c`; `make test-phase8a-integration`; `make run-wisdm-phase8b`; `make test-phase8c-integration`; `make run-wisdm-phase8c`; `make run-wisdm-final`; `make smoke`; `make check`; `make docs-check`; and `git diff --check`.

Changed files cover WISDM preparation/pipeline, CLI/Make targets, multi-sample Phase 7 adapter, focused tests, and WISDM contracts/results documentation. Generated WISDM data, checkpoints, packages, and workspaces remain ignored; no raw WISDM data was committed. Sparrow-V status was clean. No commit or push occurred. Limitation: RTL uses isolated four-partition fc1 and one fc2 runs with host bias/ReLU/requantization reconstruction; counters are not monolithic hardware latency.
