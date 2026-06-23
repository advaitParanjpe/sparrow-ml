# Milestone: Complete WISDM Phase 8B Export and Phase 8C Sparrow-V Validation

## Objective

Resume the existing WISDM real-data milestone from its current partial implementation.

Phase 8A is already complete and must be treated as an accepted prerequisite.

The measured FP32 and INT8 model-quality results already pass the required gates.

This continuation milestone must:

1. preserve and validate the completed Phase 8A implementation and artifacts;
2. diagnose and fix the existing Phase 8B implementation error;
3. complete deterministic WISDM deployment-package export;
4. complete package-reload and intermediate-trace validation;
5. add focused Phase 8B tests;
6. run aggregate regression checks;
7. mark Phase 8B complete only after all gates pass;
8. begin Phase 8C only after Phase 8B passes;
9. deterministically select held-out WISDM test samples;
10. execute selected samples through the existing Phase 7 Sparrow-V workflow;
11. validate all intermediate and final integer values exactly;
12. generate consolidated WISDM model-quality and RTL-deployment reports.

Do not repeat completed dataset ingestion, feature extraction, or model tuning unless required by a reproducibility check.

## Existing Accepted Results

Treat these results as existing measured evidence that must be reproduced or preserved:

### Phase 8A

- 51 phone-accelerometer source files discovered;
- 49 eligible subjects;
- deterministic subject-level splits using seed `20260623`;
- 25,768 accepted windows;
- 80 samples per window;
- 50% overlap;
- four classes:
  - walking;
  - jogging;
  - sitting;
  - standing;
- test class counts:
  - walking: 931;
  - jogging: 932;
  - sitting: 797;
  - standing: 797;
- Phase 8A artifacts exist under:

```text
artifacts/phase8_wisdm/phase8a/
```

### Phase 8B measured model quality

- FP32 test macro-F1:

```text
0.9287458208759758
```

- FP32 test balanced accuracy:

```text
0.9296898801135173
```

- INT8 test macro-F1:

```text
0.9197794804065271
```

- INT8 macro-F1 drop:

```text
0.008966340469448664
```

- FP32/INT8 test prediction agreement:

```text
0.9872722013306335
```

These results pass the existing quality gates.

Do not perform a hyperparameter sweep.

## First Action: Diagnose Existing Failure

Read:

- `AGENTS.md`
- `docs/codex_context.md`
- `docs/current_milestone.md`
- `docs/codex_milestone_result.md`
- existing Phase 8A/8B implementation;
- current artifacts under `artifacts/phase8_wisdm/phase8a/` and `phase8b/`;
- the exact traceback or implementation error from the failed `run-wisdm-phase8b` execution.

Reproduce the failure once with the narrowest command that triggers it.

Record:

- failing function;
- failing file;
- exception type;
- root cause;
- fix.

Do not restart the entire pipeline before understanding the existing error.

## Phase 8A Preservation Gate

Before continuing Phase 8B, verify without regenerating unnecessarily:

- Phase 8A artifacts parse;
- subject splits remain disjoint;
- feature schema contains exactly 16 features;
- all required classes exist;
- processed feature data is available;
- prior Phase 8A focused tests pass.

Run:

```text
python3 -m pytest tests/test_phase8a.py
```

Phase 8A must remain complete.

## Complete Phase 8B

### Required outputs

Complete:

```text
artifacts/phase8_wisdm/phase8b/
```

At minimum it must contain:

```text
fp32_checkpoint.pt
training_metrics.json
preprocessing.json
input_calibration.json
hidden_calibration.json
quantized_model.json
integer_evaluation.json
model_quality.json
confusion_matrices.json
prediction_agreement.json
export/
summary.md
determinism.json
```

Exact filenames may follow current conventions, but equivalent evidence is required.

### Deployment package

Reuse the Phase 6 multi-layer exporter.

The WISDM package must contain:

```text
manifest.json
model_ir.json
memory_map.json
model_data.bin
input_data.bin
input.json
expected_output.json
intermediate_reference.json
program.json
export_report.json
README.md
```

Include:

- WISDM dataset identity;
- class ordering;
- feature ordering;
- subject-split identity;
- preprocessing statistics;
- quantization parameters;
- checkpoint identity;
- selected canonical test sample;
- expected first-layer accumulators;
- expected hidden INT8 codes;
- expected second-layer accumulators;
- final prediction;
- hashes.

Do not serialize absolute dataset paths.

### Package reload validation

Reload the generated package and require exact equality for:

- decoded input INT8 vector;
- both weight tensors;
- both bias tensors;
- all scales;
- first-layer accumulators;
- hidden INT8 vector;
- second-layer accumulators;
- final prediction.

Repeated export must be byte-for-byte deterministic.

### Phase 8B focused tests

Create or complete:

```text
tests/test_phase8b.py
```

Cover:

- WISDM feature-dataset loading;
- training-only standardization;
- metric calculations;
- calibration split isolation;
- quantized artifact schema;
- export package generation;
- package reload;
- intermediate-trace equality;
- deterministic hashes;
- rejection of absolute paths.

Tests must not retrain the full model repeatedly.

### Phase 8B gate

Phase 8B passes only if:

1. Phase 8A preservation checks pass.
2. The implementation error is fixed.
3. FP32 macro-F1 remains at least `0.75`.
4. FP32 balanced accuracy remains at least `0.75`.
5. INT8 macro-F1 drop remains no more than `0.03`.
6. FP32/INT8 agreement remains at least `0.95`.
7. Input and hidden calibration use training data only.
8. All accumulators fit signed INT32.
9. Hidden codes remain in `[0,127]`.
10. Deployment package export completes.
11. Package reload reproduces every intermediate exactly.
12. Export is deterministic.
13. Focused Phase 8B tests pass.
14. Required aggregate regression checks pass.

Do not begin Phase 8C unless every Phase 8B requirement passes.

## Phase 8C Sample Selection

Select samples only from held-out test subjects.

Use deterministic rules:

### Correct examples

Select:

```text
2 correctly classified INT8 examples per class
```

Choose the lowest canonical window IDs satisfying the rule.

Expected target:

```text
8 correctly classified examples
```

### Misclassified examples

Select:

```text
up to 1 INT8-misclassified example per true class
```

Choose the lowest canonical window ID for each class where available.

Maximum preferred total:

```text
12 samples
```

Record:

- window ID;
- subject ID;
- true class;
- FP32 prediction;
- INT8 prediction;
- selection reason.

Do not select any training or validation example.

## Phase 8C Sparrow-V Execution

Reuse the Phase 7 multi-layer runtime adapter.

For each selected sample:

1. prepare four deterministic `fc1` partitions;
2. run all four through Sparrow-V RTL simulation;
3. parse raw dot-product accumulators;
4. apply documented host-side INT32 bias reconstruction;
5. validate all 16 post-bias `fc1` accumulators exactly;
6. reconstruct ReLU and hidden requantization;
7. validate all 16 hidden INT8 codes exactly;
8. execute `fc2`;
9. apply documented bias reconstruction;
10. validate all four `fc2` accumulators exactly;
11. reconstruct final logits;
12. validate the final INT8 prediction exactly;
13. retain measured and derived counters.

A model misclassification is allowed.

RTL correctness is defined as agreement with the SparrowML INT8 reference, not agreement with the ground-truth activity label.

## Phase 8C runtime efficiency

Avoid recompiling invariant RTL unnecessarily if the existing adapter can safely reuse builds.

Do not compromise workspace isolation or correctness.

Do not run the entire test set through RTL.

## Phase 8C outputs

Generate:

```text
artifacts/phase8_wisdm/phase8c/
```

including:

```text
selected_samples.json
per_sample/
rtl_validation_summary.json
counter_summary.json
model_quality_summary.json
deployment_summary.json
final_results.json
summary.md
determinism.json
```

## Consolidated final report

Generate a final real-data report containing:

### Dataset

- subject counts;
- split subject IDs or stable split hash;
- window counts;
- class counts;
- feature schema;
- leakage checks.

### Model quality

- FP32 accuracy;
- FP32 macro-F1;
- FP32 balanced accuracy;
- INT8 accuracy;
- INT8 macro-F1;
- INT8 balanced accuracy;
- per-class metrics;
- confusion matrices;
- prediction agreement.

### Quantization

- input and hidden scales;
- clipping;
- accumulator ranges;
- logit error;
- checkpoint and model sizes.

### Deployment

- IR/package identity;
- package size;
- scratchpad usage;
- selected RTL sample count;
- exact first-layer match count;
- exact hidden-code match count;
- exact second-layer match count;
- exact prediction match count.

### Runtime counters

- per-sample and aggregate partitioned cycles;
- retired instructions;
- vector loads;
- dense dot instructions;
- conceptual multiplication count;
- measured versus derived provenance.

Clearly state that partitioned simulation totals are not optimized monolithic latency.

## CLI and Make targets

Complete or add:

```text
sparrowml run-wisdm-phase8b
sparrowml select-wisdm-rtl-samples
sparrowml run-wisdm-phase8c
sparrowml run-wisdm-final
```

Make targets:

```text
run-wisdm-phase8b
test-phase8b
select-wisdm-rtl-samples
run-wisdm-phase8c
test-phase8c
test-phase8c-integration
run-wisdm-final
```

The gated final command must:

1. verify existing Phase 8A completion;
2. run and validate Phase 8B;
3. stop if Phase 8B fails;
4. run Phase 8C;
5. generate final consolidated results.

## Validation

Use focused checks during development.

At final acceptance run:

```text
python3 -m compileall src scripts
pytest
make test-phase1
make test-phase2
make test-phase3
make test-phase4
make test-phase5
make test-phase6
make test-phase7
make test-phase8a
make test-phase8b
make test-phase8c
make smoke
make check
make docs-check
git diff --check
```

Run local integrations:

```text
make test-phase8a-integration
make run-wisdm-phase8b
make test-phase8c-integration
make run-wisdm-phase8c
make run-wisdm-final
```

Verify:

```text
git -C "$SPARROWV_ROOT" status --short
```

Sparrow-V must remain clean.

## Documentation

Complete or add:

```text
docs/wisdm_data_contract.md
docs/wisdm_evaluation_protocol.md
docs/results/phase8a_wisdm_dataset.md
docs/results/phase8b_wisdm_model.md
docs/results/phase8c_wisdm_rtl.md
docs/results/final_wisdm_results.md
```

Update:

- `README.md`
- `docs/architecture.md`
- `docs/build_roadmap.md`
- `docs/codex_context.md`
- `docs/data_contracts.md`
- `docs/experiment_policy.md`

## Repository safety

Never commit:

- raw WISDM files;
- processed feature rows;
- trained checkpoints;
- generated deployment packages;
- simulator workspaces;
- raw subject sensor data.

Ensure generated paths remain ignored.

## Acceptance Criteria

This continuation milestone is complete only when:

1. Existing Phase 8A remains valid.
2. The Phase 8B implementation error is diagnosed and fixed.
3. Existing real-data quality metrics are preserved or validly reproduced.
4. Phase 8B export completes.
5. Package reload reproduces all intermediate values exactly.
6. Phase 8B focused tests pass.
7. Aggregate Phase 1–8B checks pass.
8. Phase 8C begins only after Phase 8B passes.
9. Selected samples come only from held-out subjects.
10. Selection is deterministic.
11. At least eight correct examples are selected where available.
12. Misclassified examples are included where available.
13. Every selected sample runs through Sparrow-V.
14. Every first-layer accumulator matches exactly.
15. Every hidden INT8 code matches exactly.
16. Every second-layer accumulator matches exactly.
17. Every final prediction matches the INT8 reference.
18. Sparrow-V remains clean.
19. Consolidated real-data results are generated.
20. Documentation matches implementation.
21. No raw dataset is committed.
22. `git diff --check` passes.
23. No commit or push occurs.
24. `docs/codex_milestone_result.md` is finalized.

## Stop Conditions

Stop only if:

- the existing Phase 8A artifacts are corrupt or scientifically invalid;
- the Phase 8B error exposes a fundamental incompatible package contract;
- model metrics cannot be reproduced within documented determinism limits;
- package reload cannot reproduce integer traces;
- Sparrow-V cannot execute the WISDM package;
- a Sparrow-V source modification appears necessary;
- a major prior-phase correctness defect is discovered.

Ordinary implementation errors, missing tests, schema mismatches, export bugs, and report bugs are not stop conditions.

## Token-Efficiency Instructions

Follow `AGENTS.md`.

In particular:

- do not regenerate Phase 8A unless validation fails;
- reproduce the Phase 8B failure once, then fix it;
- reuse the already trained checkpoint where valid;
- avoid retraining unless required;
- inspect only WISDM Phase 8 and reused Phase 6/7 modules;
- do not broadly inspect Sparrow-V;
- do not run RTL until Phase 8B is complete;
- use focused tests first;
- run aggregate validation once;
- keep the result concise.

## Result File

Update:

```text
docs/codex_milestone_result.md
```

Include:

```text
PHASE_8A_STATUS: COMPLETE
PHASE_8B_STATUS: COMPLETE
PHASE_8C_STATUS: COMPLETE
STATUS: COMPLETE
```

only if every phase passes.

Include:

- original Phase 8B root cause;
- fix;
- final FP32 and INT8 results;
- package/export identity;
- selected RTL samples;
- exact-match totals;
- counter totals;
- determinism evidence;
- validation results;
- changed files;
- limitations;
- confirmation that raw WISDM data was not committed;
- confirmation that Sparrow-V remained clean;
- confirmation that no commit or push occurred.