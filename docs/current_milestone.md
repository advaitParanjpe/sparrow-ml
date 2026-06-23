# Milestone: WISDM Real-Data Integration, INT8 Evaluation, Sparrow-V Validation, and Final Project Results

## Objective

Complete SparrowML’s real-data evaluation using the locally downloaded WISDM smartphone activity dataset.

This milestone is divided into three strictly gated stages:

1. **Phase 8A — WISDM ingestion, subject-safe splitting, windowing, feature extraction, and audit**
2. **Phase 8B — FP32 training, INT8 quantization, evaluation, and deployment-package export**
3. **Phase 8C — Selected real-sample Sparrow-V RTL validation and consolidated results**

Codex must complete, validate, and document each phase before moving to the next.

Do not begin Phase 8B unless every Phase 8A acceptance criterion passes.

Do not begin Phase 8C unless every Phase 8B acceptance criterion passes.

If a phase is blocked or fails, stop immediately, finalize the milestone result with the relevant status, and do not implement later phases.

## Current Project State

SparrowML already provides:

- deterministic synthetic-data training;
- fixed `16 → 16 → 4` MLP support;
- FP32 training;
- train-only input and hidden activation calibration;
- signed INT8 model inputs;
- signed INT8 per-output-channel weights;
- INT32 biases and accumulators;
- explicit multi-layer integer reference inference;
- `ReLU`;
- hidden activation requantization to `[0,127]`;
- multi-operator SparrowML IR;
- deterministic deployment packages;
- Sparrow-V external workload integration;
- exact multi-layer RTL/reference validation;
- deterministic Phase 1–7 workflows.

This milestone must reuse the existing model, quantization, compiler, package, runtime, and RTL-validation infrastructure.

Do not redesign SparrowML.

## Dataset Location

The WISDM dataset has already been downloaded locally.

Expected base location:

```text
~/Datasets/WISDM/wisdm-dataset
```

Also support:

```text
$WISDM_ROOT
```

Resolution order:

1. explicit `WISDM_ROOT`;
2. `~/Datasets/WISDM/wisdm-dataset`;
3. clear failure with actionable instructions.

Resolve the absolute path internally, but never serialize machine-specific absolute paths into canonical artifacts, manifests, hashes, or tracked documentation.

The raw dataset must remain outside Git.

## Dataset Choice

Use only:

```text
device: smartphone
sensor: accelerometer
```

Use exactly four activity classes:

```text
walking
jogging
sitting
standing
```

Read the WISDM activity-key metadata when available.

Do not silently guess activity-code mappings.

If names differ slightly in the source metadata, normalize them explicitly and document the source-to-canonical mapping.

Do not use:

- smartwatch signals;
- gyroscope signals;
- stairs;
- sensor fusion;
- biometric identification;
- unsupported classes.

## Scientific Rules

The evaluation must be subject-independent.

Never randomly split windows across train, validation, and test.

All windows from one subject must belong to exactly one split.

The test split must remain untouched until final evaluation.

No test subject, test window, test statistic, or test label may influence:

- feature standardization;
- class weighting;
- hyperparameters;
- checkpoint selection;
- calibration;
- thresholding;
- sample selection rules, except after final model evaluation for choosing representative RTL-validation examples.

Results must be described as WISDM subject-held-out evaluation.

---

# Phase 8A — WISDM Ingestion, Subject Splits, Windowing, Features, and Audit

## Phase 8A Goal

Build a deterministic, leakage-safe, auditable WISDM feature dataset with exactly 16 features per window.

Phase 8A performs no model training.

Do not begin Phase 8B until the complete Phase 8A gate passes.

## Phase 8A Relevant Files

Read first:

- `AGENTS.md`
- `docs/codex_context.md`
- `docs/current_milestone.md`
- `docs/architecture.md`
- `docs/data_contracts.md`
- `docs/experiment_policy.md`
- existing fixture, preprocessing, dataset, model, and evaluation code.

Inspect the actual WISDM directory structure narrowly.

Do not scan unrelated user directories.

## Phase 8A Dataset Discovery

Implement a diagnostic command such as:

```text
sparrowml wisdm-doctor
```

It must report:

- resolved dataset root;
- whether activity-key metadata exists;
- phone accelerometer directory discovered;
- number of subject files;
- discovered subject IDs;
- discovered activity codes and mapped names;
- malformed-row count;
- missing required activities;
- whether the dataset is usable.

Do not modify raw files.

## Phase 8A Parsing

Parse the raw smartphone accelerometer records into a canonical internal structure containing:

- subject ID;
- canonical activity name;
- source activity code;
- timestamp;
- x acceleration;
- y acceleration;
- z acceleration;
- source file identity;
- source row number.

Handle common formatting issues safely, including:

- trailing semicolons;
- blank lines;
- malformed rows;
- invalid numerical fields;
- duplicate records;
- non-monotonic timestamps;
- missing subjects;
- unknown activity codes.

Malformed records must be counted and reported.

Unknown activity codes must not be silently assigned.

## Phase 8A Subject Splits

Create deterministic subject-level splits.

Preferred split sizes for 51 subjects:

```text
35 train subjects
8 validation subjects
8 test subjects
```

Use fixed seed:

```text
20260623
```

Before finalizing splits, verify every selected subject has sufficient data for all four required activities.

If some subjects lack required activities, use a deterministic eligibility filter and then allocate approximately:

```text
70% train subjects
15% validation subjects
15% test subjects
```

Requirements:

- no subject appears in multiple splits;
- subject lists are saved;
- split generation is deterministic;
- class availability by subject is reported;
- split sizes are reported;
- the exact subject allocation is retained in artifacts;
- no window-level reassignment between splits.

## Phase 8A Segmentation

Segment continuous activity recordings into windows.

Default:

```text
sampling rate: use dataset metadata or measured nominal rate
window duration: 4 seconds
expected window length at 20 Hz: 80 samples
overlap: 50%
stride: 40 samples
```

A window must not cross:

- subject boundaries;
- activity boundaries;
- source-file boundaries;
- large timestamp discontinuities.

Reject incomplete or invalid windows.

Record for every window:

- stable window ID;
- subject ID;
- activity;
- split;
- source file identity;
- source row range;
- start timestamp;
- end timestamp;
- sample count;
- rejection status or acceptance status.

Document timestamp-gap policy explicitly.

## Phase 8A Feature Extraction

Convert each accepted triaxial accelerometer window into exactly 16 deterministic float features.

First derive magnitude:

```text
m = sqrt(x^2 + y^2 + z^2)
```

Use this feature order:

1. `x_mean`
2. `y_mean`
3. `z_mean`
4. `magnitude_mean`
5. `x_std`
6. `y_std`
7. `z_std`
8. `magnitude_std`
9. `magnitude_min`
10. `magnitude_max`
11. `magnitude_rms`
12. `magnitude_energy`
13. `magnitude_mean_absolute_difference`
14. `magnitude_zero_crossing_rate`
15. `magnitude_dominant_frequency_magnitude`
16. `magnitude_spectral_entropy`

Definitions must be explicit and tested.

### Zero-crossing rate

Subtract the window’s magnitude mean before counting crossings.

Define handling of exact zero deterministically.

### Frequency features

Use the actual or nominal sample rate consistently.

Remove the mean before the FFT where appropriate.

Exclude the DC bin when choosing the dominant-frequency magnitude.

Normalize spectral power before calculating spectral entropy.

Handle zero-energy windows safely.

## Phase 8A Feature Dataset

Produce a canonical processed feature dataset containing:

- stable window ID;
- subject ID;
- activity;
- integer class ID;
- split;
- 16 ordered feature values;
- source provenance;
- sampling-rate metadata;
- feature extractor version.

Preferred generated location:

```text
data/processed/wisdm/
```

Generated data must remain ignored unless repository policy explicitly permits a small metadata-only artifact.

Do not commit raw or processed WISDM samples.

## Phase 8A Audit

Generate an audit containing:

- raw subject count;
- eligible subject count;
- excluded subjects and reasons;
- source files parsed;
- raw rows per class;
- malformed rows;
- unknown activities;
- accepted windows per subject;
- accepted windows per class;
- accepted windows per split;
- rejected windows by reason;
- feature finiteness;
- duplicate window IDs;
- subject overlap checks;
- class balance;
- feature minimum, maximum, mean, and standard deviation;
- exact subject lists.

Flag serious imbalance.

Do not automatically rebalance before the audit is reviewed by the pipeline.

## Phase 8A Artifacts

Use:

```text
artifacts/phase8_wisdm/phase8a/
```

Generate at minimum:

```text
dataset_discovery.json
activity_mapping.json
subject_eligibility.json
subject_splits.json
recording_manifest.json
window_manifest.json
feature_schema.json
feature_statistics.json
dataset_audit.json
summary.md
```

Canonical artifacts must avoid absolute paths.

## Phase 8A CLI

Add commands such as:

```text
sparrowml wisdm-doctor
sparrowml prepare-wisdm
sparrowml audit-wisdm
sparrowml run-wisdm-phase8a
```

`run-wisdm-phase8a` must:

1. resolve the dataset;
2. parse records;
3. filter activities;
4. create subject splits;
5. create windows;
6. extract 16 features;
7. validate provenance;
8. write the audit;
9. run the Phase 8A acceptance gate.

## Phase 8A Tests

Add focused tests for:

- activity-key parsing;
- raw-row parsing;
- semicolon cleanup;
- malformed rows;
- unknown activity codes;
- deterministic subject splitting;
- no subject overlap;
- subject eligibility;
- activity-boundary handling;
- timestamp-gap handling;
- exact window length;
- stable window IDs;
- all 16 feature definitions;
- FFT edge cases;
- zero-energy windows;
- finite feature outputs;
- deterministic processed output;
- audit totals;
- no absolute paths in canonical artifacts.

Tests must use tiny synthetic WISDM-like fixtures.

Normal tests must not require the full WISDM dataset.

Add separately marked local-data integration tests.

## Phase 8A Make Targets

Add:

```text
wisdm-doctor
prepare-wisdm
audit-wisdm
run-wisdm-phase8a
test-phase8a
test-phase8a-integration
```

## Phase 8A Acceptance Gate

Phase 8A passes only if:

1. The local WISDM root is discovered.
2. Smartphone accelerometer data is discovered.
3. Activity metadata is parsed or a documented source mapping is used.
4. All four activities are available.
5. Subjects are identified.
6. Eligible-subject logic is deterministic.
7. Subject-level train, validation, and test splits exist.
8. No subject overlap exists.
9. No accepted window crosses a subject boundary.
10. No accepted window crosses an activity boundary.
11. No accepted window crosses a source-file boundary.
12. Timestamp discontinuities are handled.
13. Every accepted window has the required sample count.
14. Every example contains exactly 16 features.
15. Every feature is finite.
16. Feature order is fixed and versioned.
17. Class ordering is fixed and documented.
18. Stable window IDs exist.
19. Provenance exists for every example.
20. Audit counts reconcile.
21. Each split contains all four classes.
22. Each class has a meaningful number of windows in every split.
23. Raw and processed data remain untracked.
24. Phase 8A unit tests pass.
25. Phase 8A local-data integration tests pass.
26. Documentation accurately describes the dataset and split.
27. No model training has occurred.

If any Phase 8A requirement fails:

- do not begin Phase 8B;
- finalize `docs/codex_milestone_result.md`;
- use `STATUS: BLOCKED` for a dataset/provenance decision requiring human input;
- use `STATUS: FAILED` for an implementation failure.

---

# Phase 8B — FP32 Training, INT8 Evaluation, and Deployment Export

## Phase 8B Entry Condition

Begin Phase 8B only after Phase 8A has passed in the same run.

Record the Phase 8A gate result before performing training.

## Phase 8B Goal

Train the existing `16→16→4` MLP on the WISDM feature dataset, quantize it using training-only calibration, evaluate it on held-out subjects, and export a deterministic deployment package.

## Phase 8B Model

Reuse:

```text
Linear(16,16)
ReLU
Linear(16,4)
```

Class order:

```text
walking
jogging
sitting
standing
```

Do not change the model architecture unless the existing architecture cannot achieve the minimum acceptance level.

Such a failure is a stop condition requiring a concise report, not permission to introduce a larger model automatically.

## Phase 8B Preprocessing

Fit standardization statistics using training subjects only.

Apply the same statistics to validation and test examples.

Persist:

- feature order;
- train mean;
- train standard deviation;
- zero-variance handling;
- preprocessing version.

No validation or test statistic may influence preprocessing.

## Phase 8B Class Imbalance

First report class imbalance.

Use ordinary cross-entropy by default.

Use deterministic class-weighted cross-entropy only if the training split is materially imbalanced.

If class weights are used:

- derive them from training data only;
- record the exact formula;
- record the values;
- do not tune them on test data.

Do not use resampling unless required.

## Phase 8B Training

Use deterministic CPU training.

Suggested starting configuration:

```text
seed: 20260623
epochs: 100
batch_size: 32
learning_rate: 0.005
optimizer: Adam
selection metric: validation macro-F1, with validation loss as tie-break
```

This is one bounded training configuration, not a sweep.

Early stopping may be used with a documented patience.

Do not tune repeatedly against the test set.

## Phase 8B Real-Data Metrics

Report for train, validation, and test:

- accuracy;
- macro precision;
- macro recall;
- macro-F1;
- balanced accuracy;
- per-class precision;
- per-class recall;
- per-class F1;
- confusion matrix;
- sample count;
- subject count;
- predicted class distribution.

For test results, also report metrics by held-out subject where practical.

## Phase 8B FP32 Quality Gate

Minimum project acceptance target:

```text
test macro-F1 >= 0.75
```

Also require:

```text
test balanced accuracy >= 0.75
```

Interpretation:

- `0.75–0.84`: acceptable;
- `0.85–0.89`: good;
- `>=0.90`: strong.

These are internal project gates, not benchmark guarantees.

Do not manipulate splits to reach them.

If the gate is missed:

- report the result honestly;
- inspect for implementation or leakage errors;
- do not proceed to Phase 8C;
- stop for human review before expanding the model or feature set.

## Phase 8B Quantization

Reuse the Phase 6 quantization scheme:

- signed INT8 standardized inputs;
- per-tensor symmetric input scale;
- signed INT8 per-output-channel weights;
- INT32 biases;
- reconstructed `fc1` activations;
- ReLU;
- hidden signed INT8 codes in `[0,127]`;
- per-tensor hidden activation scale;
- signed INT8 per-output-channel `fc2` weights;
- INT32 `fc2` biases.

Input and hidden calibration must use training data only.

## Phase 8B INT8 Evaluation

Report:

- FP32 test metrics;
- INT8 test metrics;
- FP32/INT8 prediction agreement;
- disagreement count;
- per-class agreement;
- input clipping;
- hidden clipping;
- `fc1` accumulator range;
- `fc2` accumulator range;
- final-logit error.

Required quantization gates:

```text
INT8 test macro-F1 drop versus FP32 <= 0.03
INT8 prediction agreement with FP32 >= 0.95
all accumulators fit signed INT32
hidden codes remain in [0,127]
```

## Phase 8B Export

Reuse and extend the existing Phase 6 package exporter.

Generate a deterministic WISDM package containing:

- class ordering;
- preprocessing statistics;
- activity mapping;
- trained FP32 checkpoint identity;
- quantized model;
- model IR;
- memory map;
- model binary;
- selected canonical test input;
- intermediate reference trace;
- expected outputs;
- symbolic program;
- dataset/split provenance;
- package hashes.

Use:

```text
artifacts/phase8_wisdm/phase8b/
```

## Phase 8B Determinism

Require deterministic:

- subject splits;
- feature dataset;
- checkpoint selection in the tested environment;
- quantized model;
- exported package;
- reference inference;
- canonical hashes.

Document software-version limitations for exact training reproducibility.

## Phase 8B CLI

Add commands such as:

```text
sparrowml train-wisdm
sparrowml evaluate-wisdm
sparrowml quantize-wisdm
sparrowml export-wisdm
sparrowml run-wisdm-phase8b
```

## Phase 8B Tests

Add focused tests for:

- WISDM dataset loading;
- training-only preprocessing;
- class-weight calculation;
- real-data metric calculations;
- macro-F1;
- balanced accuracy;
- checkpoint selection;
- input calibration leakage prevention;
- hidden calibration leakage prevention;
- FP32/INT8 comparison;
- package provenance;
- deterministic export;
- no absolute paths.

Normal tests should use tiny prepared feature fixtures.

## Phase 8B Make Targets

Add:

```text
train-wisdm
evaluate-wisdm
quantize-wisdm
export-wisdm
run-wisdm-phase8b
test-phase8b
```

## Phase 8B Acceptance Gate

Phase 8B passes only if:

1. Phase 8A passed.
2. Preprocessing uses training subjects only.
3. Training is deterministic within documented limits.
4. Validation controls checkpoint selection.
5. Test data is not used for tuning.
6. FP32 train metrics are reported.
7. FP32 validation metrics are reported.
8. FP32 test metrics are reported.
9. Test macro-F1 is at least 0.75.
10. Test balanced accuracy is at least 0.75.
11. Per-class metrics are reported.
12. Test confusion matrix is reported.
13. Subject counts are reported.
14. Input calibration uses training data only.
15. Hidden calibration uses training data only.
16. INT8 inference works.
17. INT8 test macro-F1 drop is no more than 0.03.
18. FP32/INT8 agreement is at least 95%.
19. Accumulator ranges fit INT32.
20. Hidden codes remain in `[0,127]`.
21. Clipping is reported.
22. A deterministic WISDM deployment package is generated.
23. Package reload reproduces intermediate and final reference values exactly.
24. Phase 8B tests pass.
25. Earlier Phase 1–7 unit tests remain passing.
26. No Sparrow-V execution has occurred during Phase 8B.

If Phase 8B fails:

- do not begin Phase 8C;
- finalize the result;
- do not enlarge the model automatically.

---

# Phase 8C — Selected Real-Sample Sparrow-V Validation and Final Results

## Phase 8C Entry Condition

Begin Phase 8C only after Phase 8B passes.

## Phase 8C Goal

Execute a bounded, deterministic set of real WISDM test examples through the existing multi-layer Sparrow-V RTL-validation workflow.

Dataset-wide statistical evaluation remains a software task.

RTL simulation proves deployment equivalence.

## Phase 8C Sample Selection

Choose samples only from the held-out test subjects.

Select deterministically:

```text
2 correctly classified examples per class
```

Target:

```text
8 correct examples total
```

Additionally select:

```text
up to 1 FP32 or INT8 misclassified example per class
```

where such examples exist.

Maximum preferred total:

```text
12 examples
```

Selection rules must be deterministic and recorded.

Do not cherry-pick only unusually easy examples beyond the stated rule.

For correct examples, choose stable examples such as the lowest canonical window IDs satisfying the rule.

For misclassified examples, choose the lowest canonical window ID per class.

## Phase 8C Sparrow-V Execution

Reuse the Phase 7 workflow.

For each selected sample:

1. prepare four `fc1` partitions if required;
2. run real Sparrow-V RTL simulation;
3. parse raw `fc1` dot products;
4. reconstruct full INT32 biases with explicit provenance;
5. validate all 16 `fc1` accumulators;
6. reconstruct ReLU and hidden quantization;
7. validate all 16 hidden INT8 codes;
8. execute `fc2`;
9. validate all four final accumulators;
10. reconstruct final logits;
11. validate the INT8 software prediction;
12. retain counters and logs.

No integer mismatch is permitted.

## Phase 8C Correctness Gate

For every selected sample require:

- exact `fc1` accumulator match;
- exact hidden INT8 match;
- exact `fc2` accumulator match;
- exact SparrowML INT8 prediction match;
- successful simulator exit;
- no assertion failure;
- no unexpected trap;
- package identity verified;
- Sparrow-V working tree clean.

An RTL run may correctly reproduce a model misclassification.

Do not treat agreement with the ground-truth label as the RTL correctness criterion.

The RTL correctness criterion is agreement with the SparrowML INT8 reference.

## Phase 8C Counters

Report per sample:

- simulator invocation count;
- partitioned cycles;
- retired instructions;
- vector loads;
- dense dot-product count;
- derived conceptual multiplications;
- unavailable counters.

Aggregate carefully across the selected sample set.

Continue to label summed cycles as:

```text
partitioned simulation cycle total
```

Do not claim optimized monolithic latency.

## Phase 8C Results

Generate consolidated comparisons containing:

### Model quality

- FP32 accuracy and macro-F1;
- INT8 accuracy and macro-F1;
- per-class metrics;
- confusion matrices;
- prediction agreement.

### Model representation

- FP32 checkpoint size;
- INT8 model-data size;
- deployment-package size;
- scratchpad usage;
- parameter count.

### Compute

- `fc1` conceptual multiplications;
- `fc2` conceptual multiplications;
- total conceptual multiplications;
- simulator cycles;
- retired instructions;
- vector loads;
- dot-product instructions.

### Deployment correctness

- selected sample count;
- exact `fc1` match count;
- exact hidden match count;
- exact `fc2` match count;
- exact prediction match count;
- semantic determinism evidence.

Clearly separate:

- statistical model quality;
- compiler/package correctness;
- RTL deployment equivalence;
- partitioned simulation performance.

## Phase 8C Artifacts

Use:

```text
artifacts/phase8_wisdm/phase8c/
```

Generate:

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

## Phase 8C CLI

Add commands such as:

```text
sparrowml select-wisdm-rtl-samples
sparrowml run-wisdm-rtl
sparrowml validate-wisdm-rtl
sparrowml run-wisdm-phase8c
```

## Phase 8C Make Targets

Add:

```text
select-wisdm-rtl-samples
run-wisdm-rtl
run-wisdm-phase8c
test-phase8c
test-phase8c-integration
```

## Phase 8C Tests

Add offline tests for:

- deterministic sample selection;
- correct/misclassified selection rules;
- no non-test sample selection;
- result aggregation;
- mismatch handling;
- ground-truth versus reference distinction;
- report consistency.

Add separately marked real integration tests requiring:

- WISDM processed data;
- Phase 8B package;
- Sparrow-V checkout;
- simulator tools.

## Phase 8C Acceptance Gate

Phase 8C passes only if:

1. Phase 8A passed.
2. Phase 8B passed.
3. Selected samples come only from test subjects.
4. Selection follows the documented deterministic rule.
5. At least two correctly classified samples per class are selected where available.
6. Misclassified examples are included where available.
7. Every selected example runs through Sparrow-V.
8. Every `fc1` accumulator matches exactly.
9. Every hidden INT8 code matches exactly.
10. Every `fc2` accumulator matches exactly.
11. Every RTL prediction matches the SparrowML INT8 prediction.
12. Simulator failures are absent.
13. Counters are reported honestly.
14. Partitioned-cycle limitations are documented.
15. Sparrow-V remains clean.
16. Consolidated model-quality results exist.
17. Consolidated deployment results exist.
18. Phase 8C offline tests pass.
19. Phase 8C integration tests pass.
20. Repeated selected-sample execution is semantically deterministic for a bounded subset or all selected samples.
21. No physical-hardware claim is made.
22. No monolithic-latency claim is made.

---

# Cross-Phase Configuration

Add a primary configuration:

```text
configs/experiments/wisdm_activity_recognition.yaml
```

Include:

- dataset root resolution policy;
- device;
- sensor;
- selected classes;
- activity mapping;
- subject eligibility policy;
- split seed and proportions;
- window length;
- stride;
- timestamp-gap policy;
- feature schema version;
- training settings;
- class-weight policy;
- quantization settings;
- quality gates;
- RTL sample-selection policy;
- output directories.

Avoid absolute paths.

## Full Pipeline Command

Add one gated command:

```text
sparrowml run-wisdm-final
```

and Make target:

```text
make run-wisdm-final
```

It must:

1. run Phase 8A;
2. stop immediately if Phase 8A fails;
3. record the Phase 8A gate result;
4. run Phase 8B;
5. stop immediately if Phase 8B fails;
6. record the Phase 8B gate result;
7. run Phase 8C;
8. generate the final report.

Do not continue after a failed gate.

## Documentation

Update or add:

```text
docs/wisdm_data_contract.md
docs/wisdm_evaluation_protocol.md
docs/results/phase8a_wisdm_dataset.md
docs/results/phase8b_wisdm_model.md
docs/results/phase8c_wisdm_rtl.md
docs/results/final_wisdm_results.md
```

Also update:

- `README.md`;
- `docs/architecture.md`;
- `docs/build_roadmap.md`;
- `docs/codex_context.md`;
- `docs/data_contracts.md`;
- `docs/experiment_policy.md`.

Document:

- official dataset identity;
- local raw-data policy;
- selected sensor and classes;
- subject-held-out split;
- window policy;
- exact 16 features;
- leakage prevention;
- class balance;
- real-data metrics;
- quantization results;
- sample-selection rules;
- RTL correctness;
- limitations.

## README Status

Only after Phase 8C passes, update the project status to:

```text
Real WISDM activity-recognition model trained, quantized, compiled, and validated through Sparrow-V RTL simulation
```

Do not claim:

- medical or safety certification;
- real-time physical-device deployment;
- physical FPGA or ASIC execution;
- arbitrary activity recognition;
- arbitrary neural-network compilation;
- monolithic optimized multi-layer latency.

## Repository and Data Safety

Never commit:

- WISDM archive;
- raw WISDM files;
- processed feature samples;
- trained checkpoints;
- generated deployment packages;
- RTL logs;
- subject-specific raw data.

Ensure relevant generated paths are ignored.

Tracked documentation may contain aggregate statistics and subject IDs if repository policy permits, but no raw signal data.

## Full Validation

During development, use focused tests.

At final acceptance, run once:

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

Run local-data and RTL integrations:

```text
make test-phase8a-integration
make run-wisdm-phase8a
make run-wisdm-phase8b
make test-phase8c-integration
make run-wisdm-phase8c
make run-wisdm-final
```

Verify Sparrow-V remains clean:

```text
git -C "$SPARROWV_ROOT" status --short
```

## Global Acceptance Criteria

The milestone is complete only when:

1. Phase 8A passes before Phase 8B begins.
2. Phase 8B passes before Phase 8C begins.
3. Gating is enforced in code, not only documentation.
4. WISDM raw data is discovered locally.
5. Smartphone accelerometer data is used.
6. Exactly four documented classes are used.
7. Subject-independent splits are used.
8. No subject leakage exists.
9. Window provenance exists.
10. Exactly 16 deterministic features exist.
11. Phase 8A audit passes.
12. FP32 MLP training succeeds.
13. Real held-out test metrics are reported.
14. Test macro-F1 is at least 0.75.
15. Test balanced accuracy is at least 0.75.
16. INT8 model evaluation succeeds.
17. INT8 macro-F1 drop is no more than 0.03.
18. FP32/INT8 agreement is at least 95%.
19. Deployment package reload is exact.
20. Real test samples are selected deterministically.
21. Real selected samples execute through Sparrow-V.
22. All intermediate and final integer values match exactly.
23. Sparrow-V remains unmodified.
24. Dataset, model, quantization, and RTL results are clearly separated.
25. Earlier Phase 1–7 behavior remains passing.
26. Tests require no internet.
27. Dataset integration uses only local files.
28. No raw dataset is committed.
29. Documentation matches implementation.
30. Final consolidated results exist.
31. `git diff --check` passes.
32. No commit or push occurs.
33. `docs/codex_milestone_result.md` is finalized.

## Stop Conditions

Stop for human review if:

- the local WISDM directory cannot be identified;
- activity metadata is missing and mappings cannot be established confidently;
- too few eligible subjects contain all required activities;
- a subject-safe split cannot place all four classes in every split;
- accepted windows are too sparse for credible training;
- Phase 8A audit exposes unresolved provenance or leakage problems;
- FP32 test macro-F1 remains below 0.75 after one sound bounded training configuration;
- INT8 degradation exceeds the gate;
- Phase 8B export cannot reproduce reference traces;
- Sparrow-V cannot execute the selected Phase 8B package;
- a Sparrow-V source change appears necessary;
- a major Phase 1–7 correctness defect is found.

Do not automatically:

- change activity classes;
- alter the subject split;
- add more features;
- enlarge the model;
- add sensor modalities;
- tune repeatedly against the test set.

## Token-Efficiency Instructions

Follow `AGENTS.md`.

In particular:

- inspect only the WISDM phone-accelerometer tree;
- avoid loading the entire raw dataset repeatedly;
- cache parsed manifests and processed features;
- finish and validate Phase 8A before reading training/compiler code deeply;
- do not begin Phase 8B until the Phase 8A gate passes;
- do not inspect Sparrow-V until Phase 8C;
- reuse Phase 6 and Phase 7 infrastructure;
- do not build a generic dataset framework;
- do not perform a hyperparameter sweep;
- run focused tests during development;
- run aggregate checks once;
- keep the milestone result concise but complete.

## Result File

Update:

```text
docs/codex_milestone_result.md
```

throughout the run.

The result must contain separate sections:

```text
PHASE_8A_STATUS
PHASE_8B_STATUS
PHASE_8C_STATUS
```

Use:

```text
STATUS: COMPLETE
```

only if all three phases pass.

Include:

- resolved dataset structure;
- activity mapping;
- subject eligibility and splits;
- window counts;
- feature schema;
- audit findings;
- FP32 metrics;
- INT8 metrics;
- prediction agreement;
- quantization diagnostics;
- deployment package identity;
- RTL sample-selection summary;
- exact RTL/reference match totals;
- counter summary;
- validation commands;
- changed files;
- limitations;
- confirmation that WISDM data was not committed;
- confirmation that Sparrow-V remained clean;
- confirmation that no commit or push occurred.

Use `STATUS: BLOCKED` for a genuine dataset, scientific-protocol, or external-interface decision.

Use `STATUS: FAILED` for incomplete implementation or failed required checks.

## Next Step After Completion

After this milestone passes, the remaining project work should be limited to:

```text
final repository polish, consolidated diagrams, reproducibility guide, portfolio presentation, and optional research extensions
```

Do not implement those optional extensions during this milestone.