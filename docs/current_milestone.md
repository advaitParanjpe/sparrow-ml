# Milestone: Exact INT8 Post-Training Quantization and Integer Reference Inference

## Objective

Extend SparrowML’s deterministic FP32 baseline into a reproducible INT8 post-training quantization pipeline with exact integer reference inference.

This milestone must:

1. load the trained FP32 `Linear(16, 4)` checkpoint;
2. calibrate activation ranges using training data only;
3. quantize model inputs and weights to signed INT8;
4. quantize biases into the INT32 accumulator domain;
5. execute integer-only affine inference;
6. dequantize logits for comparison with FP32;
7. report quantization error, saturation, accuracy, and prediction agreement;
8. emit stable quantized-model artifacts for the later 2:4 pruning and Sparrow-V export milestones.

This milestone is about quantization correctness and integer semantics.

Do not implement structured pruning, sparse packing, compiler lowering, Sparrow-V execution, TinyNPU integration, or quantization-aware training.

## Baseline

Phase 1 currently provides:

- a deterministic synthetic vibration-fault-style fixture;
- 512 samples;
- 16 finite features per sample;
- four classes:
  - `normal`
  - `inner`
  - `outer`
  - `ball`
- seed `20260623`;
- balanced deterministic splits:
  - 360 train;
  - 76 validation;
  - 76 test;
- train-only standardization;
- preprocessing version `standardize_train_v1`;
- a CPU FP32 `Linear(16, 4)` model;
- 68 trainable parameters;
- deterministic training and checkpoint selection;
- FP32 fixture accuracy of 100% on train, validation, and test;
- generated Phase 1 artifacts under `artifacts/phase1_fp32/`;
- passing Phase 1 and repository checks.

The 100% result is synthetic fixture accuracy only and is not a real-world model-quality claim.

## Relevant Context

Read first:

- `AGENTS.md`
- `docs/codex_context.md`
- `docs/current_milestone.md`
- `docs/architecture.md`
- `docs/data_contracts.md`
- `docs/experiment_policy.md`
- `docs/results/phase1_fp32_baseline.md`
- `configs/experiments/fp32_sensor_baseline.yaml`
- Phase 1 fixture, preprocessing, model, training, evaluation, and artifact modules.

Inspect only directly relevant files after that.

Do not perform a broad repository audit unless a concrete failure requires it.

## Quantization Contract

Implement explicit affine quantization semantics.

For a real-valued tensor `x`:

```text
q = clamp(round(x / scale) + zero_point, qmin, qmax)
```

For dequantization:

```text
x_hat = scale × (q - zero_point)
```

Use signed INT8 ranges:

```text
qmin = -128
qmax = 127
```

Document exactly:

- rounding behavior;
- clamping behavior;
- scale derivation;
- zero-point derivation;
- accumulator width;
- bias quantization;
- requantization or dequantization behavior.

Do not rely on opaque framework quantization kernels as the only implementation.

The milestone must have an inspectable integer reference path.

## Quantization Scheme

Use a simple, Sparrow-V-compatible scheme.

### Input activations

Use signed INT8 affine or symmetric quantization.

Preferred initial scheme:

```text
per-tensor symmetric INT8
zero_point = 0
```

Calibration must use only the training split.

The activation scale should be based on a documented range policy, such as:

```text
max_abs / 127
```

If a percentile or clipping policy is used, it must be configurable and documented.

### Weights

Use signed INT8 symmetric quantization.

Preferred scheme:

```text
per-output-channel symmetric INT8
```

For `Linear(16, 4)`, each output row receives its own weight scale.

This is preferred because it is simple, accurate, and compatible with later hardware export.

If per-tensor weight quantization is chosen instead, justify it and record the accuracy trade-off.

### Biases

Quantize each FP32 bias into signed INT32 using:

```text
bias_scale[channel] = input_scale × weight_scale[channel]
```

Then:

```text
bias_int32[channel] = round(bias_fp32[channel] / bias_scale[channel])
```

Validate against signed INT32 bounds.

### Accumulation

Compute exact integer logits as:

```text
acc[channel] =
    bias_int32[channel]
    + Σ input_int8[i] × weight_int8[channel, i]
```

Use signed INT32 or wider host-side arithmetic for safety, while validating that deployed accumulator values fit signed INT32.

The integer accumulator must not use FP32 multiplication internally.

### Output reconstruction

For each output channel:

```text
logit_fp32_approx[channel] =
    acc[channel] × input_scale × weight_scale[channel]
```

Predicted class is:

```text
argmax(acc)
```

only if per-channel scale differences do not invalidate direct accumulator comparison.

Because per-channel weight scales may differ, the canonical predicted class must be computed from reconstructed real-valued logits unless a mathematically equivalent common-scale transform is implemented and documented.

Do not incorrectly apply `argmax` directly across differently scaled accumulators.

## Calibration

Implement deterministic calibration using training data only.

Calibration must report:

- minimum input value;
- maximum input value;
- maximum absolute value;
- selected activation scale;
- zero point;
- number and percentage of clipped values;
- calibration sample count;
- calibration split.

Validation and test data must not influence calibration.

Persist calibration metadata.

## Exact Integer Reference

Create a standalone integer inference implementation that accepts:

- quantized INT8 input vector;
- quantized INT8 weight matrix;
- INT32 bias vector;
- activation scale;
- per-channel weight scales;
- class metadata.

It must output:

- INT32 accumulators;
- reconstructed FP32 logits;
- predicted class;
- saturation or overflow diagnostics.

The implementation must not call `torch.nn.Linear` for the integer computation.

PyTorch or NumPy may be used for tensor storage, but the arithmetic semantics must remain explicit and testable.

## FP32 Comparison

For every train, validation, and test sample, compare:

- FP32 logits;
- dequantized INT8 logits;
- FP32 predicted class;
- INT8 predicted class;
- expected label.

Report:

- split-level FP32 fixture accuracy;
- split-level INT8 fixture accuracy;
- FP32/INT8 prediction agreement;
- number of prediction disagreements;
- maximum absolute logit error;
- mean absolute logit error;
- root mean square logit error;
- per-output-channel error statistics.

Keep metrics small and directly relevant.

## Saturation and Overflow Reporting

Report:

### Input quantization

- total values quantized;
- values clipped to `-128`;
- values clipped to `127`;
- total clipped values;
- clipping percentage.

### Weight quantization

- total values quantized;
- values at `-128`;
- values at `127`;
- per-channel scale;
- zero-scale handling.

### Bias and accumulator

- minimum and maximum quantized bias;
- minimum and maximum observed accumulator;
- whether all observed accumulators fit signed INT32;
- theoretical conservative accumulator bound;
- whether the theoretical bound fits signed INT32.

Fail clearly if any bias or accumulator exceeds the supported range.

## Quantization Error Gates

Use bounded acceptance gates.

Required:

- INT8 test fixture accuracy must be at least 95%;
- INT8 test fixture accuracy must not drop by more than 2 percentage points from FP32;
- FP32/INT8 test prediction agreement must be at least 98%;
- all observed accumulators must fit signed INT32;
- no NaN or infinity in reconstructed logits;
- quantization artifacts must be deterministic.

Because the fixture is simple, 100% INT8 fixture accuracy may occur. Report it honestly without presenting it as real-world performance.

Do not change the test set to satisfy the gate.

## Quantized Artifact Format

Define a stable machine-readable quantized model artifact.

Preferred format:

```text
JSON manifest + binary or JSON tensor payloads
```

For this small model, a single JSON file is acceptable if it remains readable and deterministic.

The artifact must include:

- format version;
- model name;
- source FP32 checkpoint identity;
- feature count;
- class count;
- class names;
- quantization scheme;
- input scale;
- input zero point;
- weight scales;
- weight zero points;
- INT8 weight matrix;
- INT32 bias vector;
- tensor shapes;
- lane/order conventions;
- preprocessing version;
- calibration split and sample count;
- accumulator type;
- creation configuration;
- optional content hashes.

Use repository-relative paths.

Do not include machine-specific absolute paths.

## Generated Artifacts

Use an output directory such as:

```text
artifacts/phase2_int8/
```

Generated artifacts should remain ignored unless repository policy explicitly tracks a small golden fixture.

At minimum generate:

- configuration snapshot;
- calibration report;
- quantized model artifact;
- integer-evaluation metrics JSON;
- error-statistics JSON;
- confusion matrix;
- prediction-agreement report;
- human-readable Markdown summary.

Do not overwrite Phase 1 artifacts.

## Configuration

Add a dedicated Phase 2 configuration, preferably:

```text
configs/experiments/int8_ptq_baseline.yaml
```

Include:

- source Phase 1 checkpoint path;
- source preprocessing metadata path;
- input quantization scheme;
- weight quantization scheme;
- calibration policy;
- clipping policy;
- accumulator type;
- acceptance thresholds;
- output directory;
- deterministic seed.

Avoid machine-specific absolute paths.

## CLI

Extend the CLI with bounded commands such as:

```text
sparrowml calibrate-int8
sparrowml quantize-int8
sparrowml evaluate-int8
sparrowml run-int8-baseline
```

Exact names may be adjusted to match existing conventions.

Preferred behavior:

### `calibrate-int8`

- load Phase 1 fixture and preprocessing;
- use training split only;
- emit activation calibration metadata.

### `quantize-int8`

- load the best FP32 checkpoint;
- quantize weights and biases;
- emit the quantized-model artifact;
- validate tensor ranges.

### `evaluate-int8`

- run exact integer reference inference;
- compare with FP32;
- emit metrics and summaries.

### `run-int8-baseline`

- perform calibration, quantization, evaluation, and reporting in one reproducible command.

Use proper exit codes.

## Package Structure

Add only the necessary modules.

A reasonable structure is:

```text
src/sparrowml/
├── quantization/
│   ├── __init__.py
│   ├── affine.py
│   ├── calibration.py
│   ├── weights.py
│   ├── bias.py
│   ├── integer_reference.py
│   └── artifacts.py
└── evaluation/
    └── quantization_metrics.py
```

Adjust to fit the existing package.

Do not add a generalized graph quantization framework.

## Tests

Add focused tests for:

### Quantization primitives

- scale calculation;
- zero-point calculation;
- symmetric INT8 behavior;
- rounding policy;
- clamping;
- `-128` and `127`;
- zero-valued tensor handling;
- deterministic output.

### Per-channel weights

- correct output-channel axis;
- one scale per output;
- correct INT8 shape;
- reconstruction error;
- zero-row handling;
- range validation.

### Bias quantization

- correct combined scale;
- correct INT32 values;
- overflow rejection;
- zero-scale rejection.

### Integer inference

- hand-computed small example;
- signed multiplication;
- negative values;
- INT32 accumulation;
- reconstructed logits;
- per-channel output scales;
- correct prediction semantics.

### Calibration

- training split only;
- stable sample count;
- deterministic scale;
- no validation/test leakage;
- clipping statistics.

### Artifact schemas

- required fields;
- tensor dimensions;
- integer ranges;
- scale positivity;
- class consistency;
- relative paths;
- unsupported format version.

### Evaluation

- confusion matrix shape;
- agreement calculation;
- error metrics;
- saturation counts;
- accumulator range checks.

### CLI

- commands parse;
- smoke quantization run succeeds;
- missing Phase 1 checkpoint fails clearly;
- invalid config fails clearly.

Tests must not:

- require internet;
- require a GPU;
- require Sparrow-V;
- modify Sparrow-V;
- retrain the full FP32 model repeatedly.

Use the existing Phase 1 checkpoint if present, or create minimal temporary models in focused tests.

## Make Targets

Add stable targets such as:

```text
calibrate-int8
quantize-int8
evaluate-int8
run-int8-baseline
test-phase2
```

Update `make help`.

Recommended:

```text
make test-phase2
```

runs focused Phase 2 tests only.

Do not include the full FP32 retraining run inside every Phase 2 test.

The aggregate baseline may depend on existing Phase 1 artifacts and fail clearly with instructions if they are missing.

## Documentation

Update or add:

- `README.md`;
- `docs/architecture.md`;
- `docs/build_roadmap.md`;
- `docs/data_contracts.md`;
- `docs/experiment_policy.md`;
- `docs/codex_context.md`;
- one Phase 2 results document.

Suggested:

```text
docs/results/phase2_int8_ptq.md
```

Document:

- quantization equations;
- rounding and clipping semantics;
- calibration policy;
- input scale and zero point;
- per-channel weight scales;
- bias quantization;
- exact integer accumulation;
- prediction semantics with per-channel scales;
- fixture accuracy;
- prediction agreement;
- error metrics;
- saturation;
- accumulator range;
- artifact format;
- limitations;
- reproduction commands.

Keep `docs/codex_context.md` concise.

## README Status

Update README status to:

```text
Phase 2 INT8 post-training quantization implemented
```

Do not claim:

- structured sparsity;
- compiler lowering;
- Sparrow-V deployment;
- hardware acceleration;
- real-world accuracy;
- quantization-aware training.

## Existing Behavior Preservation

Preserve:

- Phase 1 fixture generation;
- Phase 1 FP32 training;
- Phase 1 evaluation;
- all existing CLI commands;
- all existing tests;
- repository contracts;
- Sparrow-V target boundary.

Do not modify Sparrow-V.

## Out of Scope

Do not implement:

- 2:4 pruning;
- sparse metadata;
- compressed sparse weights;
- QAT;
- distillation;
- dynamic quantization;
- mixed precision;
- hidden layers;
- multi-layer models;
- compiler IR;
- code generation;
- Sparrow-V execution;
- TinyNPU support;
- ONNX;
- hardware cost models;
- target selection;
- research experiments;
- dataset downloads;
- hyperparameter sweeps.

## Validation

During development, run focused tests.

At final acceptance, run once:

```text
python3 -m compileall src scripts
pytest
make test-phase1
make test-phase2
make smoke
make check
make docs-check
git diff --check
```

Also run once:

```text
make run-int8-baseline
```

If Phase 1 artifacts are missing, regenerate them once with:

```text
make run-fp32-baseline
```

Do not repeatedly retrain during unrelated checks.

## Acceptance Criteria

The milestone is complete only when:

1. Training-only activation calibration exists.
2. Calibration metadata records its split and sample count.
3. Input INT8 quantization exists.
4. Input scale is positive and deterministic.
5. Input zero point is documented.
6. Weight INT8 quantization exists.
7. Weight quantization is per-output-channel or explicitly justified otherwise.
8. Weight scales are positive and deterministic.
9. Biases are quantized into the INT32 accumulator domain.
10. Bias scales equal input scale times output-channel weight scale.
11. Explicit integer affine inference exists.
12. Integer inference does not use FP32 matrix multiplication.
13. Signed INT8 products accumulate correctly.
14. Accumulators are validated against signed INT32.
15. Reconstructed logits are produced.
16. Prediction semantics account for per-channel scales correctly.
17. FP32 and INT8 metrics are compared.
18. Train INT8 fixture accuracy is reported.
19. Validation INT8 fixture accuracy is reported.
20. Test INT8 fixture accuracy is reported.
21. Prediction agreement is reported.
22. Confusion matrix is reported.
23. Maximum absolute logit error is reported.
24. Mean absolute logit error is reported.
25. RMS logit error is reported.
26. Input clipping statistics are reported.
27. Weight saturation statistics are reported.
28. Bias range is reported.
29. Accumulator range is reported.
30. Theoretical accumulator bound is reported.
31. Test INT8 fixture accuracy is at least 95%.
32. Test accuracy drop from FP32 is no more than 2 percentage points.
33. Test prediction agreement is at least 98%.
34. Quantized artifacts are deterministic.
35. Quantized model schema is validated.
36. One command reproduces calibration, quantization, and evaluation.
37. Phase 1 behavior remains passing.
38. Phase 2 focused tests pass.
39. Tests require no internet.
40. Tests require no GPU.
41. Tests require no Sparrow-V checkout.
42. Documentation matches implementation.
43. README status is accurate.
44. No pruning or sparse packing is implemented.
45. No compiler or hardware execution is implemented.
46. Sparrow-V is not modified.
47. General repository checks pass.
48. Documentation checks pass.
49. `git diff --check` passes.
50. No commit or push occurs.
51. `docs/codex_milestone_result.md` is finalized.

## Stop Conditions

Stop for human review only if:

- the Phase 1 checkpoint or preprocessing artifacts cannot be loaded;
- calibration cannot be performed without validation/test leakage;
- the chosen quantization scheme cannot be represented with signed INT8 weights and activations plus INT32 bias;
- prediction semantics cannot be made correct with per-channel scales;
- observed or theoretical accumulation exceeds signed INT32;
- the fixture accuracy gate cannot be met without modifying the test split;
- a major Phase 1 correctness defect is discovered.

Ordinary quantization error, scale bugs, artifact issues, test failures, and documentation work are not stop conditions.

## Token-Efficiency Instructions

Follow `AGENTS.md`.

In particular:

- read compact context and milestone first;
- inspect only Phase 1 and quantization-relevant files;
- do not explore future compiler or target code;
- avoid framework-wide abstractions;
- run focused tests during development;
- run aggregate checks once;
- reuse the existing Phase 1 checkpoint;
- do not repeatedly retrain;
- keep the result file concise.

## Result File

Update:

```text
docs/codex_milestone_result.md
```

throughout the run.

Finalize with `STATUS: COMPLETE` only if every required criterion and validation passes.

Include:

- quantization schemes;
- calibration split and sample count;
- input scale and zero point;
- per-channel weight scales;
- quantized bias range;
- observed and theoretical accumulator ranges;
- train/validation/test FP32 and INT8 fixture accuracy;
- test prediction agreement;
- confusion matrix;
- quantization error metrics;
- saturation statistics;
- artifact paths;
- exact commands and outcomes;
- changed files;
- remaining limitations;
- next recommended milestone;
- confirmation that Sparrow-V was not modified;
- confirmation that no commit or push occurred.

Use `STATUS: FAILED` if required work or checks remain incomplete.

Use `STATUS: BLOCKED` only for a genuine human decision or architectural blocker.

## Next Milestone

The expected next milestone is:

```text
Deterministic 2:4 structured pruning, sparse fine-tuning, and weight packing
```

Do not implement it during this milestone.