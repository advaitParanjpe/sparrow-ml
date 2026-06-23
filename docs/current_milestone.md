# Milestone: Deterministic 2:4 Structured Pruning, Sparse Fine-Tuning, and Weight Packing

## Objective

Extend SparrowML’s deterministic INT8 baseline into a reproducible 2:4 structured-sparse model pipeline.

This milestone must:

1. load the Phase 2 quantized `Linear(16, 4)` model;
2. apply deterministic 2:4 pruning to every consecutive group of four weights;
3. preserve the selected sparsity mask during optional fine-tuning;
4. compare dense INT8 and sparse INT8 inference;
5. encode all sparse groups using Sparrow-V-compatible metadata;
6. pack two retained INT8 weights per group;
7. exactly reconstruct the sparse dense-equivalent matrix;
8. run explicit compressed sparse integer reference inference;
9. report accuracy, prediction agreement, logit error, sparsity, operation count, and storage;
10. emit stable sparse artifacts for later compiler and Sparrow-V deployment milestones.

This milestone is about structured sparsity correctness, sparse-model quality, and packing contracts.

Do not implement compiler IR, program generation, Sparrow-V execution, TinyNPU integration, new hardware instructions, or hardware-aware pruning yet.

## Baseline

Phase 1 provides:

- a deterministic 512-sample synthetic sensor fixture;
- 16 input features;
- four classes:
  - `normal`
  - `inner`
  - `outer`
  - `ball`;
- train, validation, and test splits;
- train-only standardization;
- a deterministic FP32 `Linear(16, 4)` classifier.

Phase 2 provides:

- training-only activation calibration;
- signed per-tensor symmetric INT8 inputs;
- signed per-output-channel symmetric INT8 weights;
- INT32 bias quantization;
- explicit integer reference inference;
- reconstructed per-channel logits;
- deterministic quantized-model artifacts;
- 100% FP32 and INT8 synthetic-fixture accuracy and prediction agreement;
- no activation clipping in the reference run;
- observed and conservative accumulators within signed INT32.

The current model contains:

```text
4 output channels × 16 input weights = 64 INT8 weights
```

Each output row contains four consecutive groups of four weights:

```text
16 weights ÷ 4 = 4 groups per output
```

Across four outputs:

```text
4 groups × 4 outputs = 16 sparse groups
```

## Relevant Context

Read first:

- `AGENTS.md`
- `docs/codex_context.md`
- `docs/current_milestone.md`
- `docs/architecture.md`
- `docs/data_contracts.md`
- `docs/experiment_policy.md`
- `docs/results/phase1_fp32_baseline.md`
- `docs/results/phase2_int8_ptq.md`
- `configs/experiments/int8_ptq_baseline.yaml`
- Phase 2 quantization artifacts and implementation modules.

Inspect only directly relevant files after that.

Do not explore Sparrow-V implementation files during this milestone.

## Sparse Structure

Apply 2:4 structured sparsity independently to every consecutive four-weight group in each output row.

For a group:

```text
[w0, w1, w2, w3]
```

retain exactly two positions and set exactly two positions to zero.

The grouping axis must be the input-feature dimension.

For `Linear(16, 4)`:

- output channel remains unchanged;
- each output row is divided into groups `[0:4]`, `[4:8]`, `[8:12]`, `[12:16]`.

Document the axis and ordering explicitly.

## Deterministic Pruning Rule

For each group:

1. compute absolute magnitude of each weight;
2. retain the two weights with largest absolute magnitude;
3. if magnitudes tie, retain lower lane indices first;
4. order selected lanes in ascending lane-index order;
5. set all unselected positions to integer zero.

Example:

```text
weights = [5, -9, 9, 2]
magnitudes = [5, 9, 9, 2]
selected lanes = [1, 2]
```

For complete ties:

```text
weights = [4, -4, 4, -4]
selected lanes = [0, 1]
```

The implementation must be deterministic across repeated runs.

Do not use random pruning.

## Metadata Encoding

Use Sparrow-V-compatible 3-bit metadata:

| Metadata | Selected lanes |
|---|---|
| `000` | `{0,1}` |
| `001` | `{0,2}` |
| `010` | `{0,3}` |
| `011` | `{1,2}` |
| `100` | `{1,3}` |
| `101` | `{2,3}` |

Reserve:

- `110`
- `111`

as invalid.

For every group:

- compressed weight 0 maps to the lower selected lane;
- compressed weight 1 maps to the higher selected lane.

The sparse artifact must never emit invalid metadata.

## Compressed Weight Representation

For each four-weight group, store:

```text
compressed_weight_0: signed INT8
compressed_weight_1: signed INT8
metadata: 3 bits
```

The packed representation must preserve:

- output-channel ordering;
- group ordering;
- lane ordering;
- signed INT8 values;
- deterministic metadata ordering.

For the current model:

```text
16 groups × 2 weights = 32 compressed INT8 weights
```

Raw compressed-weight storage:

```text
32 bytes
```

Raw metadata storage:

```text
16 groups × 3 bits = 48 bits
```

Packed metadata storage:

```text
6 bytes
```

Total sparse weight representation:

```text
32 + 6 = 38 bytes
```

Dense weight storage:

```text
64 bytes
```

Expected reduction:

```text
(64 - 38) / 64 = 40.625%
```

Biases and scales must be reported separately and must not be silently included in only one side of the comparison.

## Sparse Dense-Equivalent Matrix

Create a sparse dense-equivalent INT8 matrix with the same shape as the dense quantized matrix:

```text
[4, 16]
```

It must contain:

- retained INT8 values in selected positions;
- exact integer zero in pruned positions.

Require:

```text
decompress(compressed_weights, metadata)
==
sparse_dense_equivalent_weights
```

for every output channel and group.

The comparison must be exact integer equality.

## Sparse Integer Reference Inference

Implement two independent sparse inference paths.

### Dense-form sparse inference

Use the sparse dense-equivalent matrix:

```text
acc_sparse_dense[channel] =
    bias_int32[channel]
    + Σ input_int8[i] × sparse_weight_int8[channel, i]
```

### Compressed sparse inference

For each four-feature group:

1. decode metadata;
2. select two input lanes;
3. multiply selected input INT8 values by the two compressed INT8 weights;
4. accumulate only those two products.

Require exact equality:

```text
compressed_sparse_accumulators
==
sparse_dense_form_accumulators
```

for every sample and output channel.

Do not implement compressed sparse inference by first decompressing and then calling the dense path.

The compressed path must explicitly decode metadata and execute two products per group.

## Operation Accounting

For the current model, per sample:

### Dense INT8

```text
64 executed multiplications
0 skipped multiplications
```

### Sparse INT8

```text
16 groups × 2 executed = 32 executed multiplications
16 groups × 2 skipped = 32 skipped multiplications
```

Report:

- executed multiplications;
- skipped multiplications;
- total possible dense multiplications;
- arithmetic reduction percentage.

Expected sparse arithmetic reduction:

```text
50%
```

The operation count must come from the sparse structure, not a hardcoded summary without validation.

## Sparse Fine-Tuning

Implement a bounded optional mask-preserving fine-tuning stage.

Preferred sequence:

1. begin from the Phase 1 FP32 checkpoint or Phase 2 quantized model, according to the simplest sound design;
2. derive a deterministic 2:4 mask;
3. apply the mask to FP32 weights;
4. fine-tune for a small bounded number of epochs;
5. reapply the same mask after every optimizer step;
6. quantize the fine-tuned sparse FP32 weights using the existing Phase 2 scheme;
7. evaluate sparse INT8 inference.

Requirements:

- the selected mask must not change during fine-tuning;
- pruned weights must remain exactly zero;
- no test data may influence checkpoint selection;
- use validation loss or validation accuracy for sparse checkpoint selection;
- fixed seeds;
- CPU-only support;
- bounded runtime.

Recommended default:

```text
epochs: 10
learning rate: 1e-3
```

Adjust only if needed for stable behavior.

If direct post-quantization pruning already meets all quality gates, still implement the fine-tuning path, but report both:

- sparse INT8 before fine-tuning;
- sparse INT8 after fine-tuning.

Do not perform a broad hyperparameter sweep.

## Mask Semantics

Define a stable mask representation containing:

- tensor shape;
- output channel;
- group index;
- selected lane indices;
- metadata value;
- binary mask values.

Require:

- exactly two ones per group;
- exactly two zeros per group;
- 32 retained weights total;
- 32 pruned weights total.

The mask must be reproducible from the same source model and configuration.

## Quality Comparisons

Compare:

1. FP32 dense;
2. INT8 dense;
3. INT8 sparse before fine-tuning;
4. INT8 sparse after fine-tuning.

For each split report:

- fixture accuracy;
- confusion matrix;
- prediction agreement with dense INT8;
- prediction agreement with FP32;
- number of disagreements;
- maximum absolute logit difference from dense INT8;
- mean absolute logit difference;
- RMS logit difference.

Keep claims explicitly scoped to the synthetic fixture.

## Acceptance Quality Gates

Required after sparse fine-tuning:

- sparse INT8 test fixture accuracy at least 95%;
- sparse INT8 test accuracy drop versus dense INT8 no more than 5 percentage points;
- sparse/dense INT8 test prediction agreement at least 95%;
- compressed and dense-form sparse accumulators match exactly;
- all observed sparse accumulators fit signed INT32;
- every group follows 2:4 structure;
- generated sparse artifacts are deterministic.

If pre-fine-tuning sparse quality already passes, report it but do not skip implementation of mask-preserving fine-tuning.

Do not modify the test set to meet the gate.

## Scale Handling

Preserve the Phase 2 quantization contract.

Use:

- existing input scale and zero point;
- per-output-channel weight scales;
- INT32 bias domain;
- reconstructed per-channel logits for prediction.

If sparse fine-tuning changes FP32 weights, regenerate sparse per-channel weight scales and sparse INT32 biases consistently.

Document whether dense and sparse models use:

- independent per-channel scales; or
- shared dense scales.

Prefer independent sparse scales if this improves correctness and is clearly represented in the artifact.

Do not compare raw accumulators across models with different per-channel scales as though they share a common real-value domain.

## Accumulator Safety

Report for sparse inference:

- minimum and maximum observed accumulator;
- conservative accumulator bound;
- signed INT32 fit status.

The conservative sparse bound should account for:

- two executed products per group;
- four groups per output;
- quantized bias.

Fail if observed or conservative values exceed signed INT32.

## Sparse Artifact Format

Define a stable machine-readable sparse-model artifact.

Preferred:

```text
JSON manifest with embedded small tensor payloads
```

The artifact must include:

- format version;
- model name;
- source dense INT8 artifact identity;
- feature count;
- class count;
- class names;
- grouping axis;
- group size;
- nonzero count per group;
- pruning rule;
- tie-breaking rule;
- mask;
- compressed INT8 weights;
- metadata values;
- packed metadata bytes;
- per-channel sparse weight scales;
- input scale and zero point;
- INT32 biases;
- tensor shapes;
- accumulator type;
- preprocessing version;
- fine-tuning configuration;
- storage accounting;
- optional hashes.

Use repository-relative paths only.

## Packing Format

Implement deterministic metadata packing.

Define and document:

- group traversal order;
- bit order inside metadata values;
- byte packing order;
- padding-bit behavior;
- expected packed length.

For 16 metadata values:

```text
16 × 3 = 48 bits = 6 bytes
```

Require:

- pack → unpack round-trip;
- no nonzero padding bits;
- deterministic byte output;
- invalid metadata rejected.

## Generated Artifacts

Use:

```text
artifacts/phase3_sparse/
```

Generated artifacts should remain ignored unless repository policy says otherwise.

At minimum generate:

- configuration snapshot;
- pruning mask;
- sparse FP32 or sparse checkpoint;
- sparse quantized-model artifact;
- packed metadata binary or JSON representation;
- storage report;
- sparse evaluation metrics;
- sparse confusion matrices;
- prediction-agreement report;
- arithmetic-count report;
- human-readable Markdown summary.

Do not overwrite Phase 1 or Phase 2 artifacts.

## Configuration

Add:

```text
configs/experiments/sparse_2of4_baseline.yaml
```

Include:

- source Phase 1 checkpoint;
- source Phase 2 quantized artifact;
- group size;
- nonzero count;
- grouping axis;
- pruning rule;
- tie-breaking rule;
- fine-tuning epochs;
- learning rate;
- seed;
- acceptance thresholds;
- output directory;
- metadata packing version.

Avoid machine-specific paths.

## CLI

Add bounded commands such as:

```text
sparrowml prune-2of4
sparrowml finetune-sparse
sparrowml pack-sparse
sparrowml evaluate-sparse
sparrowml run-sparse-baseline
```

Exact command names may follow existing conventions.

### `prune-2of4`

- load source model;
- create deterministic mask;
- emit sparse dense-equivalent weights;
- report pattern distribution.

### `finetune-sparse`

- use fixed mask;
- preserve zeros;
- select best checkpoint using validation data;
- emit sparse checkpoint.

### `pack-sparse`

- quantize sparse model;
- generate compressed weights;
- generate metadata;
- pack metadata;
- verify exact decompression.

### `evaluate-sparse`

- run dense-form sparse inference;
- run compressed sparse inference;
- compare with dense INT8;
- emit quality and storage reports.

### `run-sparse-baseline`

- perform pruning, fine-tuning, quantization, packing, evaluation, and reporting in one deterministic command.

Use clear error messages and proper exit codes.

## Package Structure

Add only necessary modules.

Suggested:

```text
src/sparrowml/
├── sparsity/
│   ├── __init__.py
│   ├── pruning.py
│   ├── masks.py
│   ├── finetune.py
│   ├── metadata.py
│   ├── packing.py
│   ├── integer_reference.py
│   └── artifacts.py
└── evaluation/
    └── sparsity_metrics.py
```

Do not build a general-purpose pruning framework.

## Pattern Distribution

Report how often each legal metadata pattern occurs:

```text
000
001
010
011
100
101
```

Require:

- total pattern count equals 16;
- invalid patterns never occur.

Do not force equal pattern distribution.

## Tests

Add focused tests for:

### Pruning

- exactly two retained weights per group;
- exactly two zeros per group;
- correct grouping axis;
- largest-magnitude selection;
- lower-index tie-breaking;
- negative values;
- zero values;
- `-128`;
- `127`;
- deterministic masks.

### Metadata

- all six legal patterns;
- lane-pair encoding;
- lane-pair decoding;
- invalid metadata rejection;
- ascending selected-lane order.

### Compression

- compressed weight order;
- exact decompression;
- shape validation;
- signed INT8 preservation;
- deterministic output.

### Metadata packing

- pack/unpack round-trip;
- expected six-byte output;
- bit order;
- padding bits;
- invalid value rejection.

### Fine-tuning

- mask remains unchanged;
- pruned weights remain zero after optimization;
- validation-based checkpoint selection;
- fixed seeds;
- short smoke run;
- no GPU.

### Sparse integer inference

- hand-computed example;
- metadata lane selection;
- two products per group;
- negative values;
- exact equality with dense-form sparse inference;
- per-channel reconstruction.

### Storage accounting

- dense 64 bytes;
- compressed weights 32 bytes;
- metadata 6 bytes;
- sparse total 38 bytes;
- reduction 40.625%;
- bias and scales reported separately.

### Evaluation

- accuracy comparison;
- prediction agreement;
- disagreement count;
- error metrics;
- pattern distribution;
- operation counts;
- accumulator safety.

### Artifacts

- required fields;
- deterministic hashes;
- correct tensor sizes;
- valid metadata;
- relative paths;
- unsupported format rejection.

### CLI

- commands parse;
- smoke sparse run succeeds;
- missing Phase 2 artifact fails clearly;
- invalid configuration fails clearly.

Tests must not:

- require internet;
- require GPU;
- require Sparrow-V;
- modify Sparrow-V;
- perform long fine-tuning runs repeatedly.

## Make Targets

Add:

```text
prune-2of4
finetune-sparse
pack-sparse
evaluate-sparse
run-sparse-baseline
test-phase3
```

Update `make help`.

`make test-phase3` should run focused Phase 3 tests only.

Do not rerun full Phase 1 training in every sparse test.

The full sparse baseline may regenerate Phase 1 or Phase 2 artifacts only when missing.

## Documentation

Update or add:

- `README.md`;
- `docs/architecture.md`;
- `docs/build_roadmap.md`;
- `docs/data_contracts.md`;
- `docs/experiment_policy.md`;
- `docs/codex_context.md`;
- one Phase 3 results document.

Suggested:

```text
docs/results/phase3_sparse_2of4.md
```

Document:

- grouping axis;
- deterministic pruning;
- tie-breaking;
- mask semantics;
- fine-tuning procedure;
- metadata mapping;
- compressed-weight ordering;
- packing format;
- dense/sparse integer inference;
- quality comparisons;
- operation reduction;
- storage reduction;
- accumulator safety;
- limitations;
- reproduction commands.

Keep `docs/codex_context.md` concise.

## README Status

Update README status to:

```text
Phase 3 deterministic 2:4 structured sparsity implemented
```

Do not claim:

- compiler lowering;
- Sparrow-V execution;
- hardware speedup;
- real-world model accuracy;
- hardware-aware pruning;
- general N:M sparsity.

## Existing Behavior Preservation

Preserve:

- Phase 1 fixture and FP32 baseline;
- Phase 2 quantization and integer inference;
- all existing CLI commands;
- all existing tests;
- artifact contracts;
- Sparrow-V boundary.

Do not modify Sparrow-V.

## Out of Scope

Do not implement:

- compiler IR;
- instruction generation;
- Sparrow-V execution;
- TinyNPU execution;
- sparse-load instructions;
- new hardware operations;
- hardware-aware pruning;
- latency-aware pruning;
- layer-adaptive sparsity;
- general N:M sparsity;
- unstructured sparsity;
- mixed precision;
- QAT;
- distillation;
- multi-layer models;
- ONNX;
- target selection;
- hardware cost models;
- real dataset download;
- hyperparameter sweeps;
- research experiments.

## Validation

During development, run focused tests.

At final acceptance, run once:

```text
python3 -m compileall src scripts
pytest
make test-phase1
make test-phase2
make test-phase3
make smoke
make check
make docs-check
git diff --check
```

Also run once:

```text
make run-sparse-baseline
```

If prerequisites are missing, regenerate once using:

```text
make run-fp32-baseline
make run-int8-baseline
```

Do not repeatedly retrain or re-quantize during unrelated validation.

## Acceptance Criteria

The milestone is complete only when:

1. Deterministic 2:4 pruning exists.
2. Grouping occurs along the input-feature axis.
3. Every group has four weights.
4. Every group retains exactly two weights.
5. Every group prunes exactly two weights.
6. Largest-magnitude selection is used.
7. Lower-lane tie-breaking is used.
8. Mask generation is deterministic.
9. Mask format is validated.
10. All six legal metadata values are supported.
11. Invalid metadata values are rejected.
12. Compressed weight ordering matches Sparrow-V.
13. Sparse dense-equivalent matrix is generated.
14. Exact decompression works.
15. Metadata packing works.
16. Packed metadata length is six bytes.
17. Pack/unpack round-trip is exact.
18. Fine-tuning preserves the mask.
19. Pruned weights remain exactly zero.
20. Validation data controls checkpoint selection.
21. Test data is not used for tuning.
22. Sparse INT8 quantization is implemented.
23. Sparse INT32 biases are consistent with sparse scales.
24. Dense-form sparse integer inference exists.
25. Compressed sparse integer inference exists.
26. Compressed inference does not simply decompress first.
27. Both sparse inference paths produce identical accumulators.
28. Reconstructed sparse logits are produced.
29. Dense INT8 and sparse INT8 metrics are compared.
30. Pre-fine-tuning sparse metrics are reported.
31. Post-fine-tuning sparse metrics are reported.
32. Sparse test fixture accuracy is at least 95%.
33. Sparse accuracy drop versus dense INT8 is no more than 5 percentage points.
34. Sparse/dense prediction agreement is at least 95%.
35. Sparse confusion matrix is reported.
36. Logit error metrics are reported.
37. Pattern distribution is reported.
38. Pattern count equals 16.
39. Invalid patterns never occur.
40. Sparse executed multiplication count is 32 per sample.
41. Sparse skipped multiplication count is 32 per sample.
42. Arithmetic reduction is 50%.
43. Dense weight storage is reported as 64 bytes.
44. Compressed weight storage is reported as 32 bytes.
45. Metadata storage is reported as 6 bytes.
46. Total sparse storage is reported as 38 bytes.
47. Storage reduction is reported as 40.625%.
48. Biases and scales are accounted for separately.
49. Sparse observed accumulators fit signed INT32.
50. Sparse conservative bound fits signed INT32.
51. Sparse artifact format is validated.
52. Sparse artifacts are deterministic.
53. One command reproduces the complete sparse baseline.
54. Phase 1 remains passing.
55. Phase 2 remains passing.
56. Phase 3 focused tests pass.
57. Tests require no internet.
58. Tests require no GPU.
59. Tests require no Sparrow-V checkout.
60. Documentation matches implementation.
61. README status is accurate.
62. No compiler or hardware execution is implemented.
63. No hardware-aware pruning is implemented.
64. Sparrow-V is not modified.
65. General repository checks pass.
66. Documentation checks pass.
67. `git diff --check` passes.
68. No commit or push occurs.
69. `docs/codex_milestone_result.md` is finalized.

## Stop Conditions

Stop for human review only if:

- the Phase 1 checkpoint or Phase 2 artifact cannot be loaded;
- the current weight layout cannot be grouped consistently into groups of four;
- preserving the 2:4 mask during fine-tuning requires a major training-system redesign;
- sparse quantization cannot be represented with the existing INT8/INT32 contract;
- compressed inference cannot reproduce dense-form sparse inference;
- sparse quality cannot meet the acceptance gate without modifying the test split;
- observed or theoretical sparse accumulation exceeds signed INT32;
- a major Phase 1 or Phase 2 correctness defect is discovered.

Ordinary pruning bugs, mask bugs, packing bugs, fine-tuning instability, test failures, and documentation work are not stop conditions.

## Token-Efficiency Instructions

Follow `AGENTS.md`.

In particular:

- read compact context and current milestone first;
- inspect only Phase 1–3 relevant files;
- avoid exploring compiler and target directories;
- use one deterministic pruning rule;
- do not perform a hyperparameter sweep;
- use bounded fine-tuning;
- run focused tests during development;
- run aggregate validation once;
- reuse existing artifacts where possible;
- keep the result file concise.

## Result File

Update:

```text
docs/codex_milestone_result.md
```

throughout the run.

Finalize with `STATUS: COMPLETE` only if every acceptance criterion and validation passes.

Include:

- pruning rule;
- grouping axis;
- tie-breaking rule;
- retained/pruned counts;
- metadata pattern distribution;
- fine-tuning configuration;
- dense INT8 accuracy;
- sparse pre-fine-tuning accuracy;
- sparse post-fine-tuning accuracy;
- prediction agreement;
- confusion matrix;
- logit error metrics;
- exact dense-form/compressed equivalence result;
- operation accounting;
- storage accounting;
- accumulator ranges;
- artifact paths and determinism evidence;
- exact commands and outcomes;
- changed files;
- remaining limitations;
- next recommended milestone;
- confirmation that Sparrow-V was not modified;
- confirmation that no commit or push occurred.

Use `STATUS: FAILED` if required work or checks remain incomplete.

Use `STATUS: BLOCKED` only for a genuine architectural or human-decision blocker.

## Next Milestone

The expected next milestone is:

```text
SparrowML intermediate representation and Sparrow-V artifact exporter
```

Do not implement it during this milestone.