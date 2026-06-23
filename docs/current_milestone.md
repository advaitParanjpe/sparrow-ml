# Milestone: Multi-Layer INT8 Model, Intermediate Activation Quantization, and Multi-Operator Compilation

## Objective

Extend SparrowML from the current single-layer `Linear(16, 4)` pipeline to a deterministic two-layer INT8 model with explicit intermediate activation quantization and multi-operator compilation.

Implement this model:

```text
Input[16]
→ Linear(16, 16)
→ ReLU
→ Linear(16, 4)
→ Output[4]
```

This milestone must:

1. train a deterministic FP32 two-layer MLP;
2. calibrate and quantize the input and intermediate activation tensors;
3. quantize both linear layers to signed INT8 weights with INT32 biases;
4. implement explicit multi-layer integer reference inference;
5. implement integer-domain ReLU and requantization between layers;
6. extend the SparrowML IR to represent multiple operators and intermediate tensors;
7. allocate deterministic intermediate memory buffers;
8. export dense Sparrow-V deployment packages for the two-layer graph;
9. validate exported artifacts against the multi-layer integer reference;
10. preserve all Phase 1–5 workflows and results.

This milestone is about **multi-layer model semantics, intermediate quantization, and multi-operator compilation**.

Do not implement Sparrow-V simulation of the multi-layer model yet unless an existing stable Sparrow-V interface already supports the complete sequence without modifying Sparrow-V. Real multi-layer RTL execution belongs in the following milestone.

## Current Baseline

Completed functionality includes:

### Phase 1

- deterministic 512-sample synthetic sensor fixture;
- 16 input features;
- four output classes;
- train-only standardization;
- deterministic FP32 `Linear(16, 4)` training.

### Phase 2

- signed symmetric INT8 inputs;
- per-output-channel signed INT8 weights;
- INT32 biases;
- exact single-layer integer reference inference.

### Phase 3

- deterministic 2:4 structured pruning;
- compressed sparse weights;
- packed sparse metadata;
- exact compressed sparse integer inference.

### Phase 4

- versioned SparrowML IR;
- dense and sparse single-operator lowering;
- deterministic memory maps and deployment packages;
- package reload validation.

### Phase 5

- Sparrow-V external runtime adapter;
- real dense and sparse RTL simulation;
- exact reference matching;
- architectural counter reporting;
- no Sparrow-V source or RTL modification.

All existing functionality must remain passing.

## Supported Model

Implement exactly:

```text
Linear(16, 16)
ReLU
Linear(16, 4)
```

Use:

- 16 input features;
- 16 hidden features;
- four output classes;
- FP32 training;
- INT8 inputs;
- INT8 hidden activations;
- INT8 weights;
- INT32 biases and accumulators;
- deterministic CPU execution.

Do not generalize to arbitrary-depth networks.

## Relevant Files

Read first:

- `AGENTS.md`
- `docs/codex_context.md`
- `docs/current_milestone.md`
- `docs/architecture.md`
- `docs/data_contracts.md`
- `docs/experiment_policy.md`
- `docs/results/phase1_fp32_baseline.md`
- `docs/results/phase2_int8_ptq.md`
- `docs/results/phase4_sparrow_v_export.md`
- `docs/results/phase5_sparrow_v_runtime.md`
- current model, training, quantization, compiler, IR, export, and evaluation modules.

Inspect only the directly relevant implementation after that.

Do not broadly inspect Sparrow-V or modify it.

## FP32 Model

Implement a minimal PyTorch model equivalent to:

```python
torch.nn.Sequential(
    torch.nn.Linear(16, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 4),
)
```

Requirements:

- deterministic initialization;
- explicit layer names;
- no dropout;
- no batch normalization;
- no hidden operations;
- CPU support;
- inspectable forward pass.

Suggested names:

```text
fc1
relu
fc2
```

Report:

- total parameter count;
- per-layer parameter count;
- checkpoint size.

## Training

Train on the existing deterministic fixture and splits.

Requirements:

- same fixture provenance;
- same train/validation/test separation;
- fixed Python, NumPy, PyTorch, and DataLoader seeds;
- cross-entropy loss;
- Adam or SGD;
- validation-based checkpoint selection;
- no test-based tuning;
- bounded epoch count;
- CPU-only default.

Recommended defaults:

```text
epochs: 50
batch_size: 32
learning_rate: 0.005
seed: 20260623
```

Adjust only if necessary for stable convergence.

Do not perform a hyperparameter sweep.

## FP32 Quality Gate

Required:

- test fixture accuracy at least 95%;
- validation and test metrics reported;
- confusion matrix reported;
- training remains deterministic within documented software limits.

Results must remain labelled as synthetic fixture accuracy.

## Quantization Domains

Represent these distinct quantization domains:

1. standardized model input;
2. first-layer weights;
3. first-layer INT32 accumulators;
4. post-ReLU hidden real values;
5. hidden INT8 activations;
6. second-layer weights;
7. second-layer INT32 accumulators;
8. reconstructed final logits.

Do not treat all tensors as though they share one scale.

## Input Quantization

Reuse the Phase 2 input quantization contract where sound:

```text
signed INT8
per-tensor symmetric
zero_point = 0
scale = max_abs(training inputs) / 127
```

Calibration must use training data only.

## Weight Quantization

For both `fc1` and `fc2`, use:

```text
signed INT8
per-output-channel symmetric
zero_point = 0
```

For `fc1`:

```text
weight shape = [16, 16]
16 output-channel scales
```

For `fc2`:

```text
weight shape = [4, 16]
4 output-channel scales
```

Use deterministic rounding and clamping consistent with Phase 2.

## Bias Quantization

For each linear layer:

```text
bias_scale[channel] =
    activation_input_scale × weight_scale[channel]
```

Quantize bias to signed INT32:

```text
bias_int32[channel] =
    round(bias_fp32[channel] / bias_scale[channel])
```

For `fc1`, the activation input scale is the model input scale.

For `fc2`, the activation input scale is the hidden activation scale.

Validate all biases against signed INT32.

## First-Layer Integer Inference

Compute:

```text
acc1[j] =
    bias1_int32[j]
    + Σ input_int8[i] × weight1_int8[j, i]
```

Requirements:

- explicit signed integer arithmetic;
- no `torch.nn.Linear` in integer inference;
- accumulator range validation;
- reconstruction using per-channel scales.

The reconstructed first-layer value for channel `j` is:

```text
hidden_pre_relu_real[j] =
    acc1[j] × input_scale × weight1_scale[j]
```

## ReLU

Apply exact ReLU in the reconstructed activation domain:

```text
hidden_relu_real[j] = max(0, hidden_pre_relu_real[j])
```

Also define an equivalent integer-aware implementation where possible.

Document clearly whether ReLU occurs:

- before requantization in reconstructed real values; or
- through an equivalent integer threshold.

Do not introduce approximation beyond the documented quantization step.

## Hidden Activation Calibration

Calibrate the hidden activation tensor using training samples only.

Preferred:

```text
per-tensor asymmetric or symmetric INT8
```

Because ReLU produces nonnegative values, preferred initial representation is:

```text
uint8, range [0, 255]
```

However, Sparrow-V’s existing vector datapath is signed INT8-oriented. Therefore, for deployment compatibility, prefer:

```text
signed INT8, range [0, 127]
zero_point = 0
scale = max(training hidden ReLU activation) / 127
```

Use only nonnegative codes.

Report:

- hidden calibration split;
- sample count;
- minimum;
- maximum;
- maximum absolute value;
- selected scale;
- zero point;
- clipped values;
- clipping percentage.

Do not use validation or test data for hidden calibration.

## Hidden Requantization

Quantize hidden ReLU activations:

```text
hidden_int8 =
    clamp(round(hidden_relu_real / hidden_scale), 0, 127)
```

Although stored in signed INT8, valid hidden values must remain in:

```text
[0, 127]
```

Requirements:

- deterministic rounding;
- explicit clamping;
- saturation reporting;
- no negative post-ReLU codes;
- finite values only.

## Second-Layer Integer Inference

Compute:

```text
acc2[k] =
    bias2_int32[k]
    + Σ hidden_int8[j] × weight2_int8[k, j]
```

Reconstruct final logits:

```text
logit[k] =
    acc2[k] × hidden_scale × weight2_scale[k]
```

Prediction must use reconstructed per-channel logits, not raw accumulators where output scales differ.

## Integer Reference Outputs

For every evaluated sample, retain:

- input INT8 vector;
- `fc1` INT32 accumulators;
- reconstructed pre-ReLU values;
- post-ReLU values;
- hidden INT8 vector;
- `fc2` INT32 accumulators;
- reconstructed final logits;
- predicted class;
- expected label.

This trace is required for compiler/export validation.

## Quantization Evaluation

Compare:

1. FP32 MLP;
2. INT8 multi-layer reference.

Report per split:

- FP32 fixture accuracy;
- INT8 fixture accuracy;
- prediction agreement;
- disagreement count;
- confusion matrix;
- final logit maximum absolute error;
- mean absolute error;
- RMS error;
- hidden activation clipping;
- accumulator ranges for both layers.

## Quality Gates

Required:

- INT8 test fixture accuracy at least 95%;
- INT8 test accuracy drop versus FP32 no more than 3 percentage points;
- FP32/INT8 prediction agreement at least 95%;
- all first-layer accumulators fit signed INT32;
- all second-layer accumulators fit signed INT32;
- hidden INT8 activations remain in `[0, 127]`;
- no NaN or infinity;
- deterministic artifacts.

Do not alter the test fixture to satisfy the gate.

## Compiler IR Extension

Extend the existing IR rather than creating a separate incompatible format.

The IR must support this operator sequence:

```text
DenseLinearInt8
ReLU
RequantizeInt8
DenseLinearInt8
```

A fused representation is permitted only if the unfused logical operations remain explicit in the IR or manifest.

Add explicit intermediate tensors:

```text
input_int8             [16]
fc1_acc_int32          [16]
hidden_relu_real       [16] or logical-only
hidden_int8            [16]
fc2_acc_int32          [4]
output_logits          [4]
```

Do not require physical storage for logical-only tensors if clearly documented.

## New Operator Support

Add narrowly scoped operators:

```text
ReLU
RequantizeInt8
```

Operator fields must explicitly record:

### ReLU

- input tensor;
- output tensor;
- threshold semantics;
- element count.

### RequantizeInt8

- input accumulator or reconstructed tensor;
- output tensor;
- input scales;
- output scale;
- output zero point;
- rounding;
- clamp minimum;
- clamp maximum.

Do not add arbitrary activation or quantization operators.

## Multi-Operator Validation

Validate:

- operator order;
- tensor producer/consumer relationships;
- no missing tensors;
- no duplicate tensor producers;
- compatible shapes;
- valid quantization domains;
- valid hidden activation range;
- final output count of four;
- static graph only;
- no cycles in the operator graph.

## Memory Layout

Extend the layout planner to allocate:

- input;
- `fc1` weights;
- `fc1` bias;
- `fc1` scales;
- hidden activation buffer;
- `fc2` weights;
- `fc2` bias;
- `fc2` scales;
- final output.

Requirements:

- deterministic order;
- four-byte alignment unless target contract requires otherwise;
- no overlap;
- hidden buffer explicitly represented;
- scratchpad capacity checked;
- total package size reported.

Use buffer reuse only if simple and clearly validated.

Do not implement a general lifetime allocator.

## Deployment Package

Generate a Phase 6 dense multi-layer package under:

```text
artifacts/phase6_multilayer/export/
```

Include:

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

The package must be self-contained and deterministic.

## Program Representation

Generate a symbolic multi-command program:

```text
load input
load fc1 weights/bias/scales
execute 16 dense dot products
apply ReLU
requantize hidden activations
load fc2 weights/bias/scales
execute four dense dot products
store final outputs
```

Do not invent raw Sparrow-V opcodes.

The symbolic representation must clearly distinguish operations expected to run:

- in vector hardware;
- in scalar runtime software;
- on the host in a future adapter.

No actual Sparrow-V execution is required in this milestone.

## Export Validation

Reload the exported package and verify:

- all binaries decode;
- tensors match source artifacts;
- graph order is valid;
- memory regions do not overlap;
- hidden buffer size is correct;
- integer reference reproduced from decoded package contents;
- both layer accumulators match exactly;
- hidden INT8 activation matches exactly;
- final prediction matches;
- repeated export is byte-for-byte deterministic.

## Configuration

Add:

```text
configs/experiments/multilayer_int8_baseline.yaml
```

Include:

- model dimensions;
- seed;
- training settings;
- calibration policies;
- input quantization;
- hidden quantization;
- weight quantization;
- acceptance gates;
- output directories;
- IR version;
- package version;
- alignment;
- target configuration.

Avoid absolute paths.

## CLI

Add bounded commands such as:

```text
sparrowml train-mlp
sparrowml quantize-mlp
sparrowml evaluate-mlp-int8
sparrowml export-mlp
sparrowml validate-mlp-export
sparrowml run-multilayer-baseline
```

Exact names may follow existing conventions.

### `train-mlp`

- train deterministic FP32 model;
- save best validation checkpoint;
- write metrics.

### `quantize-mlp`

- calibrate input and hidden activations using training data;
- quantize both layers;
- emit multi-layer quantized artifact.

### `evaluate-mlp-int8`

- run explicit multi-layer integer reference;
- compare against FP32.

### `export-mlp`

- lower to multi-operator IR;
- create deployment package.

### `validate-mlp-export`

- reload and reproduce intermediate and final reference values.

### `run-multilayer-baseline`

- train;
- calibrate;
- quantize;
- evaluate;
- export;
- validate;
- report.

## Package Structure

Add only the required modules.

Suggested additions:

```text
src/sparrowml/
├── models/
│   └── mlp_classifier.py
├── training/
│   └── mlp_trainer.py
├── quantization/
│   ├── multilayer.py
│   ├── activations.py
│   └── requantization.py
├── evaluation/
│   └── multilayer_metrics.py
└── compiler/
    └── multilayer_lowering.py
```

Reuse existing quantization and compiler utilities where sensible.

Do not create a general deep-learning framework.

## Generated Artifacts

Use:

```text
artifacts/phase6_multilayer/
```

At minimum generate:

```text
fp32_checkpoint.pt
training_metrics.json
input_calibration.json
hidden_calibration.json
quantized_model.json
integer_evaluation.json
intermediate_traces.json
export/
summary.md
determinism.json
```

Generated outputs should remain ignored.

## Tests

Add focused tests for:

### Model

- input shape `[batch, 16]`;
- hidden shape `[batch, 16]`;
- output shape `[batch, 4]`;
- deterministic initialization;
- parameter count.

### Training

- deterministic short training;
- validation checkpoint selection;
- no test leakage;
- checkpoint creation;
- loss decreases.

### Hidden calibration

- training split only;
- deterministic scale;
- no validation/test leakage;
- nonnegative range;
- clipping statistics.

### ReLU and requantization

- negative values become zero;
- zero remains zero;
- positive values quantize correctly;
- ties-to-even rounding;
- clamping to `[0, 127]`;
- deterministic results.

### Multi-layer integer inference

- hand-computed tiny example;
- first-layer accumulation;
- bias handling;
- per-channel reconstruction;
- ReLU;
- hidden requantization;
- second-layer accumulation;
- final reconstruction;
- prediction semantics.

### IR

- valid operator sequence;
- intermediate tensor definitions;
- producer/consumer checks;
- invalid order rejection;
- shape mismatch rejection;
- cyclic graph rejection.

### Layout

- hidden buffer allocation;
- deterministic offsets;
- alignment;
- no overlap;
- scratchpad capacity.

### Export

- required files;
- exact tensor decoding;
- exact intermediate trace reproduction;
- deterministic hashes;
- no absolute paths.

### CLI

- commands parse;
- smoke multi-layer run;
- missing prerequisite error;
- invalid configuration error.

Tests must not:

- require internet;
- require GPU;
- require Sparrow-V;
- modify Sparrow-V;
- run Phase 5 integration;
- perform long training repeatedly.

## Make Targets

Add:

```text
train-mlp
quantize-mlp
evaluate-mlp-int8
export-mlp
validate-mlp-export
run-multilayer-baseline
test-phase6
```

Update `make help`.

`make test-phase6` should run focused tests only.

`make run-multilayer-baseline` should perform the full Phase 6 flow once.

Do not rerun full Phase 1–5 workflows during every focused test.

## Documentation

Update or add:

- `README.md`;
- `docs/architecture.md`;
- `docs/build_roadmap.md`;
- `docs/data_contracts.md`;
- `docs/experiment_policy.md`;
- `docs/codex_context.md`;
- multi-layer quantization contract;
- Phase 6 result documentation.

Suggested files:

```text
docs/multilayer_quantization_contract.md
docs/results/phase6_multilayer_int8.md
```

Document:

- FP32 model architecture;
- training configuration;
- input calibration;
- hidden calibration;
- integer ReLU;
- hidden requantization;
- two-layer accumulation;
- per-layer quantization domains;
- IR operator sequence;
- memory map;
- package format;
- quality metrics;
- limitations;
- reproduction commands.

Keep `docs/codex_context.md` concise.

## README Status

Update README status to:

```text
Phase 6 multi-layer INT8 model and multi-operator export implemented
```

Do not claim:

- multi-layer Sparrow-V execution;
- RTL validation of intermediate activations;
- sparse multi-layer support;
- physical hardware execution;
- real-world accuracy;
- arbitrary neural-network compilation.

## Existing Behavior Preservation

Preserve:

- Phase 1 FP32 baseline;
- Phase 2 single-layer INT8;
- Phase 3 single-layer structured sparsity;
- Phase 4 single-operator export;
- Phase 5 single-layer Sparrow-V RTL validation;
- all existing CLI commands;
- all existing tests;
- Sparrow-V repository boundary.

Do not modify Sparrow-V.

## Out of Scope

Do not implement:

- multi-layer Sparrow-V simulation;
- new Sparrow-V instructions;
- RTL modifications;
- sparse hidden layer;
- 2:4 pruning of the MLP;
- mixed precision;
- QAT;
- distillation;
- arbitrary graph compilation;
- ONNX;
- convolution;
- recurrent models;
- transformer models;
- real dataset integration;
- TinyNPU integration;
- target selection;
- hardware-aware optimization;
- research experiments;
- GUI or cloud training.

## Validation

During development, run focused tests.

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
make smoke
make check
make docs-check
git diff --check
```

Also run once:

```text
make run-multilayer-baseline
```

Do not run Sparrow-V integration unless needed to ensure Phase 5 remains structurally intact.

## Acceptance Criteria

The milestone is complete only when:

1. A deterministic FP32 `16→16→4` MLP exists.
2. ReLU is the only hidden activation.
3. FP32 training uses fixed seeds.
4. Validation controls checkpoint selection.
5. Test data is not used for tuning.
6. FP32 train metrics are reported.
7. FP32 validation metrics are reported.
8. FP32 test metrics are reported.
9. FP32 test fixture accuracy is at least 95%.
10. Input calibration uses training data only.
11. Hidden calibration uses training data only.
12. Input scale is deterministic.
13. Hidden scale is deterministic.
14. Both linear layers use signed INT8 weights.
15. Both layers use per-output-channel weight scales.
16. Both layers use INT32 biases.
17. First-layer integer inference exists.
18. First-layer accumulator ranges are validated.
19. Reconstructed first-layer values are produced.
20. ReLU is applied correctly.
21. Hidden values are requantized to signed INT8 codes in `[0,127]`.
22. Hidden clipping is reported.
23. Second-layer integer inference exists.
24. Second-layer accumulator ranges are validated.
25. Final reconstructed logits are produced.
26. Prediction uses correctly scaled logits.
27. INT8 test fixture accuracy is at least 95%.
28. INT8 accuracy drop versus FP32 is no more than three percentage points.
29. FP32/INT8 prediction agreement is at least 95%.
30. Confusion matrix is reported.
31. Final logit error metrics are reported.
32. Intermediate traces are emitted.
33. Intermediate traces are deterministic.
34. Existing IR supports multiple operators.
35. `ReLU` is represented explicitly.
36. `RequantizeInt8` is represented explicitly.
37. Intermediate tensors are represented explicitly.
38. Producer/consumer relationships validate.
39. Invalid graph order is rejected.
40. Tensor shapes validate.
41. Memory layout includes a hidden buffer.
42. Memory regions do not overlap.
43. Scratchpad capacity is validated.
44. A multi-layer package is generated.
45. Model binary decodes exactly.
46. Input binary decodes exactly.
47. Decoded first-layer accumulators reproduce exactly.
48. Decoded hidden INT8 activations reproduce exactly.
49. Decoded second-layer accumulators reproduce exactly.
50. Final prediction reproduces exactly.
51. Export is byte-for-byte deterministic.
52. Symbolic multi-command program is generated.
53. No raw unsupported ISA encoding is invented.
54. One command reproduces the Phase 6 workflow.
55. Phase 1 remains passing.
56. Phase 2 remains passing.
57. Phase 3 remains passing.
58. Phase 4 remains passing.
59. Phase 5 unit tests remain passing.
60. Phase 6 focused tests pass.
61. Tests require no internet.
62. Tests require no GPU.
63. Tests require no Sparrow-V checkout.
64. Sparrow-V is not modified.
65. Documentation matches implementation.
66. README status is accurate.
67. General repository checks pass.
68. Documentation checks pass.
69. `git diff --check` passes.
70. No commit or push occurs.
71. `docs/codex_milestone_result.md` is finalized.

## Stop Conditions

Stop for human review only if:

- the existing fixture cannot train the MLP to the required quality gate;
- hidden activation quantization cannot meet the quality gate without changing the test split;
- accumulator bounds exceed signed INT32;
- the current IR cannot be extended without breaking Phase 4 packages;
- the model cannot fit the configured Sparrow-V scratchpad;
- a major Phase 1–5 correctness defect is discovered.

Ordinary training instability, calibration bugs, requantization errors, IR validation issues, memory-layout bugs, test failures, and documentation work are not stop conditions.

## Token-Efficiency Instructions

Follow `AGENTS.md`.

In particular:

- inspect only model, training, quantization, compiler, and export files relevant to Phase 6;
- do not inspect Sparrow-V broadly;
- do not run Sparrow-V simulation;
- support only `16→16→4`;
- do not build a general graph compiler;
- do not perform a hyperparameter sweep;
- run focused tests while developing;
- run aggregate validation once;
- keep result reporting concise;
- reuse existing deterministic utilities.

## Result File

Update:

```text
docs/codex_milestone_result.md
```

Finalize with:

```text
STATUS: COMPLETE
```

only when all required implementation and validation pass.

Include:

- FP32 model architecture;
- parameter count;
- training configuration;
- best epoch;
- train/validation/test FP32 metrics;
- input calibration;
- hidden calibration;
- per-layer quantization schemes;
- accumulator ranges;
- hidden clipping;
- INT8 metrics;
- prediction agreement;
- confusion matrix;
- logit error;
- IR operator sequence;
- memory layout;
- generated package files;
- determinism evidence;
- exact validation commands and outcomes;
- changed files;
- remaining limitations;
- next recommended milestone;
- confirmation that Sparrow-V was not modified;
- confirmation that no commit or push occurred.

Use `STATUS: FAILED` if required work or validation remains incomplete.

Use `STATUS: BLOCKED` only for a genuine architectural or quality-gate blocker.

## Next Milestone

The expected next milestone is:

```text
Multi-layer Sparrow-V runtime execution and intermediate RTL/reference validation
```

Do not implement it during this milestone.