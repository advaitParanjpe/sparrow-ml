# Milestone: SparrowML Intermediate Representation and Sparrow-V Artifact Exporter

## Objective

Extend SparrowML from trained/quantized model artifacts into a deterministic ahead-of-time deployment pipeline for Sparrow-V.

This milestone must:

1. define a small explicit SparrowML intermediate representation;
2. lower the current dense INT8 and sparse 2:4 linear models into that IR;
3. validate tensor shapes, quantization domains, operator legality, and memory sizes;
4. generate Sparrow-V-compatible model, input, weight, bias, scale, metadata, and expected-output artifacts;
5. generate deterministic data-memory images and deployment manifests;
6. preserve enough semantic information for a later runtime milestone to invoke Sparrow-V and compare RTL outputs;
7. validate exported artifacts against SparrowML’s integer reference implementations;
8. emit stable, documented, versioned deployment packages.

This milestone is about **compiler representation, lowering, layout, and export correctness**.

Do not execute Sparrow-V, modify Sparrow-V RTL, collect hardware counters, add new instructions, or implement a full general-purpose compiler.

## Current Baseline

Phase 1 provides:

- deterministic synthetic sensor fixture;
- 512 samples;
- 16 input features;
- four output classes;
- train-only standardization;
- FP32 `Linear(16, 4)` model.

Phase 2 provides:

- per-tensor symmetric INT8 input quantization;
- per-output-channel symmetric INT8 weights;
- INT32 biases;
- explicit dense integer reference inference;
- deterministic quantized-model artifact.

Phase 3 provides:

- deterministic 2:4 structured pruning;
- 32 retained and 32 pruned weights;
- compressed two-weight representation;
- legal Sparrow-V metadata;
- six-byte packed metadata;
- exact dense-form/compressed sparse accumulator equivalence;
- deterministic sparse artifact.

The current supported deployable models are:

```text
dense INT8 Linear(16, 4)
sparse 2:4 INT8 Linear(16, 4)
```

Do not generalize beyond these models unless a small abstraction is required for correctness.

## Repository Boundary

SparrowML owns:

- model representation;
- quantization and sparsity metadata;
- compiler IR;
- lowering;
- memory-layout planning;
- deployment-package generation;
- expected-output generation;
- export validation.

Sparrow-V remains an external deployment target.

This milestone must not:

- copy Sparrow-V RTL;
- edit Sparrow-V;
- invoke Sparrow-V simulation;
- rely on Sparrow-V being installed for normal tests;
- claim hardware execution.

Use target contracts and configuration only.

## Relevant Files

Read first:

- `AGENTS.md`
- `docs/codex_context.md`
- `docs/current_milestone.md`
- `docs/architecture.md`
- `docs/data_contracts.md`
- `docs/experiment_policy.md`
- `docs/results/phase2_int8_ptq.md`
- `docs/results/phase3_sparse_2of4.md`
- `configs/targets/sparrow_v.yaml`
- current quantization and sparsity artifact schemas;
- current dense and sparse integer reference implementations.

Inspect only directly relevant files after that.

Do not perform a broad Sparrow-V repository inspection.

## Supported Compiler Scope

Support exactly these deployment graphs:

### Dense graph

```text
Input[16, INT8]
→ DenseLinear[16 → 4, INT8 weights, INT32 bias]
→ OutputLogits[4, INT32 accumulators + per-channel scales]
```

### Sparse graph

```text
Input[16, INT8]
→ SparseLinear2of4[
    16 → 4,
    compressed INT8 weights,
    3-bit metadata,
    INT32 bias
  ]
→ OutputLogits[4, INT32 accumulators + per-channel scales]
```

No hidden layers, activations, convolutions, dynamic shapes, branching, or arbitrary graphs.

## SparrowML IR

Define a small versioned intermediate representation.

Preferred package structure:

```text
src/sparrowml/compiler/
├── __init__.py
├── ir.py
├── validation.py
├── lowering.py
├── layout.py
├── exporter.py
├── manifests.py
├── images.py
└── hashes.py
```

Adjust to match the existing repository structure.

Do not build a generic compiler framework.

## IR Requirements

The IR must explicitly represent:

### Model

- format version;
- model name;
- execution mode:
  - `dense_int8`;
  - `sparse_2of4_int8`;
- input tensor;
- output tensor;
- operators;
- constants;
- class names;
- preprocessing version;
- source artifact identity.

### Tensor

Each tensor must contain:

- unique name;
- shape;
- element type;
- role;
- logical ordering;
- storage layout;
- byte size;
- optional quantization metadata.

Supported element types should be limited to:

```text
int8
uint8
int32
float32
```

Only add another type if strictly necessary.

### Quantization

Represent:

- input scale;
- input zero point;
- per-output-channel weight scales;
- weight zero points;
- accumulator type;
- output reconstruction scales;
- rounding policy;
- clamping policy.

### Operator

Support exactly:

```text
DenseLinearInt8
SparseLinear2of4Int8
```

Operators must record:

- input tensor;
- output tensor;
- weight tensor or compressed-weight tensor;
- bias tensor;
- weight scales;
- sparse metadata tensor where applicable;
- feature count;
- output count;
- accumulation type.

### Sparse operator

For `SparseLinear2of4Int8`, also record:

- group size `4`;
- nonzero count `2`;
- grouping axis;
- group traversal order;
- metadata encoding version;
- compressed-weight ordering;
- packed metadata byte ordering.

## IR Serialization

Provide deterministic machine-readable serialization.

Preferred:

```text
JSON
```

Requirements:

- stable field ordering;
- stable list ordering;
- no timestamps inside determinism-sensitive payloads;
- no absolute paths;
- no non-deterministic identifiers;
- explicit format version;
- exact round-trip parse/serialize behavior.

Require:

```text
serialize(parse(serialize(ir))) == serialize(ir)
```

for canonical serialization.

## IR Validation

Implement strict validation.

Reject:

- unsupported operators;
- unsupported data types;
- dynamic or missing shapes;
- input feature count other than 16;
- output count other than 4;
- invalid quantization scales;
- invalid zero points;
- non-INT32 biases;
- inconsistent tensor byte sizes;
- sparse groups not equal to 2:4;
- invalid sparse metadata;
- compressed-weight count mismatches;
- class-count mismatches;
- missing preprocessing metadata;
- absolute paths;
- unsupported format versions.

Validation errors must identify the offending field clearly.

## Lowering

Implement deterministic lowering from:

### Phase 2 artifact

```text
artifacts/phase2_int8/quantized_model.json
```

to dense SparrowML IR.

### Phase 3 artifact

```text
artifacts/phase3_sparse/sparse_quantized_model.json
```

to sparse SparrowML IR.

The lowering must preserve:

- input scale and zero point;
- weight scales;
- weight values;
- sparse compressed values;
- sparse metadata;
- bias values;
- feature ordering;
- output ordering;
- class ordering;
- preprocessing version;
- source artifact identity.

Do not retrain, requantize, or reprune during lowering.

## Target Configuration

Use:

```text
configs/targets/sparrow_v.yaml
```

Extend it only as needed.

The target configuration should describe deployment-facing limits and conventions, such as:

- target name;
- architecture version;
- supported model modes;
- supported element types;
- vector width;
- feature count;
- output count;
- scratchpad size;
- endianness;
- alignment;
- base addresses or logical memory regions;
- metadata encoding version;
- instruction or command template version;
- result layout.

Do not encode machine-specific repository paths.

Keep execution commands out of the core IR.

## Memory Layout

Define deterministic memory regions for exported packages.

A reasonable logical layout is:

```text
input
weights
metadata
bias
scales
output
```

For each region record:

- name;
- byte offset;
- byte length;
- alignment;
- element type;
- logical shape;
- source tensor.

Requirements:

- no overlaps;
- deterministic ordering;
- alignment respected;
- total memory size reported;
- total size fits target limits;
- dense and sparse layouts both supported.

Use a simple alignment policy such as four-byte alignment unless Sparrow-V’s existing contract requires another value.

Do not invent a complex allocator.

## Endianness and Integer Encoding

Define and document:

- byte order;
- signed INT8 encoding;
- signed INT32 encoding;
- float32 scale encoding if scales are emitted directly;
- metadata byte packing;
- padding bytes;
- alignment padding.

Preferred:

```text
little-endian
two's-complement integers
IEEE-754 float32 scales
```

Validate all emitted values before encoding.

## Deployment Package

Generate one self-contained deployment directory per mode.

Preferred paths:

```text
artifacts/phase4_export/dense/
artifacts/phase4_export/sparse/
```

Each package must include at least:

```text
manifest.json
model_ir.json
memory_map.json
model_data.bin
input_data.bin
expected_output.json
export_report.json
README.md
```

Additional small files may be added where useful.

## Manifest

The deployment manifest must include:

- package format version;
- target name;
- model name;
- execution mode;
- source artifact hashes;
- model IR hash;
- feature count;
- output count;
- class names;
- preprocessing version;
- input quantization;
- weight quantization;
- accumulator type;
- memory-region summary;
- binary filenames;
- expected-output filename;
- export validation status;
- required future runtime interface version.

Do not include local absolute paths.

## Model Data Binary

Generate a deterministic binary containing model constants.

For dense mode, include:

- dense INT8 weights;
- INT32 biases;
- per-channel scales;
- any required target metadata.

For sparse mode, include:

- compressed INT8 weights;
- packed sparse metadata;
- INT32 biases;
- per-channel scales;
- any required target metadata.

The exact ordering must be defined in `memory_map.json`.

Do not silently include duplicated or unused data.

## Input Data

Export a bounded deterministic input set.

Preferred:

- one canonical test sample for single-run bring-up;
- optionally all 76 test samples in a separate batch-oriented artifact.

At minimum include:

- sample ID;
- quantized INT8 feature vector;
- expected label;
- class name;
- input scale;
- input zero point.

Use the same preprocessing and quantization contract as Phase 2/3.

Do not create new random inputs.

## Expected Outputs

Use SparrowML’s reference implementations to generate expected outputs.

For each exported sample, include:

- sample ID;
- expected INT32 accumulators;
- reconstructed logits;
- predicted class ID;
- predicted class name;
- expected label;
- correctness;
- execution mode.

For sparse mode, expected outputs must come from the explicit compressed sparse reference path.

For dense mode, expected outputs must come from the dense integer reference path.

Do not derive expected outputs from FP32 inference.

## Export Validation

After export, reload the generated package and verify:

1. manifest parses;
2. IR parses;
3. memory map parses;
4. binary lengths match declarations;
5. binary regions do not overlap;
6. all values decode correctly;
7. decoded tensors equal source tensors exactly;
8. expected outputs reproduce from decoded package contents;
9. dense export matches Phase 2 reference;
10. sparse export matches Phase 3 compressed reference;
11. hashes match;
12. repeated export is byte-for-byte deterministic.

Require deterministic SHA-256 evidence for at least:

- `model_ir.json`;
- `manifest.json`;
- `memory_map.json`;
- `model_data.bin`;
- `input_data.bin`;
- `expected_output.json`.

## Instruction or Command Representation

Define a minimal future-runtime command description, but do not execute it.

Preferred output:

```text
program.json
```

or:

```text
commands.json
```

It should describe a bounded abstract sequence such as:

### Dense

```text
load input
load dense weights
load bias/scales
execute dense dot products
store outputs
```

### Sparse

```text
load input
load compressed weights
load metadata
load bias/scales
execute sparse dot products
store outputs
```

Represent this as target-facing commands or symbolic operations, not raw machine code unless Sparrow-V already exposes a stable documented encoding in the target contract.

Do not reverse-engineer or invent undocumented ISA encodings.

The following milestone will connect this representation to actual Sparrow-V execution.

## CLI

Add bounded commands such as:

```text
sparrowml lower-ir
sparrowml validate-ir
sparrowml export-sparrowv
sparrowml validate-export
sparrowml run-export-baseline
```

Exact names may follow current conventions.

### `lower-ir`

- load Phase 2 or Phase 3 artifact;
- lower to canonical SparrowML IR;
- validate;
- emit IR.

### `validate-ir`

- validate one IR file;
- print concise model/operator/tensor summary;
- return nonzero on failure.

### `export-sparrowv`

- accept dense or sparse mode;
- create deployment package;
- emit manifest, binaries, memory map, expected outputs, and report.

### `validate-export`

- reload package;
- decode all data;
- rerun reference inference;
- verify hashes and equivalence.

### `run-export-baseline`

- export and validate both dense and sparse packages in one deterministic command.

Use proper exit codes.

## Configuration

Add:

```text
configs/experiments/sparrow_v_export.yaml
```

Include:

- source dense artifact;
- source sparse artifact;
- target config path;
- selected input sample IDs;
- output directory;
- alignment;
- byte order;
- scale serialization type;
- package format version;
- IR format version;
- validation policy.

Avoid absolute paths.

## Generated Artifacts

Use:

```text
artifacts/phase4_export/
```

Generated outputs should remain ignored unless repository policy explicitly tracks small golden schemas.

Do not overwrite Phase 1–3 artifacts.

At minimum generate:

```text
artifacts/phase4_export/dense/
artifacts/phase4_export/sparse/
artifacts/phase4_export/export_summary.json
artifacts/phase4_export/summary.md
artifacts/phase4_export/determinism.json
```

## Tests

Add focused tests for:

### IR

- dense IR construction;
- sparse IR construction;
- canonical serialization;
- parse/serialize round trip;
- stable ordering;
- format version;
- unsupported operator rejection;
- invalid shape rejection;
- invalid quantization rejection;
- absolute-path rejection.

### Lowering

- Phase 2 dense artifact lowers correctly;
- Phase 3 sparse artifact lowers correctly;
- values preserved exactly;
- class ordering preserved;
- preprocessing version preserved;
- source artifact hash recorded.

### Layout

- deterministic region order;
- correct offsets;
- alignment;
- no overlap;
- correct total size;
- dense and sparse size differences;
- target-capacity rejection.

### Binary encoding

- INT8 round trip;
- INT32 little-endian round trip;
- float32 scale round trip;
- metadata round trip;
- signed-value preservation;
- invalid range rejection.

### Export

- dense package generation;
- sparse package generation;
- required files exist;
- manifest fields;
- memory map consistency;
- decoded tensors equal source tensors;
- no absolute paths;
- deterministic hashes.

### Expected outputs

- dense exported package reproduces dense integer reference;
- sparse exported package reproduces compressed sparse reference;
- predictions and accumulators match exactly;
- sample IDs and labels preserved.

### CLI

- commands parse;
- dense export smoke run;
- sparse export smoke run;
- missing source artifact fails clearly;
- unsupported mode fails clearly;
- corrupt package validation fails clearly.

Tests must not:

- require internet;
- require GPU;
- require Sparrow-V checkout;
- invoke Sparrow-V;
- modify Sparrow-V;
- retrain or reprune models repeatedly.

## Make Targets

Add stable targets such as:

```text
lower-ir
validate-ir
export-sparrowv-dense
export-sparrowv-sparse
validate-export
run-export-baseline
test-phase4
```

Update:

```text
make help
```

`make test-phase4` must run focused Phase 4 tests only.

`make run-export-baseline` must export and validate both modes.

If prerequisite artifacts are missing, fail clearly or regenerate them once using existing stable targets.

Do not repeatedly rerun Phase 1–3 training during unrelated tests.

## Documentation

Update or add:

- `README.md`;
- `docs/architecture.md`;
- `docs/build_roadmap.md`;
- `docs/data_contracts.md`;
- `docs/experiment_policy.md`;
- `docs/codex_context.md`;
- one Phase 4 results document;
- one deployment-package contract document if useful.

Suggested files:

```text
docs/results/phase4_sparrow_v_export.md
docs/sparrow_v_export_contract.md
```

Document:

- supported graphs;
- IR schema;
- tensor/operator representation;
- quantization metadata;
- sparse metadata;
- lowering rules;
- memory layout;
- binary encoding;
- endianness;
- deployment package contents;
- expected-output generation;
- determinism;
- limitations;
- reproduction commands.

Keep `docs/codex_context.md` concise.

## README Status

Update README status to:

```text
Phase 4 SparrowML IR and Sparrow-V artifact export implemented
```

Do not claim:

- Sparrow-V execution;
- RTL validation;
- measured hardware cycles;
- hardware speedup;
- compiler support for arbitrary models;
- real-world accuracy;
- full ISA code generation.

## Existing Behavior Preservation

Preserve:

- Phase 1 FP32 workflow;
- Phase 2 INT8 workflow;
- Phase 3 sparse workflow;
- all existing CLI commands;
- all existing tests;
- artifact contracts;
- repository-local workflow;
- Sparrow-V boundary.

Do not modify Sparrow-V.

## Out of Scope

Do not implement:

- Sparrow-V simulation;
- subprocess execution of Sparrow-V;
- hardware result parsing;
- cycle or instruction collection;
- new Sparrow-V instructions;
- RTL modifications;
- arbitrary graph compilation;
- hidden layers;
- multi-layer models;
- ONNX;
- TinyNPU export;
- runtime scheduling;
- dynamic memory allocation;
- hardware-aware pruning;
- target selection;
- real dataset work;
- research experiments;
- GUI or dashboards.

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
make smoke
make check
make docs-check
git diff --check
```

Also run once:

```text
make run-export-baseline
```

If prerequisite artifacts are missing, regenerate them once:

```text
make run-fp32-baseline
make run-int8-baseline
make run-sparse-baseline
```

Do not repeatedly retrain during unrelated validation.

## Acceptance Criteria

The milestone is complete only when:

1. A versioned SparrowML IR exists.
2. The IR supports dense INT8 linear inference.
3. The IR supports sparse 2:4 INT8 linear inference.
4. Tensors have explicit names, shapes, types, roles, and layouts.
5. Quantization parameters are explicit.
6. Sparse metadata semantics are explicit.
7. Operators are explicitly represented.
8. Canonical deterministic JSON serialization exists.
9. Parse/serialize round-trip is stable.
10. Strict IR validation exists.
11. Unsupported operators are rejected.
12. Invalid shapes are rejected.
13. Invalid scales and zero points are rejected.
14. Absolute paths are rejected.
15. Phase 2 artifacts lower to dense IR.
16. Phase 3 artifacts lower to sparse IR.
17. Lowering preserves all integer tensor values exactly.
18. Lowering preserves class ordering.
19. Lowering preserves preprocessing version.
20. Target configuration is explicit and versioned.
21. A deterministic memory-layout planner exists.
22. Memory regions are aligned.
23. Memory regions do not overlap.
24. Total package size is reported.
25. Target capacity is validated.
26. Endianness is documented.
27. INT8 encoding is validated.
28. INT32 encoding is validated.
29. Scale encoding is validated.
30. Sparse metadata encoding is validated.
31. Dense deployment package is generated.
32. Sparse deployment package is generated.
33. Each package contains a manifest.
34. Each package contains canonical IR.
35. Each package contains a memory map.
36. Each package contains model binary data.
37. Each package contains input data.
38. Each package contains expected outputs.
39. Each package contains a human-readable README.
40. Model binary decodes exactly.
41. Input binary decodes exactly.
42. Dense decoded values match Phase 2.
43. Sparse decoded values match Phase 3.
44. Dense expected outputs match dense integer reference.
45. Sparse expected outputs match compressed sparse reference.
46. Accumulators match exactly.
47. Predicted classes match exactly.
48. Hashes are recorded.
49. Repeated export is byte-for-byte deterministic.
50. No timestamps break deterministic hashes.
51. Future-runtime commands are represented symbolically.
52. No undocumented raw ISA encoding is invented.
53. One command exports and validates both modes.
54. Phase 1 remains passing.
55. Phase 2 remains passing.
56. Phase 3 remains passing.
57. Phase 4 focused tests pass.
58. Tests require no internet.
59. Tests require no GPU.
60. Tests require no Sparrow-V checkout.
61. Sparrow-V is not executed.
62. Sparrow-V is not modified.
63. Documentation matches implementation.
64. README status is accurate.
65. General repository checks pass.
66. Documentation checks pass.
67. `git diff --check` passes.
68. No commit or push occurs.
69. `docs/codex_milestone_result.md` is finalized.

## Stop Conditions

Stop for human review only if:

- Phase 2 or Phase 3 source artifacts cannot be loaded;
- the existing Sparrow-V target contract lacks enough stable information to define export layouts without inventing unsupported behavior;
- source artifact schemas are internally inconsistent;
- exported values cannot reproduce current integer reference outputs;
- target memory limits cannot hold the current model;
- a major Phase 1–3 correctness defect is discovered.

Ordinary schema design, layout bugs, serialization issues, hash mismatches, test failures, and documentation work are not stop conditions.

## Token-Efficiency Instructions

Follow `AGENTS.md`.

In particular:

- read compact context and milestone first;
- inspect only compiler, artifact, quantization, sparsity, and target-contract files;
- do not inspect or modify Sparrow-V RTL;
- support only the two current linear model modes;
- do not build a general graph compiler;
- run focused Phase 4 tests during development;
- run aggregate checks once;
- reuse Phase 2 and Phase 3 artifacts;
- keep generated documentation and result output concise.

## Result File

Update:

```text
docs/codex_milestone_result.md
```

throughout the run.

Finalize with:

```text
STATUS: COMPLETE
```

only if all required work and validation pass.

Include:

- supported IR operators;
- IR and package format versions;
- tensor and operator summary;
- dense and sparse lowering results;
- memory-region layouts and total sizes;
- binary encoding conventions;
- exported sample IDs;
- dense expected accumulators and prediction;
- sparse expected accumulators and prediction;
- exact reference-equivalence results;
- generated files;
- deterministic hashes;
- exact commands and outcomes;
- changed files;
- remaining limitations;
- next recommended milestone;
- confirmation that Sparrow-V was not executed;
- confirmation that Sparrow-V was not modified;
- confirmation that no commit or push occurred.

Use `STATUS: FAILED` if required work or checks remain incomplete.

Use `STATUS: BLOCKED` only for a genuine target-contract or architectural blocker.

## Next Milestone

The expected next milestone is:

```text
Sparrow-V runtime adapter, simulator execution, and RTL/reference result validation
```

Do not implement it during this milestone.