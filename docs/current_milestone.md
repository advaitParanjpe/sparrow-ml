# Milestone: Multi-Layer Sparrow-V Runtime Execution and Intermediate RTL/Reference Validation

## Objective

Execute the Phase 6 dense `16→16→4` INT8 model through the existing Sparrow-V runtime and RTL simulation path, then validate all intermediate and final values against SparrowML’s multi-layer integer reference.

This milestone must:

1. extend the existing SparrowML Sparrow-V runtime adapter to accept the Phase 6 multi-layer deployment package;
2. use Sparrow-V’s documented external workload interface without modifying Sparrow-V RTL or ISA;
3. execute the first linear layer through Sparrow-V’s dense vector path;
4. validate first-layer accumulators exactly;
5. perform and record ReLU and hidden requantization with clear provenance;
6. execute the second linear layer through Sparrow-V’s dense vector path;
7. validate second-layer accumulators exactly;
8. validate hidden INT8 activations, reconstructed logits, and final prediction;
9. collect measured and derived counters for both layers;
10. emit a deterministic multi-layer runtime result and comparison report;
11. preserve all Phase 1–6 workflows and tests.

This milestone is about **real multi-stage Sparrow-V RTL execution and intermediate correctness validation**.

Do not modify Sparrow-V RTL, add new instructions, implement sparse MLP execution, or begin real-dataset evaluation.

## Current Baseline

Phase 6 provides:

- deterministic FP32 model:

```text
Linear(16,16)
ReLU
Linear(16,4)
```

- 340 parameters;
- deterministic training;
- FP32 fixture accuracy of 100%;
- input INT8 calibration;
- hidden ReLU activation calibration;
- signed INT8 weights for both layers;
- INT32 biases;
- explicit two-layer integer reference inference;
- deterministic intermediate traces;
- IR operator sequence:

```text
DenseLinearInt8
ReLU
RequantizeInt8
DenseLinearInt8
```

- deterministic 528-byte deployment package;
- explicit 16-byte hidden activation buffer;
- exact package reload validation;
- no Sparrow-V execution yet.

Phase 5 provides:

- Sparrow-V checkout discovery;
- versioned external workload interface;
- real dense and sparse single-layer RTL simulation;
- exact accumulator and prediction matching;
- measured and derived counter handling;
- semantic determinism checks;
- no Sparrow-V source or RTL modification.

## Supported Scope

Support exactly one dense multi-layer graph:

```text
Input[16, INT8]
→ DenseLinearInt8[16→16]
→ ReLU
→ RequantizeInt8[hidden scale]
→ DenseLinearInt8[16→4]
→ Output[4]
```

Do not generalize to arbitrary graphs or layer counts.

Do not implement sparse multi-layer execution.

## Repository Boundary

SparrowML owns:

- Phase 6 package loading;
- layer sequencing;
- runtime workspace generation;
- Sparrow-V command invocation;
- result parsing;
- intermediate reference comparison;
- host/runtime post-processing;
- final reporting.

Sparrow-V owns:

- RTL;
- vector instruction semantics;
- simulator;
- external workload interface;
- architectural counters.

This milestone must not:

- modify Sparrow-V RTL;
- modify Sparrow-V ISA;
- overwrite Sparrow-V tracked files;
- commit inside Sparrow-V;
- copy Sparrow-V source into SparrowML;
- claim physical hardware execution.

## Relevant Files

Read first:

- `AGENTS.md`
- `docs/codex_context.md`
- `docs/current_milestone.md`
- `docs/architecture.md`
- `docs/data_contracts.md`
- `docs/experiment_policy.md`
- `docs/multilayer_quantization_contract.md`
- `docs/sparrow_v_runtime_contract.md`
- `docs/results/phase5_sparrow_v_runtime.md`
- `docs/results/phase6_multilayer_int8.md`
- Phase 5 Sparrow-V adapter modules;
- Phase 6 package, IR, trace, and export modules;
- `configs/experiments/sparrow_v_runtime.yaml`
- `configs/experiments/multilayer_int8_baseline.yaml`
- `configs/targets/sparrow_v.yaml`

Inspect only the minimum Sparrow-V files needed to use the existing external workload interface.

Do not broadly inspect Sparrow-V RTL, synthesis, FPGA, or ASIC files.

## Execution Strategy

Use the existing Sparrow-V external fixed-shape workload interface twice:

### Stage 1

Execute:

```text
input_int8[16]
× fc1_weight_int8[16,16]
+ fc1_bias
→ fc1_acc_int32[16]
```

### Intermediate processing

Apply:

```text
fc1_acc_int32
→ reconstructed hidden pre-ReLU values
→ ReLU
→ hidden requantization
→ hidden_int8[16]
```

### Stage 2

Execute:

```text
hidden_int8[16]
× fc2_weight_int8[4,16]
+ fc2_bias
→ fc2_acc_int32[4]
```

The adapter may invoke Sparrow-V once per layer if the existing external workload interface supports only one `16→N` operation at a time.

Do not require a single monolithic RTL simulation if two documented isolated runs are simpler and preserve correctness.

## Layer 1 Execution

Prepare an external Sparrow-V workload containing:

- Phase 6 input INT8 sample;
- `fc1` INT8 weights with shape `[16,16]`;
- `fc1` INT32 biases or a documented zero-bias execution plus reconstruction path;
- 16 expected first-layer accumulators;
- source package identity;
- layer identifier `fc1`.

The existing Sparrow-V external interface may currently be fixed to four outputs. Before implementation, inspect whether it can accept 16 outputs without RTL changes.

If the interface supports only four outputs, use four deterministic `16→4` executions covering output channels:

```text
fc1 outputs 0–3
fc1 outputs 4–7
fc1 outputs 8–11
fc1 outputs 12–15
```

Then concatenate the results in canonical output-channel order.

This partitioning is acceptable only if:

- each run uses the same 16-element input;
- each weight slice is exact;
- bias handling is exact and documented;
- all 16 accumulators reproduce the Phase 6 reference;
- counters are aggregated carefully;
- the partitioning does not alter arithmetic semantics.

Prefer reuse of the existing stable `16→4` interface over modifying Sparrow-V.

## Layer 2 Execution

Prepare one external Sparrow-V workload containing:

- validated hidden INT8 vector;
- `fc2` INT8 weights `[4,16]`;
- `fc2` INT32 biases or documented reconstruction path;
- four expected final accumulators;
- source package identity;
- layer identifier `fc2`.

Require exact match with the Phase 6 integer reference.

## Bias Provenance

The existing Sparrow-V workload template may only materialize signed 12-bit biases.

Handle each layer explicitly.

Permitted approach:

1. execute the RTL workload with zero bias or representable bias;
2. parse the RTL-produced dot-product accumulators;
3. add the full Phase 6 INT32 bias in runtime or host reconstruction;
4. record both pre-bias and post-bias values;
5. clearly label value provenance.

For every accumulator field, record one of:

```text
rtl_produced
runtime_software_produced
host_reconstructed
```

Do not describe host-reconstructed values as entirely RTL-produced.

If a full bias fits the current Sparrow-V runtime interface, use it directly and record that fact.

## Intermediate ReLU and Requantization Provenance

The existing Sparrow-V interface may not directly perform the exact Phase 6 per-channel reconstruction, ReLU, and hidden requantization.

The adapter may perform this step in SparrowML using the Phase 6 contract.

It must record:

- source first-layer accumulators;
- input scale;
- 16 `fc1` weight scales;
- reconstructed pre-ReLU values;
- ReLU outputs;
- hidden scale;
- rounding mode;
- clamping range;
- resulting hidden INT8 codes;
- provenance as host/runtime post-processing.

Require exact equality with the Phase 6 `intermediate_reference.json`.

Do not claim ReLU or requantization was executed in RTL unless it actually was.

## Exact Correctness Gates

Require exact integer equality for:

- all 16 post-bias first-layer accumulators;
- all 16 hidden INT8 activation codes;
- all four post-bias second-layer accumulators;
- final predicted class.

Floating-point reconstructed values may use a documented small tolerance because scales are serialized as float32.

Any one-integer mismatch must fail the milestone.

## Runtime Result Schema

Extend or version the existing result contract.

Preferred version:

```text
sparrowml_sparrowv_multilayer_runtime_result_v1
```

Record:

- package format and hash;
- Sparrow-V commit identity;
- sample ID;
- graph identifier;
- layer execution order;
- per-run commands;
- per-run exit status;
- first-layer execution partitions;
- first-layer raw RTL accumulators;
- first-layer reconstructed bias-adjusted accumulators;
- expected first-layer accumulators;
- exact match status;
- reconstructed pre-ReLU values;
- ReLU values;
- hidden INT8 activations;
- expected hidden INT8 activations;
- hidden exact match status;
- second-layer raw RTL accumulators;
- second-layer bias-adjusted accumulators;
- expected second-layer accumulators;
- final reconstructed logits;
- final prediction;
- expected prediction;
- counter summaries;
- provenance labels;
- semantic determinism;
- overall validation status.

Canonical semantic results must not contain absolute paths or wall-clock timestamps.

## Runtime Workspace

Use an isolated generated directory such as:

```text
artifacts/phase7_multilayer_runtime/
```

Recommended structure:

```text
artifacts/phase7_multilayer_runtime/
├── compatibility.json
├── fc1/
│   ├── partition_0/
│   ├── partition_1/
│   ├── partition_2/
│   └── partition_3/
├── intermediate/
│   └── hidden_trace.json
├── fc2/
│   └── result.json
├── multilayer_result.json
├── counter_report.json
├── determinism.json
└── summary.md
```

If Sparrow-V supports 16 outputs directly, use one `fc1` workspace instead of four partitions.

Generated artifacts must remain ignored.

## Counters

Collect only counters exposed by Sparrow-V.

For each layer or partition record:

- cycles;
- retired instructions;
- vector loads;
- vector stores;
- dense dot-product instructions;
- executed multiplications;
- traps or assertion failures.

Clearly distinguish:

- measured counters;
- derived counters;
- unavailable counters.

For a four-part first layer using four-output workloads:

- each partition performs four output dot products;
- total `fc1` dot products should be 16;
- total conceptual multiplications should be 256;
- total counters should be aggregated from the four measured runs.

For `fc2`:

- four output dot products;
- 64 conceptual multiplications.

Total model arithmetic:

```text
320 dense INT8 multiplications
```

Do not hardcode measured cycle or instruction totals.

## Counter Aggregation

If multiple simulations are required for `fc1`, report:

- per-partition counters;
- aggregate measured counters;
- number of simulator invocations;
- limits of comparing this partitioned execution against a future monolithic runtime.

Do not present summed simulation startup or control overhead as intrinsic model latency without qualification.

If cycle values are summed, label them:

```text
partitioned simulation cycle total
```

not necessarily end-to-end optimized hardware latency.

## Semantic Determinism

Run the complete multi-layer workflow at least twice.

Require identical:

- first-layer accumulators;
- hidden INT8 activations;
- second-layer accumulators;
- final prediction;
- parsed architectural counters, where deterministic;
- package identities;
- validation statuses.

Exclude from semantic hashes:

- host wall-clock duration;
- temporary paths;
- timestamps;
- incidental simulator logs.

## Compatibility Audit

Before implementation, update compatibility evidence to record:

- external interface version;
- supported input size;
- supported output count;
- whether 16-output execution is directly supported;
- whether partitioning is required;
- bias-width limitation;
- simulator tools;
- commands used;
- counter availability;
- Sparrow-V commit.

Do not change Sparrow-V automatically if the four-output partitioning path is sufficient.

## Configuration

Add:

```text
configs/experiments/sparrow_v_multilayer_runtime.yaml
```

Include:

- Phase 6 package path;
- Sparrow-V discovery policy;
- target configuration;
- external interface version;
- first-layer partition size;
- timeout;
- repeat count;
- workspace root;
- required counters;
- optional counters;
- bias policy;
- hidden processing policy;
- result schema version.

Avoid machine-specific absolute paths.

## CLI

Add bounded commands such as:

```text
sparrowml prepare-sparrowv-mlp
sparrowml run-sparrowv-mlp
sparrowml validate-sparrowv-mlp
sparrowml run-sparrowv-mlp-baseline
```

Exact names may follow current conventions.

### `prepare-sparrowv-mlp`

- load and validate Phase 6 package;
- create layer workspaces;
- emit layer manifests;
- do not run simulation.

### `run-sparrowv-mlp`

- execute all required `fc1` partitions;
- reconstruct and validate hidden activations;
- execute `fc2`;
- produce result schema.

### `validate-sparrowv-mlp`

- reload saved runtime evidence;
- verify package identity;
- revalidate all intermediate and final values.

### `run-sparrowv-mlp-baseline`

- ensure Phase 6 package exists;
- run compatibility check;
- execute complete workflow twice;
- validate semantic determinism;
- emit summary.

## Package Structure

Add only required modules.

Suggested:

```text
src/sparrowml/targets/sparrow_v/
├── multilayer.py
├── multilayer_workspace.py
├── multilayer_execution.py
├── multilayer_validation.py
└── multilayer_reports.py
```

Reuse Phase 5 discovery, compatibility, subprocess, parsing, and schema utilities.

Do not duplicate the entire adapter.

## Tests

Add focused tests for:

### Package loading

- Phase 6 package validation;
- expected operator sequence;
- required tensors;
- required traces;
- package hash verification.

### First-layer partitioning

- exact `[16,16]` to four `[4,16]` slices;
- correct output-channel ordering;
- correct bias slices;
- exact reference accumulator assembly;
- deterministic manifests.

### Bias handling

- representable bias path;
- zero-bias plus reconstruction path;
- provenance labels;
- negative and large INT32 biases;
- exact post-bias values.

### Intermediate processing

- first-layer reconstruction;
- ReLU;
- hidden requantization;
- ties-to-even;
- clamp `[0,127]`;
- exact hidden-code equality.

### Second-layer preparation

- hidden vector ordering;
- `fc2` weight ordering;
- bias ordering;
- expected accumulator loading.

### Runtime parsing

- 16 first-layer values across partitions;
- four second-layer values;
- missing partition;
- duplicate output channel;
- malformed result;
- trap/assertion failure.

### Counter aggregation

- measured versus derived labels;
- partition aggregation;
- unavailable counters;
- multiplication counts;
- cycle-total labelling.

### Validation

- one first-layer mismatch fails;
- one hidden-code mismatch fails;
- one second-layer mismatch fails;
- prediction mismatch fails;
- package-hash mismatch fails;
- provenance fields required.

### Integration

With a real Sparrow-V checkout:

- complete first-layer execution;
- exact first-layer accumulator match;
- exact hidden activation match;
- complete second-layer execution;
- exact second-layer accumulator match;
- final prediction match;
- two-run semantic determinism;
- Sparrow-V working tree remains clean.

Keep real integration tests separate from offline tests.

Tests must not:

- require internet;
- require GPU;
- modify Sparrow-V;
- retrain Phase 6;
- run synthesis, FPGA, or ASIC flows.

## Make Targets

Add:

```text
prepare-sparrowv-mlp
run-sparrowv-mlp
validate-sparrowv-mlp
run-sparrowv-mlp-baseline
test-phase7
test-phase7-integration
```

Update `make help`.

Expected behavior:

```text
make test-phase7
```

runs offline focused tests.

```text
make test-phase7-integration
```

requires the Sparrow-V checkout and runs real RTL simulation.

```text
make run-sparrowv-mlp-baseline
```

runs the full deterministic Phase 7 workflow.

## Documentation

Update or add:

- `README.md`
- `docs/architecture.md`
- `docs/build_roadmap.md`
- `docs/codex_context.md`
- `docs/data_contracts.md`
- `docs/experiment_policy.md`
- multi-layer runtime contract;
- Phase 7 results.

Suggested:

```text
docs/sparrow_v_multilayer_runtime_contract.md
docs/results/phase7_sparrow_v_multilayer_runtime.md
```

Document:

- layer partitioning;
- Sparrow-V interface used;
- bias provenance;
- intermediate ReLU/requantization provenance;
- exact correctness gates;
- counters and aggregation;
- partitioned-cycle limitations;
- semantic determinism;
- reproduction commands;
- remaining limitations.

Keep `docs/codex_context.md` concise.

## README Status

Update README only after real integration succeeds:

```text
Phase 7 multi-layer Sparrow-V RTL/reference validation implemented
```

Do not claim:

- monolithic optimized multi-layer hardware execution;
- hardware ReLU unless it is truly executed there;
- physical hardware execution;
- sparse MLP support;
- real-world model accuracy;
- hardware speedup.

## Existing Behavior Preservation

Preserve:

- Phase 1 FP32 baseline;
- Phase 2 INT8 reference;
- Phase 3 structured sparsity;
- Phase 4 export;
- Phase 5 single-layer RTL validation;
- Phase 6 multi-layer model/export;
- all existing CLI commands;
- all existing tests;
- Sparrow-V external interface.

Do not modify Sparrow-V.

## Out of Scope

Do not implement:

- Sparrow-V RTL changes;
- new instructions;
- monolithic multi-layer hardware scheduler;
- sparse MLP;
- pruning of the MLP;
- QAT;
- arbitrary graph runtime;
- real dataset evaluation;
- TinyNPU integration;
- target selection;
- hardware-aware optimization;
- synthesis;
- FPGA;
- ASIC flow;
- GUI or cloud execution.

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
make test-phase7
make smoke
make check
make docs-check
git diff --check
```

Run real integration once:

```text
make test-phase7-integration
make run-sparrowv-mlp-baseline
```

Verify Sparrow-V remains clean:

```text
git -C "$SPARROWV_ROOT" status --short
```

Do not run synthesis, FPGA, or ASIC targets.

## Acceptance Criteria

The milestone is complete only when:

1. Phase 6 package loading is supported.
2. The multi-layer graph is validated.
3. Sparrow-V checkout discovery is reused.
4. Compatibility audit records output-count capability.
5. First-layer execution strategy is explicit.
6. Four-way partitioning is used if required.
7. All first-layer weight slices are exact.
8. All first-layer bias slices are exact.
9. All 16 first-layer outputs are assembled in order.
10. First-layer raw RTL accumulators are recorded.
11. First-layer post-bias accumulators are recorded.
12. Accumulator provenance is recorded.
13. All 16 first-layer accumulators match exactly.
14. Reconstructed hidden values are recorded.
15. ReLU values are recorded.
16. Hidden quantization settings are recorded.
17. Hidden INT8 codes are recorded.
18. Hidden provenance is recorded.
19. All 16 hidden INT8 values match exactly.
20. Second-layer input ordering is exact.
21. Second-layer weights are exact.
22. Second-layer biases are exact.
23. Second-layer raw RTL accumulators are recorded.
24. Second-layer post-bias accumulators are recorded.
25. All four second-layer accumulators match exactly.
26. Final logits are reconstructed.
27. Final prediction matches exactly.
28. A versioned multi-layer result schema exists.
29. Package hashes are verified.
30. Simulator commands are captured.
31. Exit statuses are captured.
32. Stdout and stderr are retained.
33. Traps and assertion failures are rejected.
34. Counters are collected per layer or partition.
35. Counter provenance is explicit.
36. Partitioned cycle totals are labelled honestly.
37. Executed multiplication counts are reported.
38. Total dense multiplication count is 320.
39. Complete workflow runs twice.
40. Semantic hashes match across repeats.
41. Host-only nondeterminism is excluded from hashes.
42. Cross-layer summary is generated.
43. Phase 1 remains passing.
44. Phase 2 remains passing.
45. Phase 3 remains passing.
46. Phase 4 remains passing.
47. Phase 5 remains passing.
48. Phase 6 remains passing.
49. Phase 7 offline tests pass.
50. Phase 7 real integration tests pass.
51. Tests require no internet.
52. Tests require no GPU.
53. Sparrow-V tracked tree remains clean.
54. Sparrow-V RTL is not modified.
55. No commit occurs inside Sparrow-V.
56. Documentation matches implementation.
57. README status is accurate.
58. General checks pass.
59. Documentation checks pass.
60. `git diff --check` passes.
61. No commit or push occurs.
62. `docs/codex_milestone_result.md` is finalized.

## Stop Conditions

Stop for human review only if:

- Sparrow-V’s external interface cannot execute the first layer through deterministic `16→4` partitions;
- output partitioning changes arithmetic semantics;
- Phase 6 package values cannot be represented by the external interface;
- bias reconstruction cannot reproduce exact reference values;
- hidden activation reconstruction cannot reproduce Phase 6 traces;
- second-layer execution cannot use the validated hidden vector;
- Sparrow-V produces irreconcilable accumulator mismatches;
- a Sparrow-V source or RTL change is genuinely required;
- a major Phase 1–6 correctness defect is discovered.

If a Sparrow-V extension appears necessary, stop and report the smallest proposed non-RTL interface change. Do not modify Sparrow-V automatically.

Ordinary adapter, slicing, parsing, provenance, counter, test, and documentation bugs are not stop conditions.

## Token-Efficiency Instructions

Follow `AGENTS.md`.

In particular:

- reuse Phase 5 adapter infrastructure;
- inspect only the Phase 6 package and the existing external workload interface;
- do not broadly audit Sparrow-V;
- do not build an arbitrary graph runtime;
- use four-output partitioning if it avoids interface changes;
- run offline focused tests during development;
- run real RTL integration only after preparation is stable;
- run aggregate checks once;
- keep the result file concise.

## Result File

Update:

```text
docs/codex_milestone_result.md
```

Finalize with:

```text
STATUS: COMPLETE
```

only if real multi-layer Sparrow-V execution succeeds and every exact correctness gate passes.

Include:

- Sparrow-V commit and interface version;
- Phase 6 package identity;
- first-layer execution strategy;
- partition count;
- first-layer expected and observed accumulators;
- bias provenance;
- hidden expected and observed INT8 activations;
- ReLU/requantization provenance;
- second-layer expected and observed accumulators;
- final prediction;
- measured counters;
- derived counters;
- unavailable counters;
- partitioned cycle and instruction totals;
- semantic determinism evidence;
- generated artifact paths;
- validation commands and outcomes;
- changed SparrowML files;
- confirmation that Sparrow-V remained clean;
- remaining limitations;
- next recommended milestone;
- confirmation that no commit or push occurred.

Use `STATUS: FAILED` if implementation or correctness checks remain incomplete.

Use `STATUS: BLOCKED` only for a genuine Sparrow-V interface or architectural blocker.

## Next Milestone

The expected next milestone is:

```text
Real vibration dataset integration and full FP32/INT8/Sparrow-V evaluation
```

Do not implement it during this milestone.