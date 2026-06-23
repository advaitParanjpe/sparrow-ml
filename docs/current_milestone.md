# Milestone: Sparrow-V Runtime Adapter, Simulator Execution, and RTL/Reference Validation

## Objective

Connect SparrowML’s deterministic Phase 4 deployment packages to the external Sparrow-V repository and validate execution against SparrowML’s integer reference models.

This milestone must:

1. discover and validate a local Sparrow-V checkout;
2. define a stable SparrowML-to-Sparrow-V runtime adapter;
3. inspect and use Sparrow-V’s existing documented simulation interfaces;
4. convert Phase 4 deployment packages into the exact files required by Sparrow-V;
5. execute dense and sparse workloads through Sparrow-V simulation;
6. parse architectural results and performance counters;
7. compare Sparrow-V results against SparrowML’s expected integer-reference outputs;
8. generate deterministic integration reports;
9. preserve a strict repository boundary with no Sparrow-V RTL modification;
10. provide one command that exports, executes, validates, and reports both dense and sparse modes.

This milestone is about **real cross-repository execution and result validation**.

Do not add new Sparrow-V instructions, modify RTL, redesign the CPU/vector interface, add multi-layer models, or perform hardware-aware optimization.

## Current Baseline

Phase 1 provides:

- deterministic synthetic sensor fixture;
- FP32 `Linear(16, 4)` baseline;
- train-only preprocessing.

Phase 2 provides:

- signed INT8 input quantization;
- per-output-channel INT8 weights;
- INT32 biases;
- exact dense integer reference inference.

Phase 3 provides:

- deterministic 2:4 structured sparsity;
- compressed INT8 weights;
- packed three-bit metadata;
- exact compressed sparse integer reference inference.

Phase 4 provides:

- versioned SparrowML IR;
- deterministic dense and sparse deployment packages;
- target memory maps;
- model and input binaries;
- symbolic programs;
- expected accumulators, logits, and predictions;
- package reload validation;
- byte-for-byte deterministic export.

Supported execution modes remain:

```text
dense_int8
sparse_2of4_int8
```

Supported model remains:

```text
Linear(16, 4)
```

## Repository Boundary

SparrowML owns:

- package discovery;
- compatibility checks;
- runtime orchestration;
- adapter-side file conversion;
- subprocess execution;
- result parsing;
- reference comparison;
- integration reports.

Sparrow-V owns:

- RTL;
- software/runtime code inside Sparrow-V;
- assembler or image generation already present there;
- simulation harness;
- architectural counters;
- instruction semantics;
- final simulated execution.

This milestone must not:

- modify Sparrow-V RTL;
- copy Sparrow-V RTL into SparrowML;
- commit files inside Sparrow-V;
- invent unsupported ISA encodings;
- silently patch Sparrow-V;
- claim FPGA or physical-hardware execution.

Temporary generated files may be placed in a documented ignored Sparrow-V build/output directory only if the existing interface requires it. Prefer an isolated temporary workspace or SparrowML-owned integration directory.

## Repository Discovery

Support these Sparrow-V discovery methods, in priority order:

1. explicit `SPARROWV_ROOT` environment variable;
2. target configuration value if repository-relative and portable;
3. sibling repository fallback such as:

```text
../sparrow-v
```

Requirements:

- resolve to an absolute path internally;
- do not serialize the absolute path into deterministic artifacts;
- verify the directory exists;
- verify expected Sparrow-V contract files exist;
- verify it is not the SparrowML repository;
- produce a concise actionable error when missing.

Add a diagnostic command such as:

```text
sparrowml sparrowv-doctor
```

It should report:

- resolved checkout status;
- compatibility status;
- discovered simulator/tool availability;
- supported dense target;
- supported sparse target;
- missing requirements;
- no repository modifications.

## Relevant Files

Read first:

- `AGENTS.md`
- `docs/codex_context.md`
- `docs/current_milestone.md`
- `docs/architecture.md`
- `docs/data_contracts.md`
- `docs/sparrow_v_export_contract.md`
- `docs/results/phase4_sparrow_v_export.md`
- `configs/targets/sparrow_v.yaml`
- Phase 4 compiler/export modules;
- existing SparrowML target adapter skeleton and contracts.

After resolving Sparrow-V, inspect only the minimum Sparrow-V files required to identify:

- supported execution commands;
- expected input/program formats;
- existing dense vector instruction path;
- existing sparse vector instruction path;
- result-output format;
- available counters;
- stable Make targets or scripts.

Do not perform a broad Sparrow-V audit.

Do not inspect unrelated RTL, ASIC-flow, synthesis, documentation, or historical experiment files.

## Preflight Contract Audit

Before implementing the adapter, produce a concise machine-readable compatibility record.

It must identify:

- Sparrow-V repository identity;
- relevant interface or schema version, if present;
- command used for dense execution;
- command used for sparse execution;
- required generated files;
- accepted input formats;
- output/result format;
- available counters;
- timeout behavior;
- simulator dependencies;
- whether existing interfaces can execute the Phase 4 model without RTL changes.

Store this under generated Phase 5 artifacts, not as a speculative hand-written claim.

If Sparrow-V does not expose a usable stable interface, stop with `STATUS: BLOCKED` and explain the smallest required human decision. Do not invent a new ISA or silently modify Sparrow-V.

## Runtime Adapter

Implement a bounded adapter under a structure such as:

```text
src/sparrowml/targets/sparrow_v/
├── __init__.py
├── discovery.py
├── compatibility.py
├── adapter.py
├── workspace.py
├── commands.py
├── execution.py
├── parsing.py
├── validation.py
└── reports.py
```

Adjust to match the existing scaffold.

The adapter must:

- accept one Phase 4 deployment package;
- determine dense or sparse mode;
- validate package and target compatibility;
- prepare a deterministic execution workspace;
- invoke the existing Sparrow-V interface;
- capture stdout, stderr, exit status, and elapsed host time;
- parse results into a typed SparrowML result object;
- compare them with `expected_output.json`;
- cleanly report success, mismatch, timeout, or tool failure.

Do not create a generalized remote-execution framework.

## Execution Workspace

Use an isolated generated directory such as:

```text
artifacts/phase5_runtime/workspaces/dense/
artifacts/phase5_runtime/workspaces/sparse/
```

or a temporary directory whose relevant outputs are copied into Phase 5 artifacts.

The workspace may contain:

- deployment package copy or links;
- generated assembly or C source if required by Sparrow-V’s existing interface;
- generated memory image;
- build logs;
- simulator stdout and stderr;
- parsed result JSON.

Requirements:

- no source files in Sparrow-V are overwritten;
- no tracked Sparrow-V files are modified;
- generated files are isolated and clearly named;
- stale outputs are removed or rejected;
- dense and sparse runs cannot collide;
- workspace paths do not enter deterministic semantic result payloads.

## Program Generation

Phase 4 emitted symbolic `program.json`.

Translate the symbolic program into the existing Sparrow-V software or image interface.

Allowed approaches, depending on Sparrow-V’s current stable workflow:

- generate bare-metal assembly;
- generate C using existing intrinsics or inline assembly;
- generate an accepted instruction/data image;
- populate an existing workload template through documented parameters.

Requirements:

- use existing documented instruction encodings or assembler support;
- preserve feature, weight, metadata, bias, and output ordering;
- use the correct dense `VDOT8` behavior;
- use the correct sparse `VSDOT8` behavior;
- use only supported load/store and scalar operations;
- do not hand-invent undocumented opcodes;
- generated program must be deterministic;
- generated source or image must be retained in the Phase 5 evidence.

## Workload Semantics

For the canonical exported sample, the program must:

1. load the 16 quantized input features;
2. load dense or compressed sparse model data;
3. execute four output-channel dot products;
4. include the appropriate INT32 bias contribution;
5. store or report four final INT32 accumulators;
6. expose predicted class if supported;
7. expose required counters;
8. terminate through the existing Sparrow-V success mechanism.

If Sparrow-V cannot directly apply biases or reconstruct per-channel logits in hardware/software, the adapter may perform clearly documented host-side post-processing only where the Phase 4 contract expects it.

The following distinction must remain explicit:

- **RTL-produced values**
- **runtime software-produced values**
- **host-side reconstructed values**

Do not label host-side calculations as RTL results.

## Dense Execution

Run the Phase 4 dense package through Sparrow-V’s dense vector path.

Required validation:

- four final accumulators exactly match Phase 4 dense expected accumulators;
- output order matches class order;
- prediction based on reconstructed logits matches the SparrowML reference;
- no simulator assertion or architectural trap occurs;
- dense instruction and load counters are recorded where available.

Reference canonical dense expected accumulators from Phase 4:

```text
[39603, -17389, -1218, -26014]
```

Do not hardcode these as the implementation oracle. Load them from the package.

## Sparse Execution

Run the Phase 4 sparse package through Sparrow-V’s compressed 2:4 path.

Required validation:

- compressed weights are consumed in defined order;
- metadata is decoded using the existing Sparrow-V encoding;
- four final accumulators exactly match Phase 4 sparse expected accumulators;
- prediction matches the SparrowML compressed sparse reference;
- no invalid metadata is emitted;
- no simulator assertion or architectural trap occurs;
- sparse instruction, executed-multiply, and skipped-multiply counters are recorded where available.

Reference canonical sparse expected accumulators from Phase 4:

```text
[27952, -9483, -738, -19017]
```

Load the oracle from the package rather than hardcoding it.

## Result Contract

Define a versioned result schema such as:

```text
sparrowml_sparrowv_runtime_result_v1
```

Each run result must record:

- mode;
- package identity and hash;
- Sparrow-V compatibility identity;
- sample ID;
- simulator command in normalized form;
- exit status;
- termination reason;
- raw output locations;
- parsed accumulators;
- reconstructed logits;
- predicted class ID;
- predicted class name;
- expected accumulators;
- exact accumulator-match status;
- expected prediction;
- prediction-match status;
- counters;
- validation status;
- failure diagnostics.

Do not include nonportable absolute paths in the canonical semantic result.

Host-specific diagnostic logs may contain local paths but should not be part of determinism hashes.

## Counters

Parse only counters actually exposed by Sparrow-V.

Desired counters include:

- total cycles;
- retired instructions;
- vector loads;
- vector stores;
- dense dot-product instructions;
- sparse dot-product instructions;
- executed multiplications;
- skipped multiplications;
- stale/wrong-path responses if relevant;
- traps or errors.

Requirements:

- distinguish measured counters from derived counters;
- label missing counters as unavailable rather than inventing values;
- record units;
- preserve raw counter values;
- validate nonnegative integer fields;
- check obvious consistency relationships where applicable.

Examples:

```text
dense executed multiplications = 64
sparse executed multiplications = 32
sparse skipped multiplications = 32
```

These may be derived only if the RTL does not directly expose them, and must then be labelled `derived`.

## Correctness Gates

For both dense and sparse execution:

- simulator exits successfully;
- no assertion failure;
- no unexpected trap;
- exactly four accumulators are parsed;
- accumulators exactly match package expectations;
- predicted class matches;
- output ordering matches;
- package hash matches the executed package;
- result schema validates.

The milestone must fail if an accumulator differs by even one integer value.

Do not use numerical tolerance for INT32 accumulator comparison.

Reconstructed FP32 logits may use a documented small tolerance because scales are serialized as float32, but prediction must match exactly.

## Cross-Mode Report

Generate a combined report comparing:

- dense expected versus Sparrow-V accumulators;
- sparse expected versus Sparrow-V accumulators;
- dense and sparse predictions;
- dense and sparse cycles;
- dense and sparse retired instructions;
- dense and sparse memory/load counts;
- dense and sparse dot-product counts;
- executed and skipped multiplications;
- model storage;
- package sizes;
- arithmetic reduction;
- observed runtime differences.

Do not claim speedup unless the cycle measurements are comparable and correctly collected.

If dense and sparse both take the same number of cycles, report that directly and explain that arithmetic reduction did not translate into latency reduction for this workload.

## Determinism

Distinguish two forms of determinism.

### Semantic determinism

Repeated runs must produce identical:

- accumulators;
- predictions;
- parsed architectural counters, assuming deterministic RTL;
- package identities;
- validation results.

### Host-log determinism

Do not require identical:

- elapsed wall-clock time;
- temporary paths;
- simulator build timestamps;
- tool-generated incidental logs.

Canonical result hashes must exclude nondeterministic host diagnostics.

Run each mode at least twice and record semantic determinism evidence.

## Configuration

Add a configuration such as:

```text
configs/experiments/sparrow_v_runtime.yaml
```

Include:

- dense package path;
- sparse package path;
- target configuration;
- repository discovery policy;
- dense execution target;
- sparse execution target;
- timeout;
- workspace root;
- retained-log policy;
- expected result schema version;
- required versus optional counters;
- semantic determinism repeat count.

Avoid hardcoded machine-specific absolute paths.

## CLI

Add bounded commands such as:

```text
sparrowml sparrowv-doctor
sparrowml prepare-sparrowv-run
sparrowml run-sparrowv
sparrowml validate-sparrowv-result
sparrowml run-sparrowv-baseline
```

Exact names may follow current conventions.

### `sparrowv-doctor`

- resolve checkout;
- inspect compatibility;
- verify required tools and targets;
- perform no build or execution unless explicitly requested.

### `prepare-sparrowv-run`

- validate one package;
- generate isolated workspace and program/image inputs;
- do not execute simulation.

### `run-sparrowv`

- execute one selected mode;
- capture logs;
- parse result;
- validate against package oracle.

### `validate-sparrowv-result`

- reload saved result and evidence;
- revalidate schema, package identity, accumulators, predictions, and counters.

### `run-sparrowv-baseline`

- ensure Phase 4 packages exist;
- run doctor;
- execute dense twice;
- execute sparse twice;
- validate semantic determinism;
- write combined report.

Use proper exit codes.

## Generated Artifacts

Use:

```text
artifacts/phase5_runtime/
```

At minimum generate:

```text
artifacts/phase5_runtime/compatibility.json
artifacts/phase5_runtime/dense/result.json
artifacts/phase5_runtime/dense/stdout.log
artifacts/phase5_runtime/dense/stderr.log
artifacts/phase5_runtime/dense/generated_program.*
artifacts/phase5_runtime/sparse/result.json
artifacts/phase5_runtime/sparse/stdout.log
artifacts/phase5_runtime/sparse/stderr.log
artifacts/phase5_runtime/sparse/generated_program.*
artifacts/phase5_runtime/cross_mode_report.json
artifacts/phase5_runtime/summary.md
artifacts/phase5_runtime/determinism.json
```

Exact generated program extension depends on the existing Sparrow-V interface.

Generated artifacts should remain ignored.

Do not overwrite Phase 1–4 artifacts.

## Tests

Add focused tests for:

### Discovery

- explicit environment variable;
- sibling fallback;
- missing checkout;
- wrong repository;
- required-file detection;
- no absolute paths serialized.

### Compatibility

- compatible interface;
- missing target;
- unsupported interface version;
- missing simulator;
- available-counter discovery;
- clear blocker diagnostics.

Use temporary fake repositories for unit tests.

### Program generation

- dense program ordering;
- sparse program ordering;
- input ordering;
- weight ordering;
- metadata ordering;
- bias ordering;
- deterministic generated source;
- no invented unsupported operation.

### Workspace

- isolated dense/sparse paths;
- stale-output cleanup;
- no source overwrite;
- no tracked repository modification;
- safe subprocess working directory.

### Execution

- command construction;
- timeout handling;
- nonzero exit handling;
- stdout/stderr capture;
- missing-output handling.

Mock subprocess execution for ordinary unit tests.

### Parsing

- valid dense result;
- valid sparse result;
- negative accumulators;
- exactly four outputs;
- malformed output;
- missing counters;
- duplicate fields;
- unexpected traps.

### Validation

- exact accumulator match;
- one-value mismatch failure;
- prediction mismatch;
- package-hash mismatch;
- result-schema validation;
- reconstructed-logit tolerance;
- measured versus derived counters.

### Integration

When a real compatible Sparrow-V checkout is available:

- dense real simulation;
- sparse real simulation;
- exact reference equivalence;
- semantic repeat determinism;
- no Sparrow-V tracked-file changes.

Mark real cross-repository integration tests separately so normal unit tests remain runnable without Sparrow-V.

Tests must not:

- require internet;
- require GPU;
- modify Sparrow-V;
- commit or push;
- rerun model training;
- rerun pruning;
- perform synthesis or ASIC flows.

## Make Targets

Add stable targets such as:

```text
sparrowv-doctor
prepare-sparrowv-dense
prepare-sparrowv-sparse
run-sparrowv-dense
run-sparrowv-sparse
run-sparrowv-baseline
test-phase5
test-phase5-integration
```

Update `make help`.

Expected behavior:

```text
make test-phase5
```

runs offline unit tests without requiring Sparrow-V.

```text
make test-phase5-integration
```

requires a compatible local Sparrow-V checkout and executes real simulation.

```text
make run-sparrowv-baseline
```

runs the complete dense/sparse integration workflow.

## Documentation

Update or add:

- `README.md`;
- `docs/architecture.md`;
- `docs/build_roadmap.md`;
- `docs/data_contracts.md`;
- `docs/experiment_policy.md`;
- `docs/codex_context.md`;
- Phase 5 runtime contract;
- Phase 5 result documentation.

Suggested:

```text
docs/sparrow_v_runtime_contract.md
docs/results/phase5_sparrow_v_runtime.md
```

Document:

- repository discovery;
- compatibility checks;
- existing Sparrow-V interfaces used;
- generated program format;
- workspace policy;
- command invocation;
- parsing contract;
- result schema;
- measured versus derived counters;
- exact correctness gates;
- determinism policy;
- limitations;
- reproduction commands.

Keep `docs/codex_context.md` concise.

## README Status

Update README status only after real simulation succeeds:

```text
Phase 5 Sparrow-V simulation and RTL/reference validation implemented
```

Do not claim:

- physical hardware execution;
- FPGA execution;
- ASIC timing or power;
- broad model support;
- multi-layer compilation;
- real-world accuracy;
- hardware speedup unless measured cycles support it.

## Existing Behavior Preservation

Preserve:

- Phase 1 FP32 baseline;
- Phase 2 INT8 quantization;
- Phase 3 structured sparsity;
- Phase 4 IR and export;
- all existing CLI commands;
- all existing tests;
- deterministic artifacts;
- repository boundary.

Do not modify Sparrow-V.

## Out of Scope

Do not implement:

- RTL changes;
- new instructions;
- assembler changes unless a pre-existing documented extension point is explicitly intended for generated workloads;
- FPGA execution;
- synthesis;
- OpenLane/OpenROAD;
- power analysis;
- multi-layer models;
- hidden activations;
- compiler optimization;
- hardware-aware pruning;
- target selection;
- TinyNPU integration;
- real dataset work;
- research experiments;
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
make smoke
make check
make docs-check
git diff --check
```

Run real integration once:

```text
make test-phase5-integration
make run-sparrowv-baseline
```

Verify Sparrow-V remains unchanged:

```text
git -C "$SPARROWV_ROOT" status --short
```

If `SPARROWV_ROOT` is not set, use the adapter-resolved checkout path.

Do not run Sparrow-V synthesis, FPGA, or ASIC targets.

## Acceptance Criteria

The milestone is complete only when:

1. Sparrow-V checkout discovery exists.
2. `SPARROWV_ROOT` is supported.
3. Sibling checkout fallback is supported.
4. Missing checkout errors are actionable.
5. A compatibility audit exists.
6. Dense execution interface is identified.
7. Sparse execution interface is identified.
8. Required tools are checked.
9. Existing Sparrow-V interfaces are used.
10. No unsupported ISA encoding is invented.
11. A versioned runtime-result schema exists.
12. Dense Phase 4 package can be prepared for execution.
13. Sparse Phase 4 package can be prepared for execution.
14. Workspaces are isolated.
15. Sparrow-V source files are not overwritten.
16. Generated dense program/image is deterministic.
17. Generated sparse program/image is deterministic.
18. Dense simulation executes successfully.
19. Sparse simulation executes successfully.
20. Simulator stdout is captured.
21. Simulator stderr is captured.
22. Exit status is captured.
23. Timeouts are handled.
24. Unexpected traps are rejected.
25. Assertion failures are rejected.
26. Four dense accumulators are parsed.
27. Four sparse accumulators are parsed.
28. Dense accumulators exactly match the Phase 4 oracle.
29. Sparse accumulators exactly match the Phase 4 oracle.
30. Dense prediction matches.
31. Sparse prediction matches.
32. Output class ordering is preserved.
33. Package hashes are verified.
34. Result schemas validate.
35. Available architectural counters are parsed.
36. Missing counters are labelled unavailable.
37. Derived counters are labelled derived.
38. Measured counters are labelled measured.
39. Dense cycles are reported if exposed.
40. Sparse cycles are reported if exposed.
41. Retired instructions are reported if exposed.
42. Dense/sparse vector operation counts are reported.
43. Executed/skipped multiplication counts are reported or carefully derived.
44. Cross-mode report is generated.
45. No unsupported speedup claim is made.
46. Dense semantic repeat run is deterministic.
47. Sparse semantic repeat run is deterministic.
48. Nondeterministic host diagnostics are excluded from semantic hashes.
49. Real integration tests are separated from offline tests.
50. Offline Phase 5 tests do not require Sparrow-V.
51. Integration tests require no internet.
52. Integration tests require no GPU.
53. Phase 1 remains passing.
54. Phase 2 remains passing.
55. Phase 3 remains passing.
56. Phase 4 remains passing.
57. Phase 5 unit tests pass.
58. Phase 5 real integration tests pass.
59. General repository checks pass.
60. Documentation checks pass.
61. `git diff --check` passes.
62. Sparrow-V tracked working tree remains unchanged.
63. Sparrow-V RTL is not modified.
64. SparrowML does not commit inside Sparrow-V.
65. Documentation matches implementation.
66. README status is accurate.
67. One command reproduces both execution modes.
68. No commit or push occurs.
69. `docs/codex_milestone_result.md` is finalized.

## Stop Conditions

Stop for human review only if:

- no compatible Sparrow-V checkout can be found;
- required simulator tools are unavailable and cannot be installed;
- Sparrow-V lacks an existing dense or sparse execution path for this model;
- Sparrow-V’s current software interface requires a source-level change that would violate the repository boundary;
- Phase 4 package semantics conflict with the real Sparrow-V ISA or runtime;
- Sparrow-V produces results that cannot be reconciled with the documented instruction semantics;
- a major Phase 1–4 correctness defect is discovered.

If a minimal non-RTL Sparrow-V harness addition appears necessary, stop and describe:

- the exact missing interface;
- the smallest proposed change;
- why it belongs in Sparrow-V rather than SparrowML;
- whether it changes architectural behavior.

Do not make that change automatically.

Ordinary adapter bugs, parser bugs, command errors, timeout handling, counter omissions, test failures, and documentation work are not stop conditions.

## Token-Efficiency Instructions

Follow `AGENTS.md`.

In particular:

- read compact SparrowML context first;
- inspect only the minimum Sparrow-V interface files;
- do not audit RTL broadly;
- do not inspect synthesis or ASIC flows;
- support only the current dense and sparse linear packages;
- use offline mocks for iterative unit testing;
- run real simulation only when focused integration is ready;
- run aggregate checks once;
- avoid repeated rebuilds where cached Sparrow-V outputs are valid;
- keep the result file concise.

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

only if real dense and sparse Sparrow-V simulation succeeds and all required correctness gates pass.

Include:

- resolved Sparrow-V compatibility identity;
- existing execution interfaces used;
- generated program/image format;
- dense simulator command and result;
- sparse simulator command and result;
- dense expected and observed accumulators;
- sparse expected and observed accumulators;
- exact-match status;
- predictions;
- measured counters;
- derived counters;
- unavailable counters;
- cycle and instruction comparison;
- semantic determinism evidence;
- generated artifact paths;
- exact validation commands and outcomes;
- changed SparrowML files;
- confirmation that Sparrow-V working tree remained unchanged;
- remaining limitations;
- next recommended milestone;
- confirmation that no commit or push occurred.

Use `STATUS: FAILED` if implementation is incomplete or correctness checks fail.

Use `STATUS: BLOCKED` only for a genuine Sparrow-V interface, toolchain, or repository-boundary blocker.

## Next Milestone

The expected next milestone is:

```text
Multi-layer INT8 model, intermediate activation quantization, and multi-operator compilation
```

Do not implement it during this milestone.