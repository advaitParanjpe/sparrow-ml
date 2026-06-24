# Milestone: Final SparrowML Portfolio Polish, Reproducibility, and Release Readiness

## Objective

Finalize SparrowML as a polished, truthful, reproducible, portfolio-ready hardware–software co-design project.

The core technical project is already complete:

- deterministic WISDM data ingestion;
- subject-held-out real-data evaluation;
- FP32 and INT8 multi-layer inference;
- structured sparsity experiments;
- versioned compiler IR;
- deterministic Sparrow-V deployment packages;
- single-layer and multi-layer RTL/reference validation;
- exact intermediate and final integer agreement;
- real held-out WISDM samples validated through Sparrow-V simulation.

This milestone must not add new model architectures, RTL, ISA features, datasets, compiler operators, optimization methods, or research experiments.

It must consolidate the existing implementation and measured results into a clear final repository that another engineer can understand, reproduce, audit, and discuss in an interview.

## Final Project Positioning

Use this concise technical framing throughout the repository:

> SparrowML is a hardware-aware edge-AI training, quantization, compilation, and runtime pipeline that deploys a subject-held-out WISDM activity-recognition model onto Sparrow-V, a custom RISC-V processor with INT8 vector execution. The system preserves exact integer semantics across software reference inference, compiler-generated deployment packages, and RTL simulation.

Do not describe SparrowML as:

- a general-purpose ML compiler;
- an ONNX or TVM replacement;
- a production wearable system;
- physical FPGA or ASIC deployment;
- a real-time optimized runtime;
- a medically certified system;
- a monolithic hardware inference engine.

## Canonical Final Results

Treat these as the canonical measured real-data results unless repository artifacts show a direct inconsistency:

### Dataset

- dataset: WISDM smartphone and smartwatch activity dataset;
- device used: smartphone;
- sensor used: accelerometer;
- classes:
  - walking;
  - jogging;
  - sitting;
  - standing;
- eligible subjects: 49;
- subject split:
  - 35 train;
  - 7 validation;
  - 7 test;
- accepted windows: 25,768;
- window length: 80 samples;
- overlap: 50%;
- input features: 16;
- evaluation: subject-held-out.

### Model

```text
Linear(16,16)
ReLU
Linear(16,4)
```

- parameters: 340;
- execution: FP32 training and explicit INT8 inference;
- compiler graph:
  - `DenseLinearInt8`;
  - `ReLU`;
  - `RequantizeInt8`;
  - `DenseLinearInt8`.

### Real-data quality

- FP32 accuracy: `0.9259473531964131`;
- FP32 macro-F1: `0.9287458208759758`;
- FP32 balanced accuracy: `0.9296898801135173`;
- INT8 accuracy: `0.9175585768006942`;
- INT8 macro-F1: `0.9197794804065271`;
- INT8 balanced accuracy: `0.920638703760132`;
- INT8 macro-F1 drop: `0.008966340469448664`;
- FP32/INT8 agreement: `0.9872722013306335`.

### RTL validation

- held-out WISDM samples selected: 12;
- `fc1` exact matches: 12/12;
- hidden INT8 exact matches: 12/12;
- `fc2` exact matches: 12/12;
- prediction exact matches: 12/12;
- Sparrow-V remained unmodified and clean.

### Runtime limitations

The current multi-layer execution uses:

- four isolated `fc1` partitions;
- one isolated `fc2` run;
- host-side full INT32 bias reconstruction;
- host-side ReLU and requantization;
- partitioned simulation cycle totals.

Do not present these counters as optimized monolithic end-to-end latency.

## Relevant Files

Read first:

- `AGENTS.md`
- `docs/codex_context.md`
- `docs/current_milestone.md`
- `README.md`
- `docs/architecture.md`
- `docs/build_roadmap.md`
- all final results documents under `docs/results/`;
- reproducibility and contract documentation;
- Make targets and CLI help;
- repository checks;
- tracked source manifest.

Inspect implementation files only where necessary to verify documentation or commands.

Do not broadly refactor working code.

## Workstream 1 — Final README

Rewrite and polish the root `README.md` so it works as the primary portfolio entry point.

It should include, in this order:

1. project title and one-sentence description;
2. concise project motivation;
3. final system overview;
4. architecture diagram;
5. supported workflow;
6. canonical WISDM results table;
7. RTL validation results;
8. quick-start reproduction;
9. repository structure;
10. limitations;
11. documentation links;
12. licence.

The README should be concise enough to skim but detailed enough for a hardware or ML-systems interviewer.

### README opening

Use a strong opening along these lines:

> SparrowML is a hardware-aware edge-AI compiler and runtime that trains and quantizes a human-activity-recognition model, lowers it into a deterministic deployment package, and validates exact integer execution through Sparrow-V RTL simulation.

Avoid inflated wording such as:

- production-grade;
- state of the art;
- industry leading;
- tapeout ready;
- real-time;
- complete general compiler.

## Workstream 2 — Architecture Diagrams

Add or refine clear Mermaid diagrams.

At minimum include:

### End-to-end pipeline

```text
WISDM raw accelerometer data
→ subject-safe windows
→ 16-feature extraction
→ FP32 MLP training
→ INT8 calibration/quantization
→ SparrowML IR
→ binary deployment package
→ Sparrow-V runtime adapter
→ RTL simulation
→ exact reference validation
```

### Multi-layer execution

```text
INT8 input
→ fc1 INT8 vector dots
→ INT32 accumulators
→ bias reconstruction
→ ReLU
→ hidden INT8 requantization
→ fc2 INT8 vector dots
→ final logits/prediction
```

Visually distinguish:

- SparrowML software;
- generated artifacts;
- Sparrow-V runtime;
- RTL execution;
- host-side reconstruction.

Do not imply host-side operations occur in RTL.

## Workstream 3 — Consolidated Final Results

Create or finalize:

```text
docs/results/final_results.md
```

It must consolidate the project into one authoritative report.

Include:

### Dataset protocol

- source dataset;
- selected device and sensor;
- classes;
- subject split;
- windowing;
- feature extraction;
- leakage prevention.

### Model results

A compact table containing:

| Model | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| FP32 MLP | measured value | measured value | measured value |
| INT8 MLP | measured value | measured value | measured value |

### Quantization

Include:

- macro-F1 degradation;
- prediction agreement;
- input and hidden calibration policy;
- clipping summary;
- accumulator safety.

### Deployment correctness

Include:

| Validation level | Result |
|---|---:|
| Package reload | exact |
| `fc1` accumulators | 12/12 exact |
| Hidden INT8 codes | 12/12 exact |
| `fc2` accumulators | 12/12 exact |
| Predictions | 12/12 exact |

### Compute and runtime

Report:

- 256 conceptual `fc1` multiplications per sample;
- 64 conceptual `fc2` multiplications per sample;
- 320 total conceptual multiplications;
- measured partitioned simulation counters;
- measured versus derived provenance;
- partitioning limitations.

### Earlier controlled experiments

Briefly summarize, without overwhelming the final story:

- synthetic fixture;
- dense INT8 reference;
- 2:4 structured sparsity;
- 50% arithmetic reduction;
- 40.625% compressed weight-storage reduction;
- equal single-layer dense/sparse cycle result;
- lack of measured sparse speedup.

Position these as controlled system-validation experiments, not the final WISDM model result.

## Workstream 4 — Reproducibility Guide

Create or finalize:

```text
docs/reproduction.md
```

It must provide an ordered reproduction path.

### Environment prerequisites

Document:

- supported Python version;
- Python dependencies;
- PyTorch;
- NumPy;
- PyYAML;
- pytest;
- Icarus Verilog;
- `vvp`;
- Sparrow-V sibling checkout;
- local WISDM dataset.

### Environment variables

```bash
export WISDM_ROOT=~/Datasets/WISDM/wisdm-dataset
export SPARROWV_ROOT=~/Desktop/projects/sparrow-v
```

### Fast verification

Provide commands that do not retrain or rerun every expensive stage unnecessarily.

For example:

```bash
make doctor
make check
make docs-check
```

### Full real-data workflow

Document:

```bash
make run-wisdm-phase8a
make run-wisdm-phase8b
make run-wisdm-phase8c
```

and:

```bash
make run-wisdm-final
```

### RTL integration

Document required commands and expected outputs.

Clearly state which commands:

- require WISDM;
- require Sparrow-V;
- run RTL simulation;
- regenerate training artifacts;
- may take longer.

Do not include machine-specific absolute paths except as examples.

## Workstream 5 — Repository Navigation

Ensure the repository has a clear source and documentation map.

Update:

```text
docs/source_manifest.md
```

or equivalent.

The map should identify:

- data ingestion;
- feature extraction;
- models;
- training;
- quantization;
- sparsity;
- compiler;
- target adapters;
- CLI;
- tests;
- configurations;
- result documents.

Remove or fix stale references.

## Workstream 6 — CLI and Make Help

Audit:

```bash
make help
python3 -m sparrowml.cli --help
```

Ensure important commands are discoverable and grouped logically.

The final help should distinguish:

- general checks;
- synthetic baseline workflows;
- compiler/export workflows;
- Sparrow-V integration;
- WISDM real-data workflow;
- final verification.

Do not add duplicate aliases unless they materially improve usability.

## Workstream 7 — Clean Repository State

Audit for unwanted tracked or untracked files:

- `.DS_Store`;
- caches;
- checkpoints;
- raw datasets;
- processed windows;
- generated packages;
- simulator logs;
- temporary workspaces;
- Python bytecode;
- virtual environments.

Ensure `.gitignore` covers all generated paths.

Do not remove legitimate source-controlled test fixtures.

## Workstream 8 — Claims Audit

Perform a strict claims audit across:

- README;
- final results;
- architecture documentation;
- phase result documents;
- comments and help text.

Correct misleading or inconsistent claims.

Required distinctions:

- synthetic fixture accuracy versus WISDM accuracy;
- FP32 versus INT8;
- software reference versus RTL output;
- RTL-produced versus host-reconstructed;
- measured versus derived counters;
- partitioned simulation totals versus optimized latency;
- RTL simulation versus physical hardware;
- structured-sparsity reduction versus measured speedup.

Do not erase useful earlier results. Label them correctly.

## Workstream 9 — Release Checklist

Create:

```text
docs/release_checklist.md
```

Include checks for:

- tests;
- documentation;
- repository cleanliness;
- dataset exclusion;
- artifact exclusion;
- Sparrow-V cleanliness;
- command verification;
- links;
- claims;
- licence;
- final results;
- reproducibility.

Mark items as complete only when verified.

Do not create a Git tag or release automatically.

## Workstream 10 — Portfolio and CV Summary

Create:

```text
docs/portfolio_summary.md
```

Include:

### One-line summary

A concise one-line description.

### Short project paragraph

Approximately 80–120 words.

### Three CV bullets

Write three truthful, metric-backed bullets suitable for CPU/RTL/ML-systems applications.

The bullets should emphasize:

1. end-to-end hardware-aware ML pipeline;
2. real WISDM model quality and quantization;
3. exact Sparrow-V RTL validation and compiler/runtime integration.

Do not overstate physical implementation or production deployment.

### Interview talking points

Include:

- why 16 handcrafted features;
- why subject-held-out splitting;
- why explicit integer reference inference;
- why per-output-channel quantization;
- why exact RTL comparison matters;
- what did and did not improve with sparsity;
- why partitioned simulation is not optimized latency;
- next research directions.

## Documentation Consistency

Ensure these documents agree:

- `README.md`;
- `docs/results/final_results.md`;
- `docs/reproduction.md`;
- `docs/portfolio_summary.md`;
- phase result documents;
- architecture documentation.

There must be one canonical source for final metrics.

Where possible, generate or validate repeated metrics from structured artifacts rather than manually duplicating numbers.

## Tests and Checks

Add small documentation/repository tests where useful, such as:

- required final documents exist;
- README links resolve;
- no forbidden claims;
- no absolute local paths;
- no generated data tracked;
- canonical metrics appear consistently;
- required Make targets exist.

Do not create brittle prose snapshot tests.

## Validation

Run focused documentation and repository checks during development.

At final acceptance run once:

```bash
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

Run the bounded final WISDM verification only if existing artifacts are available:

```bash
make run-wisdm-final
```

Do not retrain unnecessarily if the purpose is documentation verification and canonical artifacts already validate.

Verify Sparrow-V remains clean:

```bash
git -C "$SPARROWV_ROOT" status --short
```

## Acceptance Criteria

The milestone is complete only when:

1. README is polished and portfolio-ready.
2. README describes the final real-data project first.
3. Synthetic experiments are clearly secondary.
4. End-to-end architecture diagram exists.
5. Multi-layer execution diagram exists.
6. Host-side and RTL operations are distinguished.
7. Canonical final results document exists.
8. Final WISDM metrics are correct.
9. Quantization results are correct.
10. RTL exact-match totals are correct.
11. Measured and derived counters are distinguished.
12. Partitioned-cycle limitations are explicit.
13. Reproduction guide exists.
14. WISDM path configuration is documented.
15. Sparrow-V path configuration is documented.
16. Fast and full workflows are documented.
17. Repository structure is documented.
18. Make help is current.
19. CLI help is current.
20. Generated artifacts remain ignored.
21. Raw WISDM data is not tracked.
22. Processed WISDM windows are not tracked.
23. Checkpoints are not tracked.
24. No `.DS_Store` files are tracked.
25. Claims audit passes.
26. No physical hardware claim is made.
27. No monolithic latency claim is made.
28. No unsupported general compiler claim is made.
29. Release checklist exists.
30. Portfolio summary exists.
31. Three truthful CV bullets exist.
32. Interview talking points exist.
33. README links resolve.
34. Documentation references are current.
35. Phase 1–8 tests remain passing.
36. Repository checks pass.
37. Documentation checks pass.
38. `git diff --check` passes.
39. Sparrow-V remains clean.
40. No commit, push, tag, or release occurs.
41. `docs/codex_milestone_result.md` is finalized.

## Out of Scope

Do not implement:

- new datasets;
- new model architectures;
- hyperparameter tuning;
- sparse MLP;
- new compiler operators;
- new Sparrow-V instructions;
- RTL changes;
- FPGA execution;
- ASIC execution;
- monolithic runtime;
- TinyNPU integration;
- research experiments;
- benchmark comparisons requiring new external data;
- website changes;
- GitHub release creation;
- Git tagging.

## Stop Conditions

Stop only if:

- canonical metrics conflict irreconcilably across validated artifacts;
- required final results artifacts are missing or corrupt;
- repository checks expose tracked raw/private data;
- a major correctness defect is discovered;
- Sparrow-V is unexpectedly dirty due to the workflow.

Ordinary documentation inconsistencies, broken links, stale commands, claim wording, and formatting are not stop conditions.

## Token-Efficiency Instructions

Follow `AGENTS.md`.

In particular:

- prioritize README, final results, reproduction, release checklist, and portfolio summary;
- do not audit every historical file line-by-line;
- use structured artifacts as the source of truth;
- do not retrain unless required;
- do not run RTL repeatedly;
- do not inspect Sparrow-V broadly;
- avoid implementation refactors;
- run aggregate validation once;
- keep the final milestone result concise.

## Result File

Update:

```text
docs/codex_milestone_result.md
```

Finalize with:

```text
STATUS: COMPLETE
```

only when the repository is genuinely release-ready.

Include:

- final project summary;
- canonical metrics;
- final documentation created or updated;
- diagrams added;
- reproduction commands verified;
- claims corrected;
- repository-cleanliness result;
- tests and checks;
- Sparrow-V cleanliness;
- remaining limitations;
- changed files;
- confirmation that no raw WISDM data was committed;
- confirmation that no commit, push, tag, or release occurred.

## Completion Statement

If all acceptance criteria pass, state clearly:

```text
SparrowML is complete for its defined portfolio scope.
```