# Milestone: Deterministic Sensor Dataset Fixture and FP32 Baseline

## Objective

Build SparrowML’s first real ML milestone:

1. define a deterministic sensor-classification dataset fixture;
2. implement a reproducible preprocessing pipeline;
3. train a small FP32 PyTorch baseline;
4. evaluate it on fixed train, validation, and test splits;
5. save reproducible metrics and a model checkpoint;
6. establish the data and experiment contracts that later quantization, pruning, compilation, and Sparrow-V deployment milestones will consume.

This milestone is only the FP32 software baseline.

Do not implement quantization, 2:4 pruning, compiler lowering, Sparrow-V execution, TinyNPU integration, or hardware-aware optimization yet.

## Project context

SparrowML is a separate repository from Sparrow-V.

SparrowML owns:

- dataset processing;
- model training;
- quantization and pruning in future milestones;
- compiler and exporter tooling;
- runtime orchestration;
- evaluation and experiment management.

Sparrow-V remains an external deployment target.

This milestone must not modify Sparrow-V.

## Relevant files

Read first:

- `AGENTS.md`
- `docs/codex_context.md`
- `docs/current_milestone.md`
- `docs/architecture.md`
- `docs/build_roadmap.md`
- `docs/data_contracts.md`
- `docs/experiment_policy.md`
- `configs/project.yaml`
- existing package and test structure.

Inspect only directly relevant files after that.

Do not perform a broad repository audit unless a concrete failure requires it.

## Scope

Implement a deterministic four-class sensor-classification baseline suitable for later Sparrow-V deployment.

Use:

- 16 input features;
- 4 output classes;
- one linear classifier initially;
- FP32 training and inference;
- deterministic train, validation, and test splits;
- fixed random seeds;
- CPU execution only;
- offline tests;
- no internet dependency during tests.

Preferred model:

```text
Linear(16, 4)
```

The model should output four logits.

Prediction is:

```text
argmax(logits)
```

## Dataset strategy

Use a small deterministic fixture suitable for repository tests and development.

The fixture may be:

- synthetically generated from documented class prototypes and controlled noise;
- derived from a small permissively licensed dataset already available locally;
- another deterministic fixture with clear provenance.

For this milestone, prefer a synthetic but meaningful sensor fixture if no real dataset is already present.

Do not claim real-world model accuracy from synthetic data.

Label all reported performance as:

```text
fixture accuracy
```

unless real dataset provenance and splits are genuinely implemented.

## Classes

Use four classes aligned with the Sparrow-V sensor demonstration where practical:

```text
normal
inner
outer
ball
```

Document that these names are placeholders for a vibration-fault-style classification task unless backed by a real dataset.

## Fixture requirements

Create a deterministic dataset with:

- exactly 16 numerical features per sample;
- four classes;
- train split;
- validation split;
- test split;
- fixed class ordering;
- fixed seed;
- class balance or clearly reported imbalance;
- stable sample IDs.

Recommended bounded size:

- 256 to 1,024 total samples;
- enough samples for training and evaluation;
- small enough for fast CPU-only tests.

A suitable default is:

```text
512 total samples
128 samples per class
```

Suggested split:

```text
70% train
15% validation
15% test
```

Exact counts may be adjusted to keep class balance exact.

## Fixture generation

Implement one deterministic fixture generator.

It should:

- define one class prototype per class;
- generate controlled variation around each prototype;
- produce reproducible features;
- avoid perfectly trivial one-hot separation;
- keep features finite;
- avoid NaN and infinity;
- write metadata describing seed, dimensions, class order, and split sizes.

Use a fixed default seed.

Recommended:

```text
seed = 20260623
```

The generator must produce identical outputs across repeated runs in the same supported environment.

## Data representation

Define a stable internal example structure containing:

- sample ID;
- feature vector;
- integer class ID;
- class name;
- split.

Preferred persisted format:

```text
JSON Lines
```

or:

```text
CSV plus metadata JSON
```

Choose the simpler format that integrates cleanly with the package.

Document:

- feature ordering;
- class ordering;
- split policy;
- normalization policy;
- fixture provenance;
- deterministic seed.

## Data locations

Use the existing data policy.

Recommended layout:

```text
data/processed/sensor_fixture/
```

Generated large or transient artifacts should remain ignored.

Small deterministic golden fixtures may be tracked only if:

- they are documented;
- they are reproducible;
- their size is reasonable;
- repository policy permits it.

Prefer generating fixtures during tests or through a command if that keeps the repository cleaner.

## Preprocessing

Implement a simple deterministic preprocessing pipeline.

At minimum:

- validate feature count;
- validate finite values;
- fit normalization statistics on training data only;
- apply the same statistics to validation and test data;
- prevent train/test leakage;
- preserve feature ordering.

Preferred normalization:

```text
standardization using train-split mean and standard deviation
```

Handle zero-variance features safely.

Persist or report:

- feature means;
- feature standard deviations;
- class order;
- preprocessing version.

## Model

Implement a minimal PyTorch model:

```python
torch.nn.Linear(16, 4)
```

Requirements:

- deterministic initialization;
- explicit input and output dimensions;
- no hidden layers;
- no activation after final logits;
- CPU support;
- simple inspectable implementation.

Do not add a larger MLP in this milestone.

## Training

Implement one deterministic training command.

Use:

- cross-entropy loss;
- a simple optimizer such as Adam or SGD;
- bounded epoch count;
- validation loss and accuracy;
- best-checkpoint selection using validation performance only;
- fixed seeds for Python, NumPy, and PyTorch;
- CPU-only default.

Recommended initial defaults:

```text
epochs: 50
batch size: 32
learning rate: 1e-2
```

Adjust only if needed for stable convergence.

Do not tune on the test split.

## Determinism

Set and record:

- Python seed;
- NumPy seed;
- PyTorch seed;
- dataset-generation seed;
- split seed;
- DataLoader shuffle seed.

Where exact bitwise reproducibility is not guaranteed across PyTorch versions, document the limitation.

Within the tested environment, repeated runs should produce:

- identical split membership;
- identical fixture files;
- identical class counts;
- identical or tightly stable metrics.

## Evaluation

Evaluate on:

- train;
- validation;
- test.

Report:

- loss;
- accuracy;
- per-class sample count;
- per-class correct count;
- confusion matrix;
- predicted class distribution.

Also report:

- model parameter count;
- checkpoint size;
- feature count;
- class count;
- split sizes;
- training seed;
- best epoch.

Do not add AUROC, F1, or calibration metrics unless justified by the fixture and implemented cleanly.

## Expected quality gate

The fixture and model should be learnable but not fabricated as a perfect real-world benchmark.

Set a minimum test fixture accuracy gate of:

```text
>= 85%
```

If the fixture naturally produces 100%, report it honestly as fixture accuracy and explain that the fixture is deterministic and synthetic.

Do not alter the test set after observing results.

## Artifact outputs

Create a reproducible experiment output directory, for example:

```text
artifacts/phase1_fp32/
```

Generated artifacts may remain ignored.

At minimum generate:

- configuration snapshot;
- dataset metadata;
- preprocessing statistics;
- best FP32 checkpoint;
- metrics JSON;
- confusion matrix JSON or CSV;
- human-readable Markdown summary.

The summary must distinguish:

- measured values;
- fixture-only claims;
- future deployment work not yet implemented.

## Configuration

Add a dedicated Phase 1 configuration, preferably:

```text
configs/experiments/fp32_sensor_baseline.yaml
```

Include:

- seed;
- dataset size;
- feature count;
- class names;
- split ratios;
- normalization settings;
- optimizer;
- learning rate;
- epoch count;
- batch size;
- output directory.

Avoid machine-specific absolute paths.

## CLI

Extend the CLI with bounded commands such as:

```text
sparrowml generate-fixture
sparrowml train-fp32
sparrowml evaluate-fp32
sparrowml run-fp32-baseline
```

Exact command names may be adjusted.

Preferred behavior:

### `generate-fixture`

- generate or validate deterministic dataset fixture;
- print split sizes and class counts.

### `train-fp32`

- train the linear classifier;
- save the best checkpoint;
- save metrics.

### `evaluate-fp32`

- load checkpoint;
- evaluate train, validation, and test splits;
- write summary.

### `run-fp32-baseline`

- perform fixture generation, training, evaluation, and reporting in one reproducible command.

Use proper exit codes.

## Python dependencies

Add only the dependencies required for this milestone.

Expected additions:

- PyTorch;
- NumPy;
- PyYAML if not already present.

Avoid:

- pandas unless clearly useful;
- scikit-learn unless needed for a small metric utility;
- notebook-only dependencies;
- plotting libraries unless a plot is actually required;
- large ML frameworks beyond PyTorch.

Prefer lightweight, direct implementations.

## Package structure

Add only the necessary modules.

A reasonable structure is:

```text
src/sparrowml/
├── data/
│   ├── fixture.py
│   ├── dataset.py
│   └── preprocessing.py
├── models/
│   └── linear_classifier.py
├── training/
│   ├── trainer.py
│   └── seeds.py
└── evaluation/
    ├── metrics.py
    └── report.py
```

Adjust to match the existing scaffold.

Do not add a generalized training framework.

## Tests

Add focused tests for:

### Fixture generation

- deterministic output;
- correct sample count;
- exactly 16 features;
- four classes;
- stable class order;
- balanced class counts where intended;
- no duplicate sample IDs;
- no NaN or infinity;
- identical repeated generation.

### Splits

- no overlap among train, validation, and test;
- all samples assigned exactly once;
- deterministic split membership;
- class counts reported.

### Preprocessing

- statistics fit on training data only;
- validation and test use training statistics;
- zero-variance handling;
- output shape;
- finite outputs;
- reproducibility.

### Model

- input shape `(batch, 16)`;
- output shape `(batch, 4)`;
- deterministic initialization;
- parameter count.

### Training

- one short smoke-training run;
- loss decreases on a small fixture;
- checkpoint is created;
- metrics file is valid;
- no GPU required.

### Evaluation

- confusion matrix shape is 4 × 4;
- accuracy calculation is correct;
- per-class counts sum correctly;
- repeated checkpoint evaluation is stable.

### CLI

- commands parse;
- smoke run succeeds;
- invalid configuration fails clearly.

Tests must not:

- require internet;
- download a dataset;
- require Sparrow-V;
- require a GPU;
- take excessive time.

Keep normal test runtime bounded.

## Make targets

Add stable targets such as:

```text
generate-fixture
train-fp32
evaluate-fp32
run-fp32-baseline
test-phase1
```

Update:

```text
make help
```

Recommended behavior:

```text
make test-phase1
```

runs only Phase 1 focused tests.

Do not include full training in every general repository check if it materially slows iteration.

A tiny smoke-training run may remain in `make check`.

## Documentation

Update or add:

- `README.md`
- `docs/architecture.md`
- `docs/build_roadmap.md`
- `docs/data_contracts.md`
- `docs/experiment_policy.md`
- `docs/codex_context.md`
- one Phase 1 results document.

Suggested new document:

```text
docs/results/phase1_fp32_baseline.md
```

Document:

- fixture design;
- class names;
- split sizes;
- seed;
- feature representation;
- preprocessing;
- model architecture;
- training configuration;
- measured metrics;
- confusion matrix;
- limitations;
- exact reproduction commands.

Keep `docs/codex_context.md` concise.

## README status

Update the README from:

```text
scaffold only
```

to:

```text
Phase 1 FP32 baseline implemented
```

Do not claim:

- quantization;
- pruning;
- compiler support;
- Sparrow-V deployment;
- hardware metrics;
- real-world accuracy.

## Result file

Update:

```text
docs/codex_milestone_result.md
```

throughout the run.

Finalize with:

```text
STATUS: COMPLETE
```

only if every required criterion and validation passes.

Include:

- fixture size;
- split sizes;
- class names;
- seed;
- model architecture;
- parameter count;
- best epoch;
- train, validation, and test metrics;
- confusion matrix;
- generated artifacts;
- exact commands and outcomes;
- changed files;
- remaining limitations;
- next recommended milestone;
- confirmation that Sparrow-V was not modified;
- confirmation that no commit or push occurred.

Use `STATUS: FAILED` if required work or checks remain incomplete.

Use `STATUS: BLOCKED` only for a genuine human decision or architectural blocker.

## Out of scope

Do not implement:

- INT8 quantization;
- fake quantization;
- quantization-aware training;
- 2:4 pruning;
- sparse packing;
- compiler IR;
- code generation;
- Sparrow-V execution;
- TinyNPU support;
- multi-layer models;
- hidden layers;
- CNNs;
- transformers;
- ONNX;
- notebooks as the primary workflow;
- dataset downloads during tests;
- hyperparameter sweeps;
- experiment tracking services;
- web dashboards;
- cloud training;
- GPU requirements;
- hardware-aware cost models;
- research experiments.

## Validation

During development, run focused tests.

At final acceptance, run once:

```text
python3 -m compileall src scripts
pytest
make test-phase1
make smoke
make check
make docs-check
git diff --check
```

Also run the full baseline once:

```text
make run-fp32-baseline
```

Verify that the generated summary reports reproducible results.

Do not repeatedly retrain during unrelated validation.

## Acceptance criteria

The milestone is complete only when:

1. A deterministic four-class sensor fixture exists.
2. Every sample has exactly 16 features.
3. Stable train, validation, and test splits exist.
4. No sample appears in more than one split.
5. The fixture seed is recorded.
6. Class order is fixed and documented.
7. Fixture provenance is documented.
8. Results are labelled as fixture accuracy.
9. Preprocessing is fit on training data only.
10. Validation and test use training statistics.
11. A `Linear(16, 4)` PyTorch model exists.
12. FP32 training is deterministic within documented limits.
13. Training uses validation data for checkpoint selection.
14. Test data is not used for tuning.
15. Best checkpoint is saved.
16. Configuration snapshot is saved.
17. Train metrics are reported.
18. Validation metrics are reported.
19. Test metrics are reported.
20. Confusion matrix is reported.
21. Per-class counts are reported.
22. Model parameter count is reported.
23. Checkpoint size is reported.
24. Test fixture accuracy is at least 85%.
25. One command reproduces the full baseline.
26. CLI commands work.
27. Phase 1 focused tests pass.
28. Tests require no internet.
29. Tests require no GPU.
30. Tests require no Sparrow-V checkout.
31. Artifacts follow repository policy.
32. Documentation matches implementation.
33. README status is accurate.
34. No quantization or pruning is implemented.
35. No compiler or hardware execution is implemented.
36. Sparrow-V is not modified.
37. General repository checks pass.
38. Documentation checks pass.
39. `git diff --check` passes.
40. No commit or push occurs.
41. `docs/codex_milestone_result.md` is finalized.

## Stop conditions

Stop for human review only if:

- PyTorch cannot be installed or imported in the current environment;
- a licensing or provenance issue prevents use of the chosen data;
- deterministic splitting cannot be achieved;
- the fixture cannot reach the minimum quality gate without making the classes trivially separable;
- repository policy conflicts with storing required artifacts;
- a major scaffold defect prevents the milestone.

Ordinary training instability, configuration bugs, test failures, path issues, and documentation work are not stop conditions.

## Token-efficiency instructions

Follow `AGENTS.md`.

In particular:

- read only the compact context and milestone first;
- inspect only directly relevant files;
- avoid repository-wide narration;
- do not build future phases;
- run focused tests while developing;
- run aggregate checks once at final acceptance;
- keep the result file concise;
- do not perform broad hyperparameter searches;
- do not repeatedly retrain when one bounded run is enough.

## Next milestone

The expected next milestone is:

```text
Exact integer reference inference and INT8 post-training quantization
```

Do not implement it during this milestone.