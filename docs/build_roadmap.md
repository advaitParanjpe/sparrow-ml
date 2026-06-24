# Build Roadmap

| Phase | Purpose and deliverables | Explicit non-goals | Validation gate |
| --- | --- | --- | --- |
| 0 | Scaffold and target contracts | ML implementation | Offline checks and contract tests |
| 1 | Deterministic dataset and FP32 baseline (implemented) | Quantization/deployment | Reproducible split and baseline test |
| 2 | Exact integer reference and INT8 quantization (implemented) | Sparsity | Integer equivalence fixtures |
| 3 | Deterministic 2:4 pruning, fixed-mask fine-tuning, and packing (implemented) | Target execution | Packing/reference validation |
| 4 | SparrowML IR and Sparrow-V exporter (implemented) | RTL changes | Artifact contract validation |
| 5 | Sparrow-V execution adapter and runtime (implemented) | General benchmarking | Exact simulator/reference result agreement |
| 6 | Multi-layer INT8 model and multi-operator export (implemented) | Multi-layer Sparrow-V execution, arbitrary graphs | Exact intermediate/package validation |
| 7 | Multi-layer Sparrow-V RTL/reference validation (implemented) | Monolithic scheduler, hardware ReLU, sparse MLP | Exact fc1/hidden/fc2 validation and two-run semantic determinism |
| 8 | Subject-held-out WISDM preparation, INT8 package export, and bounded Sparrow-V RTL/reference validation (implemented) | New model, RTL, ISA, or runtime features | Exact package/intermediate/prediction checks |
| 9 | Portfolio polish, reproducibility, and release readiness (implemented) | New experiments or deployment claims | Documentation and repository checks |
# Phase 8

Phase 8A prepares subject-held-out WISDM windows. Phase 8B exports the measured model and validates package reload. Phase 8C executes a bounded held-out selection through Sparrow-V.
