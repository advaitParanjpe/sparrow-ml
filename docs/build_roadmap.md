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
| 7 | Experiments and target-aware optimization | Research claims | Recorded reproducible comparisons |
| 8 | Sparrow-V/TinyNPU comparison | Tight target coupling | Identical-workload comparison |
| 9 | Research-style exploration after practical completion | Replacing proven path | Reproducible study protocol |
