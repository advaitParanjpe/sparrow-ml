# Architecture

System flow: `dataset → preprocessing → PyTorch model → quantization → structured pruning → SparrowML IR → target lowering → packed artifacts → Sparrow-V runtime → RTL execution → metrics`.

Phase 1 implements the first three software stages: deterministic JSONL fixture generation, train-only standardization, a CPU FP32 `Linear(16, 4)` model, evaluation, and local artifacts. Quantization and every downstream stage remain planned. SparrowML owns model/data/optimization/export/compiler/runtime/evaluation tooling; Sparrow-V owns processor RTL, instructions, simulator, testbenches, counters, and hardware-specific execution.
