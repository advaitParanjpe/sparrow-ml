# Architecture

System flow: `dataset → preprocessing → PyTorch model → quantization → structured pruning → SparrowML IR → target lowering → packed artifacts → Sparrow-V runtime → RTL execution → metrics`.

Phase 1 implements deterministic JSONL fixture generation, train-only standardization, a CPU FP32 `Linear(16, 4)` model, evaluation, and local artifacts. Phase 2 adds train-only, per-tensor symmetric INT8 activation calibration; per-output-channel symmetric INT8 weights; INT32 accumulator-domain biases; and an explicit integer reference path. It reconstructs each output using its own scale and chooses predictions from reconstructed logits. Pruning and all downstream stages remain planned. SparrowML owns model/data/optimization/export/compiler/runtime/evaluation tooling; Sparrow-V owns processor RTL, instructions, simulator, testbenches, counters, and hardware-specific execution.
