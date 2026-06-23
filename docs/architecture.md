# Architecture

Planned system flow: `dataset → preprocessing → PyTorch model → quantization → structured pruning → SparrowML IR → target lowering → packed artifacts → Sparrow-V runtime → RTL execution → metrics`.

Only configuration, contracts, and local diagnostics exist today. Every flow component after configuration is planned, not implemented. SparrowML owns model/data/optimization/export/compiler/runtime/evaluation tooling; Sparrow-V owns processor RTL, instructions, simulator, testbenches, counters, and hardware-specific execution. A future TinyNPU adapter may use the same artifact-oriented boundary without coupling either target’s source into this repository.
