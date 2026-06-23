# Phase 7 Sparrow-V multi-layer runtime result

The local baseline executes Phase 6 sample `sensor-normal-008` through four zero-bias fc1 RTL partitions, host reconstructs bias/ReLU/requantization, and executes zero-bias fc2 RTL. It requires exact accumulator and hidden-code equality and repeats the full semantic workflow twice. The authoritative generated evidence is `artifacts/phase7_multilayer_runtime/`.

This is RTL simulation using an external fixed-shape workload interface. It does not claim physical hardware execution, monolithic multi-layer scheduling, hardware ReLU, sparse MLP execution, or hardware speedup.
