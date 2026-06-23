# Sparrow-V multi-layer runtime contract

Phase 7 supports only `Input[16] → DenseLinearInt8[16→16] → ReLU → RequantizeInt8 → DenseLinearInt8[16→4]`. The external Sparrow-V interface remains unchanged and supports four output channels, so `fc1` is four deterministic `16→4` dense workloads in channel order `0–3`, `4–7`, `8–11`, and `12–15`; `fc2` is one workload.

Every workload uses zero RTL bias. The raw dot products are `rtl_produced`; SparrowML adds serialized INT32 biases as `host_reconstructed`. It then reconstructs `fc1` with the package input and per-channel weight scales, applies ReLU, rounds with NumPy ties-to-even, clamps `[0,127]`, and emits hidden INT8 codes as `host_reconstructed`. Exact equality with `intermediate_reference.json` is required for fc1, hidden, fc2, and prediction.

Results use `sparrowml_sparrowv_multilayer_runtime_result_v1`. Counters retain Sparrow-V availability labels. Fc1 aggregate cycles are labelled `partitioned simulation cycle total`; they are not an optimized model-latency measure. Each workflow derives 256 fc1 and 64 fc2 dense multiplications, 320 total. Semantic hashes omit paths, logs, timestamps, and host timing.

Use `make prepare-sparrowv-mlp`, `make run-sparrowv-mlp`, `make validate-sparrowv-mlp`, or `make run-sparrowv-mlp-baseline`.
