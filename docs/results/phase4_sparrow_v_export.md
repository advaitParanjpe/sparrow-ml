# Phase 4 SparrowML IR and Sparrow-V artifact export

Phase 4 lowers the existing Phase 2 dense and Phase 3 sparse artifacts without retraining, requantizing, or repruning. It emits deterministic `sparrowml_ir_v1` deployment packages with canonical IR, memory map, logical model image, input payload, expected integer-reference output, manifest, hashes, and symbolic future-runtime commands.

The only supported operators are `DenseLinearInt8` and `SparseLinear2of4Int8`; sparse groups remain input-axis 2:4 with LSB-first three-bit metadata. Data uses little-endian two's-complement integers and float32 scales. Export validation establishes exact package-decoded/reference equivalence. These are compiler/export correctness checks, not Sparrow-V execution, RTL validation, cycle measurement, or hardware-speedup results.
