# Sparrow-V deployment package contract

Phase 4 supports only `dense_int8` and `sparse_2of4_int8` `Linear(16,4)` graphs. `model_ir.json` uses `sparrowml_ir_v1`, canonical JSON (sorted keys, compact separators, trailing newline), and explicit tensors, quantization, constants, and one supported operator. It contains no absolute paths or timestamps.

Packages use `sparrowml_sparrowv_package_v1`. Logical memory regions are ordered input, weights, optional metadata, bias, scales, output, each four-byte aligned. Values are little-endian two's-complement INT8/INT32 and IEEE-754 float32 scales; padding is zero. Sparse metadata is the existing legal three-bit LSB-first format. `program.json` is symbolic only and contains no ISA encoding.

`input_data.json` identifies the exported fixture sample and quantized input; `input_data.bin` is its contiguous INT8 payload. `expected_output.json` is generated exclusively by the dense or compressed-sparse integer reference implementation. `validate-export` decodes model and input binary content, re-runs that reference, verifies hashes, and rejects inconsistent layouts. No package executes or modifies Sparrow-V.
