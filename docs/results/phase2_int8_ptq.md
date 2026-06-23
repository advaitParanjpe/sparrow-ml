# Phase 2 INT8 post-training quantization

Reproduce with `make run-int8-baseline` after `make run-fp32-baseline` has produced the source checkpoint. The generated JSON and Markdown reports are in `artifacts/phase2_int8/`.

Inputs use per-tensor symmetric signed INT8: `scale = max(abs(x))/127`, `zero_point = 0`, `q = clamp(rint(x/scale), -128, 127)`. NumPy `rint` supplies deterministic nearest, ties-to-even rounding. Weights use this policy independently for each output channel. Each bias is `rint(bias_fp32 / (input_scale * weight_scale[channel]))` in signed INT32.

Reference inference explicitly computes `bias_int32 + sum(input_int8 * weight_int8)` using integer host arithmetic, verifies signed INT32 deployment range, then reconstructs logits as `acc * input_scale * weight_scale[channel]`. Because output scales differ, prediction is `argmax` over reconstructed logits, not accumulators.

The report includes train/validation/test fixture accuracy, agreement with FP32, confusion matrix, logit errors, clipping/saturation, bias range, observed accumulator range, and conservative accumulator bound. These values are measured only on the synthetic fixture and are not real-world accuracy, hardware, or deployment measurements. No pruning, sparse packing, compiler lowering, Sparrow-V execution, or QAT is included.
