# Phase 3 deterministic 2:4 structured sparsity

Reproduce with `make run-sparse-baseline`. This pipeline applies 2:4 pruning to `weight[output_channel][input_feature]`: each row's feature groups are `[0:4]`, `[4:8]`, `[8:12]`, and `[12:16]`. It retains the two largest absolute INT8 values; ties select lower lanes. The fixed mask has 32 retained and 32 pruned positions and is reapplied after each of 10 CPU Adam steps at learning rate `0.001`; validation loss selects the checkpoint.

Metadata maps `{0,1}`, `{0,2}`, `{0,3}`, `{1,2}`, `{1,3}`, `{2,3}` to `000` through `101`. Groups traverse output channel then group; compressed weights are lower selected lane then higher selected lane. Three-bit metadata values pack LSB-first, producing six bytes for 16 groups with no nonzero padding. Exact decompression recreates the `[4,16]` sparse dense-equivalent matrix, and explicit compressed reference inference decodes metadata and performs two products per group.

The reference run measured synthetic-fixture test accuracy of 100% before and after sparse fine-tuning, with 100% post-tuning dense-INT8 prediction agreement and zero test disagreements. The post-tuning test confusion matrix was `[[19,0,0,0],[0,19,0,0],[0,0,19,0],[0,0,0,19]]`. Sparse/dense logit error was maximum `4.727576`, mean absolute `1.669541`, RMS `2.160784`; values are not general model-quality claims.

Sparse execution performs 32 multiplications and skips 32 of 64 per sample (50% arithmetic reduction). Weight storage is 32 compressed bytes plus six metadata bytes, versus 64 dense bytes: 38 bytes and a 40.625% reduction. Biases (16 bytes) and per-channel scales (32 bytes as float64 artifact values) are reported separately. Observed post-tuning accumulators were `-59052` to `50654`; conservative bound was `138845`, both within signed INT32. The observed legal metadata distribution was `000:3`, `001:1`, `010:2`, `011:2`, `100:1`, `101:7`.

Artifacts are under `artifacts/phase3_sparse/`, including the mask, sparse FP32 checkpoint, sparse quantized model, metadata binary, reports, metrics, and deterministic SHA-256. No compiler lowering, Sparrow-V execution, or hardware speedup measurement is included.
