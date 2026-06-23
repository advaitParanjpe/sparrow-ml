# Multi-layer INT8 quantization contract

The sole supported Phase 6 graph is `Linear(16,16) → ReLU → Linear(16,4)`. Standardized inputs use train-only symmetric signed INT8 `max(abs(x))/127`. Both layers use per-output-channel symmetric signed INT8 weights and signed INT32 biases; a bias scale is its incoming activation scale times its output-channel weight scale.

`fc1` accumulates in INT32, reconstructs per channel, and applies exact ReLU. Training-split hidden ReLU maxima select a signed-INT8 scale. Requantization is `clip(rint(hidden_real / hidden_scale), 0, 127)`, with NumPy nearest/ties-to-even semantics. `fc2` accumulates in INT32 and reconstructs logits using hidden scale times its channel scale. Predictions use reconstructed logits.

The Phase 6 IR explicitly records dense linear, ReLU, requantization, and dense linear. The hidden real tensor is logical; the hidden INT8 tensor receives a physical 16-byte aligned buffer. Package reload validation reproduces both accumulators, hidden codes, and final prediction exactly. It is not an RTL result.
