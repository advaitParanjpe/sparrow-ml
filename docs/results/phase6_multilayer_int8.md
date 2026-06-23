# Phase 6 multi-layer INT8 result

Run `make run-multilayer-baseline`. The fixed CPU model has 340 parameters (`fc1`: 272; `fc2`: 68), trains for 50 epochs with seed 20260623, batch size 32, and Adam learning rate 0.005. Validation loss selects the checkpoint.

The bounded run measured 100% synthetic-fixture FP32 and INT8 test accuracy with 100% agreement. Input calibration used 360 training samples with scale 0.020786371756726364. Hidden calibration used the same training split with scale 0.05766356085229108, range `[0,127]`, and zero clipping. Test accumulator ranges were `fc1 [-77873, 78494]` and `fc2 [-34471, 24971]`; final-logit maximum absolute error was 0.07607368794544023. These are fixture measurements only.

The package is 528 bytes in a 4096-byte target capacity and contains a 16-byte hidden buffer. Repeated exports matched hashes; reload validation reproduced intermediate values and final predictions exactly. No Sparrow-V source was modified and no multi-layer RTL execution occurred.
