# Portfolio Summary

## One-line summary

Built SparrowML, a deterministic WISDM-to-Sparrow-V edge-AI pipeline with explicit INT8 semantics and exact RTL/reference validation.

## Project paragraph

SparrowML is a hardware-aware ML systems project that trains a fixed 16-feature activity-recognition MLP on subject-held-out WISDM smartphone accelerometer data, calibrates and runs it with explicit INT8 arithmetic, and lowers the model into deterministic Sparrow-V deployment packages. The final test protocol uses 49 eligible subjects split 35/7/7 for train/validation/test and separates all calibration from held-out evaluation. The pipeline validates package reload and compares integer intermediates through Sparrow-V RTL simulation. For 12 selected held-out samples, `fc1` accumulators, hidden INT8 codes, `fc2` accumulators, and predictions all match the software reference exactly. The project reports partitioned simulation evidence, not physical deployment or optimized latency.

## CV bullets

- Built a deterministic hardware-aware ML pipeline from subject-held-out WISDM ingestion through FP32 training, per-channel INT8 quantization, versioned IR, and Sparrow-V deployment packages for a 340-parameter `16→16→4` MLP.
- Achieved held-out WISDM FP32 macro-F1 of 0.9287 and INT8 macro-F1 of 0.9198 (0.0090 drop), with 0.9873 FP32/INT8 prediction agreement using train-only input and hidden calibration.
- Integrated an external RISC-V INT8-vector RTL workflow and validated 12/12 exact matches for `fc1` accumulators, hidden INT8 values, `fc2` accumulators, and predictions across four `fc1` partitions plus one `fc2` run.

## Interview talking points

- **Why 16 handcrafted features?** They keep the model and target interface fixed at 16 inputs, making feature provenance and integer deployment inspectable.
- **Why subject-held-out splitting?** Windows from one person are correlated; splitting by subject avoids identity leakage into test quality.
- **Why explicit integer reference inference?** It makes rounding, clipping, bias scales, and accumulator behavior testable independently of FP32 training.
- **Why per-output-channel quantization?** Each output row can use its own weight scale, reducing avoidable quantization error while retaining a simple deterministic contract.
- **Why exact RTL comparison?** Final predictions can hide intermediate divergence; checking accumulators and hidden codes localizes semantic errors.
- **What did sparsity improve?** The controlled 2:4 single-layer experiment reduced derived arithmetic by 50% and compressed weight storage by 40.625%; equal measured dense/sparse single-layer cycles showed no sparse speedup.
- **Why is partitioned simulation not optimized latency?** `fc1` is four interface-constrained runs, while bias reconstruction, ReLU, and requantization occur on the host between runs.
- **Next research directions.** A target interface supporting a fused multi-layer schedule, on-target intermediate processing, and measured performance/power studies would be separate work.
