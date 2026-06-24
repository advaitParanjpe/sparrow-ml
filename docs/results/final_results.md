# Final SparrowML Results

This is the canonical report for final real-data metrics. SparrowML is a hardware-aware edge-AI training, quantization, compilation, and runtime pipeline that deploys a subject-held-out WISDM activity-recognition model onto Sparrow-V. The results below are measured from the real-data workflow unless explicitly labelled derived or controlled synthetic evidence.

## Dataset protocol

The source is the WISDM smartphone and smartwatch activity dataset; this workflow selects the smartphone accelerometer. It retains walking, jogging, sitting, and standing. Of 49 eligible subjects, deterministic subject-level assignment uses 35 train, 7 validation, and 7 test subjects. The accepted set contains 25,768 windows of 80 samples with 50% overlap. Each window yields 16 features.

All normalization, input calibration, hidden-activation calibration, validation-based selection, and training decisions use training/validation subjects only. Test subjects remain held out until evaluation. This prevents overlap between subject identity in training and reported test quality; see the [evaluation protocol](../wisdm_evaluation_protocol.md).

## Model quality

The model is `Linear(16,16) → ReLU → Linear(16,4)` with 340 parameters. FP32 training and explicit INT8 inference use the same held-out test subjects.

| Model | Accuracy | Macro-F1 | Balanced accuracy |
| --- | ---: | ---: | ---: |
| FP32 MLP | 0.9259473531964131 | 0.9287458208759758 | 0.9296898801135173 |
| INT8 MLP | 0.9175585768006942 | 0.9197794804065271 | 0.920638703760132 |

The measured INT8 macro-F1 degradation is `0.008966340469448664`; measured FP32/INT8 prediction agreement is `0.9872722013306335`.

## Quantization and integer safety

Inputs use train-only symmetric signed-INT8 calibration. Both dense layers use per-output-channel symmetric INT8 weights and signed INT32 biases. Hidden calibration is computed from training-split post-ReLU activations; hidden codes use ties-to-even rounding and clipping to `[0, 127]`. The generated evaluation reports input and hidden clipping counts; these are calibration diagnostics, not accuracy estimates.

Software reference inference performs INT8 products and INT32 accumulation explicitly, checks that observed accumulators fit signed INT32, and reconstructs logits using the activation scale and the corresponding per-channel weight scale. Package reload validates all integer tensors and intermediate traces exactly. The detailed arithmetic contract is in the [multi-layer quantization contract](../multilayer_quantization_contract.md).

## Deployment correctness

The compiler graph is `DenseLinearInt8 → ReLU → RequantizeInt8 → DenseLinearInt8`. Twelve selected held-out WISDM samples were validated through the external Sparrow-V RTL workflow.

| Validation level | Result |
| --- | --- |
| Package reload | exact |
| `fc1` accumulators | 12/12 exact |
| Hidden INT8 codes | 12/12 exact |
| `fc2` accumulators | 12/12 exact |
| Predictions | 12/12 exact |

The RTL workload emits zero-bias dot products. Full INT32 bias reconstruction, ReLU, and hidden requantization are host-side and explicitly compared with the integer reference; they are not RTL-produced operations. Sparrow-V remains an external, unmodified target.

## Compute and runtime provenance

Per sample, the fixed graph has 256 conceptual `fc1` multiplications and 64 conceptual `fc2` multiplications, for **320 derived conceptual INT8 multiplications**. These counts describe the graph, not simulator timing.

The simulator records measured counters for each isolated invocation (where the external interface exposes them), including cycle counters. Across the 12 selected samples, the generated counter summary records **23,232 measured `fc1` partitioned simulation cycles** and **5,808 measured `fc2` simulation cycles**; it also records 5,232 and 1,308 measured retired instructions, respectively. `fc1` uses four isolated `16→4` runs and `fc2` one isolated run. These are **partitioned simulation cycle totals**, not an aggregate optimized latency: there is no hardware ReLU, no hardware requantization, and host-side bias reconstruction occurs between runs.

## Earlier controlled experiments

Earlier phases use a deterministic synthetic fixture to validate the FP32 baseline, dense INT8 reference, compiler export, and single-layer Sparrow-V integration. The 2:4 sparse experiment retained 32 of 64 single-layer weights: a derived 50% arithmetic reduction and 40.625% compressed weight-storage reduction (38 bytes versus 64 bytes for weights plus metadata). Dense and sparse single-layer RTL runs had equal measured cycle results, so the project makes **no measured sparse-speedup claim**. These are controlled system-validation experiments, not WISDM quality results or sparse multi-layer deployment results.

## Scope and limitations

This report documents RTL simulation, not FPGA/ASIC execution, power, timing, physical deployment, or real-time behavior. It describes a fixed pipeline rather than a general-purpose ML compiler. Partitioned simulator counters must not be read as optimized end-to-end latency.
