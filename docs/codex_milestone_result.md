STATUS: COMPLETE

# Phase 4 SparrowML IR and Sparrow-V artifact export result

Implemented `sparrowml_ir_v1` and package format `sparrowml_sparrowv_package_v1`. The supported operators are `DenseLinearInt8` and `SparseLinear2of4Int8` only. IR tensors explicitly record names, static shapes, element types, roles, ordering, contiguous storage, byte sizes, and quantization. Canonical JSON has stable key ordering, compact separators, no timestamps, and parse/serialize byte stability. Validation rejects invalid versions, shapes, types, scales, zero points, operators, sparse metadata, tensor sizes, and absolute source identities.

Dense lowering preserves Phase 2 INT8 weights, INT32 biases, scales, class order, preprocessing version, and source SHA-256. Sparse lowering preserves Phase 3 compressed weights, legal 2:4 metadata, packed LSB-first metadata semantics, INT32 biases, scales, class order, preprocessing version, and source SHA-256. The target configuration defines a 4096-byte logical scratchpad, four-byte alignment, little-endian encoding, and symbolic runtime interface version without target-local paths.

Generated ignored packages: `artifacts/phase4_export/dense/` and `artifacts/phase4_export/sparse/`, each with manifest, IR, memory map, model data image, input binary/JSON, expected output, report, README, and symbolic `program.json`; root outputs are `export_summary.json`, `summary.md`, and `determinism.json`. The exported sample is `sensor-normal-000`.

Dense layout is input `0:16`, weights `16:80`, bias `80:96`, scales `96:112`, output `112:128` (128 bytes). Sparse layout is input `0:16`, compressed weights `16:48`, metadata `48:54`, bias `56:72`, scales `72:88`, output `88:104` (104 bytes). Integers are two's-complement INT8/INT32, scales are IEEE-754 float32, byte order is little-endian, sparse metadata is the existing three-bit LSB-first packing, and padding bytes are zero.

Dense expected accumulators for `sensor-normal-000` are `[39603, -17389, -1218, -26014]`, prediction `0` (`normal`). Sparse expected accumulators are `[27952, -9483, -738, -19017]`, prediction `0` (`normal`). Reload validation decoded exact source tensors and inputs, reran dense and compressed-sparse references, and established exact accumulator/prediction equivalence.

Deterministic SHA-256 evidence: dense IR `4e6b0749132ed80969c70a7ae69b42736c07032afda57e0b7db8467ba91f8184`, manifest-derived model data `5c08e5f4d833d727a6f8c67751f095a0f06efe9e700250bab878a849a675e64d`, input `fbee58614ade055f6841a80f0b1b29827a80ab1ec6d54c4dd723eb462f8227cf`; sparse IR `564aeedac632f7f038b5d6c9d5aa83a7429276f762657d4dd544f1d8c63ae3a0`, model data `9b1bfe7f37b8d1774b93340c715ed8de123e4f194a32fe84a2f81134f99278a5`, input unchanged. Full required file hashes are in package manifests and `determinism.json`; repeated export produced identical content.

Validation passed: `python3 -m compileall src scripts`; `pytest` (25 passed); `make test-phase1`; `make test-phase2`; `make test-phase3`; `make test-phase4`; `make smoke`; `make check`; `make docs-check`; `make run-export-baseline`; `make validate-ir`; and `git diff --check`.

Changed compiler/exporter, CLI, Make targets, target/export configuration, focused Phase 4 tests, README, architecture, data contracts, experiment policy, context, roadmap, export contract, and Phase 4 result documentation. Removed forbidden OS metadata `docs/.DS_Store` so the pre-existing repository checker passes.

Limitations: no Sparrow-V execution, simulation, RTL validation, hardware counters, raw ISA encoding, arbitrary graphs, or general compiler support. Sparrow-V was not executed or modified. No commit or push was performed. Next recommended milestone: Sparrow-V runtime adapter, simulator execution, and RTL/reference result validation.
