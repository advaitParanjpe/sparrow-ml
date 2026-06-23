"""Deterministic Phase 5 report helpers."""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any
from sparrowml.compiler.ir import canonical_json
def _json(path: Path, value: Any) -> None: path.write_text(canonical_json(value), encoding="utf-8")
def semantic_payload(result: dict[str, Any]) -> dict[str, Any]: return {key: result[key] for key in ("format_version", "mode", "package_identity", "compatibility_identity", "sample_id", "parsed_accumulators", "predicted_class_id", "expected_accumulators", "expected_prediction", "counters", "trap_assertion_status", "validation_status")}
def semantic_hash(result: dict[str, Any]) -> str: return hashlib.sha256(canonical_json(semantic_payload(result)).encode()).hexdigest()
def write_baseline_report(root: Path, results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    hashes = {mode: [semantic_hash(item) for item in runs] for mode, runs in results.items()}
    determinism = {"format_version": "sparrowml_sparrowv_determinism_v1", "modes": {mode: {"repeat_count": len(hashes[mode]), "semantic_hashes": hashes[mode], "deterministic": len(set(hashes[mode])) == 1} for mode in hashes}}
    _json(root / "determinism.json", determinism); summary = {mode: runs[0] for mode, runs in results.items()}; report = {"format_version": "sparrowml_sparrowv_cross_mode_report_v1", "dense": summary["dense"], "sparse": summary["sparse"], "determinism": determinism["modes"]}; _json(root / "cross_mode_report.json", report)
    lines = ["# Phase 5 Sparrow-V runtime report", "", "Both modes used Sparrow-V's existing external sensor workload interface."]
    for mode in ("dense", "sparse"): lines.append(f"- {mode}: accumulators {summary[mode]['parsed_accumulators']}; prediction {summary[mode]['predicted_class_name']}; validation {summary[mode]['validation_status']}.")
    lines.append("- Cycle values are simulated measured counters; no speedup claim is made."); (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
