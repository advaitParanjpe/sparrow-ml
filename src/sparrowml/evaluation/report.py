"""Human-readable, explicitly fixture-scoped Phase 1 summary."""

from __future__ import annotations

from pathlib import Path


def write_summary(path: Path, metrics: dict[str, object]) -> None:
    evaluations = metrics["evaluations"]
    lines = [
        "# Phase 1 FP32 Sensor Baseline",
        "",
        "Measured values below are fixture accuracy on a synthetic deterministic fixture; they are not real-world accuracy claims.",
        "",
        f"- Model: `{metrics['model']['architecture']}` ({metrics['model']['parameter_count']} parameters), FP32 CPU",
        f"- Feature count: {metrics['feature_count']}; class count: {metrics['class_count']}",
        f"- Best epoch (validation loss): {metrics['best_epoch']}",
        f"- Checkpoint size: {metrics['checkpoint']['size_bytes']} bytes",
        "",
        "| Split | Loss | Fixture accuracy |",
        "| --- | ---: | ---: |",
    ]
    for split in ("train", "validation", "test"):
        result = evaluations[split]
        lines.append(f"| {split} | {result['loss']:.6f} | {result['fixture_accuracy']:.4%} |")
    lines += ["", "## Test confusion matrix", "", "Rows are true classes and columns are predicted classes in fixed order `normal, inner, outer, ball`.", "", "```json", str(evaluations["test"]["confusion_matrix"]), "```", "", "## Limitations", "", "This milestone implements no quantization, pruning, compiler lowering, Sparrow-V execution, or hardware measurements."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
