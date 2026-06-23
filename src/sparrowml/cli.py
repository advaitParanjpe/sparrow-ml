"""Minimal CLI for local diagnostics and contract validation."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

from . import __version__
from .artifacts.schemas import DeploymentManifest, ModelMetadata
from .config import load_project_config
from .targets.sparrow_v.contracts import SparrowVResult
from .data.fixture import generate_fixture, load_fixture, write_fixture
from .evaluation.report import write_summary
from .training.trainer import load_and_evaluate, train_baseline
from .quantization.pipeline import calibrate as calibrate_int8, evaluate as evaluate_int8, quantize as quantize_int8, run_baseline as run_int8_baseline
from .sparsity.pipeline import evaluate as evaluate_sparse, finetune as finetune_sparse, pack as pack_sparse, prune as prune_sparse, run_baseline as run_sparse_baseline
from .compiler.exporter import export_package, run_baseline as run_export_baseline, validate_package
from .compiler.ir import parse_ir, serialize_ir
from .compiler.lowering import lower_artifact
from .targets.sparrow_v.compatibility import audit
from .targets.sparrow_v.discovery import discover
from .targets.sparrow_v.runtime import prepare as prepare_sparrowv, run as run_sparrowv, validate_result as validate_sparrowv_result
from .targets.sparrow_v.reports import write_baseline_report


def _doctor() -> int:
    config = load_project_config()
    print(f"Python: {sys.version.split()[0]}")
    print(f"SparrowML import: {__version__}")
    print(f"Repository root: {config.root}")
    print(f"Sparrow-V root: {config.sparrow_v.root}")
    print(f"Sparrow-V exists: {config.sparrow_v.root.exists()}")
    print(f"python3 available: {shutil.which('python3') is not None}")
    print(f"git available: {shutil.which('git') is not None}")
    return 0


def _show_config() -> int:
    config = load_project_config()
    print("project:", config.name)
    print("artifacts_root:", config.artifacts_root)
    print("data_root:", config.data_root)
    print("sparrow_v.root:", config.sparrow_v.root)
    print("sparrow_v.enabled:", config.sparrow_v.enabled)
    return 0


def _validate_contracts() -> int:
    model = ModelMetadata("bootstrap-example", (1, 8), 2, "int8-planned")
    DeploymentManifest(model, "artifacts/input.bin", "artifacts/dense.bin", "artifacts/sparse.bin", "artifacts/sparse_meta.bin", "artifacts/biases.bin", "artifacts/expected.bin", "artifacts/program.bin", "artifacts/data.bin")
    SparrowVResult("complete", (4, -2), 0, 12, 8, 1, 1, 0, 64, 0)
    print("contracts: valid")
    return 0


def _sparrowv_checkout():
    config = load_project_config()
    return config, discover(config.root, "../sparrow-v")


def _sparrowv_doctor() -> int:
    config, checkout = _sparrowv_checkout(); report = audit(checkout)
    target = config.root / "artifacts/phase5_runtime/compatibility.json"; target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Sparrow-V: resolved ({report['checkout']['discovery_source']}); compatible={report['compatible']}; missing={','.join(report['missing_requirements']) or 'none'}")
    return 0 if report["compatible"] else 2


def _phase5_package(root: Path, mode: str) -> Path:
    package = root / "artifacts/phase4_export" / mode
    return package if package.is_dir() else export_package(root, mode)


def _prepare_sparrowv(mode: str, package: str | None, output: str | None) -> int:
    root = load_project_config().root; source = root / package if package else _phase5_package(root, mode)
    target = root / (output or f"artifacts/phase5_runtime/{mode}")
    prepare_sparrowv(source, target); print(f"prepared: {target.relative_to(root)}"); return 0


def _run_sparrowv(mode: str, package: str | None, output: str | None) -> int:
    config, checkout = _sparrowv_checkout(); source = config.root / package if package else _phase5_package(config.root, mode)
    target = config.root / (output or f"artifacts/phase5_runtime/{mode}")
    result = run_sparrowv(checkout, source, target, config.sparrow_v.timeout_seconds)
    print(f"Sparrow-V {mode}: {result['validation_status']}")
    return 0 if result["validation_status"] == "passed" else 1


def _validate_sparrowv(package: str, result: str) -> int:
    root = load_project_config().root; verdict = validate_sparrowv_result(json.loads((root / result).read_text()), root / package)
    print(f"Sparrow-V result validation: {verdict['valid']}"); return 0 if verdict["valid"] else 1


def _run_sparrowv_baseline() -> int:
    config, checkout = _sparrowv_checkout(); compatibility = audit(checkout)
    if not compatibility["compatible"]: raise ValueError("Sparrow-V is incompatible: " + ", ".join(compatibility["missing_requirements"]))
    root = config.root / "artifacts/phase5_runtime"; root.mkdir(parents=True, exist_ok=True)
    (root / "compatibility.json").write_text(json.dumps(compatibility, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results = {}
    for mode in ("dense", "sparse"):
        package = _phase5_package(config.root, mode); runs = []
        for _ in range(2): runs.append(run_sparrowv(checkout, package, root / mode, config.sparrow_v.timeout_seconds))
        if not all(item["validation_status"] == "passed" for item in runs): raise ValueError(f"Sparrow-V {mode} validation failed")
        results[mode] = runs
    write_baseline_report(root, results); print("Sparrow-V baseline: dense and sparse passed twice"); return 0


def _phase1_config(path: str | None) -> tuple[dict[str, object], Path]:
    root = load_project_config().root
    config_path = (root / (path or "configs/experiments/fp32_sensor_baseline.yaml")).resolve()
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        dataset, training = config["dataset"], config["training"]
        if dataset["feature_count"] != 16 or len(dataset["class_names"]) != 4 or int(training["epochs"]) < 1:
            raise ValueError("Phase 1 requires 16 features, four classes, and at least one epoch")
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid Phase 1 configuration {config_path}: {exc}") from exc
    return config, root


def _phase1_paths(config: dict[str, object], root: Path) -> tuple[Path, Path]:
    return root / config["dataset"]["directory"], root / config["output_directory"]


def _generate_fixture(config_path: str | None) -> int:
    config, root = _phase1_config(config_path)
    dataset_dir, _ = _phase1_paths(config, root)
    dataset = config["dataset"]
    examples = generate_fixture(
        seed=int(dataset["generation_seed"]), split_seed=int(dataset["split_seed"]),
        samples_per_class=int(dataset["total_samples"]) // 4,
        split_counts_per_class={key: int(value) for key, value in dataset["split_counts_per_class"].items()},
    )
    metadata = write_fixture(dataset_dir, examples, int(dataset["generation_seed"]), int(dataset["split_seed"]))
    print(f"fixture: {dataset_dir}")
    print(f"split sizes: {metadata['split_sizes']}; class counts: {metadata['class_counts']}")
    return 0


def _train_fp32(config_path: str | None) -> int:
    config, root = _phase1_config(config_path)
    dataset_dir, output_dir = _phase1_paths(config, root)
    if not (dataset_dir / "examples.jsonl").is_file():
        _generate_fixture(config_path)
    examples = load_fixture(dataset_dir)
    training = config["training"]
    metrics, _ = train_baseline(
        examples, seed=int(config["seed"]), dataloader_seed=int(training["dataloader_seed"]),
        learning_rate=float(training["learning_rate"]), epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]), output_directory=output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    shutil.copyfile(dataset_dir / "metadata.json", output_dir / "dataset_metadata.json")
    write_summary(output_dir / "summary.md", metrics)
    print(f"trained: best epoch {metrics['best_epoch']}; test fixture accuracy {metrics['evaluations']['test']['fixture_accuracy']:.4%}")
    return 0


def _evaluate_fp32(config_path: str | None) -> int:
    config, root = _phase1_config(config_path)
    dataset_dir, output_dir = _phase1_paths(config, root)
    if not (output_dir / "best_fp32.pt").is_file():
        raise ValueError(f"missing checkpoint: {output_dir / 'best_fp32.pt'}")
    evaluations = load_and_evaluate(load_fixture(dataset_dir), output_dir)
    metrics_path = output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["evaluations"] = evaluations
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "confusion_matrix.json").write_text(json.dumps(evaluations["test"]["confusion_matrix"], indent=2) + "\n", encoding="utf-8")
    write_summary(output_dir / "summary.md", metrics)
    print(f"evaluated: test fixture accuracy {evaluations['test']['fixture_accuracy']:.4%}")
    return 0


def _run_fp32_baseline(config_path: str | None) -> int:
    _generate_fixture(config_path)
    _train_fp32(config_path)
    return _evaluate_fp32(config_path)


def _phase2(command: str, config_path: str | None) -> int:
    commands = {"calibrate-int8": calibrate_int8, "quantize-int8": quantize_int8, "evaluate-int8": evaluate_int8, "run-int8-baseline": run_int8_baseline}
    result = commands[command](config_path)
    if command == "run-int8-baseline":
        test = result["evaluations"]["test"]
        print(f"INT8 baseline: test fixture accuracy {test['int8_fixture_accuracy']:.4%}; agreement {test['prediction_agreement']:.4%}")
    else:
        print(f"{command}: complete")
    return 0


def _phase3(command: str, config_path: str | None) -> int:
    commands = {"prune-2of4": prune_sparse, "finetune-sparse": finetune_sparse, "pack-sparse": pack_sparse, "evaluate-sparse": evaluate_sparse, "run-sparse-baseline": run_sparse_baseline}
    result = commands[command](config_path)
    if command == "run-sparse-baseline":
        test = result["evaluations"]["test"]["after_fine_tuning"]
        print(f"sparse baseline: test fixture accuracy {test['sparse_int8_fixture_accuracy']:.4%}; dense agreement {test['prediction_agreement_with_dense_int8']:.4%}")
    else:
        print(f"{command}: complete")
    return 0


def _lower_ir(mode: str, output: str | None) -> int:
    root = load_project_config().root
    source = root / ("artifacts/phase2_int8/quantized_model.json" if mode == "dense" else "artifacts/phase3_sparse/sparse_quantized_model.json")
    target = root / (output or f"artifacts/phase4_export/{mode}_ir.json")
    target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(serialize_ir(lower_artifact(source, root)))
    print(f"IR: {target.relative_to(root)}")
    return 0


def _export_sparrowv(mode: str, output: str | None) -> int:
    root = load_project_config().root; package = export_package(root, mode, root / output if output else None)
    print(f"export: {package.relative_to(root)}")
    return 0


def _validate_ir(ir_file: str) -> int:
    root = load_project_config().root
    ir = parse_ir((root / ir_file).read_text(encoding="utf-8"))
    print(f"IR: {ir['model_name']}; {ir['operators'][0]['op_type']}; {len(ir['tensors'])} tensors")
    return 0


def _validate_export(package: str) -> int:
    root = load_project_config().root; result = validate_package(root / package)
    print(f"export validation: {result['reference_equivalence']}")
    return 0


def _run_export_baseline() -> int:
    result = run_export_baseline(load_project_config().root)
    print(f"export baseline: dense={result['dense']['reference_equivalence']}; sparse={result['sparse']['reference_equivalence']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sparrowml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "show-config", "validate-contracts"):
        subparsers.add_parser(name)
    for name in ("generate-fixture", "train-fp32", "evaluate-fp32", "run-fp32-baseline"):
        command_parser = subparsers.add_parser(name)
        command_parser.add_argument("--config", help="repository-relative Phase 1 YAML configuration")
    for name in ("calibrate-int8", "quantize-int8", "evaluate-int8", "run-int8-baseline"):
        command_parser = subparsers.add_parser(name)
        command_parser.add_argument("--config", help="repository-relative Phase 2 YAML configuration")
    for name in ("prune-2of4", "finetune-sparse", "pack-sparse", "evaluate-sparse", "run-sparse-baseline"):
        command_parser = subparsers.add_parser(name)
        command_parser.add_argument("--config", help="repository-relative Phase 3 YAML configuration")
    lower = subparsers.add_parser("lower-ir")
    lower.add_argument("--mode", choices=("dense", "sparse"), required=True)
    lower.add_argument("--output", help="IR output path relative to repository")
    validate_ir = subparsers.add_parser("validate-ir")
    validate_ir.add_argument("ir_file", help="IR JSON path relative to repository")
    export = subparsers.add_parser("export-sparrowv")
    export.add_argument("--mode", choices=("dense", "sparse"), required=True)
    export.add_argument("--output", help="package root relative to repository")
    validate = subparsers.add_parser("validate-export")
    validate.add_argument("package", help="deployment package path relative to repository")
    subparsers.add_parser("run-export-baseline")
    subparsers.add_parser("sparrowv-doctor")
    for name in ("prepare-sparrowv-run", "run-sparrowv"):
        phase5 = subparsers.add_parser(name); phase5.add_argument("--mode", choices=("dense", "sparse"), required=True); phase5.add_argument("--package"); phase5.add_argument("--output")
    phase5_validate = subparsers.add_parser("validate-sparrowv-result"); phase5_validate.add_argument("package"); phase5_validate.add_argument("result")
    subparsers.add_parser("run-sparrowv-baseline")
    args = parser.parse_args(argv)
    try:
        commands = {"doctor": lambda: _doctor(), "show-config": lambda: _show_config(), "validate-contracts": lambda: _validate_contracts(),
                    "generate-fixture": lambda: _generate_fixture(args.config), "train-fp32": lambda: _train_fp32(args.config),
                    "evaluate-fp32": lambda: _evaluate_fp32(args.config), "run-fp32-baseline": lambda: _run_fp32_baseline(args.config),
                    "calibrate-int8": lambda: _phase2("calibrate-int8", args.config), "quantize-int8": lambda: _phase2("quantize-int8", args.config),
                    "evaluate-int8": lambda: _phase2("evaluate-int8", args.config), "run-int8-baseline": lambda: _phase2("run-int8-baseline", args.config),
                    "prune-2of4": lambda: _phase3("prune-2of4", args.config), "finetune-sparse": lambda: _phase3("finetune-sparse", args.config),
                    "pack-sparse": lambda: _phase3("pack-sparse", args.config), "evaluate-sparse": lambda: _phase3("evaluate-sparse", args.config), "run-sparse-baseline": lambda: _phase3("run-sparse-baseline", args.config),
                    "lower-ir": lambda: _lower_ir(args.mode, args.output), "validate-ir": lambda: _validate_ir(args.ir_file), "export-sparrowv": lambda: _export_sparrowv(args.mode, args.output), "validate-export": lambda: _validate_export(args.package), "run-export-baseline": lambda: _run_export_baseline()}
        commands.update({"sparrowv-doctor": _sparrowv_doctor, "prepare-sparrowv-run": lambda: _prepare_sparrowv(args.mode, args.package, args.output), "run-sparrowv": lambda: _run_sparrowv(args.mode, args.package, args.output), "validate-sparrowv-result": lambda: _validate_sparrowv(args.package, args.result), "run-sparrowv-baseline": _run_sparrowv_baseline})
        return commands[args.command]()
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
