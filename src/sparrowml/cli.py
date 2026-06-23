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
    args = parser.parse_args(argv)
    try:
        commands = {"doctor": lambda: _doctor(), "show-config": lambda: _show_config(), "validate-contracts": lambda: _validate_contracts(),
                    "generate-fixture": lambda: _generate_fixture(args.config), "train-fp32": lambda: _train_fp32(args.config),
                    "evaluate-fp32": lambda: _evaluate_fp32(args.config), "run-fp32-baseline": lambda: _run_fp32_baseline(args.config),
                    "calibrate-int8": lambda: _phase2("calibrate-int8", args.config), "quantize-int8": lambda: _phase2("quantize-int8", args.config),
                    "evaluate-int8": lambda: _phase2("evaluate-int8", args.config), "run-int8-baseline": lambda: _phase2("run-int8-baseline", args.config),
                    "prune-2of4": lambda: _phase3("prune-2of4", args.config), "finetune-sparse": lambda: _phase3("finetune-sparse", args.config),
                    "pack-sparse": lambda: _phase3("pack-sparse", args.config), "evaluate-sparse": lambda: _phase3("evaluate-sparse", args.config), "run-sparse-baseline": lambda: _phase3("run-sparse-baseline", args.config)}
        return commands[args.command]()
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
