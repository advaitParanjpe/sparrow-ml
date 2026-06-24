#!/usr/bin/env python3
"""Small deterministic repository checks; intentionally not a general linter."""
from __future__ import annotations
import argparse
import importlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
REQUIRED_FILES = ("AGENTS.md", "README.md", "LICENSE", "Makefile", "pyproject.toml", ".gitignore", ".env.example", "configs/project.yaml", "configs/targets/sparrow_v.yaml", "docs/current_milestone.md", "docs/codex_context.md", "docs/codex_milestone_prompt.md", "docs/codex_milestone_result.md", "docs/architecture.md", "docs/build_roadmap.md", "docs/data_contracts.md", "docs/experiment_policy.md", "docs/decisions/ADR-001-repository-boundary.md", "docs/results/final_results.md", "docs/reproduction.md", "docs/source_manifest.md", "docs/release_checklist.md", "docs/portfolio_summary.md", "scripts/run_milestone.sh", "scripts/smoke_test.py")
REQUIRED_DIRS = ("src/sparrowml", "src/sparrowml/artifacts", "src/sparrowml/targets/sparrow_v", "tests", "examples", "data/raw", "data/interim", "data/processed", "artifacts", "experiments")
LINK_PATTERN = re.compile(r"(?<!!)\[[^]]*\]\(([^)]+)\)")
ABSOLUTE_PATTERN = re.compile(r"(?<![\w-])/(?:Users|home|tmp)/")

def files_in_tree() -> list[Path]:
    ignored_generated = ((ROOT / "artifacts").resolve(), (ROOT / "data" / "processed" / "sensor_fixture").resolve())
    return [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts and not any(root in p.resolve().parents for root in ignored_generated)]

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-only", action="store_true")
    args = parser.parse_args()
    errors: list[str] = []
    for relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file(): errors.append(f"missing file: {relative}")
    for relative in REQUIRED_DIRS:
        if not (ROOT / relative).is_dir(): errors.append(f"missing directory: {relative}")
    for path in files_in_tree():
        relative = path.relative_to(ROOT)
        if path.name == ".DS_Store": errors.append(f"forbidden .DS_Store: {relative}")
        if path.name in {".env", "id_rsa", "credentials.json"}: errors.append(f"forbidden secret file: {relative}")
        if path.suffix in {".pt", ".pth", ".ckpt", ".onnx", ".tflite"}: errors.append(f"checkpoint: {relative}")
        if path.stat().st_size > 10 * 1024 * 1024: errors.append(f"large binary: {relative}")
        if path.suffix in {".md", ".yaml", ".yml", ".json", ".toml"}:
            text = path.read_text(encoding="utf-8")
            if ".codex/milestone_result.md" in text: errors.append(f"obsolete result path: {relative}")
            if ABSOLUTE_PATTERN.search(text): errors.append(f"machine-specific path: {relative}")
            if re.search(r"(?:api[_-]?key|password)\s*[:=]\s*['\"][^'\"]+", text, re.I): errors.append(f"possible secret: {relative}")
        if path.suffix == ".md":
            for target in LINK_PATTERN.findall(path.read_text(encoding="utf-8")):
                target = target.split("#", 1)[0].strip("<>")
                if target and "://" not in target and not target.startswith("mailto:") and not (path.parent / target).exists(): errors.append(f"broken Markdown link: {relative} -> {target}")
    canonical_metrics = ("0.9259473531964131", "0.9197794804065271", "12/12 exact")
    final_results = ROOT / "docs/results/final_results.md"
    if final_results.is_file():
        final_text = final_results.read_text(encoding="utf-8")
        for metric in canonical_metrics:
            if metric not in final_text: errors.append(f"missing canonical final metric: {metric}")
    readme = ROOT / "README.md"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8").lower()
        for metric in canonical_metrics:
            if metric not in text: errors.append(f"README missing canonical metric: {metric}")
        for claim in ("production-grade", "state of the art", "industry leading", "tapeout ready"):
            if claim in text: errors.append(f"unsupported README claim: {claim}")
    try:
        import yaml
        for path in (ROOT / "configs").rglob("*.yaml"): yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in ROOT.rglob("*.json"): json.loads(path.read_text(encoding="utf-8"))
    except ImportError: errors.append("PyYAML is required for configuration validation")
    except Exception as exc: errors.append(f"configuration parse error: {exc}")
    if not args.docs_only:
        try: importlib.import_module("sparrowml")
        except Exception as exc: errors.append(f"package import failed: {exc}")
    if errors:
        print("repository checks failed:", *[f"- {e}" for e in errors], sep="\n")
        return 1
    print("repository checks: passed")
    return 0

if __name__ == "__main__": raise SystemExit(main())
