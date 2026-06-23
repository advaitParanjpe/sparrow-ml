"""Small, explicit configuration loader for repository-local settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def repository_root(start: Path | None = None) -> Path:
    path = (start or Path(__file__)).resolve()
    for candidate in (path, *path.parents):
        if (candidate / "configs" / "project.yaml").is_file():
            return candidate
    raise RuntimeError("Unable to locate SparrowML repository root")


@dataclass(frozen=True)
class SparrowVConfig:
    root: Path
    enabled: bool
    timeout_seconds: int
    commands: dict[str, str]
    artifacts: dict[str, str]


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    name: str
    artifacts_root: Path
    data_root: Path
    sparrow_v: SparrowVConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = yaml.safe_load(source) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    return value


def load_project_config(root: Path | None = None) -> ProjectConfig:
    root = (root or repository_root()).resolve()
    project = _load_yaml(root / "configs" / "project.yaml")
    target_ref = project["targets"]["sparrow_v"]
    target = _load_yaml(root / target_ref)
    configured_root = os.environ.get(target["root_env"], target["default_root"])
    sparrow_v_root = Path(configured_root).expanduser()
    if not sparrow_v_root.is_absolute():
        sparrow_v_root = (root / sparrow_v_root).resolve()
    return ProjectConfig(
        root=root,
        name=project["project"]["name"],
        artifacts_root=root / project["artifacts_root"],
        data_root=root / project["data_root"],
        sparrow_v=SparrowVConfig(
            root=sparrow_v_root,
            enabled=bool(target["enabled"]),
            timeout_seconds=int(target["timeout_seconds"]),
            commands=dict(target["commands"]),
            artifacts=dict(target["artifacts"]),
        ),
    )
