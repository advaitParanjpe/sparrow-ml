"""Local, non-mutating Sparrow-V checkout discovery."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

REQUIRED_FILES = ("README.md", "Makefile", "scripts/run_external_sensor_workload.py", "scripts/sensor_workload.py", "docs/architecture/sensor_workload_export.md")

class DiscoveryError(ValueError):
    """A checkout is unavailable or does not provide the supported contract."""

@dataclass(frozen=True)
class SparrowVCheckout:
    root: Path
    source: str

def discover(root: Path, configured_root: str = "../sparrow-v") -> SparrowVCheckout:
    candidates: list[tuple[str, Path]] = []
    if os.environ.get("SPARROWV_ROOT"):
        candidates.append(("environment", Path(os.environ["SPARROWV_ROOT"])))
    candidates += [("configuration", Path(configured_root)), ("sibling", Path("../sparrow-v"))]
    seen: set[Path] = set()
    for source, candidate in candidates:
        path = candidate.expanduser()
        if not path.is_absolute(): path = (root / path).resolve()
        if path in seen: continue
        seen.add(path)
        if not path.is_dir():
            if source == "environment": raise DiscoveryError("SPARROWV_ROOT does not name an existing Sparrow-V checkout")
            continue
        if path == root.resolve(): raise DiscoveryError("Sparrow-V checkout resolves to the SparrowML repository")
        missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
        if not missing: return SparrowVCheckout(path, source)
        raise DiscoveryError("Sparrow-V checkout is missing required contract files: " + ", ".join(missing))
    raise DiscoveryError("Sparrow-V checkout not found; set SPARROWV_ROOT or place it at ../sparrow-v")
