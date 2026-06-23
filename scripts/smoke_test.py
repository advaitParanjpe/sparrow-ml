#!/usr/bin/env python3
"""Fast offline bootstrap validation."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sparrowml.config import load_project_config

def main() -> int:
    config = load_project_config(ROOT)
    assert config.name == "SparrowML"
    result = subprocess.run([sys.executable, "-m", "sparrowml.cli", "validate-contracts"], cwd=ROOT, capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    assert result.returncode == 0, result.stderr
    assert "contracts: valid" in result.stdout
    print("smoke: passed")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
