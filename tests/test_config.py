from pathlib import Path
from sparrowml.config import load_project_config

ROOT = Path(__file__).resolve().parents[1]

def test_project_configuration_loads(monkeypatch):
    monkeypatch.delenv("SPARROWV_ROOT", raising=False)
    config = load_project_config(ROOT)
    assert config.name == "SparrowML"
    assert config.sparrow_v.root == ROOT.parent / "sparrow-v"

def test_environment_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARROWV_ROOT", str(tmp_path / "target"))
    assert load_project_config(ROOT).sparrow_v.root == tmp_path / "target"
