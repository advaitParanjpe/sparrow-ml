import json
from pathlib import Path
import pytest

from sparrowml.targets.sparrow_v.discovery import DiscoveryError, discover
from sparrowml.targets.sparrow_v.runtime import generate_manifest, prepare, validate_result

ROOT = Path(__file__).resolve().parents[1]


def fake_checkout(path: Path) -> None:
    for name in ("README.md", "Makefile", "scripts/run_external_sensor_workload.py", "scripts/sensor_workload.py", "docs/architecture/sensor_workload_export.md"):
        target = path / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_text("x")


def test_discovery_environment_and_missing_contract(monkeypatch, tmp_path):
    fake_checkout(tmp_path / "v"); monkeypatch.setenv("SPARROWV_ROOT", str(tmp_path / "v"))
    assert discover(ROOT).source == "environment"
    monkeypatch.setenv("SPARROWV_ROOT", str(tmp_path / "bad"))
    with pytest.raises(DiscoveryError, match="SPARROWV_ROOT"): discover(ROOT)


@pytest.mark.parametrize("mode", ["dense", "sparse"])
def test_phase4_conversion_is_deterministic_and_isolated(tmp_path, mode):
    package = ROOT / "artifacts/phase4_export" / mode
    first, second = generate_manifest(package), generate_manifest(package)
    assert first == second and not any(str(value).startswith("/") for value in first.values() if isinstance(value, str))
    if mode == "sparse":
        assert len(first["compressed_weights_int8"]) == 4 and all(len(row) == 4 for row in first["compressed_weights_int8"])
        assert len(first["sparse_metadata"]) == 4 and all(len(row) == 4 for row in first["sparse_metadata"])
    output = prepare(package, tmp_path / mode)
    assert json.loads((output / "workload.json").read_text()) == first


def test_validation_rejects_one_value_mismatch(tmp_path):
    package = ROOT / "artifacts/phase4_export/dense"; output = prepare(package, tmp_path / "dense")
    manifest = generate_manifest(package); bad = {"format_version": "sparrowml_sparrowv_runtime_result_v1", "mode": "dense_int8", "package_identity": manifest["source_package_identity"], "parsed_accumulators": [0, -17389, -1218, -26014], "predicted_class_id": 0, "simulator": {"exit_status": 0}, "trap_assertion_status": "clear", "counters": {}}
    assert not validate_result(bad, package)["valid"]
