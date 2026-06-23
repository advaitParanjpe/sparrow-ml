import json
from pathlib import Path
import pytest
from sparrowml.compiler.exporter import export_package, validate_package
from sparrowml.compiler.ir import parse_ir, serialize_ir
from sparrowml.compiler.lowering import lower_artifact
ROOT = Path(__file__).resolve().parents[1]
def test_lowering_round_trip_and_validation():
    dense = lower_artifact(ROOT / "artifacts/phase2_int8/quantized_model.json", ROOT); sparse = lower_artifact(ROOT / "artifacts/phase3_sparse/sparse_quantized_model.json", ROOT)
    assert dense["operators"][0]["op_type"] == "DenseLinearInt8"; assert sparse["operators"][0]["op_type"] == "SparseLinear2of4Int8"; assert serialize_ir(parse_ir(serialize_ir(dense))) == serialize_ir(dense)
    bad = dict(dense); bad["source_artifact_identity"] = "/absolute/path"
    with pytest.raises(ValueError, match="source_artifact_identity"): serialize_ir(bad)
@pytest.mark.parametrize("mode", ["dense", "sparse"])
def test_export_reloads_and_matches_reference(tmp_path, mode):
    package = export_package(ROOT, mode, tmp_path)
    assert {"manifest.json", "model_ir.json", "memory_map.json", "model_data.bin", "input_data.bin", "expected_output.json", "README.md", "program.json"} <= {path.name for path in package.iterdir()}
    assert validate_package(package)["reference_equivalence"] == "exact"
    manifest = json.loads((package / "manifest.json").read_text()); assert not any(str(value).startswith("/") for value in manifest.values() if isinstance(value, str))
