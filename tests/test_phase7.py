import json
from pathlib import Path

import pytest

from sparrowml.targets.sparrow_v.multilayer import _hidden, _package, package_identity, prepare, validate_result

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "artifacts/phase6_multilayer/export"


def test_phase6_package_partitions_are_exact_and_deterministic(tmp_path):
    _, ir, inputs, traces = _package(PACKAGE)
    first, second = tmp_path / "first", tmp_path / "second"
    prepare(PACKAGE, first); prepare(PACKAGE, second)
    assert (first / "fc1/partition_0/workload.json").read_bytes() == (second / "fc1/partition_0/workload.json").read_bytes()
    rows = []
    for index in range(4):
        workload = json.loads((first / f"fc1/partition_{index}/workload.json").read_text())
        assert workload["input_int8"] == inputs["samples"][0]["input_int8"]
        assert workload["sparrowml_output_channels"] == list(range(4 * index, 4 * index + 4))
        assert workload["biases_int32"] == [0] * 4
        rows.extend(workload["dense_weights_int8"])
    assert rows == ir["constants"]["fc1_weights"]
    assert package_identity(PACKAGE)
    assert traces["samples"][0]["fc1_acc_int32"]


def test_hidden_processing_matches_phase6_trace_and_clamps():
    _, ir, _, traces = _package(PACKAGE)
    trace = traces["samples"][0]
    actual = _hidden(trace["fc1_acc_int32"], ir)
    assert actual["hidden_int8"] == trace["hidden_int8"]
    assert actual["provenance"] == "host_reconstructed"
    assert actual["clamp_range"] == [0, 127]


def test_validation_rejects_one_intermediate_mismatch():
    _, _, inputs, traces = _package(PACKAGE)
    trace = traces["samples"][0]
    result = {"format_version": "sparrowml_sparrowv_multilayer_runtime_result_v1", "package_identity": package_identity(PACKAGE), "sample_id": inputs["samples"][0]["sample_id"], "fc1": {"post_bias_accumulators": trace["fc1_acc_int32"]}, "intermediate": {"hidden_int8": trace["hidden_int8"][:]}, "fc2": {"post_bias_accumulators": trace["fc2_acc_int32"], "raw_accumulator_provenance": "rtl_produced", "post_bias_provenance": "host_reconstructed"}, "final_prediction": trace["predicted_class"]}
    result["intermediate"]["hidden_int8"][0] = 1
    assert "hidden" in validate_result(result, PACKAGE)["failures"]


def test_invalid_package_hash_is_rejected(tmp_path):
    copied = tmp_path / "package"; copied.mkdir()
    for source in PACKAGE.iterdir():
        if source.is_file(): (copied / source.name).write_bytes(source.read_bytes())
    (copied / "input.json").write_text("{}")
    with pytest.raises(ValueError, match="package hash mismatch"):
        _package(copied)
