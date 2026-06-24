import json
from pathlib import Path

import numpy as np

from sparrowml.wisdm_pipeline import CLASS_NAMES, OUT, _metrics, _phase8a_gate, export
from sparrowml.quantization.multilayer import validate_export


def test_wisdm_phase8a_input_and_train_only_contract():
    gate = _phase8a_gate()
    pre = json.loads((OUT / "preprocessing.json").read_text())
    assert gate["passed"] and len(pre["feature_order"]) == 16
    assert pre["fit_split"] == "train"
    assert set(CLASS_NAMES) == {"walking", "jogging", "sitting", "standing"}


def test_metrics_and_export_reload_are_deterministic():
    metric = _metrics(np.asarray([0, 1, 2, 3]), np.asarray([0, 1, 2, 3]))
    assert metric["macro_f1"] == 1.0
    result = export()
    package = OUT / "export"
    assert result["validation"]["passed"]
    assert validate_export(package)["reference_equivalence"] == "exact"
    manifest = json.loads((package / "manifest.json").read_text())
    assert manifest["no_absolute_dataset_paths"]
    assert all(not Path(value).is_absolute() for value in json.dumps(manifest).split())
