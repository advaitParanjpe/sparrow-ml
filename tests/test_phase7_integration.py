import os
from pathlib import Path

import pytest

from sparrowml.targets.sparrow_v.discovery import discover
from sparrowml.targets.sparrow_v.multilayer import run

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_real_multilayer_sparrowv_execution(tmp_path):
    try:
        checkout = discover(ROOT)
    except ValueError as exc:
        if os.environ.get("SPARROWML_REQUIRE_SPARROWV"):
            raise
        pytest.skip(str(exc))
    result = run(checkout, ROOT / "artifacts/phase6_multilayer/export", tmp_path / "phase7", 300)
    assert result["validation_status"] == "passed"
    assert result["counter_summary"]["total_dense_int8_multiplications"]["value"] == 320
