import os
from pathlib import Path
import pytest

from sparrowml.targets.sparrow_v.discovery import discover
from sparrowml.targets.sparrow_v.runtime import run

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.integration
@pytest.mark.parametrize("mode", ["dense", "sparse"])
def test_real_sparrowv_execution(tmp_path, mode):
    try: checkout = discover(ROOT)
    except ValueError as exc:
        if os.environ.get("SPARROWML_REQUIRE_SPARROWV"):
            raise
        pytest.skip(str(exc))
    result = run(checkout, ROOT / "artifacts/phase4_export" / mode, tmp_path / mode, 300)
    assert result["validation_status"] == "passed"
