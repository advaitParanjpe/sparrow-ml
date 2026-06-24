import os
import pytest

from sparrowml.wisdm_pipeline import run_phase8c


@pytest.mark.integration
def test_wisdm_rtl_execution():
    if not os.environ.get("SPARROWML_REQUIRE_SPARROWV"):
        pytest.skip("requires local Sparrow-V")
    assert run_phase8c()["all_valid"]
