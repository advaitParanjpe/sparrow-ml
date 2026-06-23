import pytest
from sparrowml.targets.sparrow_v.contracts import SparrowVResult

def test_result_schema():
    assert SparrowVResult("complete", (1, 2), 1, 1, 1, 1, 1, 1, 1, 0).cycles == 1

def test_invalid_ranges_and_relative_log_path():
    with pytest.raises(ValueError): SparrowVResult("complete", (1,), 1, 0, 0, 0, 0, 0, 0, 0)
    with pytest.raises(ValueError): SparrowVResult("complete", (1,), 0, -1, 0, 0, 0, 0, 0, 0)
    with pytest.raises(ValueError): SparrowVResult("complete", (1,), 0, 0, 0, 0, 0, 0, 0, 0, "/log")
