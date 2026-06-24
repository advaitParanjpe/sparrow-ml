import json

from sparrowml.wisdm_pipeline import OUT, select_rtl_samples


def test_wisdm_rtl_selection_is_held_out_and_deterministic():
    first = select_rtl_samples()
    second = select_rtl_samples()
    test_subjects = set(json.loads((OUT.parent / "phase8a" / "subject_splits.json").read_text())["splits"]["test"])
    assert first == second
    assert len([x for x in first if x["selection_reason"] == "lowest_canonical_correct_int8"]) >= 8
    assert {item["subject_id"] for item in first} <= test_subjects
