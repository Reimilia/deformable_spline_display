import numpy as np
import pytest

import export
import scenarios

REQUIRED_META_KEYS = {"mode", "I", "R_cas", "segments"}
REQUIRED_FRAME_KEYS = {"t", "quaternion", "M_body"}
REQUIRED_GEOMETRY_KEYS = {"verts_body", "handle_idx", "hoop_idx", "face_thickness", "face_normal_body"}


def _check_schema(doc):
    assert "meta" in doc and "frames" in doc and "background" in doc and "geometry" in doc
    assert REQUIRED_META_KEYS <= doc["meta"].keys()
    assert doc["meta"]["mode"] in ("free", "controlled")
    assert REQUIRED_FRAME_KEYS <= doc["frames"].keys()
    assert REQUIRED_GEOMETRY_KEYS <= doc["geometry"].keys()

    n = len(doc["frames"]["t"])
    assert len(doc["frames"]["quaternion"]) == n
    assert len(doc["frames"]["M_body"]) == n
    for q in doc["frames"]["quaternion"]:
        assert len(q) == 4
    for m in doc["frames"]["M_body"]:
        assert len(m) == 3

    for v in doc["geometry"]["verts_body"]:
        assert len(v) == 3
    assert len(doc["geometry"]["face_normal_body"]) == 3
    all_idx = sorted(doc["geometry"]["handle_idx"] + doc["geometry"]["hoop_idx"])
    assert all_idx == list(range(len(doc["geometry"]["verts_body"])))

    if doc["meta"]["mode"] == "controlled":
        assert "desired_H" in doc["meta"]
        assert "desired_L" in doc["meta"]
    if doc["meta"]["mode"] == "free":
        assert len(doc["meta"]["H_axis"]) == 3


def test_free_narrative_scenario_matches_schema():
    scenario = scenarios.build_free_narrative_scenario()
    doc = export.scenario_to_json(scenario)
    _check_schema(doc)
    assert doc["meta"]["mode"] == "free"


def test_controlled_scenario_matches_schema():
    scenario = scenarios.build_controlled_scenario(target_axis=2)
    doc = export.scenario_to_json(scenario)
    _check_schema(doc)
    assert doc["meta"]["mode"] == "controlled"


def test_export_rejects_inconsistent_frame_lengths():
    scenario = scenarios.build_free_narrative_scenario()
    scenario["frames"]["t"] = scenario["frames"]["t"][:-1]  # desync on purpose
    with pytest.raises(Exception):
        export.scenario_to_json(scenario)


def test_export_rejects_missing_geometry():
    scenario = scenarios.build_free_narrative_scenario()
    del scenario["geometry"]
    with pytest.raises(Exception):
        export.scenario_to_json(scenario)


def test_export_rejects_free_scenario_missing_H_axis():
    scenario = scenarios.build_free_narrative_scenario()
    del scenario["meta"]["H_axis"]
    with pytest.raises(Exception):
        export.scenario_to_json(scenario)
