import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import (CheckpointEngine, DTYPE, load, SplineCurveConfig,
    SplineCurvePHTemplate, task_from_cpu_dict, rollout_spline_ph)

ENGINE = CheckpointEngine()
CATALOG = ENGINE.catalog()["checkpoints"]


@pytest.mark.parametrize("entry", CATALOG, ids=lambda e: e["id"])
def test_every_saved_model_integrates(entry):
    response = ENGINE.run({"checkpoint": entry["id"], "horizon": 1})
    run = response["runs"][0]
    assert len(run["times"]) == entry["defaults"]["steps"] + 1
    assert not run["diverged"]
    assert len(run["body"]) == len(run["phase_q"]) == len(run["phase_p"])
    assert all(len(v) == len(run["metric_times"]) for v in run["metrics"].values())
    assert np.isfinite(np.asarray(run["phase_q"])).all()
    assert np.isfinite(np.asarray(run["world"])).all()
    assert "measurement" not in run["label"].lower()
    json.dumps(response, allow_nan=False)


def final(family, mode="FIM"):
    return next(e for e in CATALOG if e["family"] == family and e["mode_label"] == mode and e["stage"] == "Final")


def test_spline_adapter_matches_original_integrator():
    entry = final("spline")
    e = ENGINE.entry(entry["id"])
    ck = load(e["path"])
    cfg = SplineCurveConfig(**ck["config"])
    model = SplineCurvePHTemplate(cfg, hidden=ck["state_dict"]["net.0.weight"].shape[0], dtype=DTYPE)
    model.load_state_dict(ck["state_dict"])
    raw = copy.deepcopy(ENGINE.groups(e["id"])["test"][0])
    task = task_from_cpu_dict(raw, torch.device("cpu"), DTYPE)
    with torch.no_grad():
        ref = rollout_spline_ph(model, task, cfg, horizon_multiplier=1)
    run = ENGINE.run({"checkpoint": e["id"], "horizon": 1})["runs"][0]
    np.testing.assert_allclose(np.asarray(run["phase_q"])[:, :2*cfg.n_ctrl], ref["P"].flatten(1).numpy(), atol=1e-12)
    np.testing.assert_allclose(run["metrics"]["hamiltonian"], ref["energy"][:, 5].numpy(), atol=1e-12)


@pytest.mark.parametrize("family", ["multilink", "spline"])
def test_overlay_uses_same_initial_task_and_does_not_mutate_cache(family):
    a = final(family)
    b = final(family, "Open")
    raw = copy.deepcopy(ENGINE.groups(a["id"])["hard_test"][0])
    request = {"checkpoint": a["id"], "compare": b["id"], "group": "hard_test", "horizon": 1,
        "target_dx": .03, "target_angle": 8, "initial_angle": 12}
    result = ENGINE.run(request)
    np.testing.assert_allclose(result["runs"][0]["world"][0], result["runs"][1]["world"][0], atol=1e-8)
    again = ENGINE.groups(a["id"])["hard_test"][0]
    if family == "spline":
        torch.testing.assert_close(raw["pose_target"], again["pose_target"])
    else:
        torch.testing.assert_close(raw.pose_target, again.pose_target)


def test_noise_is_deterministic_and_changes_observations():
    e = next(e for e in CATALOG if e["family"] == "spline" and e["conditioning"] == "observation" and e["stage"] == "Final")
    req = {"checkpoint": e["id"], "noise": .01, "seed": 7, "horizon": 1}
    a = ENGINE.run(req)["runs"][0]
    b = ENGINE.run(req)["runs"][0]
    c = ENGINE.run({**req, "noise": 0})["runs"][0]
    assert a["world"] == b["world"]
    assert a["observations_world"] != c["observations_world"]


@pytest.mark.parametrize("overrides", [{"steps": 0}, {"steps": 12.5}, {"horizon": 100}, {"damping": float("nan")}, {"task_index": -1}, {"group": "bad"}])
def test_invalid_configuration_rejected(overrides):
    with pytest.raises(ValueError):
        ENGINE.run({"checkpoint": final("spline")["id"], **overrides})


def test_non_model_files_and_paths_rejected():
    assert len(CATALOG) == 58
    assert not any("cache.pt" in e["id"] for e in CATALOG)
    with pytest.raises(ValueError):
        ENGINE.run({"checkpoint": "../../etc/passwd"})
    with pytest.raises(ValueError):
        ENGINE.run({"checkpoint": final("spline")["id"], "compare": final("multilink")["id"]})
