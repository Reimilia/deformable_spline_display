"""CLI: build every scenario in `scenarios.py` and write it to ../data/*.json.

Usage: python export.py [--out-dir ../data]
"""
import argparse
import json
from pathlib import Path

import numpy as np

import scenarios

REQUIRED_META_KEYS = {"mode", "I", "R_cas", "segments"}
REQUIRED_FRAME_KEYS = {"t", "quaternion", "M_body"}
REQUIRED_GEOMETRY_KEYS = {"verts_body", "handle_idx", "hoop_idx", "face_thickness", "face_normal_body"}


def _tolist(x):
    return x.tolist() if hasattr(x, "tolist") else list(x)


def scenario_to_json(scenario):
    """Validate and serialize one scenario dict to the documented schema
    (see ARCHITECTURE.md). Raises if required keys are missing or array
    lengths are inconsistent."""
    meta = scenario["meta"]
    frames = scenario["frames"]
    background = scenario.get("background", {})
    geometry = scenario.get("geometry")

    if geometry is None:
        raise ValueError("scenario missing 'geometry'")
    missing_geometry = REQUIRED_GEOMETRY_KEYS - geometry.keys()
    if missing_geometry:
        raise ValueError(f"scenario geometry missing required keys: {missing_geometry}")
    verts_body = geometry["verts_body"]
    for v in verts_body:
        if len(v) != 3:
            raise ValueError(f"geometry.verts_body entry has length {len(v)}, expected 3")
    if len(geometry["face_normal_body"]) != 3:
        raise ValueError("geometry.face_normal_body must have length 3")

    missing_meta = REQUIRED_META_KEYS - meta.keys()
    if missing_meta:
        raise ValueError(f"scenario meta missing required keys: {missing_meta}")
    if meta["mode"] not in ("free", "controlled"):
        raise ValueError(f"invalid mode: {meta['mode']!r}")
    if meta["mode"] == "controlled" and not {"desired_H", "desired_L"} <= meta.keys():
        raise ValueError("controlled scenario meta missing desired_H/desired_L")
    if meta["mode"] == "free":
        if "H_axis" not in meta:
            raise ValueError("free scenario meta missing H_axis")
        if len(meta["H_axis"]) != 3:
            raise ValueError(f"H_axis has length {len(meta['H_axis'])}, expected 3")

    missing_frames = REQUIRED_FRAME_KEYS - frames.keys()
    if missing_frames:
        raise ValueError(f"scenario frames missing required keys: {missing_frames}")

    t, q, M = frames["t"], frames["quaternion"], frames["M_body"]
    n = len(t)
    if len(q) != n or len(M) != n:
        raise ValueError(
            f"frame array length mismatch: t={n}, quaternion={len(q)}, M_body={len(M)}"
        )
    for qi in q:
        if len(qi) != 4:
            raise ValueError(f"quaternion entry has length {len(qi)}, expected 4")
    for mi in M:
        if len(mi) != 3:
            raise ValueError(f"M_body entry has length {len(mi)}, expected 3")

    return {
        "meta": {k: (_tolist(v) if hasattr(v, "tolist") else v) for k, v in meta.items()},
        "frames": {
            "t": _tolist(t),
            "quaternion": [_tolist(qi) for qi in q],
            "M_body": [_tolist(mi) for mi in M],
        },
        "background": {
            "trajectories": [_tolist(c) for c in background.get("trajectories", [])],
            "separatrices": [_tolist(c) for c in background.get("separatrices", [])],
            "stable_fixed_points": _tolist(background.get("stable_fixed_points", [])),
            "unstable_fixed_points": _tolist(background.get("unstable_fixed_points", [])),
        },
        "geometry": {
            "verts_body": [_tolist(v) for v in verts_body],
            "handle_idx": _tolist(geometry["handle_idx"]),
            "hoop_idx": _tolist(geometry["hoop_idx"]),
            "face_thickness": float(geometry["face_thickness"]),
            "face_normal_body": _tolist(geometry["face_normal_body"]),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(Path(__file__).parent.parent / "data"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = {
        "dzhanibekov_free": scenario_to_json(scenarios.build_free_narrative_scenario()),
        "controlled_imin": scenario_to_json(scenarios.build_controlled_scenario(target_axis=0)),
        "controlled_imax": scenario_to_json(scenarios.build_controlled_scenario(target_axis=2)),
        "alignment": scenario_to_json(scenarios.build_alignment_scenario(R_star=np.eye(3))),
    }
    for name, doc in docs.items():
        path = out_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(doc, f)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
