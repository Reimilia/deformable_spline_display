"""Generate real checkpoint rollouts for server-free GitHub Pages playback."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from engine import CheckpointEngine


def compact(value):
    if isinstance(value, float):
        return float(f"{value:.8g}")
    if isinstance(value, list):
        return [compact(x) for x in value]
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    return value


def select_points(points, count=80):
    if len(points) <= count:
        return points
    indices = sorted({round(i*(len(points)-1)/(count-1)) for i in range(count)})
    return [points[i] for i in indices]


def prepare_result(result):
    for run in result["runs"]:
        for key in ("body", "world", "reference_body", "reference_world"):
            run[key] = [select_points(p) for p in run[key]]
        for key in ("target_body", "target_world", "observations_world"):
            if run[key] is not None:
                run[key] = select_points(run[key])
    return compact(result)


def export(destination, tasks=1, horizon=2.0):
    engine = CheckpointEngine()
    catalog = engine.catalog()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    previews = []
    for entry in catalog["checkpoints"]:
        groups = {g: min(tasks, entry["tasks"][g]) for g in ("test", "hard_test") if g in entry["tasks"]}
        entry["tasks"] = groups
        entry["comparison_group"] = engine.entry(entry["id"])["base"].relative_to(engine.root).as_posix()
        for group, count in groups.items():
            for index in range(count):
                config = {"checkpoint": entry["id"], "group": group, "task_index": index, "horizon": horizon}
                result = prepare_result(engine.run(config))
                name = hashlib.sha256(f"{entry['id']}:{group}:{index}".encode()).hexdigest()[:20] + ".json"
                (destination / name).write_text(json.dumps(result, allow_nan=False, separators=(",", ":")))
                previews.append({**config, "file": name})
        print(f"{len(previews):3d} previews · {entry['family']} · {entry['label']}", flush=True)
    manifest = {"schema_version": 1, "horizon": horizon, "catalog": catalog, "previews": previews,
        "precision": "8 significant digits", "curve_points": 80,
        "description": "Saved checkpoint inference; view/playback controls remain interactive without a server."}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    active = {item["file"] for item in previews} | {"manifest.json"}
    for old in destination.glob("*.json"):
        if old.name not in active:
            old.unlink()
    print(f"Saved {len(previews)} previews to {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "previews")
    parser.add_argument("--tasks", type=int, default=1, help="Saved tasks per checkpoint per test group")
    parser.add_argument("--horizon", type=float, default=2)
    args = parser.parse_args()
    if args.tasks < 1:
        parser.error("--tasks must be positive")
    export(args.output, args.tasks, args.horizon)
