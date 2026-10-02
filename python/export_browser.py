"""Export exact checkpoint parameters and saved tasks for NumPy/WASM inference.

No retraining, quantization, optimizer state, or recorded rollouts are exported.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from engine import CheckpointEngine, load


def plain(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def export(destination):
    destination.mkdir(parents=True, exist_ok=True)
    engine = CheckpointEngine()
    catalog = engine.catalog()
    variants = {}
    for entry in catalog["checkpoints"]:
        source = engine.entry(entry["id"])
        ck = load(source["path"])
        state = ck.get("state_dict", ck.get("model_state"))
        key = hashlib.sha256(entry["id"].encode()).hexdigest()[:20]
        filename = key + ".npz"
        np.savez_compressed(destination / filename,
                            **{k: v.detach().cpu().numpy() for k, v in state.items()})
        entry.update(weights_file=filename, mode=source["mode"], checkpoint_epoch=ck.get("epoch"),
                     source_sha256=hashlib.sha256(source["path"].read_bytes()).hexdigest(),
                     weights_sha256=hashlib.sha256((destination / filename).read_bytes()).hexdigest())
        variant = source["base"].relative_to(engine.root).as_posix()
        vkey = hashlib.sha256(variant.encode()).hexdigest()[:20]
        entry["variant_file"] = vkey + ".json.gz"
        if variant in variants:
            continue
        meta = load(source["base"] / "model.pt")
        if entry["family"] == "spline":
            metadata = {"config": meta["config"]}
            groups = engine.groups(entry["id"])
        else:
            metadata = {k: meta[k] for k in ("chain_params", "model_config", "Ksp", "closure_model", "use_fim")}
            groups = {}
            for name, tasks in engine.groups(entry["id"]).items():
                groups[name] = [{"c0": t.c0, "c_target": t.c_target, "pose0": t.initial_pose,
                                 "pose_target": t.pose_target, "task_id": t.task_id,
                                 "metadata": {k: v for k, v in (t.metadata or {}).items()
                                              if k in {"external_task", "rigid_loop_q_circle", "rigid_loop_q_target",
                                                       "rigid_loop_shared_observation_body"}}} for t in tasks]
        payload = json.dumps(plain({"metadata": metadata, "groups": groups}),
                             separators=(",", ":"), allow_nan=False).encode()
        with (destination / entry["variant_file"]).open("wb") as out:
            with gzip.GzipFile(fileobj=out, mode="wb", mtime=0) as stream:
                stream.write(payload)
        variants[variant] = entry["variant_file"]
    catalog.update(schema_version=2, runtime="numpy-float64", variants=len(variants))
    (destination / "manifest.json").write_text(json.dumps(catalog, indent=2) + "\n")
    print(f"Exported {catalog['count']} checkpoints and {len(variants)} task collections to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "web/public/models")
    export(parser.parse_args().output)
