"""Checkpoint adapters. All integration uses the supplied multilink_ph code."""
from __future__ import annotations

import copy
import json
import math
import sys
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

VENDOR = Path(__file__).resolve().parent / "vendor"
sys.path.insert(0, str(VENDOR))
from multilink_ph.chain import ChainParams, PlanarMultiLinkChain
from multilink_ph.external_forces import ExternalForceConfig
from multilink_ph.info_geometry import InformationGeometryConfig
from multilink_ph.info_ph import WorldPoseInformationPHTemplate, rollout_world_pose_info_ph, stack_world_pose_features
from multilink_ph.rigid_loop import LoopClosureModel, set_task_closure_model
from multilink_ph.spline_curve_ph import (SplineCurveConfig, SplineCurvePHTemplate,
    task_from_cpu_dict, rollout_spline_ph, evaluate_curve, apply_pose)

torch.set_num_threads(1)
DTYPE = torch.float64
STAGES = {"model.pt": "Final", "model_shape_only.pt": "Shape only",
    "checkpoint_pretrain.pt": "Pretrain snapshot", "checkpoint_shape.pt": "Shape snapshot",
    "checkpoint_pose.pt": "Pose snapshot", "model_hard_transfer.pt": "Hard transfer",
    "model_pretrained.pt": "Pretrained", "model_stage1.pt": "Stage 1", "model_transfer.pt": "Hard transfer"}


@lru_cache(maxsize=100)
def load(path):
    # These are the repository's local, trusted training checkpoints (some cache
    # LocomotionTask objects). The HTTP API never accepts a file path or upload.
    return torch.load(path, map_location="cpu", weights_only=False)


def as_list(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if not np.isfinite(arr).all():
        raise ValueError("Rollout contains nonfinite values; reduce stiffness or increase integration steps.")
    return arr.tolist()


def transform(points, pose):
    return apply_pose(torch.as_tensor(points, dtype=DTYPE), pose)


def number(data, name, default, lo, hi, integer=False):
    raw = data.get(name, default)
    try:
        val = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number") from None
    if not math.isfinite(val) or not lo <= val <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")
    if integer and val != int(val):
        raise ValueError(f"{name} must be an integer")
    return int(val) if integer else val


def options(data, defaults):
    return {
        "steps": number(data, "steps", defaults["steps"], 12, 240, True),
        "horizon": number(data, "horizon", 2, 1, 6),
        "damping": number(data, "damping", 1, .1, 5),
        "stiffness": number(data, "stiffness", 1, .2, 3),
        "target_dx": number(data, "target_dx", 0, -2, 2),
        "target_dy": number(data, "target_dy", 0, -2, 2),
        "target_angle": number(data, "target_angle", 0, -180, 180),
        "initial_angle": number(data, "initial_angle", 0, -180, 180),
        "noise": number(data, "noise", 0, 0, .1),
        "seed": number(data, "seed", 2026, 0, 2**31 - 1, True),
        "tolerance": number(data, "tolerance", defaults["tolerance"], .001, .3),
        "fim": bool(data.get("fim", defaults["fim"])),
    }


class CheckpointEngine:
    def __init__(self, root=VENDOR / "outputs"):
        self.root = Path(root)
        self.entries = {}
        for path in sorted(self.root.rglob("*.pt")):
            if path.name not in STAGES:
                continue
            relative = path.relative_to(self.root)
            family = "spline" if relative.parts[0].startswith("spline_") else "multilink"
            base = path.parent.parent if path.parent.name == "hard_ood" else path.parent
            # Resume snapshots and transfer files inherit missing architecture
            # metadata from their own run's final checkpoint.
            metadata = load(base / "model.pt")
            if family == "spline":
                sig = metadata["signature"]
                mode = sig["mode"]
                tol = metadata["config"]["closure_tolerance"]
                conditioning = sig["conditioning_case"]
                steps = metadata["config"]["n_steps"]
                fim = False
            else:
                mode = metadata["closure_model"]["mode"]
                tol = metadata["closure_model"]["tolerance_rel"]
                conditioning = "configuration"
                steps = metadata["args"]["n_steps"]
                fim = metadata.get("use_fim", True)
            mode_label = {"measurement": "FIM", "hard": "Holonomic", "periodic": "Periodic", "open": "Open"}[mode]
            stage = STAGES[path.name]
            label = f"{mode_label}{f' · ε={tol:g}' if mode == 'measurement' else ''} · {conditioning.title()} · {stage}"
            key = relative.as_posix()
            self.entries[key] = {"id": key, "label": label, "family": family,
                "variant": base.name, "mode": mode, "mode_label": mode_label,
                "stage": stage, "conditioning": conditioning, "path": path, "base": base,
                "defaults": {"steps": steps, "tolerance": tol, "fim": fim}}
        if not self.entries:
            raise RuntimeError(f"No compatible checkpoints under {self.root}")

    @lru_cache(maxsize=20)
    def groups(self, checkpoint_id):
        e = self.entry(checkpoint_id)
        if e["family"] == "spline":
            raw = load(e["base"] / "task_cache.pt")
            return {k: v for k, v in raw.items() if isinstance(v, list)}
        raw = load(e["base"] / "task_teacher_cache.pt")
        groups = dict(raw.get("converted_groups") or {"test": raw["tasks"]})
        hard = load(e["base"] / "hard_ood" / "hard_task_cache.pt")
        groups.update(hard_adapt=hard["adapt"], hard_test=hard["test"])
        return groups

    def entry(self, checkpoint_id):
        if checkpoint_id not in self.entries:
            raise ValueError("Unknown checkpoint")
        return self.entries[checkpoint_id]

    def catalog(self):
        out = []
        for e in self.entries.values():
            item = {k: v for k, v in e.items() if k not in {"base", "path", "mode", "variant"}}
            item["tasks"] = {k: len(v) for k, v in self.groups(e["id"]).items()}
            out.append(item)
        return {"schema_version": 1, "checkpoints": out, "count": len(out)}

    def run(self, data):
        e = self.entry(data.get("checkpoint"))
        opt = options(data, e["defaults"])
        group = data.get("group", "hard_test" if e["stage"] == "Hard transfer" else "test")
        if group not in self.groups(e["id"]):
            raise ValueError("Unknown task group")
        index = number(data, "task_index", 0, 0, len(self.groups(e["id"])[group]) - 1, True)
        raw = copy.deepcopy(self.groups(e["id"])[group][index])
        first = self._run(e, copy.deepcopy(raw), opt)
        compare_id = data.get("compare")
        runs = [first]
        if compare_id:
            other = self.entry(compare_id)
            if other["family"] != e["family"]:
                raise ValueError("Overlay comparison requires checkpoints in the same dynamics family")
            runs.append(self._run(other, copy.deepcopy(raw), opt, compared=True))
        return {"schema_version": 1, "family": e["family"], "group": group,
            "task_index": index, "configuration": opt, "runs": runs}

    def _run(self, e, raw, opt, compared=False):
        ck = load(e["path"])
        meta = load(e["base"] / "model.pt")
        state = ck.get("state_dict", ck.get("model_state"))
        if state is None:
            raise ValueError("Selected file does not contain model weights")
        # Exactly the stored snapshot weights; optimizer state/best_state are
        # deliberately not substituted for model_state.
        delta = torch.tensor([opt["target_dx"], opt["target_dy"], math.radians(opt["target_angle"])], dtype=DTYPE)
        initial_delta = torch.tensor([0., 0., math.radians(opt["initial_angle"])], dtype=DTYPE)
        with torch.no_grad():
            if e["family"] == "spline":
                result = self._spline(e, raw, state, meta, opt, delta, initial_delta, compared)
            else:
                result = self._multilink(e, raw, state, meta, opt, delta, initial_delta, compared)
        result.update(id=e["id"], label=e["label"], mode_label=e["mode_label"],
            checkpoint_epoch=ck.get("epoch"), conditioning=e["conditioning"])
        return result

    def _spline(self, e, raw, state, meta, opt, delta, initial_delta, compared):
        cfg = SplineCurveConfig(**meta["config"])
        cfg.n_steps = opt["steps"]
        # The comparison keeps each checkpoint's trained seam tolerance.
        if not compared:
            cfg.closure_tolerance = opt["tolerance"]
        task = task_from_cpu_dict(raw, torch.device("cpu"), DTYPE)
        task.mode = e["mode"]
        task.task_id = {"open": 0, "measurement": 1, "periodic": 2}[task.mode]
        task.conditioning_case = e["conditioning"]
        old_pose = task.pose_target.clone()
        task.pose_target += delta
        task.pose0 += initial_delta
        if task.observation_samples is not None:
            c, s = torch.cos(old_pose[2]), torch.sin(old_pose[2])
            R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
            obs_body = (task.observation_samples - old_pose[:2]) @ R
            task.observation_samples = transform(obs_body, task.pose_target)
            if opt["noise"]:
                rng = torch.Generator().manual_seed(opt["seed"])
                task.observation_samples += opt["noise"] * torch.randn(task.observation_samples.shape, generator=rng, dtype=DTYPE)
        model = SplineCurvePHTemplate(cfg, hidden=state["net.0.weight"].shape[0], dtype=DTYPE)
        model.load_state_dict(state)
        for key, param in model.named_parameters():
            if key.startswith("log_d"):
                param.add_(math.log(opt["damping"]))
            elif key.startswith("log_k"):
                param.add_(math.log(opt["stiffness"]))
        model.eval()
        r = rollout_spline_ph(model, task, cfg, horizon_multiplier=opt["horizon"])
        body = torch.stack([evaluate_curve(P, eta, cfg)[0] for P, eta in zip(r["P"], r["eta"])])
        world = torch.stack([transform(C, pose) for C, pose in zip(body, r["pose"])])
        target_body = evaluate_curve(task.P_target, task.eta_target, cfg)[0]
        target_world = transform(target_body, task.pose_target)
        x = torch.cat([r["P"].flatten(1), r["eta"], r["pose"]], dim=1)
        p = torch.cat([r["pP"].flatten(1), r["pE"], r["pPose"]], dim=1)
        labels = [f"P{i+1}.{axis}" for i in range(cfg.n_ctrl) for axis in ("x", "y")] + [f"knot {i+1}" for i in range(cfg.n_spans)] + ["pose x", "pose y", "pose θ"]
        shape_error = torch.sqrt(torch.mean((body - target_body) ** 2, dim=(1, 2)))
        mass = torch.tensor([cfg.mass_control] * (2*cfg.n_ctrl) + [cfg.mass_knot]*cfg.n_spans + [cfg.mass_pose_xy]*2 + [cfg.mass_pose_theta], dtype=DTYPE)
        speed = torch.linalg.norm(p / mass, dim=1)
        metrics = {"hamiltonian": r["energy"][:, 5], "kinetic": r["energy"][:, 0],
            "potential": r["energy"][:, 1:5].sum(1), "dissipation": r["dissipation_power"],
            "shape_error": shape_error[1:], "closure_gap": r["gap"], "speed": speed[1:],
            "sinkhorn": r["sinkhorn"], "pose_error": torch.linalg.norm(r["pose"][1:, :2] - task.pose_target[:2], dim=1)}
        ref_body = torch.stack([evaluate_curve(P, eta, cfg)[0] for P, eta in zip(r["P_ref"], r["eta_ref"])])
        ref_world = torch.stack([transform(C, pose) for C, pose in zip(ref_body, r["pose_ref"])])
        return self._serialize(body, world, r["pose"], x, p, labels, target_body, target_world,
            task.pose_target, metrics, 1/cfg.n_steps, ref_body, ref_world,
            task.observation_samples, False, tolerance=cfg.closure_tolerance,
            notes="Spline curve and control/knot momenta; energy samples are at step endpoints. FIM names the saved measurement-closure variant.")

    def _multilink(self, e, task, state, meta, opt, delta, initial_delta, compared):
        chain = PlanarMultiLinkChain(ChainParams(**meta["chain_params"]), dtype=DTYPE)
        model = WorldPoseInformationPHTemplate(dtype=DTYPE, **meta["model_config"])
        model.load_state_dict(state)
        model.log_extra_damping.add_(math.log(opt["damping"]))
        model.log_stiffness.add_(math.log(opt["stiffness"]))
        model.eval()
        task.pose0 = task.initial_pose.clone() + initial_delta
        task.pose_target += delta
        closure = LoopClosureModel(**meta["closure_model"])
        if not compared:
            closure = replace(closure, tolerance_rel=opt["tolerance"])
        set_task_closure_model(task, closure)
        use_fim = meta.get("use_fim", True) if compared else opt["fim"]
        include_pose = e["stage"] in {"Final", "Pose snapshot"}
        r = rollout_world_pose_info_ph(model, stack_world_pose_features(task, include_pose_target=include_pose),
            chain, meta["Ksp"].to(dtype=DTYPE), max(opt["steps"], round(opt["steps"]*opt["horizon"])),
            1/opt["steps"], InformationGeometryConfig(), compute_sf_diagnostics=False,
            external_force_cfg=ExternalForceConfig(case="rigid_loop"), task=task,
            control_steps=opt["steps"], use_fim_damping=use_fim, compute_terminal_sf=False)
        body = chain.kinematics(r["q"])["joint_positions"]
        world = torch.stack([transform(C, pose) for C, pose in zip(body, r["pose"])])
        q_target = torch.as_tensor(task.metadata["rigid_loop_q_target"], dtype=DTYPE)
        target_body = chain.kinematics(q_target)["joint_positions"]
        target_world = transform(target_body, task.pose_target)
        x = torch.cat([r["q"], r["pose"]], dim=1)
        p = torch.cat([r["pi"][:, 3:], r["pi"][:, :3]], dim=1)
        labels = [f"joint {i+1}" for i in range(chain.n_joints)] + ["pose x / body Px", "pose y / body Py", "pose θ / body Lz"]
        metrics = {"hamiltonian": r["hamiltonian"][1:], "kinetic": r["kinetic_energy"][1:],
            "potential": r["potential_energy"][1:], "dissipation": r["dissipation_power"],
            "shape_error": torch.sqrt(torch.mean((body[1:] - target_body)**2, dim=(1,2))),
            "closure_gap": r["loop_closure_error"], "speed": torch.linalg.norm(r["nu"], dim=1),
            "pose_error": torch.linalg.norm(r["pose"][1:, :2] - task.pose_target[:2], dim=1),
            "balance_residual": r["energy_balance_residual_cum"][1:],
            "fim_trace": r["fim_eigs"].sum(1)}
        refs = chain.kinematics(r["q_ref"])["joint_positions"]
        ref_world = torch.stack([transform(C, pose) for C, pose in zip(refs, r["pose"][1:])])
        obs_body = task.metadata.get("rigid_loop_shared_observation_body")
        observations = None if obs_body is None else transform(obs_body, task.pose_target)
        return self._serialize(body, world, r["pose"], x, p, labels, target_body, target_world,
            task.pose_target, metrics, 1/opt["steps"], refs, ref_world, observations,
            bool(r["diverged"].item()), tolerance=closure.tolerance_rel, fim_enabled=use_fim,
            notes="Joint phase coordinates use joint momenta; pose channels show world position against body momenta. Global pose follows articulated mechanics.")

    def _serialize(self, body, world, poses, x, p, labels, target_body, target_world,
                   target_pose, metrics, dt, refs, ref_world, obs, diverged, **extra):
        count = len(body)
        times = np.arange(count)*dt
        result = {"times": as_list(times), "metric_times": as_list(times[1:]),
            "body": as_list(body), "world": as_list(world), "poses": as_list(poses),
            "phase_q": as_list(x), "phase_p": as_list(p), "coordinates": labels,
            "target_body": as_list(target_body), "target_world": as_list(target_world),
            "target_pose": as_list(target_pose), "reference_body": as_list(refs),
            "reference_world": as_list(ref_world), "observations_world": None if obs is None else as_list(obs),
            "metrics": {k: as_list(v) for k, v in metrics.items()}, "diverged": diverged, **extra}
        bad = {k: len(v) for k, v in result["metrics"].items() if len(v) != count-1}
        if bad:
            raise ValueError(f"Metric frame alignment failed: expected {count-1}, got {bad}")
        return result


if __name__ == "__main__":
    engine = CheckpointEngine()
    print(json.dumps(engine.catalog(), indent=2))
