from __future__ import annotations

"""Shared spline-observation -> articulated-chain circle/crescent adapter.

The spline-only v34 experiment and the articulated ablation should differ in the
mechanism, not in which crescent was sampled.  This module converts an existing
``SplineCurveTask`` into a rigid-link task while retaining the exact target id,
alpha, empirical observation cloud and ordered fitted target curve.

A fixed-edge chain cannot in general reproduce the spline target perimeter.  The
adapter therefore exposes two explicit scale conventions:

``perimeter_normalized``
    Uniformly scale each body-frame target to the chain perimeter.  This isolates
    shape/closure representability from inextensibility-induced scale mismatch.

``literal``
    Keep the spline target at its original scale.  The resulting representation
    floor then includes the fixed-perimeter mismatch and is reported as such.
"""

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable

import numpy as np
import torch

from .chain import PlanarMultiLinkChain
from .rigid_loop import (
    LoopClosureModel,
    RigidLoopConfig,
    make_rigid_loop_task_from_desired_vertices,
    regular_polygon_joint_angles,
)
from .spline_curve_ph import (
    SplineCurveConfig,
    SplineCurveTask,
    make_feasible_tasks,
    make_hard_tasks,
    task_to_cpu_dict,
    task_from_cpu_dict,
    apply_pose,
    inverse_pose,
    ordered_circle_boundary,
    point_cloud_chamfer,
    polyline_self_intersections,
    resample_curve_samples,
    sinkhorn_divergence_points,
)


@dataclass(frozen=True)
class SharedCrescentAdapterConfig:
    scale_mode: str = "perimeter_normalized"  # perimeter_normalized | literal
    source_radius: float = 0.6
    observation_samples_for_metric: int = 128
    sinkhorn_eps: float = 0.02
    sinkhorn_n_iter: int = 18

    def normalized_scale_mode(self) -> str:
        mode = str(self.scale_mode).strip().lower()
        aliases = {"normalized": "perimeter_normalized", "perimeter": "perimeter_normalized", "raw": "literal"}
        mode = aliases.get(mode, mode)
        if mode not in {"perimeter_normalized", "literal"}:
            raise ValueError(f"unknown shared-dataset scale mode: {self.scale_mode}")
        return mode


def _closed_curve(curve: torch.Tensor) -> torch.Tensor:
    if curve.ndim != 2 or curve.shape[-1] != 2:
        raise ValueError(f"expected (N,2) curve, got {tuple(curve.shape)}")
    if curve.shape[0] < 3:
        raise ValueError("at least three curve samples are required")
    if float(torch.linalg.norm(curve[-1] - curve[0]).detach()) <= 1e-8:
        out = curve.clone()
        out[-1] = out[0]
        return out
    return torch.cat([curve, curve[:1]], dim=0)


def _curve_perimeter(curve: torch.Tensor) -> torch.Tensor:
    c = _closed_curve(curve)
    return torch.linalg.norm(c[1:] - c[:-1], dim=-1).sum()


def _rotate(points: torch.Tensor, theta: torch.Tensor | float) -> torch.Tensor:
    th = torch.as_tensor(theta, dtype=points.dtype, device=points.device)
    c, s = torch.cos(th), torch.sin(th)
    x = c * points[..., 0] - s * points[..., 1]
    y = s * points[..., 0] + c * points[..., 1]
    return torch.stack([x, y], dim=-1)


def _canonicalize_for_chain(curve_body: torch.Tensor, chain: PlanarMultiLinkChain, scale_mode: str) -> dict[str, torch.Tensor | float]:
    """Put the shared ordered material curve into the chain's body convention.

    The v34 spline target starts at the leftmost material seam and has a fixed
    orientation.  We keep that seam/order, resample to M equal material edges,
    rotate the first edge onto +x, and translate its first vertex to the chain
    origin.  No cyclic reindexing is allowed.
    """
    curve = _closed_curve(curve_body)
    perimeter = _curve_perimeter(curve)
    if float(perimeter) <= 1e-12:
        raise ValueError("degenerate target curve perimeter")
    if scale_mode == "perimeter_normalized":
        scale = float(chain.params.total_length) / float(perimeter.detach())
    elif scale_mode == "literal":
        scale = 1.0
    else:
        raise ValueError(scale_mode)
    scaled = curve * scale
    vertices = resample_curve_samples(scaled, chain.n_links + 1)
    vertices = vertices.clone()
    vertices[-1] = vertices[0]
    edge0 = vertices[1] - vertices[0]
    phi = torch.atan2(edge0[1], edge0[0])
    origin = vertices[0].clone()
    dense_chain = _rotate(scaled - origin, -phi)
    vertices_chain = _rotate(vertices - origin, -phi)
    vertices_chain[0] = 0.0
    vertices_chain[-1] = 0.0
    return {
        "dense_chain": dense_chain,
        "vertices_chain": vertices_chain,
        "scale": float(scale),
        "phi": phi,
        "origin_scaled": origin,
        "original_perimeter": float(perimeter.detach()),
        "scaled_perimeter": float(_curve_perimeter(scaled).detach()),
    }


def _transform_observation_to_chain_body(
    observation_world: torch.Tensor,
    spline_pose_target: torch.Tensor,
    *,
    scale: float,
    phi: torch.Tensor,
    origin_scaled: torch.Tensor,
) -> torch.Tensor:
    obs_body = inverse_pose(observation_world, spline_pose_target)
    obs_scaled = obs_body * float(scale)
    return _rotate(obs_scaled - origin_scaled, -phi)


def _chain_pose_for_normalized_dataset(
    spline_pose_target: torch.Tensor,
    *,
    phi: torch.Tensor,
    origin_scaled: torch.Tensor,
) -> torch.Tensor:
    """Pose mapping canonical chain body coordinates back to the normalized data frame."""
    theta_d = spline_pose_target[2]
    theta_chain = theta_d + phi
    translated_origin = _rotate(origin_scaled[None], theta_d)[0]
    xy = spline_pose_target[:2] + translated_origin
    return torch.stack([xy[0], xy[1], theta_chain])


def _ordered_source_alignment_pose(chain: PlanarMultiLinkChain, source_radius: float) -> torch.Tensor:
    """Best rigid alignment of the regular chain polygon to the v34 ordered source circle."""
    q0 = regular_polygon_joint_angles(chain)
    pred = chain.kinematics(q0)["joint_positions"][:-1]
    cfg = SplineCurveConfig(n_ctrl=max(8, chain.n_links + 3), n_samples=max(128, 8 * chain.n_links), source_radius=float(source_radius))
    src = ordered_circle_boundary(cfg, dtype=chain.dtype, device=chain.device, n_points=chain.n_links + 1)[:-1]
    # Match the fixed chain perimeter.  This also makes the initial observation
    # scale coherent with perimeter_normalized targets.
    src_perim = 2.0 * math.pi * float(source_radius)
    if src_perim > 0:
        src = src * (float(chain.params.total_length) / src_perim)
    cp = pred.mean(dim=0)
    ct = src.mean(dim=0)
    X, Y = pred - cp, src - ct
    # 2-D Kabsch rotation, constrained to SO(2).
    a = torch.sum(X[:, 0] * Y[:, 0] + X[:, 1] * Y[:, 1])
    b = torch.sum(X[:, 0] * Y[:, 1] - X[:, 1] * Y[:, 0])
    theta = torch.atan2(b, a)
    t = ct - _rotate(cp[None], theta)[0]
    return torch.stack([t[0], t[1], theta])


def _tensor_list(x: torch.Tensor) -> list:
    return x.detach().cpu().tolist()


def spline_task_to_rigid_task(
    spline_task: SplineCurveTask,
    chain: PlanarMultiLinkChain,
    Ksp: torch.Tensor,
    n_temporal_ctrl: int,
    *,
    seed: int,
    closure: LoopClosureModel,
    loop_cfg: RigidLoopConfig,
    adapter_cfg: SharedCrescentAdapterConfig | None = None,
    teacher_allowed: bool | None = None,
) -> Any:
    """Convert one exact spline-v34 task to the corresponding fixed-link task."""
    adapter_cfg = adapter_cfg or SharedCrescentAdapterConfig()
    scale_mode = adapter_cfg.normalized_scale_mode()
    geom = _canonicalize_for_chain(spline_task.target_samples, chain, scale_mode)
    desired = geom["vertices_chain"].detach().cpu().numpy()
    rigid = make_rigid_loop_task_from_desired_vertices(
        chain,
        Ksp,
        n_temporal_ctrl,
        desired,
        seed=seed,
        closure=closure,
        cfg=loop_cfg,
        target_family=("shared_spline_hard_ood" if spline_task.hard_ood else "shared_spline_feasible"),
        pose0=_ordered_source_alignment_pose(chain, adapter_cfg.source_radius),
    )
    md = rigid.metadata or {}
    obs_world = spline_task.observation_samples
    if obs_world is None:
        # Configuration-conditioned caches still carry a target curve.  Use the
        # ordered curve as a deterministic observation fallback rather than
        # inventing a different geometry.
        obs_world = apply_pose(spline_task.target_samples, spline_task.pose_target)
    obs_chain = _transform_observation_to_chain_body(
        obs_world,
        spline_task.pose_target,
        scale=float(geom["scale"]),
        phi=geom["phi"],
        origin_scaled=geom["origin_scaled"],
    )
    dataset_pose_chain = _chain_pose_for_normalized_dataset(
        spline_task.pose_target,
        phi=geom["phi"],
        origin_scaled=geom["origin_scaled"],
    )
    obs_chain_world = apply_pose(obs_chain, dataset_pose_chain)
    dense_chain = geom["dense_chain"]
    dense_chain_world = apply_pose(dense_chain, dataset_pose_chain)

    shared_hash = str(spline_task.target_hash)
    if not shared_hash:
        shared_hash = hashlib.sha1(obs_world.detach().cpu().numpy().astype(np.float64).tobytes()).hexdigest()
    md.update({
        "rigid_loop_is_hard_ood": bool(spline_task.hard_ood),
        "rigid_loop_teacher_allowed": bool((not spline_task.hard_ood) if teacher_allowed is None else teacher_allowed),
        "rigid_loop_target_family": str(spline_task.target_family),
        "rigid_loop_reference_curve_hash": shared_hash,
        "rigid_loop_shared_dataset_hash": shared_hash,
        "rigid_loop_hard_geometry_signature": shared_hash,
        "rigid_loop_shared_task_alpha": float(spline_task.target_alpha),
        "rigid_loop_shared_target_params": dict(spline_task.target_params or {}),
        "rigid_loop_shared_scale_mode": scale_mode,
        "rigid_loop_shared_scale_factor": float(geom["scale"]),
        "rigid_loop_shared_original_perimeter": float(geom["original_perimeter"]),
        "rigid_loop_shared_chain_perimeter": float(chain.params.total_length),
        "rigid_loop_shared_target_curve_body": _tensor_list(dense_chain),
        "rigid_loop_shared_observation_body": _tensor_list(obs_chain),
        "rigid_loop_shared_observation_world": _tensor_list(obs_chain_world),
        "rigid_loop_shared_target_curve_world": _tensor_list(dense_chain_world),
        "rigid_loop_shared_original_observation_world": _tensor_list(obs_world),
        "rigid_loop_shared_spline_pose_target": _tensor_list(spline_task.pose_target),
        "rigid_loop_shared_dataset_pose_target": _tensor_list(dataset_pose_chain),
        "rigid_loop_target_dense_crescent_body": _tensor_list(dense_chain),
        "rigid_loop_target_sampled_crescent_body": desired.tolist(),
        "rigid_loop_target_pose_dataset": _tensor_list(dataset_pose_chain),
        "rigid_loop_target_measurement_protocol": (
            "same v34 spline target id/alpha/observation; ordered material curve -> M equal edges; "
            f"scale_mode={scale_mode}; all-link midpoint+tangent loss; seam handled separately"
        ),
    })
    rigid.metadata = md
    # Preserve the dataset pose as the explicit observation-space target.  The
    # training script may replace ``pose_target`` with a dynamically reachable
    # teacher endpoint, but the dataset pose remains in metadata for evaluation.
    rigid.pose_target = dataset_pose_chain.detach().clone()
    return rigid


def teacher_path_geometry_report(
    task: Any,
    chain: PlanarMultiLinkChain,
    q_path: torch.Tensor,
) -> dict[str, float | int | bool]:
    """Check material-order consistency of an articulated teacher trajectory."""
    max_selfx = 0
    signed_areas: list[float] = []
    for q in q_path:
        v = chain.kinematics(q)["joint_positions"]
        closed = float(torch.linalg.norm(v[-1] - v[0]).detach()) < 1e-6
        max_selfx = max(max_selfx, int(polyline_self_intersections(v, closed=closed)))
        p = v[:-1] if closed else v
        if p.shape[0] >= 3:
            x, y = p[:, 0], p[:, 1]
            area = 0.5 * torch.sum(x * torch.roll(y, -1) - y * torch.roll(x, -1))
            signed_areas.append(float(area.detach()))
    abs_areas = [abs(a) for a in signed_areas]
    nonzero = [a for a in signed_areas if abs(a) > 1e-10]
    orientation_flip = False
    if nonzero:
        s0 = math.copysign(1.0, nonzero[0])
        orientation_flip = any(math.copysign(1.0, a) != s0 for a in nonzero[1:])
    return {
        "valid": bool(max_selfx == 0 and not orientation_flip),
        "max_self_intersections": int(max_selfx),
        "orientation_flip": bool(orientation_flip),
        "min_abs_oriented_area": float(min(abs_areas) if abs_areas else 0.0),
    }


def sample_chain_polyline(vertices: torch.Tensor, n_points: int) -> torch.Tensor:
    """Equal-arclength samples of an M-link polyline for cloud/OT metrics."""
    return resample_curve_samples(vertices, max(2, int(n_points)))


def shared_observation_metrics(
    task: Any,
    chain: PlanarMultiLinkChain,
    q: torch.Tensor,
    *,
    cfg: SharedCrescentAdapterConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Observation-space metrics against the exact shared spline target cloud."""
    cfg = cfg or SharedCrescentAdapterConfig()
    md = task.metadata or {}
    obs = torch.as_tensor(md["rigid_loop_shared_observation_body"], dtype=q.dtype, device=q.device)
    verts = chain.kinematics(q)["joint_positions"]
    pred = sample_chain_polyline(verts, cfg.observation_samples_for_metric)
    chamfer = point_cloud_chamfer(pred, obs)
    sink = sinkhorn_divergence_points(pred, obs, eps=float(cfg.sinkhorn_eps), n_iter=int(cfg.sinkhorn_n_iter))
    return {"chamfer": chamfer, "sinkhorn": sink}


def shared_dataset_group_hash(tasks: Iterable[SplineCurveTask]) -> str:
    ids = "|".join(str(t.target_hash) for t in tasks)
    return hashlib.sha1(ids.encode("utf8")).hexdigest()



def load_shared_spline_cache(path: str | Any, device: torch.device, dtype: torch.dtype) -> tuple[dict[str, list[SplineCurveTask]], dict[str, Any]]:
    """Load a ``run_spline_curve_deformation_ablation.py`` task cache verbatim."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    groups: dict[str, list[SplineCurveTask]] = {}
    for key, vals in payload.items():
        if key == "signature":
            continue
        if isinstance(vals, list):
            groups[key] = [task_from_cpu_dict(v, device, dtype) if isinstance(v, dict) else v.to(device, dtype) for v in vals]
    required = {"train_stage1", "val_stage1", "train_stage2", "val_stage2", "test"}
    missing = sorted(required.difference(groups))
    if missing:
        raise RuntimeError(f"shared spline task cache is missing groups: {missing}")
    return groups, payload.get("signature", {})


def build_shared_spline_cache(
    path: str | Any,
    *,
    seed: int = 2026,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    n_ctrl: int = 18,
    degree: int = 3,
    n_samples: int = 200,
    n_steps: int = 30,
    observation_points: int = 600,
    source_radius: float = 0.6,
    train_tasks: int = 18,
    val_tasks: int = 4,
    test_tasks: int = 5,
    hard_adapt_tasks: int = 4,
    hard_test_tasks: int = 5,
    feasible_fit_iters: int = 220,
    hard_fit_iters: int = 360,
    feasible_alpha_min: float = 0.0,
    feasible_alpha_mid: float = 0.38,
    feasible_alpha_max: float = 0.68,
    hard_alpha_min: float = 0.88,
    hard_alpha_max: float = 1.0,
    feasible_rot_max_deg: float = 25.0,
    feasible_translation: float = 0.22,
    hard_rot_max_deg: float = 55.0,
    hard_translation: float = 0.28,
    target_fit_protocol: str = "ordered_boundary",
) -> tuple[dict[str, list[SplineCurveTask]], dict[str, Any]]:
    """Generate the exact v34 spline curriculum used as the shared multi-link data source.

    The split seeds intentionally match ``build_curriculum_payload`` in the
    spline-only driver.  Therefore a cache generated here and a periodic_obs
    cache generated by that driver have the same alpha/pose/observation tasks
    when all listed dataset knobs agree.
    """
    device = torch.device(device)
    cfg = SplineCurveConfig(
        n_ctrl=n_ctrl,
        degree=degree,
        n_samples=n_samples,
        n_steps=n_steps,
        observation_points=observation_points,
        source_radius=source_radius,
        feasible_alpha_min=feasible_alpha_min,
        feasible_alpha_mid=feasible_alpha_mid,
        feasible_alpha_max=feasible_alpha_max,
        hard_alpha_min=hard_alpha_min,
        hard_alpha_max=hard_alpha_max,
        target_fit_protocol=target_fit_protocol,
    )
    easy = (feasible_alpha_min, feasible_alpha_mid)
    stage2 = (feasible_alpha_mid, feasible_alpha_max)
    hard = (hard_alpha_min, hard_alpha_max)
    groups = {
        "train_stage1": make_feasible_tasks("periodic", train_tasks, cfg, seed=seed + 11, dtype=dtype, device=device, conditioning_case="observation", rot_max_deg=feasible_rot_max_deg, translation=feasible_translation, alpha_range=easy, fit_iterations=feasible_fit_iters),
        "val_stage1": make_feasible_tasks("periodic", val_tasks, cfg, seed=seed + 101, dtype=dtype, device=device, conditioning_case="observation", rot_max_deg=feasible_rot_max_deg, translation=feasible_translation, alpha_range=easy, fit_iterations=feasible_fit_iters),
        "train_stage2": make_feasible_tasks("periodic", train_tasks, cfg, seed=seed + 21, dtype=dtype, device=device, conditioning_case="observation", rot_max_deg=feasible_rot_max_deg, translation=feasible_translation, alpha_range=stage2, fit_iterations=feasible_fit_iters),
        "val_stage2": make_feasible_tasks("periodic", val_tasks, cfg, seed=seed + 121, dtype=dtype, device=device, conditioning_case="observation", rot_max_deg=feasible_rot_max_deg, translation=feasible_translation, alpha_range=stage2, fit_iterations=feasible_fit_iters),
        "test": make_feasible_tasks("periodic", test_tasks, cfg, seed=seed + 201, dtype=dtype, device=device, conditioning_case="observation", rot_max_deg=feasible_rot_max_deg, translation=feasible_translation, alpha_range=(feasible_alpha_min, feasible_alpha_max), fit_iterations=feasible_fit_iters),
        "hard_adapt": make_hard_tasks("periodic", hard_adapt_tasks, cfg, seed=seed + 401, dtype=dtype, device=device, conditioning_case="observation", rot_max_deg=hard_rot_max_deg, translation=hard_translation, alpha_range=hard, fit_iterations=hard_fit_iters),
        "hard_test": make_hard_tasks("periodic", hard_test_tasks, cfg, seed=seed + 801, dtype=dtype, device=device, conditioning_case="observation", rot_max_deg=hard_rot_max_deg, translation=hard_translation, alpha_range=hard, fit_iterations=hard_fit_iters),
    }
    signature = {
        "protocol": "v35_shared_from_spline_v34",
        "seed": seed,
        "n_ctrl": n_ctrl,
        "degree": degree,
        "n_samples": n_samples,
        "n_steps": n_steps,
        "observation_points": observation_points,
        "source_radius": source_radius,
        "train_tasks": train_tasks,
        "val_tasks": val_tasks,
        "test_tasks": test_tasks,
        "hard_adapt_tasks": hard_adapt_tasks,
        "hard_test_tasks": hard_test_tasks,
        "feasible_fit_iters": feasible_fit_iters,
        "hard_fit_iters": hard_fit_iters,
        "feasible_alpha_range": [feasible_alpha_min, feasible_alpha_mid, feasible_alpha_max],
        "hard_alpha_range": [hard_alpha_min, hard_alpha_max],
        "feasible_rot_max_deg": feasible_rot_max_deg,
        "feasible_translation": feasible_translation,
        "hard_rot_max_deg": hard_rot_max_deg,
        "hard_translation": hard_translation,
        "target_fit_protocol": target_fit_protocol,
    }
    serial = {"signature": signature}
    serial.update({k: [task_to_cpu_dict(t) for t in v] for k, v in groups.items()})
    path = str(path)
    from pathlib import Path as _Path
    _p = _Path(path)
    _p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(serial, _p)
    return groups, signature
