from __future__ import annotations

from pathlib import Path
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from .chain import PlanarMultiLinkChain
from .geometry import se2_endpoint_error
from .spline_curve_ph import apply_pose, point_cloud_chamfer
from .shared_crescent import sample_chain_polyline, shared_observation_metrics


def _arr(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _world_vertices(chain: PlanarMultiLinkChain, q: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    return apply_pose(chain.kinematics(q)["joint_positions"], pose)


def plot_shared_training_gallery(
    groups: dict[str, list[Any]],
    teachers: dict[str, list[dict[str, torch.Tensor]]],
    chain: PlanarMultiLinkChain,
    path: str | Path,
    *,
    n_per_group: int = 3,
    title: str = "Multi-link: shared spline circle-to-crescent targets",
) -> dict[str, Any]:
    """Spline-v34-style gallery for the chain representation and teacher path."""
    rows = [(name, tasks[:n_per_group]) for name, tasks in groups.items() if tasks]
    if not rows:
        return {}
    ncols = max(len(ts) for _, ts in rows)
    fig, axes = plt.subplots(len(rows), ncols, figsize=(4.4 * ncols, 4.3 * len(rows)), squeeze=False)
    reports: dict[str, list[dict[str, float]]] = {}
    for ir, (name, tasks) in enumerate(rows):
        reports[name] = []
        group_teachers = teachers.get(name, [])
        for ic in range(ncols):
            ax = axes[ir, ic]
            if ic >= len(tasks):
                ax.axis("off")
                continue
            task = tasks[ic]
            md = task.metadata or {}
            desired = np.asarray(md["rigid_loop_shared_target_curve_body"], dtype=float)
            obs = np.asarray(md["rigid_loop_shared_observation_body"], dtype=float)
            oracle = np.asarray(md["rigid_loop_target_rigid_vertices_body"], dtype=float)
            initial = np.asarray(md["rigid_loop_initial_vertices_body"], dtype=float)
            ax.scatter(obs[:, 0], obs[:, 1], s=6, alpha=.22, label="shared obs cloud")
            ax.plot(desired[:, 0], desired[:, 1], linestyle="--", linewidth=1.7, label="shared ordered target")
            ax.plot(initial[:, 0], initial[:, 1], linewidth=1.2, label="initial chain")
            ax.plot(oracle[:, 0], oracle[:, 1], marker="o", linewidth=1.6, label="best fixed-link fit")
            teacher_selfx = np.nan
            if ic < len(group_teachers):
                tr = group_teachers[ic]
                mid = min(max(tr["q"].shape[0] // 2, 0), tr["q"].shape[0] - 1)
                vm = _arr(chain.kinematics(tr["q"][mid])["joint_positions"])
                ax.plot(vm[:, 0], vm[:, 1], linestyle=":", linewidth=1.4, label="teacher midpoint")
                rep = md.get("rigid_loop_shared_teacher_validation") or {}
                teacher_selfx = float(rep.get("max_self_intersections", np.nan))
            ax.axis("equal")
            ax.grid(True, alpha=.25)
            alpha = float(md.get("rigid_loop_shared_task_alpha", np.nan))
            floor = float(md.get("rigid_loop_representation_error", np.nan))
            scale = float(md.get("rigid_loop_shared_scale_factor", np.nan))
            ax.set_title(f"{name} #{ic}\nalpha={alpha:.2f}, floor={floor:.3f}, scale={scale:.3f}, selfX={teacher_selfx:g}")
            if ir == 0 and ic == 0:
                ax.legend(fontsize=7, loc="best")
            reports[name].append({"alpha": alpha, "representation_floor": floor, "scale_factor": scale, "teacher_max_self_intersections": teacher_selfx})
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, .96])
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return reports


def plot_shared_rollout_diagnostics(
    task: Any,
    chain: PlanarMultiLinkChain,
    rollout: dict[str, torch.Tensor],
    path: str | Path,
    *,
    dt: float,
    title_suffix: str = "",
) -> dict[str, float]:
    """Observation-first diagnostics analogous to the spline-only panel."""
    md = task.metadata or {}
    q = rollout["q"]
    pose = rollout["pose"]
    T = q.shape[0]
    t = np.arange(T) * float(dt)
    desired_body = torch.as_tensor(md["rigid_loop_shared_target_curve_body"], dtype=q.dtype, device=q.device)
    obs_body = torch.as_tensor(md["rigid_loop_shared_observation_body"], dtype=q.dtype, device=q.device)
    desired_world = torch.as_tensor(md["rigid_loop_shared_target_curve_world"], dtype=q.dtype, device=q.device)
    obs_world = torch.as_tensor(md["rigid_loop_shared_observation_world"], dtype=q.dtype, device=q.device)
    oracle = torch.as_tensor(md["rigid_loop_target_rigid_vertices_body"], dtype=q.dtype, device=q.device)
    dataset_pose = torch.as_tensor(md["rigid_loop_target_pose_dataset"], dtype=q.dtype, device=q.device)

    chamfer_body = []
    chamfer_world_dataset = []
    joint_rms = []
    pose_err = []
    for k in range(T):
        verts = chain.kinematics(q[k])["joint_positions"]
        dense = sample_chain_polyline(verts, min(128, max(48, obs_body.shape[0])))
        chamfer_body.append(float(point_cloud_chamfer(dense, obs_body).detach()))
        dense_world = apply_pose(dense, pose[k])
        chamfer_world_dataset.append(float(point_cloud_chamfer(dense_world, obs_world).detach()))
        qT = torch.as_tensor(md["rigid_loop_q_target"], dtype=q.dtype, device=q.device)
        joint_rms.append(float(torch.sqrt(torch.mean((q[k] - qT) ** 2) + 1e-16).detach()))
        pose_err.append(float(torch.linalg.norm(se2_endpoint_error(pose[k], dataset_pose)).detach()))

    final_body = chain.kinematics(q[-1])["joint_positions"]
    final_world = apply_pose(final_body, pose[-1])
    oracle_world = apply_pose(oracle, dataset_pose)
    closure = _arr(rollout.get("loop_closure_error", torch.zeros(max(T - 1, 1), dtype=q.dtype, device=q.device)))
    speed = _arr(torch.linalg.norm(rollout["nu"], dim=-1))
    H = _arr(rollout["hamiltonian"])
    diss = _arr(rollout["dissipation_power"])

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    ax = axes[0, 0]
    ax.scatter(_arr(obs_body)[:, 0], _arr(obs_body)[:, 1], s=7, alpha=.24, label="shared obs")
    ax.plot(_arr(desired_body)[:, 0], _arr(desired_body)[:, 1], linestyle="--", label="ordered target")
    ax.plot(_arr(oracle)[:, 0], _arr(oracle)[:, 1], marker="o", linestyle=":", label="best fixed-link fit")
    ax.plot(_arr(final_body)[:, 0], _arr(final_body)[:, 1], marker="o", label="pH final")
    ax.axis("equal"); ax.grid(True, alpha=.25); ax.set_title("Intrinsic target / representation floor"); ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.scatter(_arr(obs_world)[:, 0], _arr(obs_world)[:, 1], s=7, alpha=.24, label="dataset observation")
    ax.plot(_arr(desired_world)[:, 0], _arr(desired_world)[:, 1], linestyle="--", label="dataset ordered target")
    ax.plot(_arr(oracle_world)[:, 0], _arr(oracle_world)[:, 1], linestyle=":", marker="o", label="oracle @ dataset pose")
    ax.plot(_arr(final_world)[:, 0], _arr(final_world)[:, 1], marker="o", label="pH final world")
    ax.axis("equal"); ax.grid(True, alpha=.25); ax.set_title("Shared observation-space comparison"); ax.legend(fontsize=7)

    ax = axes[0, 2]
    ax.plot(t, chamfer_body, label="intrinsic Chamfer")
    ax.plot(t, chamfer_world_dataset, label="world Chamfer to dataset pose")
    ax.axhline(float(md.get("rigid_loop_representation_error", np.nan)), linestyle=":", label="link measurement floor")
    ax.set_xlabel("time"); ax.set_ylabel("error"); ax.grid(True, alpha=.25); ax.set_title("Observation convergence"); ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.plot(t, joint_rms)
    ax.set_xlabel("time"); ax.set_ylabel("joint RMS"); ax.grid(True, alpha=.25); ax.set_title("Intrinsic coordinate error")

    ax = axes[1, 1]
    tc = t[1:1 + len(closure)] if len(closure) <= T - 1 else np.arange(len(closure)) * dt
    ax.plot(tc, closure)
    ax.set_xlabel("time"); ax.set_ylabel("seam gap"); ax.grid(True, alpha=.25); ax.set_title("Closure measurement")

    ax = axes[1, 2]
    ax.plot(t[:len(speed)], speed)
    ax.set_xlabel("time"); ax.set_ylabel("generalized speed"); ax.grid(True, alpha=.25); ax.set_title("Settling")

    ax = axes[2, 0]
    ax.plot(t[:len(H)], H, label="Hamiltonian")
    ax.set_xlabel("time"); ax.grid(True, alpha=.25); ax.set_title("Hamiltonian"); ax.legend(fontsize=7)

    ax = axes[2, 1]
    td = np.arange(len(diss)) * dt
    ax.plot(td, diss)
    ax.set_xlabel("time"); ax.set_ylabel("dissipation power"); ax.grid(True, alpha=.25); ax.set_title("Dissipation")

    ax = axes[2, 2]
    ax.plot(t, pose_err, label="error to shared dataset pose")
    ax.set_xlabel("time"); ax.set_ylabel("SE(2) log error"); ax.grid(True, alpha=.25); ax.set_title("Pose diagnostic")
    ax.legend(fontsize=7)

    mode = md.get("rigid_loop_closure_mode", "")
    alpha = float(md.get("rigid_loop_shared_task_alpha", np.nan))
    fig.suptitle(f"Multi-link shared circle→crescent — {mode}, alpha={alpha:.3f} {title_suffix}".strip())
    fig.tight_layout(rect=[0, 0, 1, .965])
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)

    terminal = shared_observation_metrics(task, chain, q[-1])
    return {
        "final_intrinsic_chamfer": float(chamfer_body[-1]),
        "final_world_chamfer_to_dataset_pose": float(chamfer_world_dataset[-1]),
        "final_sinkhorn": float(terminal["sinkhorn"].detach()),
        "final_dataset_pose_error": float(pose_err[-1]),
    }
