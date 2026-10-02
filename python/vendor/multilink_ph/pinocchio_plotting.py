from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .chain import PlanarMultiLinkChain
from .pinocchio_experiments import CASE_LABELS


def _vertices_from_joint_angles(chain: PlanarMultiLinkChain, joints: np.ndarray, pose: np.ndarray) -> np.ndarray:
    import torch
    q = torch.as_tensor(joints, dtype=chain.dtype, device=chain.device)
    p = torch.as_tensor(pose, dtype=chain.dtype, device=chain.device)
    with torch.no_grad():
        body = chain.kinematics(q)["joint_positions"]
        c, s = torch.cos(p[2]), torch.sin(p[2])
        R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
        world = (R @ body.T).T + p[:2]
    return world.detach().cpu().numpy()


def _draw_rollout(ax, chain: PlanarMultiLinkChain, result: dict, target_pose: np.ndarray, title: str) -> None:
    pose = np.asarray(result["pose"])
    joints = np.asarray(result["joints"])
    ids = np.unique(np.linspace(0, len(pose) - 1, 5).round().astype(int))
    com = []
    for k in range(len(pose)):
        verts = _vertices_from_joint_angles(chain, joints[k], pose[k])
        com.append(verts.mean(axis=0))
    com = np.asarray(com)
    ax.plot(com[:, 0], com[:, 1], linewidth=1.7, label="CoM")
    for k in ids:
        verts = _vertices_from_joint_angles(chain, joints[k], pose[k])
        ax.plot(verts[:, 0], verts[:, 1], marker="o", markersize=2, alpha=0.55)
    ax.scatter([target_pose[0]], [target_pose[1]], marker="x", s=50, label="target")
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(True, alpha=0.25)


def plot_backend_trajectory_grid(task, chain: PlanarMultiLinkChain, results: dict[str, dict], path: str | Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    target = task.pose_target.detach().cpu().numpy()
    for ax, case in zip(axes.flat, ["A", "B", "C", "D"]):
        if case in results:
            _draw_rollout(ax, chain, results[case], target, CASE_LABELS[case])
        else:
            ax.text(0.5, 0.5, f"{CASE_LABELS[case]}\nnot available", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
    fig.suptitle(f"{task.kind}: same learned pH template across physical backends")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_contact_diagnostics(results: dict[str, dict], path: str | Path, dt: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for case, r in results.items():
        n = len(r.get("actuator_power", []))
        t = np.arange(n) * dt
        if n:
            axes[0, 0].plot(t, r["actuator_power"], label=case)
        if len(r.get("contact_power", [])):
            axes[0, 1].plot(t, r["contact_power"], label=case)
        if len(r.get("hamiltonian", [])):
            th = np.arange(len(r["hamiltonian"])) * dt
            axes[1, 0].plot(th, r["hamiltonian"], label=case)
        if len(r.get("max_lateral_slip_step", [])):
            axes[1, 1].plot(t, r["max_lateral_slip_step"], label=f"{case} lateral")
        if len(r.get("contact_velocity_norm", [])):
            axes[1, 1].plot(t, r["contact_velocity_norm"], linestyle="--", label=f"{case} rigid Jv")
    axes[0, 0].set_title("Actuator power $u^T v_j$")
    axes[0, 1].set_title("Contact/friction power")
    axes[1, 0].set_title("Mechanical Hamiltonian diagnostic")
    axes[1, 1].set_title("Slip / rigid-contact velocity")
    for ax in axes.flat:
        ax.grid(True, alpha=0.25)
        if ax.lines:
            ax.legend(fontsize=8)
        ax.set_xlabel("time")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_metric_bars(rows: list[dict], path: str | Path) -> None:
    # Matplotlib output for the repository; this intentionally keeps one figure
    # with independent panels because it is a saved experiment report rather than
    # a user-facing python_user_visible chart.
    metrics = ["pose_error", "actuator_energy", "friction_energy", "energy_balance_residual"]
    labels = ["Pose error", "Actuator energy", "Friction energy", "|Energy residual|"]
    cases = [c for c in ["A", "B", "C", "D"] if any(r["case"] == c for r in rows)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, metric, label in zip(axes.flat, metrics, labels):
        vals = []
        for c in cases:
            x = [abs(float(r[metric])) if metric == "energy_balance_residual" else float(r[metric]) for r in rows if r["case"] == c]
            vals.append(float(np.mean(x)) if x else np.nan)
        ax.bar(cases, vals)
        ax.set_title(label)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Mean transfer metrics across BVP / isoholonomic / periodic tasks")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
