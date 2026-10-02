from __future__ import annotations
import math
from pathlib import Path
from typing import Mapping, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation

from .chain import PlanarMultiLinkChain
from .tasks import LocomotionTask


def _world_vertices(chain: PlanarMultiLinkChain, q: torch.Tensor, pose: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        q = torch.as_tensor(q)
        pose = torch.as_tensor(pose, dtype=q.dtype, device=q.device)
        kin = chain.kinematics(q)
        v = kin["joint_positions"]
        c, s = torch.cos(pose[2]), torch.sin(pose[2])
        R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
        world = (R @ v.T).T + pose[:2]
    return world.detach().cpu().numpy()


def plot_trajectory(ax, chain: PlanarMultiLinkChain, q: torch.Tensor, pose: torch.Tensor, title: str) -> None:
    n = q.shape[0]
    ids = np.unique(np.linspace(0, n - 1, 5).round().astype(int))
    com_path = []
    for k in range(n):
        verts = _world_vertices(chain, q[k], pose[k])
        com_path.append(verts.mean(axis=0))
    com_path = np.asarray(com_path)
    ax.plot(com_path[:, 0], com_path[:, 1], linewidth=1.5)
    for k in ids:
        verts = _world_vertices(chain, q[k], pose[k])
        ax.plot(verts[:, 0], verts[:, 1], marker="o", markersize=2, alpha=0.55)
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(True, alpha=0.25)


def plot_benchmark(task: LocomotionTask, chain: PlanarMultiLinkChain, bench: dict, path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plot_trajectory(axes[0, 0], chain, bench["cold"].trajectory["q"], bench["cold"].trajectory["pose"], "Cold variational, fixed budget")
    plot_trajectory(axes[0, 1], chain, bench["ph"].trajectory["q"], bench["ph"].trajectory["pose"], "Learned pH direct")
    plot_trajectory(axes[1, 0], chain, bench["warm"].trajectory["q"], bench["warm"].trajectory["pose"], "pH warm-start + same budget")
    plot_trajectory(axes[1, 1], chain, bench["full"].trajectory["q"], bench["full"].trajectory["pose"], "Variational reference")
    fig.suptitle(f"{task.kind}: target displacement={task.pose_target.detach().cpu().numpy()}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------- Locomotion-focused visualizations ----------

def _get_traj_entry(entry: Any) -> dict[str, torch.Tensor]:
    return entry.trajectory if hasattr(entry, "trajectory") else entry


def _collect_world_vertices(chain: PlanarMultiLinkChain, traj: Mapping[str, torch.Tensor]) -> tuple[list[np.ndarray], np.ndarray]:
    q = traj["q"]
    pose = traj["pose"]
    frames: list[np.ndarray] = []
    com = []
    for k in range(q.shape[0]):
        verts = _world_vertices(chain, q[k], pose[k])
        frames.append(verts)
        com.append(verts.mean(axis=0))
    return frames, np.asarray(com)


def _global_xy_limits(frame_sets: list[list[np.ndarray]], pad_frac: float = 0.12) -> tuple[float, float, float, float]:
    pts = np.concatenate([np.concatenate(frames, axis=0) for frames in frame_sets], axis=0)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    dx = xmax - xmin
    dy = ymax - ymin
    pad = pad_frac * max(dx, dy, 1e-3)
    return xmin - pad, xmax + pad, ymin - pad, ymax + pad


def _task_goal_label(task: LocomotionTask) -> str:
    goal = task.pose_target.detach().cpu().numpy()
    if task.kind == "bvp":
        return f"BVP target pose = ({goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f})"
    if task.kind == "iso":
        return f"Isoholonomic displacement = ({goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f})"
    return f"Periodic gait holonomy = ({goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f})"


METHOD_LABELS = {
    "ph": "Learned pH rollout",
    "full": "Variational minimization",
    "cold": "Variational, short budget",
    "warm": "pH warm start + short budget",
}


METHOD_COLORS = {
    "ph": "tab:blue",
    "full": "tab:red",
    "cold": "tab:green",
    "warm": "tab:purple",
}


def plot_locomotion_storyboard(
    task: LocomotionTask,
    chain: PlanarMultiLinkChain,
    methods: Mapping[str, Any],
    path: str | Path,
    n_snapshots: int = 8,
    include_generator: bool = True,
) -> None:
    """Static filmstrip showing locomotion over time for each method."""
    selected = list(methods.keys())
    rows = len(selected) + (1 if include_generator else 0)
    fig, axes = plt.subplots(rows, n_snapshots, figsize=(2.0 * n_snapshots, 2.2 * rows), squeeze=False)

    ordered: list[tuple[str, dict[str, torch.Tensor], str]] = []
    if include_generator:
        # Recreate the generating locomotion when available.
        from .bspline import TemporalBasis
        from .simulation import geometric_rollout
        # Infer an open vs periodic temporal basis directly from generator controls.
        n_steps = _get_traj_entry(next(iter(methods.values())))["q"].shape[0]
        n_ctrl = task.generator_controls.shape[0]
        periodic = task.kind == "periodic"
        basis = TemporalBasis(
            n_steps=n_steps,
            n_ctrl=n_ctrl,
            degree=3,
            periodic=periodic,
            dtype=task.generator_controls.dtype,
            device=task.generator_controls.device,
        )
        c, cdot = basis.evaluate(task.generator_controls)
        # Build a pseudoinverse map from c0 -> q0 dimensions when possible via the first method q.
        sample_traj = _get_traj_entry(next(iter(methods.values())))
        q_dim = sample_traj["q"].shape[-1]
        mode_dim = c.shape[-1]
        # Estimate Ksp from the sample trajectories using least squares when the dimensions line up poorly.
        # Prefer an exact map from the first method if c/q history exists.
        if "c" in sample_traj and sample_traj["c"].shape[-1] == mode_dim and sample_traj["q"].shape[-1] == q_dim:
            # q = Ksp c, solve on the whole trajectory.
            C = sample_traj["c"].detach()
            Q = sample_traj["q"].detach()
            Ksp = torch.linalg.lstsq(C, Q).solution.T
        else:
            Ksp = torch.eye(q_dim, mode_dim, dtype=c.dtype, device=c.device)
        gen_traj = geometric_rollout(chain, Ksp, c, cdot, dt=1.0 / max(n_steps - 1, 1))
        ordered.append(("generator", gen_traj, "Task generator / nominal motion"))
    for key in selected:
        ordered.append((key, _get_traj_entry(methods[key]), METHOD_LABELS.get(key, key)))

    frame_sets = []
    com_sets = []
    for _, traj, _ in ordered:
        frames, com = _collect_world_vertices(chain, traj)
        frame_sets.append(frames)
        com_sets.append(com)
    xmin, xmax, ymin, ymax = _global_xy_limits(frame_sets)

    for row, (key, traj, label) in enumerate(ordered):
        frames = frame_sets[row]
        com = com_sets[row]
        ids = np.unique(np.linspace(0, len(frames) - 1, n_snapshots).round().astype(int))
        color = METHOD_COLORS.get(key, "0.15")
        for col, idx in enumerate(ids):
            ax = axes[row, col]
            verts = frames[idx]
            ax.plot(com[: idx + 1, 0], com[: idx + 1, 1], color=color, linewidth=1.2, alpha=0.85)
            ax.plot(verts[:, 0], verts[:, 1], marker="o", markersize=3, color=color)
            ax.scatter([task.pose_target[0].item()], [task.pose_target[1].item()], marker="x", s=20, color="black")
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.2)
            if row == 0:
                ax.set_title(f"t = {idx/(len(frames)-1):.2f}")
            if col == 0:
                ax.set_ylabel(label)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f"Locomotion storyboard — {task.kind.upper()}\n{_task_goal_label(task)}")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def animate_locomotion_comparison(
    task: LocomotionTask,
    chain: PlanarMultiLinkChain,
    methods: Mapping[str, Any],
    path: str | Path,
    fps: int = 8,
    trail: int = 0,
) -> None:
    """Animated side-by-side locomotion for sanity checking how motion unfolds."""
    ordered = [(k, _get_traj_entry(v), METHOD_LABELS.get(k, k)) for k, v in methods.items()]
    if not ordered:
        raise ValueError("methods cannot be empty")
    frame_sets = []
    com_sets = []
    n_frames = max(_get_traj_entry(v)["q"].shape[0] for v in methods.values())
    for _, traj, _ in ordered:
        frames, com = _collect_world_vertices(chain, traj)
        frame_sets.append(frames)
        com_sets.append(com)
    xmin, xmax, ymin, ymax = _global_xy_limits(frame_sets)
    ncols = len(ordered)
    fig, axes = plt.subplots(1, ncols, figsize=(5.0 * ncols, 5.2), squeeze=False)
    axes = axes[0]

    line_objs = []
    path_objs = []
    title_objs = []
    target_xy = task.pose_target.detach().cpu().numpy()[:2]

    for ax, (key, traj, label), frames, com in zip(axes, ordered, frame_sets, com_sets):
        ax.scatter([target_xy[0]], [target_xy[1]], marker="x", s=48, color="black", label="target")
        path_line, = ax.plot([], [], color=METHOD_COLORS.get(key, "tab:blue"), linewidth=1.5, alpha=0.9)
        body_line, = ax.plot([], [], marker="o", markersize=4, color=METHOD_COLORS.get(key, "tab:blue"), linewidth=2.2)
        title = ax.set_title(label)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
        line_objs.append(body_line)
        path_objs.append(path_line)
        title_objs.append(title)

    suptitle = fig.suptitle(f"Locomotion comparison — {task.kind.upper()}\n{_task_goal_label(task)}")

    def _frame_data(frames: list[np.ndarray], com: np.ndarray, i: int):
        idx = min(i, len(frames) - 1)
        verts = frames[idx]
        if trail > 0:
            j0 = max(0, idx - trail)
            path = com[j0: idx + 1]
        else:
            path = com[: idx + 1]
        return verts, path, idx

    def init():
        for body, path in zip(line_objs, path_objs):
            body.set_data([], [])
            path.set_data([], [])
        return [*line_objs, *path_objs, suptitle]

    def update(i: int):
        artists = []
        for body, path, title, frames, com, (key, traj, label) in zip(line_objs, path_objs, title_objs, frame_sets, com_sets, ordered):
            verts, p, idx = _frame_data(frames, com, i)
            body.set_data(verts[:, 0], verts[:, 1])
            path.set_data(p[:, 0], p[:, 1])
            title.set_text(f"{label}\nstep {idx+1}/{len(frames)}")
            artists.extend([body, path, title])
        return artists

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=n_frames, interval=1000 / max(fps, 1), blit=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".gif":
        writer = animation.PillowWriter(fps=fps)
    else:
        writer = animation.PillowWriter(fps=fps)
    anim.save(path, writer=writer, dpi=130)
    plt.close(fig)


def write_locomotion_case_summary(task: LocomotionTask, bench: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    rows = []
    for key, obj in bench.items():
        if key == "times":
            continue
        if hasattr(obj, "pose_error"):
            pose_error = float(obj.pose_error)
            shape_err = float(getattr(obj, "shape_or_cycle_error", 0.0))
            if hasattr(obj, "dissipation"):
                objective = float(obj.dissipation)
            else:
                objective = float(getattr(obj, "energy", 0.0))
        else:
            continue
        rows.append((key, pose_error, shape_err, objective))
    with path.open("w") as f:
        f.write(f"task_kind,{task.kind}\n")
        f.write(f"pose_target,{task.pose_target.detach().cpu().numpy().tolist()}\n")
        f.write("method,pose_error,shape_or_cycle_error,objective\n")
        for key, p, s, o in rows:
            f.write(f"{key},{p:.8e},{s:.8e},{o:.8e}\n")
