from __future__ import annotations
from dataclasses import dataclass
import torch

from .tasks import LocomotionTask
from .bspline import TemporalBasis
from .chain import PlanarMultiLinkChain
from .simulation import geometric_rollout
from .geometry import se2_endpoint_error


@dataclass
class VariationalResult:
    controls: torch.Tensor
    loss: float
    energy: float
    pose_error: float
    trajectory: dict[str, torch.Tensor]
    history: list[dict[str, float]]


def _effective_controls(raw: torch.Tensor, task: LocomotionTask, basis: TemporalBasis) -> torch.Tensor:
    if task.kind == "bvp":
        if raw.shape[0] < 2:
            raise ValueError("need at least two spline controls")
        return torch.cat([task.c0[None], raw[1:-1], task.c_target[None]], dim=0)
    if task.kind == "iso":
        return torch.cat([task.c0[None], raw[1:]], dim=0)
    # Periodic basis analytically closes c and cdot.  Apply a common modal shift
    # so the prescribed initial shape is also exact; partition of unity preserves derivatives.
    c_start = basis.B[0] @ raw
    shift = task.c0 - c_start
    return raw + shift[None, :]


def default_initial_controls(task: LocomotionTask, basis: TemporalBasis) -> torch.Tensor:
    n, d = basis.n_ctrl, task.c0.numel()
    if task.kind == "bvp":
        u = torch.linspace(0.0, 1.0, n, dtype=task.c0.dtype, device=task.c0.device)[:, None]
        z = (1.0 - u) * task.c0[None] + u * task.c_target[None]
    else:
        z = task.c0[None].repeat(n, 1)
    return z


def fit_controls_to_modal_trajectory(
    c_samples: torch.Tensor,
    task: LocomotionTask,
    basis: TemporalBasis,
) -> torch.Tensor:
    """Least-squares fit of spline controls to a modal trajectory, respecting boundary parameterization."""
    B = basis.B
    if task.kind == "bvp":
        rhs = c_samples - B[:, :1] * task.c0[None] - B[:, -1:] * task.c_target[None]
        if basis.n_ctrl > 2:
            x = torch.linalg.lstsq(B[:, 1:-1], rhs).solution
            raw = torch.cat([task.c0[None], x, task.c_target[None]], dim=0)
        else:
            raw = torch.stack([task.c0, task.c_target])
    elif task.kind == "iso":
        rhs = c_samples - B[:, :1] * task.c0[None]
        x = torch.linalg.lstsq(B[:, 1:], rhs).solution
        raw = torch.cat([task.c0[None], x], dim=0)
    else:
        raw = torch.linalg.lstsq(B, c_samples).solution
        raw = _effective_controls(raw, task, basis)
    return raw.detach()


def solve_variational(
    task: LocomotionTask,
    chain: PlanarMultiLinkChain,
    Ksp: torch.Tensor,
    basis: TemporalBasis,
    dt: float,
    iterations: int = 300,
    lr: float = 3e-2,
    pose_weight: float = 10000.0,
    joint_limit_weight: float = 100.0,
    init_controls: torch.Tensor | None = None,
    record_every: int = 20,
) -> VariationalResult:
    if init_controls is None:
        init_controls = default_initial_controls(task, basis)
    raw = torch.nn.Parameter(init_controls.clone())
    opt = torch.optim.Adam([raw], lr=lr)
    history: list[dict[str, float]] = []
    best = None
    best_score = float("inf")

    for it in range(iterations):
        opt.zero_grad(set_to_none=True)
        controls = _effective_controls(raw, task, basis)
        c, cdot = basis.evaluate(controls)
        traj = geometric_rollout(chain, Ksp, c, cdot, dt, pose0=task.initial_pose)
        err = se2_endpoint_error(traj["pose"][-1], task.pose_target)
        pose_loss = (err**2).sum()
        q = traj["q"]
        excess = torch.relu(q.abs() - chain.params.joint_limit)
        limit_pen = (excess**2).mean()
        loss = traj["energy"] + pose_weight * pose_loss + joint_limit_weight * limit_pen
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw], max_norm=50.0)
        opt.step()

        score = float(loss.detach())
        if score < best_score:
            best_score = score
            best = raw.detach().clone()
        if it % record_every == 0 or it == iterations - 1:
            history.append({
                "iter": float(it),
                "loss": score,
                "energy": float(traj["energy"].detach()),
                "pose_error": float(torch.linalg.norm(err).detach()),
                "limit_pen": float(limit_pen.detach()),
            })

    assert best is not None
    controls = _effective_controls(best, task, basis)
    c, cdot = basis.evaluate(controls)
    traj = geometric_rollout(chain, Ksp, c, cdot, dt, pose0=task.initial_pose)
    err = se2_endpoint_error(traj["pose"][-1], task.pose_target)
    final_loss = float((traj["energy"] + pose_weight * (err**2).sum()).detach())
    return VariationalResult(
        controls=controls.detach(),
        loss=final_loss,
        energy=float(traj["energy"].detach()),
        pose_error=float(torch.linalg.norm(err).detach()),
        trajectory={k: v.detach() if torch.is_tensor(v) else v for k, v in traj.items()},
        history=history,
    )
