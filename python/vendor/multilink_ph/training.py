from __future__ import annotations
from dataclasses import dataclass
import time
import torch

from .tasks import LocomotionTask, stack_task_features
from .bspline import TemporalBasis
from .chain import PlanarMultiLinkChain
from .variational import solve_variational, VariationalResult
from .ph_model import TaskConditionedPHTemplate, PHLossConfig, rollout_ph, ph_task_loss


@dataclass
class TrainConfig:
    teacher_iterations: int = 160
    teacher_lr: float = 3e-2
    pretrain_epochs: int = 300
    pretrain_lr: float = 2e-3
    finetune_epochs: int = 350
    finetune_lr: float = 8e-4
    grad_clip: float = 20.0
    log_every: int = 25


def compute_variational_teachers(
    tasks: list[LocomotionTask],
    chain: PlanarMultiLinkChain,
    Ksp: torch.Tensor,
    basis_open: TemporalBasis,
    basis_periodic: TemporalBasis,
    dt: float,
    iterations: int,
    lr: float,
) -> list[VariationalResult]:
    teachers = []
    for i, task in enumerate(tasks):
        basis = basis_periodic if task.kind == "periodic" else basis_open
        result = solve_variational(
            task, chain, Ksp, basis, dt,
            iterations=iterations, lr=lr,
            pose_weight=10000.0,
        )
        teachers.append(result)
    return teachers


def pretrain_reference(
    model: TaskConditionedPHTemplate,
    tasks: list[LocomotionTask],
    teachers: list[VariationalResult],
    epochs: int,
    lr: float,
    log_every: int = 50,
) -> list[float]:
    features = stack_task_features(tasks)
    # Teachers may use open or periodic bases but all use same N+1 sample count.
    teacher_c = torch.stack([r.trajectory["c"] for r in teachers], dim=0)
    T = teacher_c.shape[1]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for ep in range(epochs):
        opt.zero_grad(set_to_none=True)
        preds = []
        for k in range(T):
            tau = torch.full((len(tasks),), k / (T - 1), dtype=teacher_c.dtype, device=teacher_c.device)
            preds.append(model.reference(features, tau))
        pred = torch.stack(preds, dim=1)
        # Reference need not equal realized teacher exactly, but this provides a stable initialization.
        loss = ((pred - teacher_c) ** 2).mean()
        loss = loss + 1e-4 * (model.log_stiffness.pow(2).mean() + model.log_extra_damping.pow(2).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
        opt.step()
        hist.append(float(loss.detach()))
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"[pretrain] epoch={ep:04d} mse={hist[-1]:.6e}")
    return hist


def finetune_ph(
    model: TaskConditionedPHTemplate,
    tasks: list[LocomotionTask],
    chain: PlanarMultiLinkChain,
    Ksp: torch.Tensor,
    n_steps: int,
    dt: float,
    epochs: int,
    lr: float,
    loss_cfg: PHLossConfig,
    grad_clip: float = 20.0,
    log_every: int = 25,
) -> list[dict[str, float]]:
    features = stack_task_features(tasks)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for ep in range(epochs):
        opt.zero_grad(set_to_none=True)
        rollout = rollout_ph(model, features, chain, Ksp, n_steps, dt)
        loss, metrics = ph_task_loss(rollout, features, chain, Ksp, loss_cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        rec = {
            "epoch": float(ep),
            "loss": float(loss.detach()),
            "pose_error": float(metrics["pose_error"].mean().detach()),
            "dissipation": float(metrics["dissipation"].mean().detach()),
            "periodic_state": float(metrics["periodic_state"].mean().detach()),
        }
        hist.append(rec)
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(
                f"[finetune] epoch={ep:04d} loss={rec['loss']:.5e} "
                f"pose={rec['pose_error']:.4e} diss={rec['dissipation']:.4e}"
            )
    return hist


def train_ph_template(
    model: TaskConditionedPHTemplate,
    tasks: list[LocomotionTask],
    chain: PlanarMultiLinkChain,
    Ksp: torch.Tensor,
    basis_open: TemporalBasis,
    basis_periodic: TemporalBasis,
    n_steps: int,
    dt: float,
    train_cfg: TrainConfig,
    loss_cfg: PHLossConfig | None = None,
) -> dict:
    if loss_cfg is None:
        loss_cfg = PHLossConfig()
    t0 = time.perf_counter()
    teachers = compute_variational_teachers(
        tasks, chain, Ksp, basis_open, basis_periodic, dt,
        iterations=train_cfg.teacher_iterations, lr=train_cfg.teacher_lr,
    )
    teacher_time = time.perf_counter() - t0
    pre_hist = pretrain_reference(
        model, tasks, teachers,
        epochs=train_cfg.pretrain_epochs,
        lr=train_cfg.pretrain_lr,
        log_every=max(train_cfg.log_every * 2, 1),
    )
    fine_hist = finetune_ph(
        model, tasks, chain, Ksp, n_steps, dt,
        epochs=train_cfg.finetune_epochs,
        lr=train_cfg.finetune_lr,
        loss_cfg=loss_cfg,
        grad_clip=train_cfg.grad_clip,
        log_every=train_cfg.log_every,
    )
    return {
        "teachers": teachers,
        "pretrain_history": pre_hist,
        "finetune_history": fine_hist,
        "teacher_time_s": teacher_time,
    }
