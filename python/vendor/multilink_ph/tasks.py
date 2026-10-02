from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
import torch

from .bspline import TemporalBasis
from .simulation import geometric_rollout
from .chain import PlanarMultiLinkChain

TaskKind = Literal["bvp", "iso", "periodic"]
TASK_TO_ID = {"bvp": 0, "iso": 1, "periodic": 2}


@dataclass
class LocomotionTask:
    kind: TaskKind
    c0: torch.Tensor
    c_target: torch.Tensor
    pose_target: torch.Tensor
    generator_controls: torch.Tensor
    pose0: torch.Tensor | None = None
    metadata: dict | None = None

    @property
    def initial_pose(self) -> torch.Tensor:
        if self.pose0 is None:
            return torch.zeros(3, dtype=self.c0.dtype, device=self.c0.device)
        return self.pose0

    @property
    def task_id(self) -> int:
        return TASK_TO_ID[self.kind]


def _random_open_controls(
    rng: np.random.Generator,
    n_ctrl: int,
    n_modes: int,
    amplitude: float,
) -> np.ndarray:
    c0 = rng.normal(scale=0.25 * amplitude, size=n_modes)
    c1 = rng.normal(scale=0.35 * amplitude, size=n_modes)
    u = np.linspace(0.0, 1.0, n_ctrl)[:, None]
    controls = (1.0 - u) * c0[None, :] + u * c1[None, :]
    controls[1:-1] += rng.normal(scale=0.35 * amplitude, size=(n_ctrl - 2, n_modes))
    return controls


def _random_periodic_controls(
    rng: np.random.Generator,
    n_ctrl: int,
    n_modes: int,
    amplitude: float,
) -> np.ndarray:
    # Smooth-ish random loop in coefficient space; periodic basis handles closure.
    phases = np.linspace(0, 2 * np.pi, n_ctrl, endpoint=False)
    controls = np.zeros((n_ctrl, n_modes), dtype=float)
    for m in range(n_modes):
        a = rng.normal(scale=0.55 * amplitude)
        b = rng.normal(scale=0.55 * amplitude)
        phi = rng.uniform(0, 2 * np.pi)
        controls[:, m] = a * np.cos(phases + phi) + b * np.sin(phases + phi)
        controls[:, m] += 0.12 * amplitude * rng.normal(size=n_ctrl)
    return controls


def generate_tasks(
    kind: TaskKind,
    count: int,
    chain: PlanarMultiLinkChain,
    Ksp: torch.Tensor,
    basis_open: TemporalBasis,
    basis_periodic: TemporalBasis,
    dt: float,
    seed: int = 0,
    amplitude: float = 2.0,
    min_translation: float = 0.005,
    max_attempts: int = 80,
) -> list[LocomotionTask]:
    rng = np.random.default_rng(seed)
    tasks: list[LocomotionTask] = []
    basis = basis_periodic if kind == "periodic" else basis_open
    for _ in range(count):
        accepted = False
        for _attempt in range(max_attempts):
            if kind == "periodic":
                z = _random_periodic_controls(rng, basis.n_ctrl, Ksp.shape[1], amplitude)
            else:
                z = _random_open_controls(rng, basis.n_ctrl, Ksp.shape[1], amplitude)
            controls = torch.as_tensor(z, dtype=Ksp.dtype, device=Ksp.device)
            c, cdot = basis.evaluate(controls)
            rollout = geometric_rollout(chain, Ksp, c, cdot, dt)
            pose_target = rollout["pose"][-1]
            q = rollout["q"]
            within_limits = bool((q.abs().max() < 0.8 * chain.params.joint_limit).item())
            trans = float(torch.linalg.norm(pose_target[:2]).item())
            if within_limits and trans >= min_translation:
                accepted = True
                break
        if not accepted:
            # Keep the last feasible-ish sample rather than fail a smoke run.
            pass
        c0 = c[0].detach().clone()
        c_target = c[-1].detach().clone() if kind == "bvp" else torch.zeros_like(c0)
        tasks.append(
            LocomotionTask(
                kind=kind,
                c0=c0,
                c_target=c_target,
                pose_target=pose_target.detach().clone(),
                generator_controls=controls.detach().clone(),
            )
        )
    return tasks


def stack_task_features(tasks: list[LocomotionTask]) -> dict[str, torch.Tensor]:
    if not tasks:
        raise ValueError("tasks cannot be empty")
    dtype = tasks[0].c0.dtype
    device = tasks[0].c0.device
    onehot = torch.zeros(len(tasks), 3, dtype=dtype, device=device)
    ids = torch.tensor([t.task_id for t in tasks], dtype=torch.long, device=device)
    onehot[torch.arange(len(tasks), device=device), ids] = 1.0
    return {
        "task_id": ids,
        "task_onehot": onehot,
        "c0": torch.stack([t.c0 for t in tasks]),
        "c_target": torch.stack([t.c_target for t in tasks]),
        "pose0": torch.stack([t.initial_pose for t in tasks]),
        "pose_target": torch.stack([t.pose_target for t in tasks]),
    }
