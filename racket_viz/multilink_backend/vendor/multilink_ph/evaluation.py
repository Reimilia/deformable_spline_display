from __future__ import annotations
from dataclasses import dataclass
import time
import torch

from .tasks import LocomotionTask, stack_task_features
from .bspline import TemporalBasis
from .chain import PlanarMultiLinkChain
from .variational import solve_variational, fit_controls_to_modal_trajectory, VariationalResult
from .ph_model import TaskConditionedPHTemplate, rollout_ph
from .geometry import se2_endpoint_error


@dataclass
class PHDirectResult:
    trajectory: dict[str, torch.Tensor]
    pose_error: float
    shape_or_cycle_error: float
    terminal_momentum: float
    dissipation: float
    solve_time_s: float


def evaluate_ph_direct(
    model: TaskConditionedPHTemplate,
    task: LocomotionTask,
    chain: PlanarMultiLinkChain,
    Ksp: torch.Tensor,
    n_steps: int,
    dt: float,
) -> PHDirectResult:
    f = stack_task_features([task])
    t0 = time.perf_counter()
    with torch.no_grad():
        r = rollout_ph(model, f, chain, Ksp, n_steps, dt)
    elapsed = time.perf_counter() - t0
    pose_err = float(torch.linalg.norm(se2_endpoint_error(r["pose"][0, -1], task.pose_target)))
    if task.kind == "bvp":
        qtarget = Ksp @ task.c_target
        shape_cycle = float(torch.linalg.norm(r["q"][0, -1] - qtarget))
    elif task.kind == "periodic":
        shape_cycle = float(torch.linalg.norm(r["q"][0, -1] - r["q"][0, 0]))
    else:
        shape_cycle = 0.0
    mom = float(torch.linalg.norm(r["pi"][0, -1]))
    traj = {k: (v[0].detach() if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == 1 else v) for k, v in r.items()}
    return PHDirectResult(
        trajectory=traj,
        pose_error=pose_err,
        shape_or_cycle_error=shape_cycle,
        terminal_momentum=mom,
        dissipation=float(r["dissipation"][0]),
        solve_time_s=elapsed,
    )


def projected_modal_path_from_ph(ph: PHDirectResult, Ksp: torch.Tensor) -> torch.Tensor:
    pinv = torch.linalg.pinv(Ksp)
    q = ph.trajectory["q"]
    return torch.einsum("mj,tj->tm", pinv, q)


def benchmark_task(
    model: TaskConditionedPHTemplate,
    task: LocomotionTask,
    chain: PlanarMultiLinkChain,
    Ksp: torch.Tensor,
    basis_open: TemporalBasis,
    basis_periodic: TemporalBasis,
    n_steps: int,
    dt: float,
    short_iterations: int = 60,
    full_iterations: int = 300,
    variational_lr: float = 3e-2,
) -> dict:
    basis = basis_periodic if task.kind == "periodic" else basis_open

    t0 = time.perf_counter()
    cold = solve_variational(
        task, chain, Ksp, basis, dt,
        iterations=short_iterations, lr=variational_lr,
    )
    cold_time = time.perf_counter() - t0

    ph = evaluate_ph_direct(model, task, chain, Ksp, n_steps, dt)
    c_ph = projected_modal_path_from_ph(ph, Ksp)
    warm_controls = fit_controls_to_modal_trajectory(c_ph, task, basis)

    t0 = time.perf_counter()
    warm = solve_variational(
        task, chain, Ksp, basis, dt,
        iterations=short_iterations, lr=variational_lr,
        init_controls=warm_controls,
    )
    warm_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    full = solve_variational(
        task, chain, Ksp, basis, dt,
        iterations=full_iterations, lr=variational_lr,
    )
    full_time = time.perf_counter() - t0

    return {
        "cold": cold,
        "ph": ph,
        "warm": warm,
        "full": full,
        "times": {"cold": cold_time, "ph": ph.solve_time_s, "warm": warm_time, "full": full_time},
    }


def rows_from_benchmark(kind: str, index: int, bench: dict) -> list[dict[str, float | str | int]]:
    rows = []
    for method in ["cold", "warm", "full"]:
        r: VariationalResult = bench[method]
        rows.append({
            "task": kind,
            "index": index,
            "method": method,
            "pose_error": r.pose_error,
            "shape_or_cycle_error": 0.0,
            "terminal_momentum": 0.0,
            "dissipation_or_energy": r.energy,
            "solve_time_s": bench["times"][method],
        })
    p: PHDirectResult = bench["ph"]
    rows.append({
        "task": kind,
        "index": index,
        "method": "ph_direct",
        "pose_error": p.pose_error,
        "shape_or_cycle_error": p.shape_or_cycle_error,
        "terminal_momentum": p.terminal_momentum,
        "dissipation_or_energy": p.dissipation,
        "solve_time_s": p.solve_time_s,
    })
    return rows
