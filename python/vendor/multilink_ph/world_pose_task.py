from __future__ import annotations

from dataclasses import dataclass, replace
import math
import numpy as np
import torch

from .tasks import LocomotionTask
from .bspline import TemporalBasis
from .chain import PlanarMultiLinkChain
from .simulation import geometric_rollout


def _wrap_angle_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def _wrap_angle_float(x: float) -> float:
    return float(math.atan2(math.sin(x), math.cos(x)))


@dataclass
class WorldPoseTaskSpec:
    """Specification for one positioned multi-link locomotion task.

    Two sampling regimes are supported.

    * ``selection_mode='first_feasible'`` preserves the original perturbation-style
      behavior and accepts the first trajectory above ``min_translation`` and
      ``min_rotation``.
    * ``selection_mode='targeted_salient'`` first samples a desired terminal-motion
      magnitude from explicit body-length/rotation ranges, then searches a pool of
      feasible spline motions and selects a trajectory near that sampled target.

    The second regime is intended for locomotion training.  It prevents a dataset
    from clustering just above a tiny feasibility threshold.
    """

    kind: str = "bvp"  # bvp | iso | periodic
    pose0_xytheta: tuple[float, float, float] = (-0.25, 0.15, 0.25)
    amplitude: float = 2.0

    # Legacy/perturbation feasibility thresholds (absolute world units and radians).
    min_translation: float = 0.015
    min_rotation: float = 0.06

    # Salient terminal-motion distribution. Translation is normalized by body length.
    selection_mode: str = "first_feasible"  # first_feasible | targeted_salient
    translation_fraction_range: tuple[float, float] | None = None
    rotation_magnitude_range: tuple[float, float] | None = None
    min_shape_excursion: float = 0.0  # RMS joint excursion in radians over the rollout
    min_endpoint_shape_change: float = 0.0  # useful for BVP; radians in joint-space norm
    strict_salience: bool = False
    rotation_sign_balance: bool = True

    # Candidate search.  ``candidate_pool`` defaults to max_attempts in targeted mode.
    seed: int = 19
    max_attempts: int = 400
    candidate_pool: int | None = None

    # Optional initial-pose diversity for scale-up datasets.
    randomize_initial_pose: bool = False
    initial_xy_range: tuple[float, float] = (-0.35, 0.35)
    initial_theta_range: tuple[float, float] = (-math.pi, math.pi)


def _open_controls(rng: np.random.Generator, n_ctrl: int, n_modes: int, amplitude: float) -> np.ndarray:
    c0 = rng.normal(scale=0.20 * amplitude, size=n_modes)
    cT = rng.normal(scale=0.25 * amplitude, size=n_modes)
    u = np.linspace(0.0, 1.0, n_ctrl)[:, None]
    z = (1.0 - u) * c0 + u * cT
    # Correlated traveling-wave structure generates significantly more global motion
    # than independent control-point jitter while remaining random task-to-task.
    phase = np.linspace(0.0, 2.0 * np.pi, n_ctrl)[:, None]
    mode_phase = np.linspace(0.0, np.pi, n_modes)[None, :]
    z += 0.40 * amplitude * np.sin(phase + mode_phase + rng.uniform(-0.6, 0.6))
    z += rng.normal(scale=0.10 * amplitude, size=z.shape)
    z[0] = c0
    z[-1] = cT
    return z


def _periodic_controls(rng: np.random.Generator, n_ctrl: int, n_modes: int, amplitude: float) -> np.ndarray:
    ph = np.linspace(0.0, 2.0 * np.pi, n_ctrl, endpoint=False)
    z = np.zeros((n_ctrl, n_modes), dtype=float)
    for m in range(n_modes):
        a = rng.normal(scale=0.45 * amplitude)
        b = rng.normal(scale=0.45 * amplitude)
        shift = rng.uniform(0.0, 2.0 * np.pi)
        z[:, m] = a * np.cos(ph + shift) + b * np.sin(ph + shift)
    return z


def _sample_pose0(spec: WorldPoseTaskSpec, rng: np.random.Generator, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if not spec.randomize_initial_pose:
        return torch.tensor(spec.pose0_xytheta, dtype=dtype, device=device)
    lo, hi = spec.initial_xy_range
    th0, th1 = spec.initial_theta_range
    return torch.tensor(
        [rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(th0, th1)],
        dtype=dtype,
        device=device,
    )


def _trajectory_salience(
    r: dict[str, torch.Tensor],
    pose0: torch.Tensor,
    chain: PlanarMultiLinkChain,
) -> dict[str, float]:
    q = r["q"]
    dxy = r["pose"][-1, :2] - pose0[:2]
    dtheta_t = _wrap_angle_tensor(r["pose"][-1, 2] - pose0[2])
    trans = float(torch.linalg.norm(dxy))
    trans_fraction = trans / max(float(chain.params.total_length), 1e-12)
    rot_signed = float(dtheta_t)
    rot = abs(rot_signed)
    q0 = q[0]
    shape_excursion = float(torch.sqrt(torch.mean((q - q0) ** 2)))
    endpoint_shape_change = float(torch.linalg.norm(q[-1] - q0))
    qmax = float(q.abs().max())
    heading = float(math.atan2(float(dxy[1]), float(dxy[0]))) if trans > 1e-12 else 0.0
    return {
        "translation": trans,
        "translation_fraction": trans_fraction,
        "rotation": rot,
        "rotation_signed": rot_signed,
        "translation_heading": heading,
        "shape_excursion": shape_excursion,
        "endpoint_shape_change": endpoint_shape_change,
        "qmax": qmax,
    }


def _targeted_cost(
    metrics: dict[str, float],
    desired_trans_fraction: float,
    desired_rot: float,
    desired_rot_sign: float,
    spec: WorldPoseTaskSpec,
) -> tuple[float, bool]:
    tr = spec.translation_fraction_range
    rr = spec.rotation_magnitude_range
    assert tr is not None and rr is not None
    tw = max(tr[1] - tr[0], 1e-6)
    rw = max(rr[1] - rr[0], 1e-6)
    cost = ((metrics["translation_fraction"] - desired_trans_fraction) / tw) ** 2
    cost += ((metrics["rotation"] - desired_rot) / rw) ** 2
    if spec.rotation_sign_balance and metrics["rotation"] > 1e-8:
        if math.copysign(1.0, metrics["rotation_signed"]) != desired_rot_sign:
            cost += 1.0
    if metrics["shape_excursion"] < spec.min_shape_excursion:
        denom = max(spec.min_shape_excursion, 1e-6)
        cost += 4.0 * ((spec.min_shape_excursion - metrics["shape_excursion"]) / denom) ** 2
    if spec.kind == "bvp" and metrics["endpoint_shape_change"] < spec.min_endpoint_shape_change:
        denom = max(spec.min_endpoint_shape_change, 1e-6)
        cost += 2.0 * ((spec.min_endpoint_shape_change - metrics["endpoint_shape_change"]) / denom) ** 2

    in_band = (
        tr[0] <= metrics["translation_fraction"] <= tr[1]
        and rr[0] <= metrics["rotation"] <= rr[1]
        and metrics["shape_excursion"] >= spec.min_shape_excursion
        and (spec.kind != "bvp" or metrics["endpoint_shape_change"] >= spec.min_endpoint_shape_change)
    )
    return cost, in_band


def _make_task_from_candidate(
    spec: WorldPoseTaskSpec,
    pose0: torch.Tensor,
    z: torch.Tensor,
    c: torch.Tensor,
    r: dict[str, torch.Tensor],
    metrics: dict[str, float],
    *,
    desired: dict[str, float] | None = None,
    candidate_rank: int | None = None,
) -> LocomotionTask:
    c0 = c[0].detach().clone()
    c_target = c[-1].detach().clone() if spec.kind == "bvp" else torch.zeros_like(c0)
    metadata = {
        "sampling_mode": spec.selection_mode,
        "salience": metrics,
        "desired_salience": desired or {},
        "candidate_rank": candidate_rank,
        "amplitude": float(spec.amplitude),
    }
    return LocomotionTask(
        kind=spec.kind,
        c0=c0,
        c_target=c_target,
        pose_target=r["pose"][-1].detach().clone(),
        generator_controls=z.detach().clone(),
        pose0=pose0.detach().clone(),
        metadata=metadata,
    )


def make_world_pose_task(
    spec: WorldPoseTaskSpec,
    chain: PlanarMultiLinkChain,
    Ksp: torch.Tensor,
    basis_open: TemporalBasis,
    basis_periodic: TemporalBasis,
    dt: float,
) -> LocomotionTask:
    """Build one feasible positioned locomotion task.

    In salient mode the *terminal motion itself* is sampled from a distribution.
    Candidate spline motions are then searched for a feasible realization near
    that sampled terminal-motion target.  This avoids the previous pathology in
    which random tasks accumulated immediately above a tiny acceptance threshold.
    """

    if spec.kind not in {"bvp", "iso", "periodic"}:
        raise ValueError("kind must be bvp, iso, or periodic")
    if spec.selection_mode not in {"first_feasible", "targeted_salient"}:
        raise ValueError("selection_mode must be first_feasible or targeted_salient")
    if spec.selection_mode == "targeted_salient":
        if spec.translation_fraction_range is None or spec.rotation_magnitude_range is None:
            raise ValueError("targeted_salient requires translation_fraction_range and rotation_magnitude_range")

    rng = np.random.default_rng(spec.seed)
    basis = basis_periodic if spec.kind == "periodic" else basis_open
    pose0 = _sample_pose0(spec, rng, dtype=Ksp.dtype, device=Ksp.device)

    if spec.selection_mode == "targeted_salient":
        tr = spec.translation_fraction_range
        rr = spec.rotation_magnitude_range
        assert tr is not None and rr is not None
        desired_tf = float(rng.uniform(tr[0], tr[1]))
        desired_rot = float(rng.uniform(rr[0], rr[1]))
        desired_sign = float(rng.choice([-1.0, 1.0])) if spec.rotation_sign_balance else 1.0
        desired = {
            "translation_fraction": desired_tf,
            "rotation": desired_rot,
            "rotation_sign": desired_sign,
        }
        pool = int(spec.candidate_pool or spec.max_attempts)
        feasible: list[tuple[float, bool, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, float]]] = []
        for _ in range(pool):
            z_np = _periodic_controls(rng, basis.n_ctrl, Ksp.shape[1], spec.amplitude) if spec.kind == "periodic" else _open_controls(rng, basis.n_ctrl, Ksp.shape[1], spec.amplitude)
            z = torch.as_tensor(z_np, dtype=Ksp.dtype, device=Ksp.device)
            c, cdot = basis.evaluate(z)
            r = geometric_rollout(chain, Ksp, c, cdot, dt, pose0=pose0)
            metrics = _trajectory_salience(r, pose0, chain)
            within = metrics["qmax"] < 0.82 * chain.params.joint_limit
            if not within:
                continue
            cost, in_band = _targeted_cost(metrics, desired_tf, desired_rot, desired_sign, spec)
            feasible.append((cost, in_band, z, c, r, metrics))

        if not feasible:
            raise RuntimeError("could not construct any feasible positioned locomotion candidate")
        in_band = [x for x in feasible if x[1]]
        if spec.strict_salience and not in_band:
            best = min(feasible, key=lambda x: x[0])
            raise RuntimeError(
                "could not realize sampled salient terminal-pose band; "
                f"desired={desired}, closest={best[-1]}. Increase amplitude/candidate_pool or relax the range."
            )
        candidates = in_band if in_band else feasible
        candidates.sort(key=lambda x: x[0])
        _, _, z, c, r, metrics = candidates[0]
        return _make_task_from_candidate(spec, pose0, z, c, r, metrics, desired=desired, candidate_rank=0)

    # Legacy first-feasible behavior kept for old tests and perturbation baselines.
    best = None
    best_score = -float("inf")
    for _ in range(spec.max_attempts):
        z_np = _periodic_controls(rng, basis.n_ctrl, Ksp.shape[1], spec.amplitude) if spec.kind == "periodic" else _open_controls(rng, basis.n_ctrl, Ksp.shape[1], spec.amplitude)
        z = torch.as_tensor(z_np, dtype=Ksp.dtype, device=Ksp.device)
        c, cdot = basis.evaluate(z)
        r = geometric_rollout(chain, Ksp, c, cdot, dt, pose0=pose0)
        metrics = _trajectory_salience(r, pose0, chain)
        within = metrics["qmax"] < 0.82 * chain.params.joint_limit
        score = metrics["translation"] + 0.25 * metrics["rotation"]
        if within and score > best_score:
            best = (z, c, r, metrics)
            best_score = score
        if within and metrics["translation"] >= spec.min_translation and metrics["rotation"] >= spec.min_rotation:
            best = (z, c, r, metrics)
            break
    if best is None:
        raise RuntimeError("could not construct a feasible positioned locomotion task")
    z, c, r, metrics = best
    return _make_task_from_candidate(spec, pose0, z, c, r, metrics)


def make_world_pose_tasks(
    spec: WorldPoseTaskSpec,
    count: int,
    chain: PlanarMultiLinkChain,
    Ksp: torch.Tensor,
    basis_open: TemporalBasis,
    basis_periodic: TemporalBasis,
    dt: float,
    *,
    stratified: bool = True,
    translation_bins: int = 3,
    rotation_bins: int = 3,
) -> list[LocomotionTask]:
    """Generate a scale-up dataset whose terminal-pose magnitudes cover the requested range.

    When ``stratified`` is enabled, translation and rotation ranges are partitioned
    into a grid and tasks cycle through those cells.  This prevents large training
    sets from drifting back toward only the easiest terminal motions.
    """
    if count <= 0:
        return []
    if not stratified or spec.selection_mode != "targeted_salient":
        return [
            make_world_pose_task(replace(spec, seed=spec.seed + 1009 * i), chain, Ksp, basis_open, basis_periodic, dt)
            for i in range(count)
        ]

    tr = spec.translation_fraction_range
    rr = spec.rotation_magnitude_range
    assert tr is not None and rr is not None
    t_edges = np.linspace(tr[0], tr[1], translation_bins + 1)
    r_edges = np.linspace(rr[0], rr[1], rotation_bins + 1)
    cells = [(i, j) for i in range(translation_bins) for j in range(rotation_bins)]
    rng = np.random.default_rng(spec.seed + 7717)
    rng.shuffle(cells)

    tasks: list[LocomotionTask] = []
    for k in range(count):
        i, j = cells[k % len(cells)]
        local = replace(
            spec,
            seed=spec.seed + 1009 * k,
            translation_fraction_range=(float(t_edges[i]), float(t_edges[i + 1])),
            rotation_magnitude_range=(float(r_edges[j]), float(r_edges[j + 1])),
            # Individual strata can be narrow relative to the reachable set.
            # Select the closest feasible candidate to the sampled cell rather
            # than failing the entire dataset; the outer profile remains the
            # hard definition of what constitutes salient locomotion.
            strict_salience=False,
        )
        task = make_world_pose_task(local, chain, Ksp, basis_open, basis_periodic, dt)
        if task.metadata is not None:
            task.metadata["stratum"] = {"translation_bin": i, "rotation_bin": j}
        tasks.append(task)
    return tasks


def world_pose_salience_profile(profile: str, kind: str, *, smoke: bool = False) -> dict[str, object]:
    """Canonical task-distribution settings used by focused and scale-up experiments."""
    if profile == "perturbation":
        return {
            "selection_mode": "first_feasible",
            "amplitude": 1.7 if smoke else 1.9,
            "min_translation": 0.012 if smoke else 0.025,
            "min_rotation": 0.035 if smoke else 0.08,
        }
    if profile == "strong":
        amp = 4.5 if smoke else 5.5
        tr = (0.07, 0.15) if smoke else (0.10, 0.22)
        rr = (0.25, 0.75) if smoke else (0.35, 1.00)
        shape = 0.12 if smoke else 0.16
    elif profile == "salient":
        amp = 3.5 if smoke else 4.5
        tr = (0.04, 0.10) if smoke else (0.07, 0.17)
        rr = (0.15, 0.55) if smoke else (0.22, 0.80)
        shape = 0.08 if smoke else 0.12
    else:
        raise ValueError("profile must be perturbation, salient, or strong")

    # In the present planar drag model, periodic closed shape loops have nearly
    # zero yaw holonomy.  Translation remains a true locomotion signal, so the
    # periodic profile does not impose an artificial large-rotation target.
    if kind == "periodic":
        rr = (0.0, 0.08 if smoke else 0.10)

    return {
        "selection_mode": "targeted_salient",
        "amplitude": amp,
        "translation_fraction_range": tr,
        "rotation_magnitude_range": rr,
        "min_shape_excursion": shape,
        "min_endpoint_shape_change": 0.06 if kind == "bvp" else 0.0,
        "strict_salience": True,
        "candidate_pool": 180 if smoke else 500,
    }
