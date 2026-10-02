from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
import copy
import hashlib
import math
import random

import numpy as np
import torch
from torch import nn

ClosureMode = Literal["open", "measurement", "periodic"]
ConditioningCase = Literal["configuration", "observation"]


@dataclass
class SplineCurveConfig:
    n_ctrl: int = 18
    degree: int = 3
    n_samples: int = 200
    n_steps: int = 30
    min_knot_span: float = 0.003

    mass_control: float = 1.0
    mass_knot: float = 0.25
    mass_pose_xy: float = 1.0
    mass_pose_theta: float = 0.45
    damping_control: float = 0.45
    damping_knot: float = 0.15
    damping_pose_xy: float = 0.65
    damping_pose_theta: float = 0.35

    closure_tolerance: float = 0.04
    closure_energy_scale: float = 0.0025
    closure_damping_scale: float = 0.05
    tangent_closure_weight: float = 0.15

    observation_points: int = 600
    source_radius: float = 0.6

    feasible_alpha_min: float = 0.15
    feasible_alpha_mid: float = 0.38
    feasible_alpha_max: float = 0.68
    hard_alpha_min: float = 0.88
    hard_alpha_max: float = 1.00

    # v29: train through a post-command settling interval.
    training_horizon_multiplier: float = 1.75
    tail_start: float = 1.0
    tail_samples: int = 5

    # v29: continuation fitter preserves a material branch.
    continuation_steps: int = 5
    continuation_branch_weight: float = 2.5e-3
    continuation_anchor_weight: float = 2.0e-2

    # v30: OT/Sinkhorn target geometry.  The learned moving rest path remains
    # a trajectory prior; endpoint geometry is no longer supervised by direct
    # control-point/knot-vector quadratic matching.
    geometry_objective: str = "sinkhorn"
    sinkhorn_eps_start: float = 0.05
    sinkhorn_eps_end: float = 0.015
    sinkhorn_n_iter: int = 10
    sinkhorn_force_n_iter: int = 4
    sinkhorn_symmetric: bool = True
    sinkhorn_source_points: int = 96
    sinkhorn_target_points: int = 192
    sinkhorn_force_source_points: int = 48
    sinkhorn_force_target_points: int = 96
    sinkhorn_loss_weight: float = 110.0
    sinkhorn_potential_weight: float = 0.45
    use_sinkhorn_potential: bool = True
    sinkhorn_force_backprop: bool = False

    # v32: robust target-match search from weak supervision only (point clouds /
    # geometric observations).  A target spline is selected by multi-start OT
    # fitting with smoothness/repulsion regularization, instead of trusting one
    # possibly self-intersecting least-squares fit.
    fit_objective: str = "sinkhorn"
    fit_sinkhorn_eps: float = 0.02
    fit_sinkhorn_n_iter: int = 18
    fit_multistart: int = 4
    fit_multistart_noise: float = 0.04
    fit_curve_repulsion_weight: float = 1.5e-3
    fit_curve_smoothness_weight: float = 2.5e-3
    fit_polygon_smoothness_weight: float = 8.0e-4
    fit_area_prior_weight: float = 4.0e-4
    fit_branch_weight: float = 2.5e-3
    fit_reject_self_intersection: bool = True
    fit_self_intersection_penalty: float = 4.0

    # v33: for the synthetic legacy crescent family we know the generating
    # circle-difference geometry.  Use its ordered simple boundary to construct
    # a safe teacher configuration; retain point-cloud search as a generic
    # fallback for real unordered observations.
    target_fit_protocol: str = "ordered_boundary"  # ordered_boundary | point_cloud_search
    ordered_fit_curve_weight: float = 1.0
    ordered_fit_ot_weight: float = 0.0
    ordered_fit_smoothness_weight: float = 1.0e-4

    # v34: the initial circle is fitted as a *materially phased* closed spline
    # with uniform knot spans.  This removes the visually isolated seam slave
    # control point produced by projecting a regular polygon after the fact.
    source_fit_iterations: int = 260
    source_fit_lr: float = 0.03
    source_fit_smoothness_weight: float = 1.0e-4

    # v34 target/path validation.  Synthetic targets and their conventional
    # teacher homotopies must stay in one simple orientation class.
    validate_teacher_path: bool = True
    teacher_area_floor_ratio: float = 0.08

    @property
    def dense_obs_samples(self) -> int:
        return self.observation_points

    @property
    def n_spans(self) -> int:
        return self.n_ctrl - self.degree

    @property
    def n_state(self) -> int:
        return 2 * self.n_ctrl + self.n_spans + 3


@dataclass
class SplineCurveTask:
    mode: ClosureMode
    # Intrinsic/body-frame spline coordinates.
    P0: torch.Tensor
    eta0: torch.Tensor
    P_target: torch.Tensor
    eta_target: torch.Tensor
    # Explicit global SE(2) factor [tx, ty, theta].
    pose0: torch.Tensor
    pose_target: torch.Tensor
    # Intrinsic fitted target spline samples in the body frame.
    target_samples: torch.Tensor
    task_id: int
    hard_ood: bool = False
    target_family: str = "legacy_crescent_point_cloud"
    representation_floor: float = 0.0
    target_hash: str = ""
    conditioning_case: ConditioningCase = "configuration"
    # Exact empirical target measure in world coordinates.
    observation_samples: torch.Tensor | None = None
    target_alpha: float = 0.0
    target_params: dict[str, float] | None = None

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "SplineCurveTask":
        dtype = self.P0.dtype if dtype is None else dtype
        return SplineCurveTask(
            mode=self.mode,
            P0=self.P0.to(device=device, dtype=dtype),
            eta0=self.eta0.to(device=device, dtype=dtype),
            P_target=self.P_target.to(device=device, dtype=dtype),
            eta_target=self.eta_target.to(device=device, dtype=dtype),
            pose0=self.pose0.to(device=device, dtype=dtype),
            pose_target=self.pose_target.to(device=device, dtype=dtype),
            target_samples=self.target_samples.to(device=device, dtype=dtype),
            task_id=self.task_id,
            hard_ood=self.hard_ood,
            target_family=self.target_family,
            representation_floor=self.representation_floor,
            target_hash=self.target_hash,
            conditioning_case=self.conditioning_case,
            observation_samples=None if self.observation_samples is None else self.observation_samples.to(device=device, dtype=dtype),
            target_alpha=self.target_alpha,
            target_params=None if self.target_params is None else dict(self.target_params),
        )


def mode_id(mode: ClosureMode) -> int:
    return {"open": 0, "measurement": 1, "periodic": 2}[mode]


def conditioning_id(case: ConditioningCase) -> int:
    return {"configuration": 0, "observation": 1}[case]


# -----------------------------------------------------------------------------
# B-spline utilities


def knot_spans_from_logits(eta: torch.Tensor, cfg: SplineCurveConfig) -> torch.Tensor:
    n = cfg.n_spans
    if cfg.min_knot_span * n >= 1.0:
        raise ValueError("min_knot_span is too large for the number of spans")
    slack = 1.0 - cfg.min_knot_span * n
    return cfg.min_knot_span + slack * torch.softmax(eta, dim=-1)


def full_knot_vector(eta: torch.Tensor, cfg: SplineCurveConfig) -> torch.Tensor:
    spans = knot_spans_from_logits(eta, cfg)
    internal = torch.cumsum(spans, dim=-1)[..., :-1]
    z = torch.zeros((*eta.shape[:-1], cfg.degree + 1), dtype=eta.dtype, device=eta.device)
    o = torch.ones((*eta.shape[:-1], cfg.degree + 1), dtype=eta.dtype, device=eta.device)
    return torch.cat([z, internal, o], dim=-1)


def _basis_degree(samples: torch.Tensor, knots: torch.Tensor, degree: int, n_ctrl: int) -> torch.Tensor:
    u = samples
    batch_shape = knots.shape[:-1]
    while u.ndim < len(batch_shape) + 1:
        u = u.unsqueeze(0)
    u = u.expand(*batch_shape, samples.shape[0])
    left = knots[..., :-1].unsqueeze(-2)
    right = knots[..., 1:].unsqueeze(-2)
    uu = u.unsqueeze(-1)
    N = ((uu >= left) & (uu < right)).to(knots.dtype)
    eps = torch.finfo(knots.dtype).eps * 16
    for k in range(1, degree + 1):
        count = n_ctrl + degree - k
        a_num = uu[..., :count] - knots[..., :count].unsqueeze(-2)
        a_den = knots[..., k:k + count] - knots[..., :count]
        b_num = knots[..., k + 1:k + 1 + count].unsqueeze(-2) - uu[..., :count]
        b_den = knots[..., k + 1:k + 1 + count] - knots[..., 1:1 + count]
        a = torch.where(a_den.abs().unsqueeze(-2) > eps, a_num / a_den.unsqueeze(-2), torch.zeros_like(a_num))
        b = torch.where(b_den.abs().unsqueeze(-2) > eps, b_num / b_den.unsqueeze(-2), torch.zeros_like(b_num))
        N = a * N[..., :count] + b * N[..., 1:count + 1]
    return N[..., :n_ctrl]


def bspline_basis_torch(samples: torch.Tensor, eta: torch.Tensor, cfg: SplineCurveConfig) -> tuple[torch.Tensor, torch.Tensor]:
    knots = full_knot_vector(eta, cfg)
    B = _basis_degree(samples, knots, cfg.degree, cfg.n_ctrl)
    end_mask = torch.isclose(samples, torch.ones((), dtype=samples.dtype, device=samples.device), atol=1e-12, rtol=0.0)
    if end_mask.any():
        Bend = torch.zeros((*B.shape[:-2], int(end_mask.sum().item()), cfg.n_ctrl), dtype=B.dtype, device=B.device)
        Bend[..., -1] = 1.0
        B = B.clone()
        B[..., end_mask, :] = Bend
    if cfg.degree == 0:
        return B, torch.zeros_like(B)
    Bm = _basis_degree(samples, knots, cfg.degree - 1, cfg.n_ctrl + 1)
    p = cfg.degree
    eps = torch.finfo(knots.dtype).eps * 16
    d_terms = []
    for i in range(cfg.n_ctrl):
        den1 = knots[..., i + p] - knots[..., i]
        den2 = knots[..., i + p + 1] - knots[..., i + 1]
        t1 = torch.where(den1.abs() > eps, p * Bm[..., i] / den1.unsqueeze(-1), torch.zeros_like(Bm[..., i]))
        t2 = torch.where(den2.abs() > eps, p * Bm[..., i + 1] / den2.unsqueeze(-1), torch.zeros_like(Bm[..., i + 1]))
        d_terms.append(t1 - t2)
    dB = torch.stack(d_terms, dim=-1)
    start_mask = torch.isclose(samples, torch.zeros((), dtype=samples.dtype, device=samples.device), atol=1e-12, rtol=0.0)
    spans = knot_spans_from_logits(eta, cfg)
    if start_mask.any():
        dB = dB.clone()
        dB[..., start_mask, :] = 0.0
        a = cfg.degree / spans[..., 0].clamp_min(1e-12)
        dB[..., start_mask, 0] = -a.unsqueeze(-1)
        dB[..., start_mask, 1] = a.unsqueeze(-1)
    if end_mask.any():
        dB = dB.clone()
        dB[..., end_mask, :] = 0.0
        a = cfg.degree / spans[..., -1].clamp_min(1e-12)
        dB[..., end_mask, -2] = -a.unsqueeze(-1)
        dB[..., end_mask, -1] = a.unsqueeze(-1)
    return B, dB


def evaluate_curve(P: torch.Tensor, eta: torch.Tensor, cfg: SplineCurveConfig, samples: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if samples is None:
        samples = torch.linspace(0.0, 1.0, cfg.n_samples, dtype=P.dtype, device=P.device)
    B, dB = bspline_basis_torch(samples, eta, cfg)
    C = torch.einsum("...si,...id->...sd", B, P)
    dC = torch.einsum("...si,...id->...sd", dB, P)
    return C, dC


def periodic_project_controls(P: torch.Tensor, eta: torch.Tensor, cfg: SplineCurveConfig) -> torch.Tensor:
    """Project clamped control coordinates onto the endpoint/C1 closure manifold.

    The last two entries are dependent seam coordinates for the clamped spline
    representation.  They should therefore not be interpreted as independent
    material control points.
    """
    if cfg.degree < 1:
        Q = P.clone()
        Q[..., -1, :] = Q[..., 0, :]
        return Q
    Q = P.clone()
    spans = knot_spans_from_logits(eta, cfg)
    first = spans[..., 0].unsqueeze(-1)
    last = spans[..., -1].unsqueeze(-1)
    Q[..., -1, :] = Q[..., 0, :]
    ratio = last / first.clamp_min(1e-9)
    Q[..., -2, :] = Q[..., -1, :] - ratio * (Q[..., 1, :] - Q[..., 0, :])
    return Q


def periodic_project_velocity(
    P: torch.Tensor,
    eta: torch.Tensor,
    vP: torch.Tensor,
    vEta: torch.Tensor,
    cfg: SplineCurveConfig,
) -> torch.Tensor:
    """Project a velocity into the tangent space of the periodic constraint.

    ``periodic_project_controls(vP, eta, cfg)`` is only correct when the knot
    logits are frozen.  When ``eta`` moves, the endpoint tangent constraint also
    contains ``d(last_span/first_span)/dt``.  Omitting that term injects a seam
    impulse and can make a deformation appear to flip before settling.
    """
    V = vP.clone()
    V[..., -1, :] = V[..., 0, :]
    if cfg.degree < 1:
        return V

    n = cfg.n_spans
    slack = 1.0 - cfg.min_knot_span * n
    w = torch.softmax(eta, dim=-1)
    spans = cfg.min_knot_span + slack * w
    sf = spans[..., 0].clamp_min(1e-9)
    sl = spans[..., -1]
    ratio = sl / sf

    # Jacobians of first/last span with respect to the knot logits.
    eye_first = torch.zeros_like(w); eye_first[..., 0] = 1.0
    eye_last = torch.zeros_like(w); eye_last[..., -1] = 1.0
    dsf = slack * w[..., 0:1] * (eye_first - w)
    dsl = slack * w[..., -1:] * (eye_last - w)
    dr = dsl / sf.unsqueeze(-1) - (sl / sf.pow(2)).unsqueeze(-1) * dsf
    rdot = torch.sum(dr * vEta, dim=-1)

    V[..., -2, :] = (
        V[..., -1, :]
        - ratio.unsqueeze(-1) * (V[..., 1, :] - V[..., 0, :])
        - rdot.unsqueeze(-1) * (P[..., 1, :] - P[..., 0, :])
    )
    return V


def closure_measurement(P: torch.Tensor, eta: torch.Tensor, cfg: SplineCurveConfig) -> tuple[torch.Tensor, torch.Tensor]:
    gap = P[..., -1, :] - P[..., 0, :]
    spans = knot_spans_from_logits(eta, cfg)
    a0 = cfg.degree / spans[..., 0].clamp_min(1e-8)
    a1 = cfg.degree / spans[..., -1].clamp_min(1e-8)
    d0 = a0.unsqueeze(-1) * (P[..., 1, :] - P[..., 0, :])
    d1 = a1.unsqueeze(-1) * (P[..., -1, :] - P[..., -2, :])
    t0 = d0 / d0.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    t1 = d1 / d1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return gap, t1 - t0


# -----------------------------------------------------------------------------
# SE(2) pose factorization


def wrap_angle(theta: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(theta), torch.cos(theta))


def apply_pose(X: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """Apply pose [...,3] = (tx,ty,theta) to body-frame points [...,N,2]."""
    th = pose[..., 2]
    c = torch.cos(th)
    s = torch.sin(th)
    R = torch.stack([torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)], dim=-2)
    Y = torch.einsum("...ij,...nj->...ni", R, X)
    return Y + pose[..., None, :2]


def inverse_pose(X: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    th = -pose[..., 2]
    c = torch.cos(th)
    s = torch.sin(th)
    R = torch.stack([torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)], dim=-2)
    return torch.einsum("...ij,...nj->...ni", R, X - pose[..., None, :2])


def pose_errors(pose: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    trans = torch.linalg.norm(pose[..., :2] - target[..., :2], dim=-1)
    rot = wrap_angle(pose[..., 2] - target[..., 2]).abs()
    return trans, rot


# -----------------------------------------------------------------------------
# Geometry / point-cloud targets


def _canonicalize_curve(C: torch.Tensor) -> torch.Tensor:
    c = C.mean(dim=-2, keepdim=True)
    X = C - c
    e = X[..., -1, :] - X[..., 0, :]
    closed = e.norm(dim=-1, keepdim=True) < 1e-5
    j = max(1, C.shape[-2] // 4)
    e_alt = X[..., j, :] - X[..., 0, :]
    e = torch.where(closed, e_alt, e)
    ang = torch.atan2(e[..., 1], e[..., 0])
    ca, sa = torch.cos(-ang), torch.sin(-ang)
    R = torch.stack([torch.stack([ca, -sa], dim=-1), torch.stack([sa, ca], dim=-1)], dim=-2)
    return torch.einsum("...ij,...sj->...si", R, X)


def resample_curve_samples(C: torch.Tensor, n_out: int) -> torch.Tensor:
    if C.shape[-2] == n_out:
        return C
    X = C
    seg = X[..., 1:, :] - X[..., :-1, :]
    seg_len = seg.norm(dim=-1)
    zero = torch.zeros((*X.shape[:-2], 1), dtype=X.dtype, device=X.device)
    arc = torch.cat([zero, torch.cumsum(seg_len, dim=-1)], dim=-1)
    total = arc[..., -1:].clamp_min(1e-8)
    tgt = torch.linspace(0.0, 1.0, n_out, dtype=X.dtype, device=X.device) * total
    outs = []
    for u in tgt:
        idx = (arc <= u).to(torch.int64).sum(dim=-1) - 1
        idx = idx.clamp(0, X.shape[-2] - 2)
        a0 = torch.gather(arc, -1, idx.unsqueeze(-1)).squeeze(-1)
        a1 = torch.gather(arc, -1, (idx + 1).unsqueeze(-1)).squeeze(-1)
        p0 = torch.gather(X, -2, idx[..., None, None].expand(*X.shape[:-2], 1, X.shape[-1])).squeeze(-2)
        p1 = torch.gather(X, -2, (idx + 1)[..., None, None].expand(*X.shape[:-2], 1, X.shape[-1])).squeeze(-2)
        w = ((u - a0) / (a1 - a0).clamp_min(1e-8))[..., None]
        outs.append((1.0 - w) * p0 + w * p1)
    return torch.stack(outs, dim=-2)


def curve_rms(pred: torch.Tensor, target: torch.Tensor, pose_invariant: bool = True) -> torch.Tensor:
    if pred.shape[-2] != target.shape[-2]:
        target = resample_curve_samples(target, pred.shape[-2])
    if pose_invariant:
        pred = _canonicalize_curve(pred)
        target = _canonicalize_curve(target)
    return torch.sqrt(torch.mean((pred - target) ** 2, dim=(-2, -1)).clamp_min(0.0))


def point_cloud_chamfer(curve: torch.Tensor, cloud: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(curve, cloud)
    return torch.sqrt(0.5 * (d.min(dim=-1).values.pow(2).mean() + d.min(dim=-2).values.pow(2).mean()).clamp_min(0.0))


def _deterministic_subsample(points: torch.Tensor, n_points: int) -> torch.Tensor:
    """Even deterministic subsampling for OT costs.

    The old crescent target contains 600 points.  Computing a differentiable
    Sinkhorn solve at every pH integration step on all 600 points is wasteful,
    especially because the pH force requires higher-order derivatives during
    training.  Evenly spaced samples preserve the empirical measure well enough
    for the force while keeping the full cloud available for diagnostics.
    """
    n = points.shape[-2]
    if n_points <= 0 or n <= n_points:
        return points
    idx = torch.linspace(0, n - 1, n_points, device=points.device).round().long()
    return points.index_select(-2, idx)


def _pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x.unsqueeze(-2) - y.unsqueeze(-3)).pow(2).sum(dim=-1)


def _sinkhorn_log_cost(x: torch.Tensor, y: torch.Tensor, eps: float, n_iter: int) -> torch.Tensor:
    """Finite-iteration entropy-regularized OT dual value.

    This mirrors the log-domain implementation in the uploaded darboux_igph
    repository, using uniform empirical masses.  x and y are two-dimensional
    point sets with shapes (n,d) and (m,d).
    """
    if eps <= 0.0:
        raise ValueError(f"sinkhorn eps must be positive; got {eps}")
    if n_iter <= 0:
        raise ValueError(f"sinkhorn n_iter must be positive; got {n_iter}")
    C = _pairwise_sq_dist(x, y)
    n, m = C.shape[-2], C.shape[-1]
    log_a = torch.full((n,), -math.log(float(n)), dtype=x.dtype, device=x.device)
    log_b = torch.full((m,), -math.log(float(m)), dtype=y.dtype, device=y.device)
    f = torch.zeros(n, dtype=x.dtype, device=x.device)
    g = torch.zeros(m, dtype=y.dtype, device=y.device)
    for _ in range(n_iter):
        M = (-C + g.unsqueeze(0)) / eps
        f = -eps * torch.logsumexp(M + log_b.unsqueeze(0), dim=1)
        M = (-C + f.unsqueeze(1)) / eps
        g = -eps * torch.logsumexp(M + log_a.unsqueeze(1), dim=0)
    return (f * log_a.exp()).sum() + (g * log_b.exp()).sum()


def sinkhorn_divergence_points(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float,
    n_iter: int,
    symmetric_finite_iter: bool = True,
    source_points: int = 0,
    target_points: int = 0,
) -> torch.Tensor:
    """Debiased Sinkhorn divergence between two empirical point measures.

    This is the geometry objective used by v30.  It is invariant to point order
    and therefore appropriate for the old-repo empirical crescent cloud.
    """
    x = _deterministic_subsample(x, source_points)
    y = _deterministic_subsample(y, target_points)
    xy = _sinkhorn_log_cost(x, y, eps, n_iter)
    if symmetric_finite_iter:
        yx = _sinkhorn_log_cost(y, x, eps, n_iter)
        cross = 0.5 * (xy + yx)
    else:
        cross = xy
    xx = _sinkhorn_log_cost(x, x, eps, n_iter)
    yy = _sinkhorn_log_cost(y, y, eps, n_iter)
    # Finite iterations may yield tiny negative values near equality.
    return (cross - 0.5 * xx - 0.5 * yy).clamp_min(0.0)


def sinkhorn_state_potential(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float,
    n_iter: int,
    symmetric_finite_iter: bool = True,
    source_points: int = 0,
    target_points: int = 0,
) -> torch.Tensor:
    """State-dependent part of Sinkhorn divergence for Hamiltonian forces.

    The target self-cost S(y,y) is constant with respect to the spline state,
    so it is omitted here.  The resulting gradient is exactly the gradient of
    the full finite-iteration divergence while saving one OT solve per pH step.
    """
    x = _deterministic_subsample(x, source_points)
    y = _deterministic_subsample(y, target_points)
    xy = _sinkhorn_log_cost(x, y, eps, n_iter)
    if symmetric_finite_iter:
        yx = _sinkhorn_log_cost(y, x, eps, n_iter)
        cross = 0.5 * (xy + yx)
    else:
        cross = xy
    xx = _sinkhorn_log_cost(x, x, eps, n_iter)
    return cross - 0.5 * xx


def sinkhorn_eps_at_progress(cfg: SplineCurveConfig, progress: float) -> float:
    a = max(0.0, min(1.0, float(progress)))
    return float(cfg.sinkhorn_eps_start + a * (cfg.sinkhorn_eps_end - cfg.sinkhorn_eps_start))


def target_hash(samples: torch.Tensor) -> str:
    arr = samples.detach().cpu().numpy().astype(np.float64)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def legacy_crescent_point_cloud(
    n_points: int = 600,
    outer_radius: float = 1.0,
    inner_radius: float = 0.9,
    offset: float = 0.35,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    device = torch.device(device)
    n_outer = n_points * 2
    n_inner = n_points * 2
    theta_outer = torch.linspace(0.0, 2.0 * math.pi, n_outer + 1, dtype=dtype, device=device)[:-1]
    outer = torch.stack([outer_radius * torch.cos(theta_outer), outer_radius * torch.sin(theta_outer)], dim=-1)
    mask_outer = ((outer[:, 0] - offset) ** 2 + outer[:, 1] ** 2).sqrt() >= inner_radius
    theta_inner = torch.linspace(0.0, 2.0 * math.pi, n_inner + 1, dtype=dtype, device=device)[:-1]
    inner = torch.stack([offset + inner_radius * torch.cos(theta_inner), inner_radius * torch.sin(theta_inner)], dim=-1)
    mask_inner = (inner[:, 0] ** 2 + inner[:, 1] ** 2).sqrt() <= outer_radius
    boundary = torch.cat([outer[mask_outer], inner[mask_inner]], dim=0)
    idx = torch.linspace(0, boundary.shape[0] - 1, n_points, device=device).long()
    return boundary[idx]


def legacy_crescent_family_point_cloud(
    alpha: float,
    cfg: SplineCurveConfig,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    a = float(np.clip(alpha, 0.0, 1.0))
    outer = 1.0
    inner = 0.62 + a * (0.90 - 0.62)
    offset = 0.58 + a * (0.35 - 0.58)
    cloud = legacy_crescent_point_cloud(
        cfg.observation_points,
        outer_radius=outer,
        inner_radius=inner,
        offset=offset,
        dtype=dtype,
        device=device,
    )
    # Body frame: remove translation induced by the asymmetric crescent measure.
    cloud = cloud - cloud.mean(dim=-2, keepdim=True)
    return cloud, {"outer_radius": outer, "inner_radius": inner, "offset": offset}


def circle_point_cloud(n_points: int, radius: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    theta = torch.linspace(0.0, 2.0 * math.pi, n_points + 1, dtype=dtype, device=device)[:-1]
    return torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=-1)


def _sample_target_pose(rng: np.random.Generator, rot_max_deg: float, translation: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    theta = math.radians(float(rng.uniform(-rot_max_deg, rot_max_deg)))
    tx = float(rng.uniform(-translation, translation))
    ty = float(rng.uniform(-translation, translation))
    return torch.tensor([tx, ty, theta], dtype=dtype, device=device)


def _base_control_polygon(cfg: SplineCurveConfig, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    # Explicit material seam at the unique leftmost point of the source circle.
    th = torch.linspace(math.pi, math.pi + 2.0 * math.pi, cfg.n_ctrl, dtype=dtype, device=device)
    return torch.stack([cfg.source_radius * torch.cos(th), cfg.source_radius * torch.sin(th)], dim=-1)


# -----------------------------------------------------------------------------
# Configuration fitting with material continuation


_SOURCE_CIRCLE_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_FAMILY_FIT_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]] = {}


def _cloud_material_anchor(cloud: torch.Tensor) -> torch.Tensor:
    # A deterministic point-cloud landmark fixes periodic phase without requiring ordering.
    return cloud[torch.argmin(cloud[:, 0])]


def _orientation_np(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_intersect_np(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, eps: float = 1e-12) -> bool:
    def on_segment(p, q, r):
        return (min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps and min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps)
    o1 = _orientation_np(a, b, c)
    o2 = _orientation_np(a, b, d)
    o3 = _orientation_np(c, d, a)
    o4 = _orientation_np(c, d, b)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True
    if abs(o1) <= eps and on_segment(a, c, b):
        return True
    if abs(o2) <= eps and on_segment(a, d, b):
        return True
    if abs(o3) <= eps and on_segment(c, a, d):
        return True
    if abs(o4) <= eps and on_segment(c, b, d):
        return True
    return False


def polyline_self_intersections(points: torch.Tensor, *, closed: bool = True) -> int:
    pts = points.detach().cpu().numpy()
    if closed and pts.shape[0] >= 2 and np.linalg.norm(pts[0] - pts[-1]) < 1e-9:
        pts = pts[:-1]
    if pts.shape[0] < 4:
        return 0
    n = pts.shape[0]
    seg_count = n if closed else n - 1
    count = 0
    for i in range(seg_count):
        a = pts[i]
        b = pts[(i + 1) % n] if closed else pts[i + 1]
        for j in range(i + 1, seg_count):
            if j == i + 1:
                continue
            if closed and i == 0 and j == seg_count - 1:
                continue
            c = pts[j]
            d = pts[(j + 1) % n] if closed else pts[j + 1]
            if _segments_intersect_np(a, b, c, d):
                count += 1
    return count


def signed_curve_area(points: torch.Tensor) -> torch.Tensor:
    """Signed polygonal area for a sampled closed curve (batch-safe)."""
    x = points[..., :, 0]
    y = points[..., :, 1]
    return 0.5 * torch.sum(x * torch.roll(y, shifts=-1, dims=-1) - y * torch.roll(x, shifts=-1, dims=-1), dim=-1)


def _fit_geometry_loss(C: torch.Tensor, cloud: torch.Tensor, cfg: SplineCurveConfig) -> torch.Tensor:
    if cfg.fit_objective == "chamfer":
        return point_cloud_chamfer(C, cloud)
    if cfg.fit_objective != "sinkhorn":
        raise ValueError(f"unknown fit_objective={cfg.fit_objective!r}")
    return sinkhorn_divergence_points(
        C, cloud,
        eps=cfg.fit_sinkhorn_eps,
        n_iter=cfg.fit_sinkhorn_n_iter,
        symmetric_finite_iter=True,
        source_points=min(cfg.sinkhorn_target_points, C.shape[0]),
        target_points=min(cfg.observation_points, cloud.shape[0]),
    )


def _curve_smoothness(C: torch.Tensor) -> torch.Tensor:
    prev = torch.roll(C, shifts=1, dims=0)
    nxt = torch.roll(C, shifts=-1, dims=0)
    return torch.mean((nxt - 2.0 * C + prev) ** 2)


def _polygon_smoothness(P: torch.Tensor) -> torch.Tensor:
    prev = torch.roll(P, shifts=1, dims=0)
    nxt = torch.roll(P, shifts=-1, dims=0)
    return torch.mean((nxt - 2.0 * P + prev) ** 2)


def _curve_repulsion(C: torch.Tensor, *, sigma: float = 0.09, min_sep: int = 5) -> torch.Tensor:
    n = C.shape[0]
    if n <= 2 * min_sep + 1:
        return torch.zeros((), dtype=C.dtype, device=C.device)
    idx = torch.arange(n, device=C.device)
    diff = torch.abs(idx[:, None] - idx[None, :])
    diff = torch.minimum(diff, n - diff)
    mask = diff > min_sep
    d2 = torch.sum((C[:, None, :] - C[None, :, :]) ** 2, dim=-1)
    rep = torch.exp(-d2 / (2.0 * sigma * sigma))
    return rep[mask].mean()


def _single_target_fit(
    cloud: torch.Tensor,
    cfg: SplineCurveConfig,
    *,
    seed: int,
    iterations: int,
    lr: float,
    init_P: torch.Tensor | None,
    init_eta: torch.Tensor | None,
    branch_P: torch.Tensor | None,
    branch_eta: torch.Tensor | None,
    noise_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float | int]]:
    device, dtype = cloud.device, cloud.dtype
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    if init_P is None:
        P0 = _base_control_polygon(cfg, dtype, device)
        P_init = P0 + noise_scale * torch.randn(cfg.n_ctrl, 2, generator=gen, dtype=dtype, device=device)
    else:
        P_init = init_P.detach().clone()
        if noise_scale > 0:
            P_init = P_init + noise_scale * torch.randn(cfg.n_ctrl, 2, generator=gen, dtype=dtype, device=device)
    if init_eta is None:
        eta_init = 0.03 * torch.randn(cfg.n_spans, generator=gen, dtype=dtype, device=device)
    else:
        eta_init = init_eta.detach().clone()
        if noise_scale > 0:
            eta_init = eta_init + 0.5 * noise_scale * torch.randn(cfg.n_spans, generator=gen, dtype=dtype, device=device)
    P = P_init.detach().clone().requires_grad_(True)
    eta = eta_init.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([P, eta], lr=lr)
    u = torch.linspace(0.0, 1.0, cfg.n_samples, dtype=dtype, device=device)
    anchor = _cloud_material_anchor(cloud)
    for _ in range(iterations):
        opt.zero_grad(set_to_none=True)
        Puse = periodic_project_controls(P, eta, cfg)
        C, _ = evaluate_curve(Puse, eta, cfg, u)
        geom = _fit_geometry_loss(C, cloud, cfg)
        spans = knot_spans_from_logits(eta, cfg)
        knot_reg = 1e-4 * torch.mean((torch.log(spans) - torch.log(spans).mean()) ** 2)
        anchor_loss = torch.mean((C[0] - anchor) ** 2)
        branch = torch.zeros((), dtype=dtype, device=device)
        if branch_P is not None:
            branch = branch + torch.mean((Puse - branch_P) ** 2)
        if branch_eta is not None:
            branch = branch + 0.2 * torch.mean((knot_spans_from_logits(eta, cfg) - knot_spans_from_logits(branch_eta, cfg)) ** 2)
        smooth = cfg.fit_curve_smoothness_weight * _curve_smoothness(C) + cfg.fit_polygon_smoothness_weight * _polygon_smoothness(Puse)
        repulsion = cfg.fit_curve_repulsion_weight * _curve_repulsion(C)
        loss = geom + knot_reg + cfg.continuation_anchor_weight * anchor_loss + cfg.fit_branch_weight * branch + smooth + repulsion
        loss.backward()
        torch.nn.utils.clip_grad_norm_([P, eta], 20.0)
        opt.step()
    with torch.no_grad():
        Pfit = periodic_project_controls(P, eta, cfg)
        Cfit, _ = evaluate_curve(Pfit, eta, cfg, u)
        floor = float(point_cloud_chamfer(Cfit, cloud).item())
        geom = float(_fit_geometry_loss(Cfit, cloud, cfg).item())
        curve_selfx = int(polyline_self_intersections(Cfit, closed=True))
        poly_selfx = int(polyline_self_intersections(Pfit, closed=True))
        stats = {
            "floor": float(floor),
            "geom": geom,
            "curve_selfx": curve_selfx,
            "poly_selfx": poly_selfx,
            "score": float(geom + cfg.fit_self_intersection_penalty * (curve_selfx + poly_selfx)),
        }
    return Pfit.detach(), eta.detach(), Cfit.detach(), stats


def fit_point_cloud_target(
    cloud: torch.Tensor,
    cfg: SplineCurveConfig,
    *,
    seed: int,
    iterations: int = 300,
    lr: float = 0.025,
    init_P: torch.Tensor | None = None,
    init_eta: torch.Tensor | None = None,
    branch_P: torch.Tensor | None = None,
    branch_eta: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Fit a periodic adaptive spline while preserving one material branch.

    v32 uses a weak-supervision target search: multi-start fitting against the
    empirical target measure with OT/Chamfer, plus smoothness/repulsion priors.
    The selected target is the best non-self-intersecting candidate whenever
    possible, which is more reliable than a single least-squares fit.
    """
    candidates = []
    nstart = max(1, int(cfg.fit_multistart))
    for k in range(nstart):
        noise = 0.0 if k == 0 else float(cfg.fit_multistart_noise) * (0.7 + 0.3 * k)
        cand = _single_target_fit(
            cloud, cfg, seed=seed + 997 * k, iterations=iterations, lr=lr,
            init_P=init_P, init_eta=init_eta, branch_P=branch_P, branch_eta=branch_eta,
            noise_scale=noise,
        )
        candidates.append(cand)
    if cfg.fit_reject_self_intersection:
        valid = [c for c in candidates if (c[3]["curve_selfx"] + c[3]["poly_selfx"]) == 0]
        if valid:
            candidates = valid
    best = min(candidates, key=lambda c: (c[3]["score"], c[3]["floor"]))
    return best[0], best[1], best[2], float(best[3]["floor"])


def ordered_circle_boundary(
    cfg: SplineCurveConfig,
    *,
    dtype: torch.dtype,
    device: torch.device,
    n_points: int | None = None,
) -> torch.Tensor:
    """Closed source circle with the material seam at the leftmost point.

    The parameter advances from the leftmost point downward, matching the
    orientation used by the corrected crescent boundary below.
    """
    n = int(n_points or cfg.n_samples)
    u = torch.linspace(0.0, 1.0, n, dtype=dtype, device=device)
    th = math.pi + 2.0 * math.pi * u
    return torch.stack([cfg.source_radius * torch.cos(th), cfg.source_radius * torch.sin(th)], dim=-1)


def _fit_source_circle(cfg: SplineCurveConfig, dtype: torch.dtype, device: torch.device, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    # Include every source-construction knob in the cache key.  v33 omitted the
    # protocol, which allowed a point-cloud-fitted source to leak into an ordered
    # run in the same process.
    key = (
        cfg.n_ctrl, cfg.degree, cfg.n_samples, cfg.source_radius, cfg.min_knot_span,
        cfg.target_fit_protocol, cfg.source_fit_iterations, cfg.source_fit_lr,
        cfg.source_fit_smoothness_weight, str(dtype), str(device),
    )
    if key in _SOURCE_CIRCLE_CACHE:
        P, eta = _SOURCE_CIRCLE_CACHE[key]
        return P.clone(), eta.clone()

    if cfg.target_fit_protocol == "ordered_boundary":
        # Keep the source parameterization physically clean: uniform knot spans
        # and a pointwise material fit to the exact circle.  Directly projecting a
        # regular polygon makes only P[-2] jump off the circle because it is the
        # dependent C1 seam coordinate.
        eta = torch.zeros(cfg.n_spans, dtype=dtype, device=device)
        target = ordered_circle_boundary(cfg, dtype=dtype, device=device, n_points=cfg.n_samples)
        Pvar = _base_control_polygon(cfg, dtype, device).detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([Pvar], lr=float(cfg.source_fit_lr))
        for _ in range(max(20, int(cfg.source_fit_iterations))):
            opt.zero_grad(set_to_none=True)
            Puse = periodic_project_controls(Pvar, eta, cfg)
            C, _ = evaluate_curve(Puse, eta, cfg)
            loss = torch.mean((C - target) ** 2)
            loss = loss + cfg.source_fit_smoothness_weight * (
                _polygon_smoothness(Puse) + _curve_smoothness(C)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([Pvar], 20.0)
            opt.step()
        with torch.no_grad():
            P = periodic_project_controls(Pvar, eta, cfg)
            # The source target is exactly reflection-symmetric about the x-axis.
            # Enforce that symmetry after optimization so the two seam-adjacent
            # dependent controls are a matched pair rather than leaving one
            # visually displaced by optimizer noise.
            Psym = P.clone()
            for i in range((cfg.n_ctrl + 1) // 2):
                j = cfg.n_ctrl - 1 - i
                xavg = 0.5 * (Psym[i, 0] + Psym[j, 0])
                yanti = 0.5 * (Psym[i, 1] - Psym[j, 1])
                Psym[i, 0] = xavg
                Psym[j, 0] = xavg
                Psym[i, 1] = yanti
                Psym[j, 1] = -yanti
            P = periodic_project_controls(Psym, eta, cfg).detach()
    else:
        cloud = circle_point_cloud(cfg.observation_points, cfg.source_radius, dtype, device)
        P, eta, _, _ = fit_point_cloud_target(cloud, cfg, seed=seed, iterations=100, lr=0.03)

    _SOURCE_CIRCLE_CACHE[key] = (P.detach().clone(), eta.detach().clone())
    return P, eta


def ordered_crescent_boundary(
    alpha: float,
    cfg: SplineCurveConfig,
    *,
    dtype: torch.dtype,
    device: torch.device,
    n_points: int | None = None,
) -> torch.Tensor:
    """Materially ordered simple boundary for the synthetic crescent family.

    v33 started at the outer/inner circle intersection while the source circle
    starts at its leftmost point.  Linear teacher interpolation therefore paired
    unrelated material points and produced transient self-intersections.  v34
    uses the same leftmost seam and orientation for source and target, then
    equal-arclength resamples the analytic boundary.
    """
    a = float(np.clip(alpha, 0.0, 1.0))
    R = 1.0
    r = 0.62 + a * (0.90 - 0.62)
    d = 0.58 + a * (0.35 - 0.58)
    if d <= 0.0:
        raise ValueError("crescent offset must be positive")
    x = (R * R - r * r + d * d) / (2.0 * d)
    x = float(np.clip(x, -R, R))
    y = math.sqrt(max(R * R - x * x, 0.0))
    theta_top = math.acos(x / R)
    phi_top = math.atan2(y, x - d)
    n = int(n_points or cfg.n_samples)

    dense = max(8 * n, 1024)
    n_lower = max(8, dense // 4)
    n_inner = max(8, dense // 2)
    n_upper = max(8, dense - n_lower - n_inner + 2)

    # Seam = exact leftmost outer point.  Direction is downward, matching the
    # source circle: leftmost -> lower outer arc -> inner arc -> upper outer arc.
    th_lower = torch.linspace(math.pi, 2.0 * math.pi - theta_top, n_lower, dtype=dtype, device=device)
    th_inner = torch.linspace(2.0 * math.pi - phi_top, phi_top, n_inner, dtype=dtype, device=device)
    th_upper = torch.linspace(theta_top, math.pi, n_upper, dtype=dtype, device=device)
    lower = torch.stack([R * torch.cos(th_lower), R * torch.sin(th_lower)], dim=-1)
    inner = torch.stack([d + r * torch.cos(th_inner), r * torch.sin(th_inner)], dim=-1)
    upper = torch.stack([R * torch.cos(th_upper), R * torch.sin(th_upper)], dim=-1)
    boundary = torch.cat([lower, inner[1:], upper[1:]], dim=0)

    # Equal-arclength material samples avoid over-weighting either analytic arc.
    boundary = resample_curve_samples(boundary, n)
    boundary = boundary - boundary.mean(dim=0, keepdim=True)
    # Resampling preserves closure, but enforce it to numerical precision so the
    # teacher and diagnostics do not see a spurious endpoint gap.
    boundary = boundary.clone()
    boundary[-1] = boundary[0]
    return boundary



def fit_ordered_crescent_target(
    alpha: float,
    cfg: SplineCurveConfig,
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    iterations: int,
    init_P: torch.Tensor | None = None,
    init_eta: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Fit a valid periodic spline to the known ordered synthetic boundary.

    This is used only to construct training/configuration targets.  The learned
    pH objective can still use the empirical cloud/Sinkhorn divergence.
    """
    target = ordered_crescent_boundary(alpha, cfg, dtype=dtype, device=device, n_points=cfg.n_samples)
    if init_eta is None:
        eta0 = torch.zeros(cfg.n_spans, dtype=dtype, device=device)
    else:
        eta0 = init_eta.detach().clone()
    if init_P is None:
        idx = torch.linspace(0, target.shape[0] - 1, cfg.n_ctrl, dtype=dtype, device=device).round().long()
        P0 = target[idx].clone()
        P0 = periodic_project_controls(P0, eta0, cfg)
    else:
        P0 = init_P.detach().clone()
    P = P0.requires_grad_(True)
    eta = eta0.requires_grad_(True)
    opt = torch.optim.Adam([P, eta], lr=0.02)
    u = torch.linspace(0.0, 1.0, cfg.n_samples, dtype=dtype, device=device)
    cloud, _ = legacy_crescent_family_point_cloud(alpha, cfg, dtype=dtype, device=device)
    for _ in range(max(20, int(iterations))):
        opt.zero_grad(set_to_none=True)
        Puse = periodic_project_controls(P, eta, cfg)
        C, _ = evaluate_curve(Puse, eta, cfg, u)
        curve_loss = torch.mean((C - target) ** 2)
        if cfg.ordered_fit_ot_weight != 0.0:
            ot_loss = _fit_geometry_loss(C, cloud, cfg)
        else:
            ot_loss = torch.zeros((), dtype=C.dtype, device=C.device)
        smooth = _polygon_smoothness(Puse) + _curve_smoothness(C)
        spans = knot_spans_from_logits(eta, cfg)
        knot_reg = torch.mean((torch.log(spans) - torch.log(spans).mean()) ** 2)
        loss = (
            cfg.ordered_fit_curve_weight * curve_loss
            + cfg.ordered_fit_ot_weight * ot_loss
            + cfg.ordered_fit_smoothness_weight * smooth
            + 1e-5 * knot_reg
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([P, eta], 20.0)
        opt.step()
    with torch.no_grad():
        Pfit = periodic_project_controls(P, eta, cfg)
        Cfit, _ = evaluate_curve(Pfit, eta, cfg, u)
        curve_selfx = polyline_self_intersections(Cfit, closed=True)
        poly_selfx = polyline_self_intersections(Pfit, closed=True)
        if curve_selfx or poly_selfx:
            raise RuntimeError(
                f"ordered target fit unexpectedly self-intersects: curve={curve_selfx}, polygon={poly_selfx}, alpha={alpha:.4f}"
            )
        floor = float(point_cloud_chamfer(Cfit, cloud).item())
    return Pfit.detach(), eta.detach(), Cfit.detach(), floor


def fit_family_configuration(
    alpha: float,
    cfg: SplineCurveConfig,
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    total_iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Continuation from source circle to one crescent configuration.

    This is a conventional static optimizer used to construct feasible teachers
    and an OOD representation floor.  It is not a hard-task trajectory teacher.
    """
    cache_key = (
        round(float(alpha), 5), cfg.n_ctrl, cfg.degree, cfg.n_samples,
        cfg.observation_points, cfg.source_radius, cfg.min_knot_span,
        cfg.continuation_steps, str(dtype), str(device), total_iterations,
        cfg.fit_objective, cfg.fit_sinkhorn_eps, cfg.fit_sinkhorn_n_iter,
        cfg.fit_multistart, cfg.fit_multistart_noise,
        cfg.fit_curve_repulsion_weight, cfg.fit_curve_smoothness_weight,
        cfg.fit_polygon_smoothness_weight, cfg.fit_reject_self_intersection,
        cfg.target_fit_protocol, cfg.ordered_fit_curve_weight,
        cfg.ordered_fit_ot_weight, cfg.ordered_fit_smoothness_weight,
    )
    if cache_key in _FAMILY_FIT_CACHE:
        return tuple(x.clone() if torch.is_tensor(x) else x for x in _FAMILY_FIT_CACHE[cache_key])
    P, eta = _fit_source_circle(cfg, dtype, device, seed=seed + 50000)
    if cfg.target_fit_protocol == "ordered_boundary":
        # The ordered analytic boundary already supplies a globally consistent
        # material branch.  Running a continuation loop while reinitializing at
        # every stage (v33) merely divided the requested fit budget by the number
        # of continuation steps.  Use the full budget on the actual target.
        P, eta, Cfit, floor = fit_ordered_crescent_target(
            float(alpha), cfg, dtype=dtype, device=device,
            seed=seed + 101, iterations=max(20, int(total_iterations)),
            init_P=None, init_eta=None,
        )
    elif cfg.target_fit_protocol == "point_cloud_search":
        steps = max(1, cfg.continuation_steps)
        alphas = np.linspace(0.0, float(alpha), steps + 1)[1:]
        per_step = max(20, int(math.ceil(total_iterations / steps)))
        Cfit = None
        floor = float("inf")
        for k, a in enumerate(alphas):
            cloud, _ = legacy_crescent_family_point_cloud(float(a), cfg, dtype=dtype, device=device)
            Pnew, enew, Cfit, floor = fit_point_cloud_target(
                cloud, cfg, seed=seed + 101 * (k + 1), iterations=per_step, lr=0.025,
                init_P=P, init_eta=eta, branch_P=P, branch_eta=eta,
            )
            P, eta = Pnew, enew
        assert Cfit is not None
    else:
        raise ValueError(f"unknown target_fit_protocol={cfg.target_fit_protocol!r}")
    out = (P.detach(), eta.detach(), Cfit.detach(), float(floor))
    _FAMILY_FIT_CACHE[cache_key] = out
    return tuple(x.clone() if torch.is_tensor(x) else x for x in out)


# -----------------------------------------------------------------------------
# Task construction / curriculum


def make_feasible_tasks(
    mode: ClosureMode,
    count: int,
    cfg: SplineCurveConfig,
    *,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
    conditioning_case: ConditioningCase = "configuration",
    rot_max_deg: float = 20.0,
    translation: float = 0.16,
    alpha_range: tuple[float, float] = (0.15, 0.60),
    fit_iterations: int = 180,
) -> list[SplineCurveTask]:
    rng = np.random.default_rng(seed)
    tasks: list[SplineCurveTask] = []
    P0_base, eta0_base = _fit_source_circle(cfg, dtype, device, seed=seed + 9000)
    pose0 = torch.zeros(3, dtype=dtype, device=device)
    for i in range(count):
        alpha = float(rng.uniform(alpha_range[0], alpha_range[1]))
        cloud_body, params = legacy_crescent_family_point_cloud(alpha, cfg, dtype=dtype, device=device)
        Pt, et, Cfit, floor = fit_family_configuration(alpha, cfg, dtype=dtype, device=device, seed=seed + 71 * (i + 1), total_iterations=fit_iterations)
        pose_target = _sample_target_pose(rng, rot_max_deg, translation, dtype, device)
        observation_world = apply_pose(cloud_body, pose_target)
        params = dict(params)
        params.update({
            "pose_tx": float(pose_target[0].item()), "pose_ty": float(pose_target[1].item()), "pose_theta": float(pose_target[2].item()),
            "fit_objective": 0.0 if cfg.fit_objective == "chamfer" else 1.0,
            "fit_multistart": float(cfg.fit_multistart),
        })
        tasks.append(SplineCurveTask(
            mode=mode,
            P0=P0_base.clone(),
            eta0=eta0_base.clone(),
            P_target=Pt,
            eta_target=et,
            pose0=pose0.clone(),
            pose_target=pose_target,
            target_samples=Cfit,
            task_id=mode_id(mode),
            hard_ood=False,
            target_family="legacy_crescent_point_cloud_feasible_pose_factored",
            representation_floor=floor,
            target_hash=target_hash(observation_world),
            conditioning_case=conditioning_case,
            observation_samples=observation_world.detach(),
            target_alpha=alpha,
            target_params=params,
        ))
    return tasks


def make_hard_tasks(
    mode: ClosureMode,
    count: int,
    cfg: SplineCurveConfig,
    *,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
    fit_iterations: int = 350,
    conditioning_case: ConditioningCase = "configuration",
    rot_max_deg: float = 50.0,
    translation: float = 0.25,
    alpha_range: tuple[float, float] = (0.88, 1.00),
) -> list[SplineCurveTask]:
    rng = np.random.default_rng(seed)
    tasks: list[SplineCurveTask] = []
    P0_base, eta0_base = _fit_source_circle(cfg, dtype, device, seed=seed + 11000)
    pose0 = torch.zeros(3, dtype=dtype, device=device)
    for j in range(count):
        alpha = float(rng.uniform(alpha_range[0], alpha_range[1]))
        cloud_body, params = legacy_crescent_family_point_cloud(alpha, cfg, dtype=dtype, device=device)
        Pt, et, Cfit, floor = fit_family_configuration(alpha, cfg, dtype=dtype, device=device, seed=seed + 101 * (j + 1), total_iterations=fit_iterations)
        pose_target = _sample_target_pose(rng, rot_max_deg, translation, dtype, device)
        observation_world = apply_pose(cloud_body, pose_target)
        params = dict(params)
        params.update({
            "pose_tx": float(pose_target[0].item()), "pose_ty": float(pose_target[1].item()), "pose_theta": float(pose_target[2].item()),
            "fit_objective": 0.0 if cfg.fit_objective == "chamfer" else 1.0,
            "fit_multistart": float(cfg.fit_multistart),
        })
        tasks.append(SplineCurveTask(
            mode=mode,
            P0=P0_base.clone(),
            eta0=eta0_base.clone(),
            P_target=Pt,
            eta_target=et,
            pose0=pose0.clone(),
            pose_target=pose_target,
            target_samples=Cfit,
            task_id=mode_id(mode),
            hard_ood=True,
            target_family="legacy_crescent_point_cloud_hard_pose_factored",
            representation_floor=floor,
            target_hash=target_hash(observation_world),
            conditioning_case=conditioning_case,
            observation_samples=observation_world.detach(),
            target_alpha=alpha,
            target_params=params,
        ))
    return tasks


def teacher_trajectory(task: SplineCurveTask, cfg: SplineCurveConfig) -> dict[str, torch.Tensor]:
    dtype, device = task.P0.dtype, task.P0.device
    tau = torch.linspace(0.0, 1.0, cfg.n_steps + 1, dtype=dtype, device=device)
    a = 3 * tau ** 2 - 2 * tau ** 3
    P = task.P0[None] + a[:, None, None] * (task.P_target - task.P0)[None]
    eta = task.eta0[None] + a[:, None] * (task.eta_target - task.eta0)[None]
    dtheta = wrap_angle(task.pose_target[2] - task.pose0[2])
    pose = task.pose0[None].expand(cfg.n_steps + 1, -1).clone()
    pose[:, :2] = task.pose0[:2][None] + a[:, None] * (task.pose_target[:2] - task.pose0[:2])[None]
    pose[:, 2] = wrap_angle(task.pose0[2] + a * dtheta)
    if task.mode == "periodic":
        P = torch.stack([periodic_project_controls(P[k], eta[k], cfg) for k in range(P.shape[0])], dim=0)
    dt = 1.0 / cfg.n_steps
    Pdot = torch.gradient(P, spacing=(dt,), dim=(0,))[0]
    etadot = torch.gradient(eta, spacing=(dt,), dim=(0,))[0]
    posedot = torch.gradient(pose, spacing=(dt,), dim=(0,))[0]
    return {"P": P, "eta": eta, "pose": pose, "Pdot": Pdot, "etadot": etadot, "posedot": posedot, "tau": tau}


# -----------------------------------------------------------------------------
# Learned pH template with permutation-invariant observation encoder


class SplineCurvePHTemplate(nn.Module):
    def __init__(self, cfg: SplineCurveConfig, hidden: int = 160, ref_scale: float = 0.22, knot_ref_scale: float = 0.9, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.cfg = cfg
        self.ref_scale = ref_scale
        self.knot_ref_scale = knot_ref_scale
        self.obs_embed_dim = 64
        phi_hidden = max(32, hidden // 3)
        self.point_phi = nn.Sequential(
            nn.Linear(2, phi_hidden, dtype=dtype), nn.Tanh(),
            nn.Linear(phi_hidden, self.obs_embed_dim, dtype=dtype), nn.Tanh(),
        )
        self.point_rho = nn.Sequential(
            nn.Linear(2 * self.obs_embed_dim + 5, self.obs_embed_dim, dtype=dtype), nn.Tanh(),
            nn.Linear(self.obs_embed_dim, self.obs_embed_dim, dtype=dtype), nn.Tanh(),
        )
        flatP = 2 * cfg.n_ctrl
        base_in = 3 + flatP + cfg.n_spans + 3 + flatP + cfg.n_spans + 3 + 2 + self.obs_embed_dim + 3
        out_block = flatP + cfg.n_spans + 3
        self.net = nn.Sequential(
            nn.Linear(base_in, hidden, dtype=dtype), nn.Tanh(),
            nn.Linear(hidden, hidden, dtype=dtype), nn.Tanh(),
            nn.Linear(hidden, 2 * out_block, dtype=dtype),
        )
        self.log_kP = nn.Parameter(torch.tensor(math.log(7.0), dtype=dtype))
        self.log_kK = nn.Parameter(torch.tensor(math.log(1.5), dtype=dtype))
        self.log_kPoseXY = nn.Parameter(torch.tensor(math.log(5.0), dtype=dtype))
        self.log_kPoseTheta = nn.Parameter(torch.tensor(math.log(2.5), dtype=dtype))
        self.log_dP = nn.Parameter(torch.tensor(math.log(cfg.damping_control), dtype=dtype))
        self.log_dK = nn.Parameter(torch.tensor(math.log(cfg.damping_knot), dtype=dtype))
        self.log_dPoseXY = nn.Parameter(torch.tensor(math.log(cfg.damping_pose_xy), dtype=dtype))
        self.log_dPoseTheta = nn.Parameter(torch.tensor(math.log(cfg.damping_pose_theta), dtype=dtype))

    @property
    def kP(self):
        return torch.exp(self.log_kP).clamp(0.1, 80.0)

    @property
    def kK(self):
        return torch.exp(self.log_kK).clamp(0.02, 30.0)

    @property
    def kPoseXY(self):
        return torch.exp(self.log_kPoseXY).clamp(0.05, 80.0)

    @property
    def kPoseTheta(self):
        return torch.exp(self.log_kPoseTheta).clamp(0.02, 40.0)

    @property
    def dP(self):
        return torch.exp(self.log_dP).clamp(1e-4, 10.0)

    @property
    def dK(self):
        return torch.exp(self.log_dK).clamp(1e-4, 10.0)

    @property
    def dPoseXY(self):
        return torch.exp(self.log_dPoseXY).clamp(1e-4, 15.0)

    @property
    def dPoseTheta(self):
        return torch.exp(self.log_dPoseTheta).clamp(1e-4, 15.0)

    def encode_observation(self, task: SplineCurveTask) -> torch.Tensor:
        if task.observation_samples is None:
            return torch.zeros(self.obs_embed_dim, dtype=task.P0.dtype, device=task.P0.device)
        pts = task.observation_samples
        h = self.point_phi(pts)
        mean = h.mean(dim=-2)
        maxv = h.max(dim=-2).values
        centroid = pts.mean(dim=-2)
        X = pts - centroid
        cov_xx = (X[:, 0] ** 2).mean()
        cov_xy = (X[:, 0] * X[:, 1]).mean()
        cov_yy = (X[:, 1] ** 2).mean()
        stats = torch.cat([centroid, torch.stack([cov_xx, cov_xy, cov_yy])], dim=-1)
        return self.point_rho(torch.cat([mean, maxv, stats], dim=-1))

    def task_context(self, task: SplineCurveTask) -> torch.Tensor:
        if task.conditioning_case == "observation":
            return self.encode_observation(task)
        return torch.zeros(self.obs_embed_dim, dtype=task.P0.dtype, device=task.P0.device)

    def reference(self, task: SplineCurveTask, tau: torch.Tensor, context: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tauv = tau.reshape(-1) if tau.ndim > 0 else tau[None]
        B = tauv.shape[0]
        if context is None:
            context = self.task_context(task)
        mode_oh = torch.zeros(B, 3, dtype=task.P0.dtype, device=task.P0.device)
        mode_oh[:, task.task_id] = 1.0
        case_oh = torch.zeros(B, 2, dtype=task.P0.dtype, device=task.P0.device)
        case_oh[:, conditioning_id(task.conditioning_case)] = 1.0
        P0f = task.P0.reshape(1, -1).expand(B, -1)
        e0f = task.eta0.reshape(1, -1).expand(B, -1)
        pose0f = task.pose0.reshape(1, -1).expand(B, -1)
        if task.conditioning_case == "configuration":
            Ptf = task.P_target.reshape(1, -1).expand(B, -1)
            etf = task.eta_target.reshape(1, -1).expand(B, -1)
            pose_tf = task.pose_target.reshape(1, -1).expand(B, -1)
        else:
            Ptf = torch.zeros(B, 2 * self.cfg.n_ctrl, dtype=task.P0.dtype, device=task.P0.device)
            etf = torch.zeros(B, self.cfg.n_spans, dtype=task.P0.dtype, device=task.P0.device)
            pose_tf = torch.zeros(B, 3, dtype=task.P0.dtype, device=task.P0.device)
        ctx = context.reshape(1, -1).expand(B, -1)
        phase = torch.stack([torch.sin(2 * math.pi * tauv), torch.cos(2 * math.pi * tauv), tauv], dim=-1)
        x = torch.cat([mode_oh, P0f, e0f, pose0f, Ptf, etf, pose_tf, case_oh, ctx, phase], dim=-1)
        raw = self.net(x)
        flatP = 2 * self.cfg.n_ctrl
        ne = self.cfg.n_spans
        bsz = flatP + ne + 3
        goal = raw[:, :bsz]
        path = raw[:, bsz:]
        goalP_raw = goal[:, :flatP].reshape(B, self.cfg.n_ctrl, 2)
        goalE_raw = goal[:, flatP:flatP + ne]
        goalPose_raw = goal[:, flatP + ne:]
        pathP_raw = path[:, :flatP].reshape(B, self.cfg.n_ctrl, 2)
        pathE_raw = path[:, flatP:flatP + ne]
        pathPose_raw = path[:, flatP + ne:]

        if task.conditioning_case == "configuration":
            Pgoal = task.P_target[None] + 0.03 * torch.tanh(goalP_raw)
            Egoal = task.eta_target[None] + 0.12 * torch.tanh(goalE_raw)
            pose_goal = task.pose_target[None] + torch.stack([
                0.03 * torch.tanh(goalPose_raw[:, 0]),
                0.03 * torch.tanh(goalPose_raw[:, 1]),
                0.06 * torch.tanh(goalPose_raw[:, 2]),
            ], dim=-1)
        else:
            Pgoal = task.P0[None] + 1.8 * torch.tanh(goalP_raw)
            Egoal = task.eta0[None] + 2.0 * torch.tanh(goalE_raw)
            pose_goal = task.pose0[None] + torch.stack([
                1.2 * torch.tanh(goalPose_raw[:, 0]),
                1.2 * torch.tanh(goalPose_raw[:, 1]),
                math.pi * torch.tanh(goalPose_raw[:, 2]),
            ], dim=-1)

        a1 = 3 * tauv ** 2 - 2 * tauv ** 3
        a = a1[:, None, None]
        ae = a1[:, None]
        env = 4 * tauv * (1 - tauv)
        Pref = (1 - a) * task.P0[None] + a * Pgoal + env[:, None, None] * self.ref_scale * torch.tanh(pathP_raw)
        eref = (1 - ae) * task.eta0[None] + ae * Egoal + env[:, None] * self.knot_ref_scale * torch.tanh(pathE_raw)
        dtheta = wrap_angle(pose_goal[:, 2] - task.pose0[2])
        pose_xy = task.pose0[:2][None] + a1[:, None] * (pose_goal[:, :2] - task.pose0[:2][None])
        pose_theta = wrap_angle(task.pose0[2] + a1 * dtheta)
        path_pose = env[:, None] * torch.stack([
            0.20 * torch.tanh(pathPose_raw[:, 0]),
            0.20 * torch.tanh(pathPose_raw[:, 1]),
            0.35 * torch.tanh(pathPose_raw[:, 2]),
        ], dim=-1)
        pose_ref = torch.cat([pose_xy + path_pose[:, :2], wrap_angle(pose_theta + path_pose[:, 2])[:, None]], dim=-1)
        if task.mode == "periodic":
            Pref = torch.stack([periodic_project_controls(Pref[k], eref[k], self.cfg) for k in range(B)], dim=0)
        return Pref, eref, pose_ref


def _measurement_closure_energy(P: torch.Tensor, eta: torch.Tensor, cfg: SplineCurveConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gap, tan = closure_measurement(P, eta, cfg)
    sig = cfg.closure_tolerance
    kpos = cfg.closure_energy_scale / max(sig * sig, 1e-8)
    V = 0.5 * kpos * gap.pow(2).sum() + 0.5 * (cfg.tangent_closure_weight * kpos) * tan.pow(2).sum()
    return V, gap, tan


def _sinkhorn_world_error(
    P: torch.Tensor, eta: torch.Tensor, pose: torch.Tensor, task: SplineCurveTask, cfg: SplineCurveConfig,
    *, eps: float | None = None, n_iter: int | None = None, force_resolution: bool = False,
) -> torch.Tensor:
    if task.observation_samples is None:
        return torch.zeros((), dtype=P.dtype, device=P.device)
    Cw = _world_curve(P, eta, pose, cfg)
    eps = cfg.sinkhorn_eps_end if eps is None else float(eps)
    if force_resolution:
        n_iter = cfg.sinkhorn_force_n_iter if n_iter is None else int(n_iter)
        ns = cfg.sinkhorn_force_source_points
        nt = cfg.sinkhorn_force_target_points
    else:
        n_iter = cfg.sinkhorn_n_iter if n_iter is None else int(n_iter)
        ns = cfg.sinkhorn_source_points
        nt = cfg.sinkhorn_target_points
    return sinkhorn_divergence_points(
        Cw, task.observation_samples, eps=eps, n_iter=n_iter,
        symmetric_finite_iter=cfg.sinkhorn_symmetric,
        source_points=ns, target_points=nt,
    )


def _sinkhorn_world_state_potential(
    P: torch.Tensor, eta: torch.Tensor, pose: torch.Tensor, task: SplineCurveTask, cfg: SplineCurveConfig,
    *, eps: float,
) -> torch.Tensor:
    if task.observation_samples is None:
        return torch.zeros((), dtype=P.dtype, device=P.device)
    Cw = _world_curve(P, eta, pose, cfg)
    return sinkhorn_state_potential(
        Cw, task.observation_samples, eps=eps, n_iter=cfg.sinkhorn_force_n_iter,
        symmetric_finite_iter=cfg.sinkhorn_symmetric,
        source_points=cfg.sinkhorn_force_source_points,
        target_points=cfg.sinkhorn_force_target_points,
    )


def rollout_spline_ph(model: SplineCurvePHTemplate, task: SplineCurveTask, cfg: SplineCurveConfig, *, horizon_multiplier: float = 1.0) -> dict[str, torch.Tensor]:
    dtype, device = task.P0.dtype, task.P0.device
    n = max(cfg.n_steps, int(round(cfg.n_steps * horizon_multiplier)))
    dt = 1.0 / cfg.n_steps
    P = task.P0.clone()
    eta = task.eta0.clone()
    pose = task.pose0.clone()
    pP = torch.zeros_like(P)
    pE = torch.zeros_like(eta)
    pG = torch.zeros_like(pose)
    Ps, Es, Gs = [P], [eta], [pose]
    pPs, pEs, pGs = [pP], [pE], [pG]
    refsP, refsE, refsG = [], [], []
    energies, gap_hist, tan_hist, powers, ot_hist = [], [], [], [], []
    context = model.task_context(task)
    for k in range(n):
        progress = min(k / cfg.n_steps, 1.0)
        tau = torch.tensor(progress, dtype=dtype, device=device)
        Pref, Eref, Gref = model.reference(task, tau, context=context)
        Pref, Eref, Gref = Pref[0], Eref[0], Gref[0]

        # Learned moving-rest-path potential remains a trajectory prior.
        gradP = model.kP * (P - Pref)
        gradE = model.kK * (eta - Eref)
        dpose = pose - Gref
        dpose = torch.stack([dpose[0], dpose[1], wrap_angle(dpose[2])])
        gradG = torch.stack([model.kPoseXY * dpose[0], model.kPoseXY * dpose[1], model.kPoseTheta * dpose[2]])

        gap = torch.zeros(2, dtype=dtype, device=device)
        tan = torch.zeros(2, dtype=dtype, device=device)
        Vot_state = torch.zeros((), dtype=dtype, device=device)

        # v30: direct target-measure potential.  The numerical OT force is
        # exact for the finite-iteration Sinkhorn potential.  By default its
        # Jacobian is stop-gradiented during network training, which avoids a
        # prohibitively expensive second derivative through every Sinkhorn
        # iteration while preserving the actual pH rollout force.
        use_ot_force = bool(cfg.use_sinkhorn_potential and cfg.sinkhorn_potential_weight != 0.0 and task.observation_samples is not None)

        # Measurement-closure force keeps the historical differentiable path.
        if task.mode == "measurement":
            outer_grad = torch.is_grad_enabled()
            with torch.enable_grad():
                P_req = (P if outer_grad else P.detach()).requires_grad_(True)
                E_req = (eta if outer_grad else eta.detach()).requires_grad_(True)
                Vc, gap, tan = _measurement_closure_energy(P_req, E_req, cfg)
                gPc, gEc = torch.autograd.grad(Vc, (P_req, E_req), create_graph=outer_grad)
            if not outer_grad:
                gPc, gEc = gPc.detach(), gEc.detach()
                gap, tan = gap.detach(), tan.detach()
                P_req, E_req = P_req.detach(), E_req.detach()
            gradP = gradP + gPc
            gradE = gradE + gEc
            P, eta = P_req, E_req

        if use_ot_force:
            outer_grad = torch.is_grad_enabled()
            # Stop-gradient mode must use detached proxy states.  Calling
            # autograd.grad directly on the live rollout state would consume
            # parts of the graph that the final training loss still needs.
            live_graph = bool(outer_grad and cfg.sinkhorn_force_backprop)
            with torch.enable_grad():
                P_req = (P if live_graph else P.detach()).requires_grad_(True)
                E_req = (eta if live_graph else eta.detach()).requires_grad_(True)
                G_req = (pose if live_graph else pose.detach()).requires_grad_(True)
                eps_force = sinkhorn_eps_at_progress(cfg, progress)
                Vot_state = cfg.sinkhorn_potential_weight * _sinkhorn_world_state_potential(
                    P_req, E_req, G_req, task, cfg, eps=eps_force,
                )
                gPo, gEo, gGo = torch.autograd.grad(
                    Vot_state, (P_req, E_req, G_req),
                    create_graph=live_graph, allow_unused=True,
                )
            gPo = torch.zeros_like(P_req) if gPo is None else gPo
            gEo = torch.zeros_like(E_req) if gEo is None else gEo
            gGo = torch.zeros_like(G_req) if gGo is None else gGo
            if not live_graph:
                gPo, gEo, gGo = gPo.detach(), gEo.detach(), gGo.detach()
                Vot_state = Vot_state.detach()
            gradP = gradP + gPo
            gradE = gradE + gEo
            gradG = gradG + gGo
            # The actual rollout state remains the live state in stop-gradient
            # mode; only the computed force is detached.
            if live_graph:
                P, eta, pose = P_req, E_req, G_req

        pP = (pP - dt * gradP) / (1.0 + dt * model.dP / cfg.mass_control)
        pE = (pE - dt * gradE) / (1.0 + dt * model.dK / cfg.mass_knot)
        pGxy = (pG[:2] - dt * gradG[:2]) / (1.0 + dt * model.dPoseXY / cfg.mass_pose_xy)
        pGt = (pG[2] - dt * gradG[2]) / (1.0 + dt * model.dPoseTheta / cfg.mass_pose_theta)
        pG = torch.cat([pGxy, pGt[None]])
        P = P + dt * pP / cfg.mass_control
        eta = eta + dt * pE / cfg.mass_knot
        pose_xy = pose[:2] + dt * pG[:2] / cfg.mass_pose_xy
        pose_theta = wrap_angle(pose[2] + dt * pG[2] / cfg.mass_pose_theta)
        pose = torch.cat([pose_xy, pose_theta[None]])
        if task.mode == "periodic":
            P = periodic_project_controls(P, eta, cfg)
            # Project momentum through the *tangent map* of the closure
            # constraint.  The old position-style projection ignored moving knot
            # spans and generated an artificial seam impulse.
            vP = pP / cfg.mass_control
            vE = pE / cfg.mass_knot
            vP = periodic_project_velocity(P, eta, vP, vE, cfg)
            pP = cfg.mass_control * vP

        T = 0.5 * (
            pP.pow(2).sum() / cfg.mass_control
            + pE.pow(2).sum() / cfg.mass_knot
            + pG[:2].pow(2).sum() / cfg.mass_pose_xy
            + pG[2].pow(2) / cfg.mass_pose_theta
        )
        Vshape = 0.5 * model.kP * (P - Pref).pow(2).sum() + 0.5 * model.kK * (eta - Eref).pow(2).sum()
        Vpose = 0.5 * model.kPoseXY * (pose[:2] - Gref[:2]).pow(2).sum() + 0.5 * model.kPoseTheta * wrap_angle(pose[2] - Gref[2]).pow(2)
        if task.mode == "measurement":
            Vc2, gap, tan = _measurement_closure_energy(P, eta, cfg)
        else:
            Vc2 = torch.zeros((), dtype=dtype, device=device)
            gap, tan = closure_measurement(P, eta, cfg)
        diss = (
            model.dP * (pP / cfg.mass_control).pow(2).sum()
            + model.dK * (pE / cfg.mass_knot).pow(2).sum()
            + model.dPoseXY * (pG[:2] / cfg.mass_pose_xy).pow(2).sum()
            + model.dPoseTheta * (pG[2] / cfg.mass_pose_theta).pow(2)
        )
        # Vot_state is evaluated at the beginning of the semi-implicit step.
        # Its constant target self-term is intentionally omitted, because it has
        # no force.  A nonnegative full Sinkhorn divergence is logged separately.
        Vot_energy = Vot_state
        Ps.append(P)
        Es.append(eta)
        Gs.append(pose)
        pPs.append(pP)
        pEs.append(pE)
        pGs.append(pG)
        refsP.append(Pref)
        refsE.append(Eref)
        refsG.append(Gref)
        energies.append(torch.stack([T, Vshape, Vpose, Vot_energy, Vc2, T + Vshape + Vpose + Vot_energy + Vc2, diss]))
        gap_hist.append(gap.norm())
        tan_hist.append(tan.norm())
        powers.append(diss)
        if task.observation_samples is not None:
            # Cheap diagnostic at force resolution; no target-control coordinates.
            ot_diag = _sinkhorn_world_error(
                P, eta, pose, task, cfg, eps=cfg.sinkhorn_eps_end,
                n_iter=cfg.sinkhorn_force_n_iter, force_resolution=True,
            )
        else:
            ot_diag = torch.zeros((), dtype=dtype, device=device)
        ot_hist.append(ot_diag)
    return {
        "P": torch.stack(Ps), "eta": torch.stack(Es), "pose": torch.stack(Gs),
        "pP": torch.stack(pPs), "pE": torch.stack(pEs), "pPose": torch.stack(pGs),
        "P_ref": torch.stack(refsP), "eta_ref": torch.stack(refsE), "pose_ref": torch.stack(refsG),
        "energy": torch.stack(energies), "gap": torch.stack(gap_hist), "tangent_gap": torch.stack(tan_hist),
        "sinkhorn": torch.stack(ot_hist), "dissipation_power": torch.stack(powers),
    }

def _world_curve(P: torch.Tensor, eta: torch.Tensor, pose: torch.Tensor, cfg: SplineCurveConfig) -> torch.Tensor:
    C, _ = evaluate_curve(P, eta, cfg)
    return apply_pose(C, pose)


def _intrinsic_shape_error(P: torch.Tensor, eta: torch.Tensor, task: SplineCurveTask, cfg: SplineCurveConfig) -> torch.Tensor:
    C, _ = evaluate_curve(P, eta, cfg)
    return curve_rms(C, task.target_samples, False)


def _observation_error(P: torch.Tensor, eta: torch.Tensor, pose: torch.Tensor, task: SplineCurveTask, cfg: SplineCurveConfig) -> torch.Tensor:
    """Chamfer diagnostic retained for backward-compatible plots/reporting."""
    if task.observation_samples is None:
        return torch.zeros((), dtype=P.dtype, device=P.device)
    Cw = _world_curve(P, eta, pose, cfg)
    return point_cloud_chamfer(Cw, task.observation_samples)


def _sinkhorn_training_error(P: torch.Tensor, eta: torch.Tensor, pose: torch.Tensor, task: SplineCurveTask, cfg: SplineCurveConfig) -> torch.Tensor:
    if task.observation_samples is None:
        return torch.zeros((), dtype=P.dtype, device=P.device)
    if cfg.geometry_objective == "chamfer":
        # Compatibility/debugging mode.  v30 defaults to Sinkhorn.
        obs = _observation_error(P, eta, pose, task, cfg)
        return obs.pow(2)
    if cfg.geometry_objective != "sinkhorn":
        raise ValueError(f"unknown geometry_objective={cfg.geometry_objective!r}")
    return _sinkhorn_world_error(P, eta, pose, task, cfg, eps=cfg.sinkhorn_eps_end)


def _tail_indices(rollout: dict[str, torch.Tensor], cfg: SplineCurveConfig) -> list[int]:
    n = rollout["P"].shape[0]
    start = min(cfg.n_steps, n - 1)
    if n - start <= cfg.tail_samples:
        return list(range(start, n))
    return torch.linspace(start, n - 1, cfg.tail_samples).round().to(torch.int64).tolist()


def task_loss(model: SplineCurvePHTemplate, task: SplineCurveTask, cfg: SplineCurveConfig, rollout: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Settling-tail loss driven by target-measure geometry.

    v30 removes direct target-control-point and target-knot quadratic matching
    from end-to-end training.  Those quantities are still reported as diagnostics
    and still define feasible teacher paths during reference pretraining, but the
    terminal deformation goal is the empirical target measure.
    """
    idxs = _tail_indices(rollout, cfg)
    losses = []
    shape_vals, obs_vals, ot_vals, pose_t_vals, pose_r_vals, speed_vals = [], [], [], [], [], []
    ctrl_vals, knot_vals, gap_vals, tan_vals = [], [], [], []
    for idx in idxs:
        P = rollout["P"][idx]
        eta = rollout["eta"][idx]
        pose = rollout["pose"][idx]
        intrinsic = _intrinsic_shape_error(P, eta, task, cfg)
        obs = _observation_error(P, eta, pose, task, cfg)
        ot = _sinkhorn_training_error(P, eta, pose, task, cfg)
        pP = rollout["pP"][idx]
        pE = rollout["pE"][idx]
        pG = rollout["pPose"][idx]
        speed2 = pP.pow(2).mean() + pE.pow(2).mean() + pG.pow(2).mean()
        gap, tan = closure_measurement(P, eta, cfg)
        trans_e, rot_e = pose_errors(pose, task.pose_target)
        # Kept only for debugging whether the pH state follows the conventional
        # feasible teacher; neither term enters the v30 terminal objective.
        ctrl = torch.sqrt(torch.mean((P - task.P_target) ** 2).clamp_min(0))
        knot = torch.sqrt(torch.mean((knot_spans_from_logits(eta, cfg) - knot_spans_from_logits(task.eta_target, cfg)) ** 2).clamp_min(0))

        l = cfg.sinkhorn_loss_weight * ot + 2.5 * speed2
        if task.conditioning_case == "configuration":
            # The configuration-only experiment is allowed to know global pose,
            # but intrinsic shape is still judged by OT rather than P/eta matching.
            l = l + 22.0 * trans_e ** 2 + 8.0 * rot_e ** 2
        # Observation-only receives no privileged P_target/eta_target/pose_target
        # supervision.  World-space OT jointly drives shape and SE(2) placement.
        if task.mode == "measurement":
            l = l + 2.0 * (gap.norm() / max(cfg.closure_tolerance, 1e-5)) ** 2 + 0.15 * tan.norm() ** 2
        if task.mode == "periodic":
            l = l + 80.0 * gap.norm() ** 2 + 2.0 * tan.norm() ** 2

        losses.append(l)
        shape_vals.append(intrinsic)
        obs_vals.append(obs)
        ot_vals.append(ot)
        pose_t_vals.append(trans_e)
        pose_r_vals.append(rot_e)
        speed_vals.append(torch.sqrt(speed2.clamp_min(0)))
        ctrl_vals.append(ctrl)
        knot_vals.append(knot)
        gap_vals.append(gap.norm())
        tan_vals.append(tan.norm())
    loss = torch.stack(losses).mean()
    mean = lambda xs: torch.stack(xs).mean()
    return loss, {
        "shape": mean(shape_vals), "observation": mean(obs_vals), "sinkhorn": mean(ot_vals),
        "pose_translation": mean(pose_t_vals), "pose_rotation": mean(pose_r_vals),
        "control": mean(ctrl_vals), "knot": mean(knot_vals), "momentum": mean(speed_vals),
        "gap": mean(gap_vals), "tangent": mean(tan_vals),
    }

def reference_pretrain(model: SplineCurvePHTemplate, tasks: list[SplineCurveTask], cfg: SplineCurveConfig, *, epochs: int = 80, lr: float = 1e-3) -> list[float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for task in tasks:
            teacher = teacher_trajectory(task, cfg)
            context = model.task_context(task)
            for k in range(0, cfg.n_steps + 1, max(1, cfg.n_steps // 8)):
                Pref, Eref, Gref = model.reference(task, teacher["tau"][k], context=context)
                dtheta = wrap_angle(Gref[0, 2] - teacher["pose"][k, 2])
                loss = (
                    loss
                    + torch.mean((Pref[0] - teacher["P"][k]) ** 2)
                    + 0.15 * torch.mean((Eref[0] - teacher["eta"][k]) ** 2)
                    + 0.8 * torch.mean((Gref[0, :2] - teacher["pose"][k, :2]) ** 2)
                    + 0.25 * dtheta ** 2
                )
        loss = loss / max(len(tasks), 1)
        loss.backward()
        opt.step()
        hist.append(float(loss.detach().cpu()))
    return hist


def train_end_to_end(
    model: SplineCurvePHTemplate,
    train_tasks: list[SplineCurveTask],
    val_tasks: list[SplineCurveTask],
    cfg: SplineCurveConfig,
    *,
    epochs: int = 120,
    lr: float = 6e-4,
    seed: int = 0,
    horizon_multiplier: float | None = None,
) -> tuple[list[dict[str, float]], dict[str, torch.Tensor]]:
    horizon_multiplier = cfg.training_horizon_multiplier if horizon_multiplier is None else horizon_multiplier
    rng = random.Random(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    best = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    for ep in range(epochs):
        task = rng.choice(train_tasks)
        opt.zero_grad(set_to_none=True)
        ro = rollout_spline_ph(model, task, cfg, horizon_multiplier=horizon_multiplier)
        loss, m = task_loss(model, task, cfg, ro)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        with torch.no_grad():
            vals = []
            for vt in val_tasks:
                vro = rollout_spline_ph(model, vt, cfg, horizon_multiplier=horizon_multiplier)
                vl, _ = task_loss(model, vt, cfg, vro)
                vals.append(float(vl.detach().cpu()))
            v = float(np.mean(vals)) if vals else float(loss.detach().cpu())
        if v < best:
            best = v
            best_state = copy.deepcopy(model.state_dict())
        hist.append({
            "epoch": ep, "train_loss": float(loss.detach().cpu()), "val_loss": v,
            "shape": float(m["shape"].detach().cpu()), "observation": float(m["observation"].detach().cpu()),
            "sinkhorn": float(m["sinkhorn"].detach().cpu()),
            "pose_translation": float(m["pose_translation"].detach().cpu()), "pose_rotation": float(m["pose_rotation"].detach().cpu()),
            "settling_speed": float(m["momentum"].detach().cpu()),
        })
    model.load_state_dict(best_state)
    return hist, best_state


def evaluate_tasks(model: SplineCurvePHTemplate, tasks: list[SplineCurveTask], cfg: SplineCurveConfig, *, horizon_multiplier: float = 2.0) -> list[dict[str, Any]]:
    rows = []
    for i, t in enumerate(tasks):
        ro = rollout_spline_ph(model, t, cfg, horizon_multiplier=horizon_multiplier)
        P, eta, pose = ro["P"][-1], ro["eta"][-1], ro["pose"][-1]
        intrinsic = float(_intrinsic_shape_error(P, eta, t, cfg).detach().cpu())
        obs_err = float(_observation_error(P, eta, pose, t, cfg).detach().cpu())
        sink_err = float(_sinkhorn_training_error(P, eta, pose, t, cfg).detach().cpu())
        trans_e, rot_e = pose_errors(pose, t.pose_target)
        C0, _ = evaluate_curve(ro["P"][0], ro["eta"][0], cfg)
        orient = 1.0 if float(signed_curve_area(C0).detach().cpu()) >= 0.0 else -1.0
        rollout_selfx = 0
        rollout_min_area = float("inf")
        for k in range(ro["P"].shape[0]):
            Ck, _ = evaluate_curve(ro["P"][k], ro["eta"][k], cfg)
            rollout_selfx = max(rollout_selfx, int(polyline_self_intersections(Ck, closed=True)))
            rollout_min_area = min(rollout_min_area, orient * float(signed_curve_area(Ck).detach().cpu()))
        rows.append({
            "task": i, "mode": t.mode, "hard_ood": t.hard_ood,
            "target_family": t.target_family, "conditioning_case": t.conditioning_case,
            "representation_floor": float(t.representation_floor),
            "shape_error": obs_err if t.conditioning_case == "observation" else intrinsic,
            "observation_error": obs_err, "sinkhorn_error": sink_err, "aligned_shape_error": intrinsic,
            "pose_translation_error": float(trans_e.detach().cpu()),
            "pose_rotation_error": float(rot_e.detach().cpu()),
            "excess_error": max(0.0, obs_err - float(t.representation_floor)),
            "closure_gap": float(ro["gap"][-1].detach().cpu()),
            "tangent_gap": float(ro["tangent_gap"][-1].detach().cpu()),
            "terminal_speed": float(torch.sqrt(ro["pP"][-1].pow(2).mean() + ro["pE"][-1].pow(2).mean() + ro["pPose"][-1].pow(2).mean()).detach().cpu()),
            "rollout_max_self_intersections": int(rollout_selfx),
            "rollout_min_oriented_area": float(rollout_min_area),
            "target_hash": t.target_hash, "target_alpha": float(t.target_alpha),
        })
    return rows


def teacher_free_transfer(model: SplineCurvePHTemplate, adapt_tasks: list[SplineCurveTask], cfg: SplineCurveConfig, *, epochs: int = 60, lr: float = 2e-4) -> SplineCurvePHTemplate:
    tuned = copy.deepcopy(model)
    for name, p in tuned.named_parameters():
        p.requires_grad_(name.startswith("net.") or name.startswith("point_phi.") or name.startswith("point_rho."))
    opt = torch.optim.Adam([p for p in tuned.parameters() if p.requires_grad], lr=lr)
    for ep in range(epochs):
        task = adapt_tasks[ep % len(adapt_tasks)]
        opt.zero_grad(set_to_none=True)
        ro = rollout_spline_ph(tuned, task, cfg, horizon_multiplier=cfg.training_horizon_multiplier)
        loss, _ = task_loss(tuned, task, cfg, ro)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tuned.parameters(), 8.0)
        opt.step()
    return tuned



def validate_spline_task_target(task: SplineCurveTask, cfg: SplineCurveConfig) -> dict[str, float | int | bool]:
    spans = knot_spans_from_logits(task.eta_target, cfg)
    C, _ = evaluate_curve(task.P_target, task.eta_target, cfg)
    C0, _ = evaluate_curve(task.P0, task.eta0, cfg)
    curve_selfx = int(polyline_self_intersections(C, closed=True))
    polygon_selfx = int(polyline_self_intersections(task.P_target, closed=True))
    source_curve_selfx = int(polyline_self_intersections(C0, closed=True))
    gap, tan = closure_measurement(task.P_target, task.eta_target, cfg)

    teacher_max_selfx = 0
    teacher_min_oriented_area = float("inf")
    teacher_area_floor = 0.0
    if cfg.validate_teacher_path:
        tr = teacher_trajectory(task, cfg)
        source_area = float(signed_curve_area(C0).detach().cpu())
        target_area = float(signed_curve_area(C).detach().cpu())
        orient = 1.0 if source_area >= 0.0 else -1.0
        teacher_area_floor = cfg.teacher_area_floor_ratio * min(abs(source_area), abs(target_area))
        vals = []
        for k in range(tr["P"].shape[0]):
            Ck, _ = evaluate_curve(tr["P"][k], tr["eta"][k], cfg)
            teacher_max_selfx = max(teacher_max_selfx, int(polyline_self_intersections(Ck, closed=True)))
            vals.append(orient * float(signed_curve_area(Ck).detach().cpu()))
        teacher_min_oriented_area = min(vals) if vals else float("inf")

    valid = bool(
        torch.isfinite(task.P_target).all()
        and torch.isfinite(task.eta_target).all()
        and float(spans.min().detach().cpu()) >= cfg.min_knot_span - 1e-10
        and curve_selfx == 0
        and polygon_selfx == 0
        and source_curve_selfx == 0
        and (not cfg.validate_teacher_path or teacher_max_selfx == 0)
        and (not cfg.validate_teacher_path or teacher_min_oriented_area >= teacher_area_floor)
    )
    return {
        "valid": valid,
        "curve_self_intersections": curve_selfx,
        "polygon_self_intersections": polygon_selfx,
        "source_curve_self_intersections": source_curve_selfx,
        "teacher_max_self_intersections": teacher_max_selfx,
        "teacher_min_oriented_area": float(teacher_min_oriented_area),
        "teacher_area_floor": float(teacher_area_floor),
        "min_knot_span": float(spans.min().detach().cpu()),
        "max_knot_span": float(spans.max().detach().cpu()),
        "closure_gap": float(gap.norm().detach().cpu()),
        "tangent_mismatch": float(tan.norm().detach().cpu()),
        "representation_floor": float(task.representation_floor),
        "alpha": float(task.target_alpha),
    }



def validate_task_collection(tasks: list[SplineCurveTask], cfg: SplineCurveConfig) -> list[dict[str, float | int | bool]]:
    return [validate_spline_task_target(t, cfg) for t in tasks]


def task_to_cpu_dict(task: SplineCurveTask) -> dict[str, Any]:
    return {
        "mode": task.mode,
        "P0": task.P0.detach().cpu(), "eta0": task.eta0.detach().cpu(),
        "P_target": task.P_target.detach().cpu(), "eta_target": task.eta_target.detach().cpu(),
        "pose0": task.pose0.detach().cpu(), "pose_target": task.pose_target.detach().cpu(),
        "target_samples": task.target_samples.detach().cpu(), "task_id": task.task_id,
        "hard_ood": task.hard_ood, "target_family": task.target_family,
        "representation_floor": task.representation_floor, "target_hash": task.target_hash,
        "conditioning_case": task.conditioning_case,
        "observation_samples": None if task.observation_samples is None else task.observation_samples.detach().cpu(),
        "target_alpha": task.target_alpha, "target_params": task.target_params,
    }


def task_from_cpu_dict(d: dict[str, Any], device: torch.device, dtype: torch.dtype) -> SplineCurveTask:
    obs = d.get("observation_samples")
    pose0 = d.get("pose0", torch.zeros(3))
    pose_target = d.get("pose_target", torch.zeros(3))
    return SplineCurveTask(
        mode=d["mode"], P0=d["P0"].to(device=device, dtype=dtype), eta0=d["eta0"].to(device=device, dtype=dtype),
        P_target=d["P_target"].to(device=device, dtype=dtype), eta_target=d["eta_target"].to(device=device, dtype=dtype),
        pose0=pose0.to(device=device, dtype=dtype), pose_target=pose_target.to(device=device, dtype=dtype),
        target_samples=d["target_samples"].to(device=device, dtype=dtype), task_id=int(d["task_id"]),
        hard_ood=bool(d["hard_ood"]), target_family=d["target_family"],
        representation_floor=float(d["representation_floor"]), target_hash=d["target_hash"],
        conditioning_case=d.get("conditioning_case", "configuration"),
        observation_samples=None if obs is None else obs.to(device=device, dtype=dtype),
        target_alpha=float(d.get("target_alpha", 0.0)), target_params=d.get("target_params"),
    )
