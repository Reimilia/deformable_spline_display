from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from typing import Iterable

import numpy as np
import torch

from .chain import PlanarMultiLinkChain
from .tasks import LocomotionTask
from .geometry import se2_compose, se2_exp_increment


@dataclass(frozen=True)
class RigidLoopConfig:
    """Numerical settings for a closed, inextensible articulated loop."""

    closure_position_gain: float = 10.0
    closure_regularization: float = 1e-8
    position_projection_iterations: int = 5
    target_fit_iterations: int = 500
    target_fit_lr: float = 2e-2
    closure_loss_weight: float = 2500.0
    target_shape_weight: float = 1.0
    joint_limit_weight: float = 20.0
    # v18 target/measurement protocol.  The crescent is represented by all M
    # physical link midpoints and tangents, while seam continuity is measured
    # independently by x_M-x_0.
    target_tangent_weight: float = 0.06
    target_joint_margin: float = 0.96
    target_max_measurement_rmse: float = 0.16
    crescent_bulge_min: float = 0.38
    crescent_bulge_max: float = 0.44
    crescent_vertical_min: float = 0.90
    crescent_vertical_max: float = 1.10
    strict_target_validation: bool = True



@dataclass(frozen=True)
class LoopClosureModel:
    """How the material seam is treated during a rigid-loop rollout.

    ``hard`` reproduces the holonomic velocity/configuration projection used in
    v14. ``measurement`` removes those projections and replaces them with a
    compliant endpoint measurement potential. ``open`` applies neither.

    ``tolerance_rel`` is the one-sigma endpoint-position tolerance normalized by
    the total material perimeter.  The measurement potential uses
    K_c = energy_scale / sigma^2, so decreasing the tolerance tightens the seam
    without changing the target-shape loss itself.
    """

    mode: str = "hard"
    tolerance_rel: float = 0.05
    energy_scale: float = 1.0e-3
    damping_scale: float = 0.12
    eps: float = 1.0e-10

    def normalized_mode(self) -> str:
        mode = self.mode.lower().strip()
        aliases = {"close": "hard", "closed": "hard", "exact": "hard", "soft": "measurement", "measured": "measurement"}
        mode = aliases.get(mode, mode)
        if mode not in {"hard", "measurement", "open"}:
            raise ValueError(f"unknown loop closure mode: {self.mode}")
        return mode


def closure_model_from_task(task, *, default: LoopClosureModel | None = None) -> LoopClosureModel:
    """Recover the closure model stored in task metadata."""
    base = default or LoopClosureModel()
    md = getattr(task, "metadata", None) or {}
    return LoopClosureModel(
        mode=str(md.get("rigid_loop_closure_mode", base.mode)),
        tolerance_rel=float(md.get("rigid_loop_closure_tolerance_rel", base.tolerance_rel)),
        energy_scale=float(md.get("rigid_loop_closure_energy_scale", base.energy_scale)),
        damping_scale=float(md.get("rigid_loop_closure_damping_scale", base.damping_scale)),
        eps=float(md.get("rigid_loop_closure_eps", base.eps)),
    )


def set_task_closure_model(task, closure: LoopClosureModel) -> None:
    """Attach a closure treatment to a rigid-loop task in-place."""
    if task.metadata is None:
        task.metadata = {}
    task.metadata.update({
        "rigid_loop_closure_mode": closure.normalized_mode(),
        "rigid_loop_closure_tolerance_rel": float(closure.tolerance_rel),
        "rigid_loop_closure_energy_scale": float(closure.energy_scale),
        "rigid_loop_closure_damping_scale": float(closure.damping_scale),
        "rigid_loop_closure_eps": float(closure.eps),
    })


def measurement_closure_terms(
    chain: PlanarMultiLinkChain,
    q: torch.Tensor,
    qdot: torch.Tensor,
    closure: LoopClosureModel,
) -> dict[str, torch.Tensor]:
    """Compliant measurement-driven seam terms in intrinsic coordinates.

    The measurement is y_c = x_M(q)-x_0(q) with covariance sigma^2 I.
    Its pullback stiffness is J_c^T K_c J_c.  The returned damping matrix is
    positive semidefinite and can be added directly to the pH dissipation block.
    """
    mode = closure.normalized_mode()
    dtype, device = q.dtype, q.device
    nj = chain.n_joints
    gap = loop_closure(chain, q)
    J = loop_closure_jacobian(chain, q)
    perimeter = loop_perimeter(chain, q).to(dtype=dtype, device=device)
    sigma = torch.as_tensor(max(float(closure.tolerance_rel), 1e-8), dtype=dtype, device=device) * perimeter
    zeroj = torch.zeros(nj, dtype=dtype, device=device)
    zeroM = torch.zeros(nj, nj, dtype=dtype, device=device)
    if mode != "measurement":
        return {
            "gap": gap,
            "jacobian": J,
            "sigma": sigma,
            "stiffness": torch.zeros((), dtype=dtype, device=device),
            "damping": torch.zeros((), dtype=dtype, device=device),
            "potential": torch.zeros((), dtype=dtype, device=device),
            "measurement_loss": torch.zeros((), dtype=dtype, device=device),
            "spring_force_joint": zeroj,
            "damping_force_joint": zeroj,
            "damping_matrix_joint": zeroM,
        }
    k = torch.as_tensor(float(closure.energy_scale), dtype=dtype, device=device) / (sigma * sigma + closure.eps)
    # Damping is defined in the 2-D endpoint measurement space.  sqrt(k) keeps
    # the damping growth milder than the stiffness growth as tolerance shrinks.
    d = torch.as_tensor(float(closure.damping_scale), dtype=dtype, device=device) * torch.sqrt(k.clamp_min(0.0))
    vgap = J @ qdot
    fs = -J.transpose(0, 1) @ (k * gap)
    fd = -J.transpose(0, 1) @ (d * vgap)
    Dj = d * (J.transpose(0, 1) @ J)
    potential = 0.5 * k * torch.dot(gap, gap)
    measurement_loss = 0.5 * torch.dot(gap, gap) / (sigma * sigma + closure.eps)
    return {
        "gap": gap,
        "jacobian": J,
        "sigma": sigma,
        "stiffness": k,
        "damping": d,
        "potential": potential,
        "measurement_loss": measurement_loss,
        "spring_force_joint": fs,
        "damping_force_joint": fd,
        "damping_matrix_joint": Dj,
    }

def loop_closure(chain: PlanarMultiLinkChain, q: torch.Tensor) -> torch.Tensor:
    """Body-frame closure residual x_M-x_0 for the serial representation."""
    return chain.kinematics(q)["joint_positions"][..., -1, :]


def loop_closure_jacobian(chain: PlanarMultiLinkChain, q: torch.Tensor) -> torch.Tensor:
    """Analytical dc/dq for c(q)=sum_i l_i t_i(q), shape (2,M-1)."""
    kin = chain.kinematics(q)
    tangents = kin["tangents"]
    Jt = chain._Jvec(tangents)
    J = torch.zeros(2, chain.n_joints, dtype=q.dtype, device=q.device)
    for j in range(chain.n_joints):
        # q_j rotates links j+1,...,M-1.
        for a in range(j + 1, chain.n_links):
            J[:, j] = J[:, j] + chain.lengths[a] * Jt[a]
    return J


def loop_perimeter(chain: PlanarMultiLinkChain, q: torch.Tensor | None = None) -> torch.Tensor:
    # Rigid links make this independent of q. Keep q argument for a uniform diagnostics API.
    return chain.lengths.sum()


def loop_body_vertices(chain: PlanarMultiLinkChain, q: torch.Tensor, *, unique: bool = True) -> torch.Tensor:
    """Return body-frame polygon vertices generated by the *physical* fixed-edge chain.

    ``PlanarMultiLinkChain`` already fixes the material gauge: vertex zero is the
    body origin and the first link points along +x.  For shape comparison we drop
    the duplicate closing vertex by default.
    """
    pts = chain.kinematics(q)["joint_positions"]
    return pts[..., :-1, :] if unique else pts


def canonicalize_loop_vertices(
    points: torch.Tensor,
    *,
    drop_duplicate_endpoint: bool = True,
) -> torch.Tensor:
    """Remove global translation/orientation from an ordered perimeter point set.

    ``drop_duplicate_endpoint`` is kept for backward compatibility with the hard
    closed-loop representation, where the final kinematic point duplicates the
    first material vertex.  Measurement/open seam rollouts must *not* decide the
    vertex count from the instantaneous endpoint gap, because an open seam has
    ``M+1`` kinematic points while the local-shape objective is defined on the
    ``M`` material perimeter vertices.  ``cyclic_local_shape_loss`` therefore
    normalizes the material vertex convention explicitly before calling this
    routine with ``drop_duplicate_endpoint=False``.
    """
    pts = points
    if drop_duplicate_endpoint and pts.shape[-2] >= 2:
        # Use a detached scalar only to select the representation; this branch is
        # not part of the differentiable geometry calculation itself.
        if bool(torch.linalg.norm((pts[..., -1, :] - pts[..., 0, :]).detach()).item() < 1e-9):
            pts = pts[..., :-1, :]
    if pts.shape[-2] < 2:
        raise ValueError("at least two material vertices are required")
    centered = pts - pts.mean(dim=-2, keepdim=True)
    e0 = centered[..., 1, :] - centered[..., 0, :]
    theta = torch.atan2(e0[..., 1], e0[..., 0])
    c, s = torch.cos(theta), torch.sin(theta)
    # Row-vector multiplication by R(-theta)^T.
    x = centered[..., 0] * c.unsqueeze(-1) + centered[..., 1] * s.unsqueeze(-1)
    y = -centered[..., 0] * s.unsqueeze(-1) + centered[..., 1] * c.unsqueeze(-1)
    return torch.stack([x, y], dim=-1)


def _material_vertices_for_shape_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    closure_tol: float = 1e-9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize full/unique seam conventions to the same material vertex count.

    The rigid-chain forward kinematics returns ``M+1`` points.  In hard closure,
    point ``M`` duplicates point ``0``; in measurement/open modes it is the second
    side of the material seam and is generally distinct.  The *local shape* loss
    is intentionally evaluated on vertices ``0,...,M-1`` while endpoint
    continuity is handled separately by the closure measurement/loss.

    Callers in the plotting code sometimes provide the full ``M+1`` rollout and
    callers in the training code sometimes provide the already-unique ``M``
    perimeter.  This helper makes both conventions safe and prevents a relaxed
    seam from changing the tensor size merely because the endpoints no longer
    coincide.
    """
    p, t = predicted, target
    np_, nt_ = p.shape[-2], t.shape[-2]
    if np_ < 2 or nt_ < 2:
        raise ValueError("shape loss requires at least two vertices")

    # If counts differ by one, the longer representation is the full kinematic
    # chain with an explicit seam endpoint.  Drop that endpoint for local shape.
    if np_ == nt_ + 1:
        p = p[..., :-1, :]
    elif nt_ == np_ + 1:
        t = t[..., :-1, :]
    elif np_ == nt_:
        # Equal counts can still mean both callers supplied full M+1 arrays.  A
        # closed target has an explicit duplicated endpoint.  Drop the final point
        # from *both* arrays so relaxed/open prediction is compared on the same M
        # material vertices and its seam gap remains a separate measurement.
        p_closed = bool(torch.linalg.norm((p[..., -1, :] - p[..., 0, :]).detach()).item() < closure_tol)
        t_closed = bool(torch.linalg.norm((t[..., -1, :] - t[..., 0, :]).detach()).item() < closure_tol)
        if p.shape[-2] > 2 and (p_closed or t_closed):
            p = p[..., :-1, :]
            t = t[..., :-1, :]
    else:
        raise ValueError(
            f"incompatible perimeter vertex counts for shape loss: predicted={np_}, target={nt_}; "
            "expected equal counts or a single explicit seam endpoint"
        )

    if p.shape[-2] != t.shape[-2]:
        raise RuntimeError(
            f"failed to normalize perimeter vertex counts: predicted={p.shape[-2]}, target={t.shape[-2]}"
        )
    return p, t


def cyclic_local_shape_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    allow_cyclic_shift: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pose-invariant local geometry loss for a rigid-loop material perimeter.

    Local deformation and seam continuity are intentionally decoupled.  The
    geometry term compares the ``M`` material vertices and ignores the explicit
    kinematic seam endpoint; ``measurement_closure_terms`` (or the hard closure
    constraint) evaluates ``x_M-x_0`` separately.  This convention is essential
    for ``closure_mode=measurement`` and ``closure_mode=open``.

    When ``allow_cyclic_shift`` is true, the target seam is treated as a gauge and
    all cyclic material-index shifts are compared.  Reversal is deliberately not
    allowed: the material orientation of the rod is preserved.
    """
    p_raw, t_raw = _material_vertices_for_shape_loss(predicted, target)
    p = canonicalize_loop_vertices(p_raw, drop_duplicate_endpoint=False)
    variants = []
    shifts = range(t_raw.shape[-2]) if allow_cyclic_shift else range(1)
    for k in shifts:
        tk = torch.roll(t_raw, shifts=-int(k), dims=-2)
        tk = canonicalize_loop_vertices(tk, drop_duplicate_endpoint=False)
        variants.append(torch.mean((p - tk) ** 2))
    vals = torch.stack(variants)
    idx = torch.argmin(vals)
    return vals[idx], idx


def rigid_target_representation_error(
    chain: PlanarMultiLinkChain,
    q_target: torch.Tensor,
    sampled_target: torch.Tensor,
) -> torch.Tensor:
    """Best pose/seam-invariant error between fixed-edge target and sampled crescent."""
    rigid = loop_body_vertices(chain, q_target, unique=True)
    loss, _ = cyclic_local_shape_loss(rigid, sampled_target, allow_cyclic_shift=True)
    return torch.sqrt(loss + 1e-16)


def project_loop_configuration(
    chain: PlanarMultiLinkChain,
    q: torch.Tensor,
    *,
    iterations: int = 5,
    regularization: float = 1e-8,
) -> torch.Tensor:
    """Newton projection of q onto c(q)=0 while preserving every link length."""
    x = q
    eye2 = torch.eye(2, dtype=q.dtype, device=q.device)
    for _ in range(iterations):
        c = loop_closure(chain, x)
        J = loop_closure_jacobian(chain, x)
        W = J @ J.transpose(0, 1) + regularization * eye2
        dq = -J.transpose(0, 1) @ torch.linalg.solve(W, c)
        x = x + dq
    return x


def project_loop_velocity_mass_metric(
    chain: PlanarMultiLinkChain,
    q: torch.Tensor,
    M: torch.Tensor,
    nu_free: torch.Tensor,
    *,
    closure_position_gain: float = 10.0,
    regularization: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mass-metric velocity projection for the two loop-closure constraints.

    The closure is invariant under the floating SE(2) pose, hence the first three
    columns of J_full vanish. A Baumgarte-like target velocity -k*c stabilizes
    position drift between configuration projections.
    """
    Jq = loop_closure_jacobian(chain, q)
    J = torch.zeros(2, 3 + chain.n_joints, dtype=q.dtype, device=q.device)
    J[:, 3:] = Jq
    c = loop_closure(chain, q)
    v_target = -closure_position_gain * c
    Minv_JT = torch.linalg.solve(M, J.transpose(0, 1))
    W = J @ Minv_JT + regularization * torch.eye(2, dtype=q.dtype, device=q.device)
    lam = torch.linalg.solve(W, J @ nu_free - v_target)
    nu = nu_free - Minv_JT @ lam
    # generalized reaction force that maps the unconstrained velocity to nu over one unit time;
    # caller can divide the corresponding momentum change by dt for an actual force diagnostic.
    reaction_direction = -J.transpose(0, 1) @ lam
    return nu, reaction_direction


def regular_polygon_joint_angles(chain: PlanarMultiLinkChain) -> torch.Tensor:
    """M equal edges forming a regular closed polygon; q has M-1 turning angles."""
    ang = 2.0 * math.pi / float(chain.n_links)
    q = torch.full((chain.n_joints,), ang, dtype=chain.dtype, device=chain.device)
    return project_loop_configuration(chain, q, iterations=8)


def _closed_arc_resample(points: np.ndarray, n_edges: int) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-12:
        pts = np.vstack([pts, pts[0]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    ss = np.linspace(0.0, total, n_edges + 1)
    out = np.zeros((n_edges + 1, 2), dtype=float)
    for d in range(2):
        out[:, d] = np.interp(ss, s, pts[:, d])
    out[-1] = out[0]
    return out


def _curve_samples_at_arclength(points: np.ndarray, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate positions and unit tangents of a dense closed polyline."""
    pts = np.asarray(points, dtype=float)
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-12:
        pts = np.vstack([pts, pts[0]])
    d = np.diff(pts, axis=0)
    seg = np.linalg.norm(d, axis=1)
    seg_safe = np.maximum(seg, 1e-12)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    uu = np.mod(np.asarray(samples, dtype=float), total)
    idx = np.searchsorted(s, uu, side="right") - 1
    idx = np.clip(idx, 0, len(seg) - 1)
    a = (uu - s[idx]) / seg_safe[idx]
    pos = pts[idx] + a[:, None] * d[idx]
    tan = d[idx] / seg_safe[idx, None]
    return pos, tan


def _canonicalize_reference_measurements(
    centers: np.ndarray,
    tangents: np.ndarray,
    dense_curve: np.ndarray,
    sampled_vertices: np.ndarray,
    first_link_length: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Put target measurements in the material gauge used by PlanarMultiLinkChain.

    The first physical link points along +x and its midpoint lies at l_0/2.  We
    rotate the first target tangent to +x and translate the first target center to
    the same point.  This removes global SE(2) placement without deleting any
    physical link measurement.
    """
    theta = math.atan2(float(tangents[0, 1]), float(tangents[0, 0]))
    c, sn = math.cos(-theta), math.sin(-theta)
    R = np.array([[c, -sn], [sn, c]], dtype=float)
    centers_r = centers @ R.T
    tangents_r = tangents @ R.T
    dense_r = dense_curve @ R.T
    vertices_r = sampled_vertices @ R.T
    shift = np.array([0.5 * first_link_length, 0.0]) - centers_r[0]
    centers_r = centers_r + shift
    dense_r = dense_r + shift
    vertices_r = vertices_r + shift
    return centers_r, tangents_r, dense_r, vertices_r


def smooth_crescent_reference_data(
    n_links: int,
    perimeter: float,
    *,
    bulge: float = 0.40,
    vertical_scale: float = 1.0,
    dense: int = 2400,
) -> dict[str, np.ndarray | float]:
    """Smooth bounded-curvature crescent-like reference for rigid-link fitting.

    The legacy two-arc moon had a sharp inner tip whose discrete turning angle
    could exceed the robot joint range by more than a factor of two.  v18 uses a
    smooth second-harmonic closed curve

        x(theta)=cos(theta)+a cos(2 theta),  y(theta)=b sin(theta),

    which can be made visibly concave while keeping turning angles bounded.  The
    *continuous* reference remains a measurement target; the physical training
    target is still the best fixed-edge articulated fit under the selected seam
    model.
    """
    th = np.linspace(0.0, 2.0 * np.pi, dense, endpoint=False)
    pts = np.column_stack([
        np.cos(th) + float(bulge) * np.cos(2.0 * th),
        float(vertical_scale) * np.sin(th),
    ])
    pts = np.vstack([pts, pts[0]])
    L0 = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    pts = pts * (float(perimeter) / max(float(L0), 1e-12))

    # M desired link measurements are taken at equal material arc bins.  They do
    # not encode endpoint closure; closure is a separate measurement/constraint.
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    edge_s = np.linspace(0.0, total, n_links + 1)
    center_s = (np.arange(n_links, dtype=float) + 0.5) * total / float(n_links)
    centers, tangents = _curve_samples_at_arclength(pts, center_s)
    vertices, _ = _curve_samples_at_arclength(pts, edge_s[:-1])
    vertices = np.vstack([vertices, vertices[0]])
    centers, tangents, dense_body, vertices_body = _canonicalize_reference_measurements(
        centers, tangents, pts, vertices, float(perimeter) / float(n_links)
    )
    angles = np.unwrap(np.arctan2(tangents[:, 1], tangents[:, 0]))
    raw_q = np.arctan2(np.sin(np.diff(angles)), np.cos(np.diff(angles)))
    curve_hash = hashlib.sha1(np.round(dense_body, 11).tobytes()).hexdigest()
    return {
        "dense": dense_body,
        "vertices": vertices_body,
        "link_centers": centers,
        "link_tangents": tangents,
        "raw_q": raw_q,
        "curve_hash": curve_hash,
        "bulge": float(bulge),
        "vertical_scale": float(vertical_scale),
    }


def crescent_reference_vertices(
    n_links: int,
    perimeter: float,
    *,
    outer_bulge: float = 1.0,
    inner_ratio: float = 0.30,
    vertical_scale: float = 1.0,
    dense: int = 2400,
) -> np.ndarray:
    """Backward-compatible wrapper returning the smooth v18 crescent samples.

    ``inner_ratio`` now controls the smooth second-harmonic bulge.  Keeping this
    wrapper avoids breaking older plotting/tests while removing the sharp-tip
    two-arc target from the training protocol.
    """
    bulge = float(np.clip(0.24 + 0.40 * (0.42 - float(inner_ratio)) / 0.24, 0.22, 0.48))
    data = smooth_crescent_reference_data(
        n_links, perimeter, bulge=bulge, vertical_scale=vertical_scale, dense=dense
    )
    return np.asarray(data["vertices"], dtype=float)


def _turning_angles_from_vertices(points: np.ndarray) -> np.ndarray:
    e = np.diff(points, axis=0)
    ang = np.unwrap(np.arctan2(e[:, 1], e[:, 0]))
    q = np.diff(ang)
    q = np.arctan2(np.sin(q), np.cos(q))
    return q


def link_measurements_from_vertices(vertices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """All M physical link midpoint/tangent measurements from M+1 vertices."""
    e = vertices[..., 1:, :] - vertices[..., :-1, :]
    lengths = torch.linalg.norm(e, dim=-1, keepdim=True).clamp_min(1e-12)
    tangents = e / lengths
    centers = 0.5 * (vertices[..., 1:, :] + vertices[..., :-1, :])
    return centers, tangents


def rigid_link_measurement_loss(
    predicted_vertices: torch.Tensor,
    target_centers: torch.Tensor,
    target_tangents: torch.Tensor,
    *,
    perimeter: float,
    tangent_weight: float = 0.06,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mode-independent shape measurement on all M physical links.

    This replaces the v16/v17 vertex loss that dropped x_M.  Every physical link
    now contributes through its midpoint and tangent, while x_M-x_0 remains an
    independent seam measurement.  The positional term is normalized by total
    perimeter so the loss is dimensionless.
    """
    centers, tangents = link_measurements_from_vertices(predicted_vertices)
    if centers.shape[-2:] != target_centers.shape[-2:]:
        raise ValueError(f"link-center measurement shape mismatch: {centers.shape} vs {target_centers.shape}")
    if tangents.shape[-2:] != target_tangents.shape[-2:]:
        raise ValueError(f"link-tangent measurement shape mismatch: {tangents.shape} vs {target_tangents.shape}")
    scale = max(float(perimeter), 1e-12)
    pos_mse = torch.mean(((centers - target_centers) / scale) ** 2)
    tan_mse = torch.mean((tangents - target_tangents) ** 2)
    total = pos_mse + float(tangent_weight) * tan_mse
    return total, pos_mse, tan_mse


def rigid_target_measurement_errors(
    chain: PlanarMultiLinkChain,
    q_target: torch.Tensor,
    target_centers: torch.Tensor,
    target_tangents: torch.Tensor,
    *,
    tangent_weight: float = 0.06,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    verts = chain.kinematics(q_target)["joint_positions"]
    loss, pos, tan = rigid_link_measurement_loss(
        verts, target_centers, target_tangents,
        perimeter=float(chain.params.total_length), tangent_weight=tangent_weight,
    )
    return torch.sqrt(loss + 1e-16), torch.sqrt(pos + 1e-16), torch.sqrt(tan + 1e-16)

def _bounded_loop_projection(
    chain: PlanarMultiLinkChain,
    q: torch.Tensor,
    *,
    bound: float,
    iterations: int = 16,
    regularization: float = 1e-8,
) -> torch.Tensor:
    """Damped Newton closure projection that never leaves the joint-limit box."""
    x = q.clone()
    eye2 = torch.eye(2, dtype=q.dtype, device=q.device)
    for _ in range(iterations):
        c = loop_closure(chain, x)
        if float(torch.linalg.norm(c).detach()) < 1e-10:
            break
        J = loop_closure_jacobian(chain, x)
        W = J @ J.transpose(0, 1) + regularization * eye2
        dq = -J.transpose(0, 1) @ torch.linalg.solve(W, c)
        old = float(torch.dot(c, c).detach())
        step = 1.0
        accepted = False
        for _ls in range(14):
            trial = x + step * dq
            if float(trial.abs().max().detach()) >= bound:
                step *= 0.5
                continue
            ct = loop_closure(chain, trial)
            if float(torch.dot(ct, ct).detach()) < old:
                x = trial
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    return x


def fit_rigid_crescent_joint_angles(
    chain: PlanarMultiLinkChain,
    target_vertices: np.ndarray | None = None,
    cfg: RigidLoopConfig | None = None,
    *,
    closure: LoopClosureModel | None = None,
    target_link_centers: np.ndarray | None = None,
    target_link_tangents: np.ndarray | None = None,
) -> torch.Tensor:
    """Fit a fixed-edge target using *all M link measurements* plus seam physics.

    v18 no longer fits only vertices 0,...,M-1.  The target is specified by M
    link midpoint/tangent measurements, so the final physical link is observable
    in open and measurement modes.  Endpoint continuity remains a separate seam
    objective/constraint.
    """
    cfg = cfg or RigidLoopConfig()
    closure = closure or LoopClosureModel(mode="hard")
    mode = closure.normalized_mode()

    if target_link_centers is None or target_link_tangents is None:
        if target_vertices is None:
            raise ValueError("target measurements or target_vertices are required")
        tv = torch.as_tensor(target_vertices, dtype=chain.dtype, device=chain.device)
        cc, tt = link_measurements_from_vertices(tv)
        target_link_centers = cc.detach().cpu().numpy()
        target_link_tangents = tt.detach().cpu().numpy()

    target_centers = torch.as_tensor(target_link_centers, dtype=chain.dtype, device=chain.device)
    target_tangents = torch.as_tensor(target_link_tangents, dtype=chain.dtype, device=chain.device)
    if target_centers.shape != (chain.n_links, 2) or target_tangents.shape != (chain.n_links, 2):
        raise ValueError("target link measurements must have shape (n_links,2)")

    # Initialize directly from desired consecutive link directions.  This is much
    # closer to the physical target than converting unequal target chords.
    a = torch.atan2(target_tangents[:, 1], target_tangents[:, 0])
    raw_t = torch.atan2(torch.sin(a[1:] - a[:-1]), torch.cos(a[1:] - a[:-1]))
    bound = float(cfg.target_joint_margin) * float(chain.params.joint_limit)
    raw_t = raw_t.clamp(-0.96 * bound, 0.96 * bound)
    z0 = torch.atanh((raw_t / max(bound, 1e-8)).clamp(-0.999, 0.999))
    z = torch.nn.Parameter(z0.clone())
    opt = torch.optim.Adam([z], lr=cfg.target_fit_lr)
    best_q = None
    best_score = float("inf")

    for it in range(cfg.target_fit_iterations):
        q = bound * torch.tanh(z)
        verts = chain.kinematics(q)["joint_positions"]
        fit, _, _ = rigid_link_measurement_loss(
            verts, target_centers, target_tangents,
            perimeter=float(chain.params.total_length),
            tangent_weight=cfg.target_tangent_weight,
        )
        seam = verts[-1] - verts[0]
        frac = (it + 1) / max(cfg.target_fit_iterations, 1)
        if mode == "hard":
            seam_penalty = cfg.closure_loss_weight * (0.10 + 0.90 * frac**2) * torch.dot(seam, seam)
        elif mode == "measurement":
            sigma = max(float(closure.tolerance_rel) * float(chain.params.total_length), 1e-8)
            seam_penalty = (0.10 + 0.90 * frac**2) * 0.5 * torch.dot(seam, seam) / (sigma * sigma)
        else:
            seam_penalty = torch.zeros((), dtype=q.dtype, device=q.device)
        # Only penalize proximity to the *configured* joint margin.  The target
        # family is validated separately instead of silently rounding it away.
        margin = torch.relu(q.abs() - 0.985 * bound).pow(2).mean()
        loss = cfg.target_shape_weight * fit + seam_penalty + cfg.joint_limit_weight * margin
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([z], 20.0)
        opt.step()
        score = float(loss.detach())
        if score < best_score:
            best_score = score
            best_q = q.detach().clone()
    assert best_q is not None
    q = best_q
    if mode == "hard":
        q = _bounded_loop_projection(
            chain, q, bound=bound, iterations=24,
            regularization=cfg.closure_regularization,
        )
    return q.detach()


def validate_rigid_loop_target(
    chain: PlanarMultiLinkChain,
    q_target: torch.Tensor,
    target_centers: torch.Tensor,
    target_tangents: torch.Tensor,
    cfg: RigidLoopConfig,
) -> dict[str, float | bool]:
    """Pre-training target validation; failures are never hidden by training."""
    total_rmse, pos_rmse, tan_rmse = rigid_target_measurement_errors(
        chain, q_target, target_centers, target_tangents,
        tangent_weight=cfg.target_tangent_weight,
    )
    max_joint = float(q_target.abs().max())
    joint_cap = float(cfg.target_joint_margin) * float(chain.params.joint_limit)
    return {
        "measurement_rmse": float(total_rmse),
        "position_rmse": float(pos_rmse),
        "tangent_rmse": float(tan_rmse),
        "max_abs_joint": max_joint,
        "joint_cap": joint_cap,
        "joint_margin_ok": bool(max_joint <= joint_cap + 1e-7),
        "representation_ok": bool(float(total_rmse) <= float(cfg.target_max_measurement_rmse)),
        "valid": bool(max_joint <= joint_cap + 1e-7 and float(total_rmse) <= float(cfg.target_max_measurement_rmse)),
    }

def make_rigid_loop_task(
    chain: PlanarMultiLinkChain,
    Ksp: torch.Tensor,
    n_temporal_ctrl: int,
    *,
    seed: int,
    pose0: torch.Tensor | None = None,
    cfg: RigidLoopConfig | None = None,
    closure: LoopClosureModel | None = None,
    reference_spec: dict | None = None,
) -> LocomotionTask:
    """Random circle -> smooth crescent task with validated physical target.

    The underlying continuous crescent is sampled independently of the seam mode.
    Each closure variant then fits its own physically realizable fixed-edge target
    to the *same* M link measurements.  This is required for a fair open / soft /
    hard closure ablation.
    """
    cfg = cfg or RigidLoopConfig()
    closure = closure or LoopClosureModel(mode="hard")
    rng = np.random.default_rng(seed)
    q0 = regular_polygon_joint_angles(chain)

    if reference_spec is None:
        bulge = float(rng.uniform(cfg.crescent_bulge_min, cfg.crescent_bulge_max))
        vertical = float(rng.uniform(cfg.crescent_vertical_min, cfg.crescent_vertical_max))
    else:
        bulge = float(reference_spec["bulge"])
        vertical = float(reference_spec["vertical_scale"])

    # Keep the reference itself within the joint-range scale before any seam-mode
    # fit. This adaptation depends only on geometry/joint limits, so all closure
    # variants with the same seed still use the same reference curve.
    ref = None
    for _ in range(10):
        ref = smooth_crescent_reference_data(
            chain.n_links, float(chain.params.total_length),
            bulge=bulge, vertical_scale=vertical,
        )
        raw_max = float(np.max(np.abs(np.asarray(ref["raw_q"])))) if chain.n_joints else 0.0
        if raw_max <= 0.98 * float(cfg.target_joint_margin) * float(chain.params.joint_limit):
            break
        bulge *= 0.94
    assert ref is not None

    target_centers_np = np.asarray(ref["link_centers"], dtype=float)
    target_tangents_np = np.asarray(ref["link_tangents"], dtype=float)
    target_ref = np.asarray(ref["vertices"], dtype=float)
    target_dense = np.asarray(ref["dense"], dtype=float)

    qT = fit_rigid_crescent_joint_angles(
        chain, target_ref, cfg, closure=closure,
        target_link_centers=target_centers_np,
        target_link_tangents=target_tangents_np,
    )
    target_centers = torch.as_tensor(target_centers_np, dtype=Ksp.dtype, device=Ksp.device)
    target_tangents = torch.as_tensor(target_tangents_np, dtype=Ksp.dtype, device=Ksp.device)
    validation = validate_rigid_loop_target(chain, qT, target_centers, target_tangents, cfg)
    if cfg.strict_target_validation and not bool(validation["valid"]):
        raise RuntimeError(
            "rigid-loop crescent target failed pre-training validation: "
            f"measurement_rmse={validation['measurement_rmse']:.4f} "
            f"(limit {cfg.target_max_measurement_rmse:.4f}), "
            f"max|q|={validation['max_abs_joint']:.4f} "
            f"(cap {validation['joint_cap']:.4f}). "
            "Increase n_links/joint range or reduce crescent bulge before training."
        )

    Kpinv = torch.linalg.pinv(Ksp)
    c0 = Kpinv @ q0
    cT = Kpinv @ qT
    u = torch.linspace(0.0, 1.0, n_temporal_ctrl, dtype=Ksp.dtype, device=Ksp.device)[:, None]
    controls = (1.0 - u) * c0[None, :] + u * cT[None, :]
    controls[1:-1] += 0.03 * torch.randn_like(controls[1:-1])
    if pose0 is None:
        if reference_spec is not None and "pose0" in reference_spec:
            pose0 = torch.as_tensor(reference_spec["pose0"], dtype=Ksp.dtype, device=Ksp.device)
        else:
            pose0 = torch.tensor([
                float(rng.uniform(-0.20, 0.20)),
                float(rng.uniform(-0.20, 0.20)),
                float(rng.uniform(-math.pi, math.pi)),
            ], dtype=Ksp.dtype, device=Ksp.device)

    rigid_vertices = chain.kinematics(qT)["joint_positions"].detach()
    rep_total, rep_pos, rep_tan = rigid_target_measurement_errors(
        chain, qT, target_centers, target_tangents,
        tangent_weight=cfg.target_tangent_weight,
    )
    metadata = {
        "external_task": "rigid_loop",
        "rigid_loop": True,
        "joint_limit": float(chain.params.joint_limit),
        "rigid_loop_q_circle": q0.detach().cpu().tolist(),
        "rigid_loop_q_target": qT.detach().cpu().tolist(),
        "rigid_loop_target_vertices_body": rigid_vertices.cpu().tolist(),
        "rigid_loop_target_rigid_vertices_body": rigid_vertices.cpu().tolist(),
        "rigid_loop_target_sampled_crescent_body": target_ref.tolist(),
        "rigid_loop_target_dense_crescent_body": target_dense.tolist(),
        "rigid_loop_target_link_centers_body": target_centers_np.tolist(),
        "rigid_loop_target_link_tangents_body": target_tangents_np.tolist(),
        "rigid_loop_initial_vertices_body": chain.kinematics(q0)["joint_positions"].detach().cpu().tolist(),
        "rigid_loop_perimeter": float(chain.params.total_length),
        "rigid_loop_target_family": "smooth_fourier_crescent_v18",
        "rigid_loop_reference_spec": {"bulge": float(bulge), "vertical_scale": float(vertical)},
        "rigid_loop_reference_curve_hash": str(ref["curve_hash"]),
        "rigid_loop_raw_reference_max_abs_turn": float(np.max(np.abs(np.asarray(ref["raw_q"])))),
        "rigid_loop_initial_closure_error": float(torch.linalg.norm(loop_closure(chain, q0))),
        "rigid_loop_target_closure_error": float(torch.linalg.norm(loop_closure(chain, qT))),
        "rigid_loop_target_max_abs_joint": float(qT.abs().max()),
        "rigid_loop_target_joint_limit": float(chain.params.joint_limit),
        "rigid_loop_target_within_joint_limit": bool(float(qT.abs().max()) < float(chain.params.joint_limit)),
        "rigid_loop_modal_projection_error": float(torch.linalg.norm(Ksp @ cT - qT)),
        "rigid_loop_representation_error": float(rep_total),
        "rigid_loop_representation_position_rmse": float(rep_pos),
        "rigid_loop_representation_tangent_rmse": float(rep_tan),
        "rigid_loop_target_validation": validation,
        "rigid_loop_target_measurement_protocol": "all M link midpoints+tangents; seam x_M-x_0 separate",
        "rigid_loop_target_tangent_weight": float(cfg.target_tangent_weight),
        "rigid_loop_allow_cyclic_seam": False,
        "rigid_loop_target_is_rigid_polygon": True,
        "rigid_loop_closure_mode": closure.normalized_mode(),
        "rigid_loop_closure_tolerance_rel": float(closure.tolerance_rel),
        "rigid_loop_closure_energy_scale": float(closure.energy_scale),
        "rigid_loop_closure_damping_scale": float(closure.damping_scale),
    }
    return LocomotionTask(
        kind="bvp",
        c0=c0,
        c_target=cT,
        pose_target=pose0.clone(),
        generator_controls=controls.detach(),
        pose0=pose0.detach(),
        metadata=metadata,
    )


def make_rigid_loop_task_from_desired_vertices(
    chain: PlanarMultiLinkChain,
    Ksp: torch.Tensor,
    n_temporal_ctrl: int,
    desired_vertices: np.ndarray,
    *,
    seed: int,
    closure: LoopClosureModel,
    cfg: RigidLoopConfig | None = None,
    target_family: str = "legacy_two_arc_hard",
    pose0: torch.Tensor | None = None,
) -> LocomotionTask:
    """Build an OOD rigid/measurement task from a fixed desired M-link polyline.

    No trajectory teacher is implied.  The static best-fit q is used only as the
    finite-dimensional task descriptor and to measure the rigid representation
    floor.  This is used by feasible->hard transfer protocols.
    """
    cfg = cfg or RigidLoopConfig(strict_target_validation=False)
    desired_np = np.asarray(desired_vertices, dtype=float)
    if desired_np.shape != (chain.n_links + 1, 2):
        raise ValueError(f"desired_vertices must have shape {(chain.n_links+1,2)}, got {desired_np.shape}")
    desired = torch.as_tensor(desired_np, dtype=Ksp.dtype, device=Ksp.device)
    tc, tt = link_measurements_from_vertices(desired)
    q0 = regular_polygon_joint_angles(chain)
    qT = fit_rigid_crescent_joint_angles(
        chain, desired_np, cfg, closure=closure,
        target_link_centers=tc.detach().cpu().numpy(),
        target_link_tangents=tt.detach().cpu().numpy(),
    )
    Kpinv = torch.linalg.pinv(Ksp)
    c0 = Kpinv @ q0
    cT = Kpinv @ qT
    u = torch.linspace(0.0, 1.0, n_temporal_ctrl, dtype=Ksp.dtype, device=Ksp.device)[:, None]
    controls = (1.0-u)*c0[None] + u*cT[None]
    if pose0 is None:
        rng=np.random.default_rng(seed)
        pose0=torch.tensor([
            float(rng.uniform(-0.20,0.20)), float(rng.uniform(-0.20,0.20)),
            float(rng.uniform(-math.pi,math.pi)),
        ],dtype=Ksp.dtype,device=Ksp.device)
    rigid=chain.kinematics(qT)["joint_positions"].detach()
    rep_total,rep_pos,rep_tan=rigid_target_measurement_errors(
        chain,qT,tc,tt,tangent_weight=cfg.target_tangent_weight,
    )
    raw=_turning_angles_from_vertices(desired_np)
    desired_gap=desired[-1]-desired[0]
    rigid_gap=loop_closure(chain,qT)
    curve_hash=hashlib.sha1(np.round(desired_np,11).tobytes()).hexdigest()
    validation=validate_rigid_loop_target(chain,qT,tc,tt,cfg)
    md={
        "external_task":"rigid_loop", "rigid_loop":True,
        "rigid_loop_is_hard_ood":True, "rigid_loop_teacher_allowed":False,
        "rigid_loop_target_family":str(target_family),
        "rigid_loop_reference_curve_hash":curve_hash,
        "rigid_loop_q_circle":q0.detach().cpu().tolist(),
        "rigid_loop_q_target":qT.detach().cpu().tolist(),
        "rigid_loop_target_vertices_body":rigid.cpu().tolist(),
        "rigid_loop_target_rigid_vertices_body":rigid.cpu().tolist(),
        "rigid_loop_target_sampled_crescent_body":desired_np.tolist(),
        "rigid_loop_target_dense_crescent_body":desired_np.tolist(),
        "rigid_loop_target_link_centers_body":tc.detach().cpu().tolist(),
        "rigid_loop_target_link_tangents_body":tt.detach().cpu().tolist(),
        "rigid_loop_initial_vertices_body":chain.kinematics(q0)["joint_positions"].detach().cpu().tolist(),
        "rigid_loop_perimeter":float(chain.params.total_length),
        "rigid_loop_raw_reference_max_abs_turn":float(np.max(np.abs(raw))) if raw.size else 0.0,
        "rigid_loop_target_max_abs_joint":float(qT.abs().max()),
        "rigid_loop_target_closure_error":float(torch.linalg.norm(rigid_gap)),
        "rigid_loop_desired_endpoint_gap":float(torch.linalg.norm(desired_gap)),
        "rigid_loop_desired_endpoint_gap_vector":desired_gap.detach().cpu().tolist(),
        "rigid_loop_representation_error":float(rep_total),
        "rigid_loop_representation_position_rmse":float(rep_pos),
        "rigid_loop_representation_tangent_rmse":float(rep_tan),
        "rigid_loop_target_validation":validation,
        "rigid_loop_target_measurement_protocol":"all M link midpoints+tangents; seam separate; OOD static fit only",
        "rigid_loop_allow_cyclic_seam":False,
        "rigid_loop_target_is_rigid_polygon":True,
        "rigid_loop_closure_mode":closure.normalized_mode(),
        "rigid_loop_closure_tolerance_rel":float(closure.tolerance_rel),
        "rigid_loop_closure_energy_scale":float(closure.energy_scale),
        "rigid_loop_closure_damping_scale":float(closure.damping_scale),
    }
    return LocomotionTask(
        kind="bvp",c0=c0,c_target=cT,pose_target=pose0.clone(),
        generator_controls=controls.detach(),pose0=pose0.detach(),metadata=md,
    )


def make_rigid_loop_tasks(
    count: int,
    chain: PlanarMultiLinkChain,
    Ksp: torch.Tensor,
    n_temporal_ctrl: int,
    *,
    seed: int,
    cfg: RigidLoopConfig | None = None,
    closure: LoopClosureModel | None = None,
    reference_specs: list[dict] | None = None,
) -> list[LocomotionTask]:
    if reference_specs is not None and len(reference_specs) != count:
        raise ValueError("reference_specs length must equal count")
    return [
        make_rigid_loop_task(
            chain, Ksp, n_temporal_ctrl, seed=seed + 7919 * i, cfg=cfg, closure=closure,
            reference_spec=None if reference_specs is None else reference_specs[i],
        )
        for i in range(count)
    ]

def projected_loop_reference_path(
    chain: PlanarMultiLinkChain,
    q0: torch.Tensor,
    qT: torch.Tensor,
    n_steps: int,
    *,
    cfg: RigidLoopConfig | None = None,
    closure: LoopClosureModel | None = None,
) -> torch.Tensor:
    cfg = cfg or RigidLoopConfig()
    closure = closure or LoopClosureModel(mode="hard")
    mode = closure.normalized_mode()
    vals = []
    for k in range(n_steps + 1):
        s = k / max(n_steps, 1)
        # cubic smoothstep gives zero endpoint shape speed before projection.
        a = 3.0 * s**2 - 2.0 * s**3
        q = (1.0 - a) * q0 + a * qT
        if mode == "hard":
            q = project_loop_configuration(chain, q, iterations=cfg.position_projection_iterations, regularization=cfg.closure_regularization)
        vals.append(q)
    return torch.stack(vals)


def make_rigid_loop_teacher_trajectory(
    task: LocomotionTask,
    chain: PlanarMultiLinkChain,
    Ksp: torch.Tensor,
    n_steps: int,
    dt: float,
    *,
    cfg: RigidLoopConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Projected inextensible intrinsic teacher used for initial pH supervision."""
    md = task.metadata or {}
    q0 = torch.as_tensor(md["rigid_loop_q_circle"], dtype=Ksp.dtype, device=Ksp.device)
    qT = torch.as_tensor(md["rigid_loop_q_target"], dtype=Ksp.dtype, device=Ksp.device)
    closure = closure_model_from_task(task)
    q = projected_loop_reference_path(chain, q0, qT, n_steps, cfg=cfg, closure=closure)
    qdot = torch.zeros_like(q)
    qdot[1:-1] = (q[2:] - q[:-2]) / (2.0 * dt)
    qdot[0] = (q[1] - q[0]) / dt
    qdot[-1] = (q[-1] - q[-2]) / dt
    Kpinv = torch.linalg.pinv(Ksp)
    c = torch.einsum("mj,tj->tm", Kpinv, q)
    cdot = torch.einsum("mj,tj->tm", Kpinv, qdot)

    pose = task.initial_pose.clone()
    poses = [pose]
    xis = []
    power = []
    for k in range(n_steps):
        pw, xi = chain.geometric_power(q[k], qdot[k])
        pose = se2_compose(pose, se2_exp_increment(xi, dt))
        poses.append(pose)
        xis.append(xi)
        power.append(pw)
    return {
        "c": c,
        "cdot": cdot,
        "q": q,
        "qdot": qdot,
        "pose": torch.stack(poses),
        "xi": torch.stack(xis),
        "power": torch.stack(power),
        "energy": 0.5 * dt * torch.stack(power).sum(),
        "loop_closure": torch.stack([loop_closure(chain, qq) for qq in q]),
    }


def loop_target_error(task: LocomotionTask, q: torch.Tensor) -> torch.Tensor:
    md = task.metadata or {}
    qT = torch.as_tensor(md["rigid_loop_q_target"], dtype=q.dtype, device=q.device)
    return torch.sqrt(torch.mean((q - qT) ** 2))


def loop_local_shape_error(task: LocomotionTask, chain: PlanarMultiLinkChain, q: torch.Tensor) -> torch.Tensor:
    """RMS all-link measurement error to the continuous crescent reference."""
    md = task.metadata or {}
    if "rigid_loop_target_link_centers_body" in md:
        centers = torch.as_tensor(md["rigid_loop_target_link_centers_body"], dtype=q.dtype, device=q.device)
        tangents = torch.as_tensor(md["rigid_loop_target_link_tangents_body"], dtype=q.dtype, device=q.device)
        verts = chain.kinematics(q)["joint_positions"]
        loss, _, _ = rigid_link_measurement_loss(
            verts, centers, tangents, perimeter=float(md.get("rigid_loop_perimeter", chain.params.total_length)),
            tangent_weight=float(md.get("rigid_loop_target_tangent_weight", 0.06)),
        )
        return torch.sqrt(loss + 1e-16)
    target = torch.as_tensor(md["rigid_loop_target_rigid_vertices_body"], dtype=q.dtype, device=q.device)
    pred = loop_body_vertices(chain, q, unique=True)
    loss, _ = cyclic_local_shape_loss(pred, target, allow_cyclic_shift=bool(md.get("rigid_loop_allow_cyclic_seam", True)))
    return torch.sqrt(loss + 1e-16)
