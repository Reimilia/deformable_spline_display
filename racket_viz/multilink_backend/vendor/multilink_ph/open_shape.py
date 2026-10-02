from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch

from .chain import PlanarMultiLinkChain
from .tasks import LocomotionTask
from .geometry import se2_compose, se2_exp_increment
from .rigid_loop import crescent_reference_vertices, canonicalize_loop_vertices, smooth_crescent_reference_data


@dataclass(frozen=True)
class OpenShapeConfig:
    """Settings for the open-chain circle-arc -> crescent-arc diagnostic.

    This task deliberately removes the loop-closure constraint while keeping the
    same rigid links, pH template, anisotropic environment, and local/global
    decomposition. It is a control experiment for diagnosing whether the closed
    constraint manifold is what prevents shape alignment.
    """

    target_fit_iterations: int = 500
    target_fit_lr: float = 2e-2
    target_joint_margin: float = 0.90
    target_shape_weight: float = 1.0
    joint_limit_weight: float = 5.0
    # Build the open target by cutting one edge from an (M+1)-edge crescent.
    missing_edge_count: int = 1
    # Target-family protocol. ``smooth_feasible`` is used for teacher-supervised
    # training; ``legacy_two_arc_hard`` deliberately retains the sharp concavity
    # used in the early experiments and is intended for teacher-free OOD tests.
    target_family: str = "smooth_feasible"
    max_representation_error: float = 0.055
    max_sampling_attempts: int = 40
    hard_inner_ratio_min: float = 0.10
    hard_inner_ratio_max: float = 0.22
    hard_outer_bulge_min: float = 0.96
    hard_outer_bulge_max: float = 1.08
    hard_vertical_min: float = 0.90
    hard_vertical_max: float = 1.08
    # OOD selection gates.  Hard tasks are sampled *toward* poor rigid-link
    # representability / high turning demand rather than accidentally becoming
    # another easy training-like crescent.
    hard_min_representation_error: float = 0.025
    hard_min_raw_turn: float = 2.35
    # Hard-transfer geometry can either leave a large material opening (legacy_open)
    # or deliberately approach the holonomic seam without enforcing it.
    hard_target_style: str = "near_holonomic"
    hard_seam_gap_rel_min: float = 0.008
    hard_seam_gap_rel_max: float = 0.025
    hard_coarse_edges: int = 8


def open_circle_arc_joint_angles(chain: PlanarMultiLinkChain) -> torch.Tensor:
    """M rigid links following an almost-complete regular circle with one missing edge."""
    # M links are M consecutive edges of a regular (M+1)-gon. The missing edge
    # makes the target genuinely open while preserving a circle-like initial arc.
    ang = 2.0 * math.pi / float(chain.n_links + 1)
    return torch.full((chain.n_joints,), ang, dtype=chain.dtype, device=chain.device)


def open_body_vertices(chain: PlanarMultiLinkChain, q: torch.Tensor) -> torch.Tensor:
    return chain.kinematics(q)["joint_positions"]


def canonicalize_open_vertices(points: torch.Tensor) -> torch.Tensor:
    """Pose-invariant representation of an ordered open material chain."""
    # canonicalize_loop_vertices only drops the last point if it duplicates the
    # first; for a genuinely open chain this is exactly the desired operation.
    return canonicalize_loop_vertices(points)


def open_local_shape_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = canonicalize_open_vertices(predicted)
    t = canonicalize_open_vertices(target)
    if p.shape != t.shape:
        raise ValueError(f"open shape point-count mismatch: {p.shape} vs {t.shape}")
    return torch.mean((p - t) ** 2)


def _turning_angles_from_vertices(points: np.ndarray) -> np.ndarray:
    e = np.diff(np.asarray(points, dtype=float), axis=0)
    ang = np.unwrap(np.arctan2(e[:, 1], e[:, 0]))
    q = np.diff(ang)
    return np.arctan2(np.sin(q), np.cos(q))



def legacy_two_arc_closed_reference_vertices(
    n_edges: int,
    perimeter: float,
    *,
    outer_bulge: float = 1.0,
    inner_ratio: float = 0.16,
    vertical_scale: float = 1.0,
    dense: int = 2400,
) -> np.ndarray:
    """Original sharp two-arc crescent used for OOD reachability tests.

    This function intentionally does *not* auto-soften the target when the joint
    range or fixed-edge discretization cannot reproduce it.  It therefore serves
    as the challenging geometry for zero-shot/transfer evaluation rather than as
    a teacher-supervised training target.
    """
    eps = 1e-7
    n1 = dense // 2
    y1 = np.linspace(-1.0 + eps, 1.0 - eps, n1)
    x1 = float(outer_bulge) * np.sqrt(np.maximum(0.0, 1.0 - y1**2))
    y2 = np.linspace(1.0 - eps, -1.0 + eps, dense - n1)
    x2 = float(outer_bulge) * float(inner_ratio) * np.sqrt(np.maximum(0.0, 1.0 - y2**2))
    pts = np.column_stack([
        np.concatenate([x1, x2]),
        float(vertical_scale) * np.concatenate([y1, y2]),
    ])
    pts = np.vstack([pts, pts[0]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    pts *= float(perimeter) / max(float(seg.sum()), 1e-12)

    # Uniform arc-length resampling to n_edges physical intervals.
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    ss = np.linspace(0.0, float(s[-1]), n_edges + 1)
    out = np.zeros((n_edges + 1, 2), dtype=float)
    for d in range(2):
        out[:, d] = np.interp(ss, s, pts[:, d])
    out[-1] = out[0]
    out -= out[0]
    e0 = out[1] - out[0]
    th = math.atan2(float(e0[1]), float(e0[0]))
    c, sn = math.cos(-th), math.sin(-th)
    R = np.array([[c, -sn], [sn, c]], dtype=float)
    return out @ R.T


def hard_open_crescent_reference_vertices(
    chain: PlanarMultiLinkChain,
    *,
    outer_bulge: float = 1.0,
    inner_ratio: float = 0.16,
    vertical_scale: float = 1.0,
    coarse_edges: int = 8,
) -> np.ndarray:
    """Challenging open crescent with sharp corners preserved as OOD geometry.

    The desired geometry is first built with only ``coarse_edges`` perimeter
    segments, so its high-curvature two-arc junction is not smoothed away when
    the robot itself has many links.  One coarse edge is cut away from the
    sharpest corner.  The remaining coarse polyline is then subdivided into
    exactly M material intervals while *preserving every coarse corner*.

    The resulting M+1 target measurements generally have unequal interval
    lengths and can demand turns beyond the mechanism's training distribution.
    This is intentional: it is an OOD desired geometry, not a teacher path.
    """
    coarse_edges = max(5, min(int(coarse_edges), chain.n_links))
    closed = legacy_two_arc_closed_reference_vertices(
        coarse_edges,
        float(chain.params.total_length),
        outer_bulge=outer_bulge,
        inner_ratio=inner_ratio,
        vertical_scale=vertical_scale,
    )
    u = np.asarray(closed[:-1], dtype=float)
    cyc_edges = np.roll(u, -1, axis=0) - u
    edge_ang = np.unwrap(np.arctan2(cyc_edges[:, 1], cyc_edges[:, 0]))
    turns = np.arctan2(np.sin(np.roll(edge_ang, -1) - edge_ang), np.cos(np.roll(edge_ang, -1) - edge_ang))
    sharp_vertex = int(np.argmax(np.abs(turns)))
    cut_edge = int((sharp_vertex + coarse_edges // 2) % coarse_edges)
    # Open coarse path: start just after the removed edge and end just before it.
    coarse_idx = [int((cut_edge + 1 + j) % coarse_edges) for j in range(coarse_edges)]
    path = u[coarse_idx]
    n_coarse_seg = len(path) - 1
    if n_coarse_seg <= 0:
        raise RuntimeError("hard crescent coarse path is empty")

    # Allocate M subdivisions while preserving all coarse vertices.  Every coarse
    # segment gets at least one physical interval; extras follow coarse length.
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    counts = np.ones(n_coarse_seg, dtype=int)
    remaining = int(chain.n_links - counts.sum())
    if remaining < 0:
        raise ValueError("coarse hard target has more segments than physical links")
    if remaining > 0:
        weights = lengths / max(float(lengths.sum()), 1e-12)
        raw = weights * remaining
        add = np.floor(raw).astype(int)
        counts += add
        left = remaining - int(add.sum())
        if left > 0:
            order = np.argsort(-(raw - add))
            counts[order[:left]] += 1

    pts = [path[0]]
    for j in range(n_coarse_seg):
        a, b = path[j], path[j + 1]
        for k in range(1, int(counts[j]) + 1):
            pts.append(a + (k / float(counts[j])) * (b - a))
    pts = np.asarray(pts, dtype=float)
    if pts.shape[0] != chain.n_links + 1:
        raise RuntimeError(f"hard target subdivision produced {pts.shape[0]} points, expected {chain.n_links+1}")
    pts -= pts[0]
    e0 = pts[1] - pts[0]
    th = math.atan2(float(e0[1]), float(e0[0]))
    c, sn = math.cos(-th), math.sin(-th)
    R = np.array([[c, -sn], [sn, c]], dtype=float)
    return pts @ R.T



def hard_closed_crescent_reference_vertices(
    chain: PlanarMultiLinkChain,
    *,
    outer_bulge: float = 1.0,
    inner_ratio: float = 0.16,
    vertical_scale: float = 1.0,
    coarse_edges: int = 8,
) -> np.ndarray:
    """Sharp legacy crescent sampled as an exactly closed M-link desired geometry.

    A coarse two-arc polygon is subdivided without deleting its corners.  This is
    used for hard holonomic OOD evaluation, where the desired curve is allowed to
    be poorly representable but the *desired seam* is exactly zero.
    """
    M = int(chain.n_links)
    n = max(5, min(int(coarse_edges), M))
    coarse = legacy_two_arc_closed_reference_vertices(
        n, float(chain.params.total_length), outer_bulge=outer_bulge,
        inner_ratio=inner_ratio, vertical_scale=vertical_scale,
    )
    u = np.asarray(coarse[:-1], dtype=float)
    segs = [(u[j], u[(j + 1) % n]) for j in range(n)]
    lengths = np.asarray([np.linalg.norm(b-a) for a,b in segs], dtype=float)
    counts = np.ones(n, dtype=int)
    rem = M - int(counts.sum())
    if rem < 0:
        raise ValueError("hard closed target has more coarse segments than physical links")
    if rem:
        w = lengths / max(float(lengths.sum()), 1e-12)
        raw = w * rem
        add = np.floor(raw).astype(int)
        counts += add
        left = rem - int(add.sum())
        if left:
            order = np.argsort(-(raw-add))
            counts[order[:left]] += 1
    pts=[u[0]]
    for j,(a,b) in enumerate(segs):
        for k in range(1, int(counts[j])+1):
            pts.append(a + (k/float(counts[j]))*(b-a))
    pts=np.asarray(pts,dtype=float)
    if pts.shape[0] != M+1:
        raise RuntimeError(f"hard closed subdivision produced {pts.shape[0]} points, expected {M+1}")
    pts[-1]=pts[0]
    pts -= pts[0]
    e0=pts[1]-pts[0]
    th=math.atan2(float(e0[1]),float(e0[0]))
    c,sn=math.cos(-th),math.sin(-th)
    R=np.array([[c,-sn],[sn,c]],dtype=float)
    return pts @ R.T


def near_holonomic_hard_open_crescent_reference_vertices(
    chain: PlanarMultiLinkChain,
    *,
    seam_gap_rel: float = 0.015,
    outer_bulge: float = 1.0,
    inner_ratio: float = 0.16,
    vertical_scale: float = 1.0,
    coarse_edges: int = 8,
) -> np.ndarray:
    """Sharp legacy crescent whose material seam is deliberately *almost* closed.

    The desired path traverses a closed two-arc crescent except for a short gap
    centered on a low-curvature coarse edge.  The missing arc is chosen so that
    the Euclidean endpoint gap is approximately ``seam_gap_rel * L``.  The open
    path is then subdivided into exactly M rigid-link measurement intervals while
    preserving all coarse crescent corners.

    This target is ideal for transfer tests: it lies close to the holonomic
    manifold but no hard closure projection or teacher trajectory is supplied.
    """
    M=int(chain.n_links)
    L=float(chain.params.total_length)
    rho=float(np.clip(seam_gap_rel, 1e-5, 0.20))
    n=max(5, min(int(coarse_edges), max(M-1,5)))
    # Make the complete curve slightly longer so removing a rho*L seam leaves an
    # open desired path with approximately the same material length L.
    coarse=legacy_two_arc_closed_reference_vertices(
        n, L*(1.0+rho), outer_bulge=outer_bulge,
        inner_ratio=inner_ratio, vertical_scale=vertical_scale,
    )
    u=np.asarray(coarse[:-1],dtype=float)
    cyc=np.roll(u,-1,axis=0)-u
    ea=np.unwrap(np.arctan2(cyc[:,1],cyc[:,0]))
    turns=np.arctan2(np.sin(np.roll(ea,-1)-ea),np.cos(np.roll(ea,-1)-ea))
    sharp=int(np.argmax(np.abs(turns)))
    cut=int((sharp+n//2)%n)  # seam away from the sharp indentation
    A=u[cut]; B=u[(cut+1)%n]
    edge=B-A; el=max(float(np.linalg.norm(edge)),1e-12)
    gap=min(rho*L,0.70*el)
    frac=gap/el
    lo=0.5-0.5*frac; hi=0.5+0.5*frac
    start=A+hi*edge
    end=A+lo*edge
    cycle=[u[(cut+1+j)%n] for j in range(n)]  # B,...,A
    path=np.asarray([start,*cycle,end],dtype=float)
    seg=np.diff(path,axis=0)
    lengths=np.linalg.norm(seg,axis=1)
    ns=len(lengths)
    if ns > M:
        # Reduce coarse resolution rather than deleting the hard corner.
        return near_holonomic_hard_open_crescent_reference_vertices(
            chain, seam_gap_rel=rho, outer_bulge=outer_bulge,
            inner_ratio=inner_ratio, vertical_scale=vertical_scale,
            coarse_edges=max(5, n-1),
        )
    counts=np.ones(ns,dtype=int)
    rem=M-int(counts.sum())
    if rem:
        w=lengths/max(float(lengths.sum()),1e-12)
        raw=w*rem
        add=np.floor(raw).astype(int); counts+=add
        left=rem-int(add.sum())
        if left:
            order=np.argsort(-(raw-add)); counts[order[:left]]+=1
    pts=[path[0]]
    for j in range(ns):
        a,b=path[j],path[j+1]
        for k in range(1,int(counts[j])+1):
            pts.append(a+(k/float(counts[j]))*(b-a))
    pts=np.asarray(pts,dtype=float)
    if pts.shape[0] != M+1:
        raise RuntimeError(f"near-holonomic target produced {pts.shape[0]} points, expected {M+1}")
    pts-=pts[0]
    e0=pts[1]-pts[0]
    th=math.atan2(float(e0[1]),float(e0[0]))
    c,sn=math.cos(-th),math.sin(-th)
    R=np.array([[c,-sn],[sn,c]],dtype=float)
    return pts @ R.T


def open_crescent_reference_vertices(
    chain: PlanarMultiLinkChain,
    *,
    outer_bulge: float = 1.0,
    inner_ratio: float = 0.30,
    vertical_scale: float = 1.0,
) -> np.ndarray:
    """Open crescent-like target obtained by cutting one material edge.

    We first build an (M+1)-edge perimeter-matched closed crescent, then remove
    the closing edge. The remaining M consecutive segments form a genuinely
    open target with the same nominal link length as the physical M-link chain.
    """
    link_length = float(chain.params.total_length) / float(chain.n_links)
    parent_edges = chain.n_links + 1
    parent_perimeter = parent_edges * link_length
    closed = crescent_reference_vertices(
        parent_edges,
        parent_perimeter,
        outer_bulge=outer_bulge,
        inner_ratio=inner_ratio,
        vertical_scale=vertical_scale,
    )
    # closed has parent_edges+1 = M+2 points. Keep M+1 points => M links,
    # leaving the final parent edge out of the material chain.
    # Choose the material seam *away* from the sharpest crescent turn.  Earlier
    # open targets always cut the final parent edge, which could accidentally
    # remove the very concave joint we intended to use for OOD testing.
    u = np.asarray(closed[:-1], dtype=float)
    cyc_edges = np.roll(u, -1, axis=0) - u
    edge_ang = np.unwrap(np.arctan2(cyc_edges[:, 1], cyc_edges[:, 0]))
    turns = np.arctan2(np.sin(np.roll(edge_ang, -1) - edge_ang), np.cos(np.roll(edge_ang, -1) - edge_ang))
    sharp_vertex = int(np.argmax(np.abs(turns)))
    cut_edge = int((sharp_vertex + parent_edges // 2) % parent_edges)
    idx = [int((cut_edge + 1 + j) % parent_edges) for j in range(chain.n_links + 1)]
    pts = u[idx].copy()
    pts -= pts[0]
    e0 = pts[1] - pts[0]
    th = math.atan2(e0[1], e0[0])
    c, s = math.cos(-th), math.sin(-th)
    R = np.array([[c, -s], [s, c]], dtype=float)
    return pts @ R.T


def fit_open_crescent_joint_angles(
    chain: PlanarMultiLinkChain,
    target_vertices: np.ndarray,
    cfg: OpenShapeConfig | None = None,
) -> torch.Tensor:
    """Fit fixed-length open articulated chain to the open crescent target.

    Joint bounds are enforced by a tanh parameterization; unlike the closed-loop
    fit there is no closure projection that can push the target outside the
    joint-limit set after optimization.
    """
    cfg = cfg or OpenShapeConfig()
    target = torch.as_tensor(target_vertices, dtype=chain.dtype, device=chain.device)
    if target.shape != (chain.n_links + 1, 2):
        raise ValueError("open target must contain n_links+1 points")
    raw = torch.as_tensor(_turning_angles_from_vertices(target_vertices), dtype=chain.dtype, device=chain.device)
    bound = cfg.target_joint_margin * float(chain.params.joint_limit)
    raw = raw.clamp(-0.95 * bound, 0.95 * bound)
    z0 = torch.atanh((raw / max(bound, 1e-8)).clamp(-0.999, 0.999))
    z = torch.nn.Parameter(z0.clone())
    opt = torch.optim.Adam([z], lr=cfg.target_fit_lr)
    best_q = None
    best = float("inf")
    for _ in range(cfg.target_fit_iterations):
        q = bound * torch.tanh(z)
        verts = open_body_vertices(chain, q)
        fit = torch.mean((verts - target) ** 2)
        # tanh already enforces the hard bound; keep a soft margin penalty to
        # discourage the target from living at saturation.
        lim = torch.relu(q.abs() - 0.85 * bound).pow(2).mean()
        loss = cfg.target_shape_weight * fit + cfg.joint_limit_weight * lim
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([z], 20.0)
        opt.step()
        val = float(loss.detach())
        if val < best:
            best = val
            best_q = q.detach().clone()
    assert best_q is not None
    return best_q


def open_target_representation_error(
    chain: PlanarMultiLinkChain,
    q_target: torch.Tensor,
    sampled_target: torch.Tensor,
) -> torch.Tensor:
    return torch.sqrt(open_local_shape_loss(open_body_vertices(chain, q_target), sampled_target) + 1e-16)


def make_open_shape_task(
    chain: PlanarMultiLinkChain,
    Ksp: torch.Tensor,
    n_temporal_ctrl: int,
    *,
    seed: int,
    pose0: torch.Tensor | None = None,
    cfg: OpenShapeConfig | None = None,
) -> LocomotionTask:
    """Construct one open circle->crescent task under the requested target family.

    ``smooth_feasible`` performs rejection sampling on the best fixed-link
    representation error and is intended for teacher-supervised training.
    ``legacy_two_arc_hard`` keeps the original sharp two-arc geometry even when
    the rigid mechanism cannot reproduce it exactly; its best-fit q is used only
    as a target descriptor / representation-floor oracle, not as a test teacher.
    """
    cfg = cfg or OpenShapeConfig()
    rng = np.random.default_rng(seed)
    q0 = open_circle_arc_joint_angles(chain)
    family = str(cfg.target_family).lower().strip()
    best = None
    hard_family = family in {"legacy_two_arc_hard", "legacy", "hard"}
    attempts = max(int(cfg.max_sampling_attempts), 1)
    for _ in range(attempts):
        if family in {"legacy_two_arc_hard", "legacy", "hard"}:
            outer = float(rng.uniform(cfg.hard_outer_bulge_min, cfg.hard_outer_bulge_max))
            inner = float(rng.uniform(cfg.hard_inner_ratio_min, cfg.hard_inner_ratio_max))
            vertical = float(rng.uniform(cfg.hard_vertical_min, cfg.hard_vertical_max))
            style = str(cfg.hard_target_style).lower().strip()
            if style in {"near_holonomic", "near_closed", "almost_closed"}:
                seam_rel = float(rng.uniform(cfg.hard_seam_gap_rel_min, cfg.hard_seam_gap_rel_max))
                sampled = near_holonomic_hard_open_crescent_reference_vertices(
                    chain, seam_gap_rel=seam_rel, outer_bulge=outer, inner_ratio=inner,
                    vertical_scale=vertical, coarse_edges=cfg.hard_coarse_edges,
                )
            else:
                seam_rel = float("nan")
                sampled = hard_open_crescent_reference_vertices(
                    chain, outer_bulge=outer, inner_ratio=inner, vertical_scale=vertical,
                    coarse_edges=cfg.hard_coarse_edges,
                )
        else:
            # Smooth feasible family used by the supervised curriculum.
            outer = float(rng.uniform(0.84, 1.04))
            inner = float(rng.uniform(0.26, 0.40))
            vertical = float(rng.uniform(0.88, 1.08))
            sampled = open_crescent_reference_vertices(
                chain, outer_bulge=outer, inner_ratio=inner, vertical_scale=vertical
            )
        qT = fit_open_crescent_joint_angles(chain, sampled, cfg)
        sampled_t = torch.as_tensor(sampled, dtype=Ksp.dtype, device=Ksp.device)
        repr_err = float(open_target_representation_error(chain, qT, sampled_t))
        raw_turn_here = _turning_angles_from_vertices(sampled)
        raw_max_here = float(np.max(np.abs(raw_turn_here))) if raw_turn_here.size else 0.0
        cand = (repr_err, raw_max_here, outer, inner, vertical, sampled, qT, seam_rel if hard_family else float("nan"))
        if hard_family:
            # Keep the hardest candidate seen; stop once either explicit OOD gate
            # is satisfied.  This prevents a high-joint-limit experiment from
            # silently sampling another easy crescent.
            score = repr_err + 0.02 * max(0.0, raw_max_here - 0.75 * float(chain.params.joint_limit))
            if best is None:
                best = (score, cand)
            else:
                if score > best[0]:
                    best = (score, cand)
            if repr_err >= float(cfg.hard_min_representation_error) or raw_max_here >= float(cfg.hard_min_raw_turn):
                best = (score, cand)
                break
        else:
            if best is None or repr_err < best[0]:
                best = cand
            if repr_err <= float(cfg.max_representation_error):
                best = cand
                break
    assert best is not None
    if hard_family:
        _, cand = best
        repr_err, raw_max_selected, outer, inner, vertical, sampled, qT, seam_rel_selected = cand
    else:
        repr_err, raw_max_selected, outer, inner, vertical, sampled, qT, seam_rel_selected = best

    Kpinv = torch.linalg.pinv(Ksp)
    c0 = Kpinv @ q0
    cT = Kpinv @ qT
    u = torch.linspace(0.0, 1.0, n_temporal_ctrl, dtype=Ksp.dtype, device=Ksp.device)[:, None]
    controls = (1.0 - u) * c0[None] + u * cT[None]
    if n_temporal_ctrl > 2:
        gen = torch.Generator(device=Ksp.device).manual_seed(seed + 331)
        controls[1:-1] += 0.015 * torch.randn(controls[1:-1].shape, generator=gen, dtype=Ksp.dtype, device=Ksp.device)
    if pose0 is None:
        pose0 = torch.tensor([
            float(rng.uniform(-0.20, 0.20)),
            float(rng.uniform(-0.20, 0.20)),
            float(rng.uniform(-math.pi, math.pi)),
        ], dtype=Ksp.dtype, device=Ksp.device)
    rigid = open_body_vertices(chain, qT).detach()
    initial = open_body_vertices(chain, q0).detach()
    sampled_t = torch.as_tensor(sampled, dtype=Ksp.dtype, device=Ksp.device)
    raw_turn = _turning_angles_from_vertices(sampled)
    endpoint_gap_initial = float(torch.linalg.norm(initial[-1] - initial[0]))
    endpoint_gap_target = float(torch.linalg.norm(rigid[-1] - rigid[0]))
    hard = family in {"legacy_two_arc_hard", "legacy", "hard"}
    md = {
        "external_task": "open_shape",
        "open_shape": True,
        "joint_limit": float(chain.params.joint_limit),
        "open_shape_target_family": "legacy_two_arc_hard" if hard else "smooth_feasible",
        "open_shape_teacher_allowed": not hard,
        "open_shape_q_initial": q0.detach().cpu().tolist(),
        # Best physically representable descriptor under the current joint box.
        "open_shape_q_target": qT.detach().cpu().tolist(),
        "open_shape_initial_vertices_body": initial.cpu().tolist(),
        "open_shape_target_rigid_vertices_body": rigid.cpu().tolist(),
        # Desired geometry may lie outside the rigid-chain reachable set.
        "open_shape_target_desired_vertices_body": sampled_t.cpu().tolist(),
        "open_shape_target_sampled_crescent_body": sampled_t.cpu().tolist(),
        "open_shape_target_params": {"outer_bulge": outer, "inner_ratio": inner, "vertical_scale": vertical},
        "open_shape_hard_target_style": str(cfg.hard_target_style) if hard else None,
        "open_shape_desired_endpoint_gap": float(np.linalg.norm(sampled[-1]-sampled[0])),
        "open_shape_desired_endpoint_gap_rel": float(np.linalg.norm(sampled[-1]-sampled[0]) / max(float(chain.params.total_length), 1e-12)),
        "open_shape_requested_endpoint_gap_rel": (float(seam_rel_selected) if hard and np.isfinite(seam_rel_selected) else None),
        "open_shape_representation_error": float(repr_err),
        "open_shape_endpoint_gap_initial": endpoint_gap_initial,
        "open_shape_endpoint_gap_target": endpoint_gap_target,
        "open_shape_target_max_abs_joint": float(qT.abs().max()),
        "open_shape_raw_reference_max_abs_turn": float(np.max(np.abs(raw_turn))) if raw_turn.size else 0.0,
        "open_shape_hard_gate_representation_error": float(cfg.hard_min_representation_error) if hard else None,
        "open_shape_hard_gate_raw_turn": float(cfg.hard_min_raw_turn) if hard else None,
        "open_shape_target_is_fixed_edge_open_polygon": True,
        "open_shape_is_hard_ood": hard,
    }
    return LocomotionTask(
        kind="bvp",
        c0=c0,
        c_target=cT,
        pose_target=pose0.clone(),
        generator_controls=controls.detach(),
        pose0=pose0.detach(),
        metadata=md,
    )


def make_open_shape_tasks(
    count: int,
    chain: PlanarMultiLinkChain,
    Ksp: torch.Tensor,
    n_temporal_ctrl: int,
    *,
    seed: int,
    cfg: OpenShapeConfig | None = None,
) -> list[LocomotionTask]:
    return [make_open_shape_task(chain, Ksp, n_temporal_ctrl, seed=seed + 7919 * i, cfg=cfg) for i in range(count)]


def make_open_shape_teacher_trajectory(
    task: LocomotionTask,
    chain: PlanarMultiLinkChain,
    Ksp: torch.Tensor,
    n_steps: int,
    dt: float,
) -> dict[str, torch.Tensor]:
    """Smooth fixed-edge open-chain intrinsic teacher; no closure projection."""
    md = task.metadata or {}
    q0 = torch.as_tensor(md["open_shape_q_initial"], dtype=Ksp.dtype, device=Ksp.device)
    qT = torch.as_tensor(md["open_shape_q_target"], dtype=Ksp.dtype, device=Ksp.device)
    qs = []
    for k in range(n_steps + 1):
        s = k / max(n_steps, 1)
        a = 3.0 * s**2 - 2.0 * s**3
        qs.append((1.0 - a) * q0 + a * qT)
    q = torch.stack(qs)
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
    powers = []
    for k in range(n_steps):
        pw, xi = chain.geometric_power(q[k], qdot[k])
        pose = se2_compose(pose, se2_exp_increment(xi, dt))
        poses.append(pose)
        xis.append(xi)
        powers.append(pw)
    return {
        "c": c,
        "cdot": cdot,
        "q": q,
        "qdot": qdot,
        "pose": torch.stack(poses),
        "xi": torch.stack(xis),
        "power": torch.stack(powers),
        "energy": 0.5 * dt * torch.stack(powers).sum(),
    }


def open_target_error(task: LocomotionTask, q: torch.Tensor) -> torch.Tensor:
    qT = torch.as_tensor((task.metadata or {})["open_shape_q_target"], dtype=q.dtype, device=q.device)
    return torch.sqrt(torch.mean((q - qT) ** 2) + 1e-16)


def open_shape_error(task: LocomotionTask, chain: PlanarMultiLinkChain, q: torch.Tensor) -> torch.Tensor:
    target = torch.as_tensor((task.metadata or {})["open_shape_target_rigid_vertices_body"], dtype=q.dtype, device=q.device)
    return torch.sqrt(open_local_shape_loss(open_body_vertices(chain, q), target) + 1e-16)
