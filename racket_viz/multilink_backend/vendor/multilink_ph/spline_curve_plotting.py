from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from .spline_curve_ph import (
    SplineCurveConfig,
    SplineCurveTask,
    SplineCurvePHTemplate,
    evaluate_curve,
    knot_spans_from_logits,
    curve_rms,
    point_cloud_chamfer,
    rollout_spline_ph,
    apply_pose,
    pose_errors,
    polyline_self_intersections,
    signed_curve_area,
    validate_spline_task_target,
    teacher_trajectory,
)
from .chain import ChainParams, PlanarMultiLinkChain
from .rigid_loop import RigidLoopConfig, LoopClosureModel, fit_rigid_crescent_joint_angles, rigid_target_measurement_errors, link_measurements_from_vertices


def _np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def plot_spline_task_diagnostics(model: SplineCurvePHTemplate, task: SplineCurveTask, cfg: SplineCurveConfig, path: str | Path, *, horizon_multiplier: float = 2.0, title_suffix: str = "") -> None:
    ro = rollout_spline_ph(model, task, cfg, horizon_multiplier=horizon_multiplier)
    C0b, _ = evaluate_curve(ro["P"][0], ro["eta"][0], cfg)
    Cfb, _ = evaluate_curve(ro["P"][-1], ro["eta"][-1], cfg)
    Ct_body = task.target_samples
    C0w = apply_pose(C0b, ro["pose"][0])
    Cfw = apply_pose(Cfb, ro["pose"][-1])
    Ct_world = apply_pose(Ct_body, task.pose_target)
    Obs = task.observation_samples
    t = np.arange(ro["P"].shape[0]) / cfg.n_steps

    obs_err, intrinsic_err, pose_t_err, pose_r_err = [], [], [], []
    rollout_selfx, rollout_area = [], []
    source_area = float(signed_curve_area(C0b).detach().cpu())
    orient = 1.0 if source_area >= 0.0 else -1.0
    for k in range(ro["P"].shape[0]):
        Cb, _ = evaluate_curve(ro["P"][k], ro["eta"][k], cfg)
        Cw = apply_pose(Cb, ro["pose"][k])
        obs_err.append(float(point_cloud_chamfer(Cw, Obs).detach().cpu()) if Obs is not None else np.nan)
        intrinsic_err.append(float(curve_rms(Cb, Ct_body, False).detach().cpu()))
        et, er = pose_errors(ro["pose"][k], task.pose_target)
        pose_t_err.append(float(et.detach().cpu()))
        pose_r_err.append(float(er.detach().cpu()))
        rollout_selfx.append(int(polyline_self_intersections(Cb, closed=True)))
        rollout_area.append(orient * float(signed_curve_area(Cb).detach().cpu()))
    obs_err = np.asarray(obs_err)
    intrinsic_err = np.asarray(intrinsic_err)
    pose_t_err = np.asarray(pose_t_err)
    pose_r_err = np.asarray(pose_r_err)
    energy = _np(ro["energy"])
    spans0 = _np(knot_spans_from_logits(ro["eta"][0], cfg))
    spansf = _np(knot_spans_from_logits(ro["eta"][-1], cfg))
    spanst = _np(knot_spans_from_logits(task.eta_target, cfg))

    fig, ax = plt.subplots(3, 3, figsize=(15, 13))
    a = ax[0, 0]
    if Obs is not None:
        a.scatter(_np(Obs)[:, 0], _np(Obs)[:, 1], s=7, alpha=0.28, label="empirical target cloud")
    a.plot(_np(Ct_world)[:, 0], _np(Ct_world)[:, 1], linestyle="--", linewidth=1.5, label="fitted target spline")
    a.plot(_np(C0w)[:, 0], _np(C0w)[:, 1], label="initial curve")
    a.plot(_np(Cfw)[:, 0], _np(Cfw)[:, 1], label="pH final")
    Pworld = apply_pose(ro["P"][-1], ro["pose"][-1])
    a.plot(_np(Pworld)[:, 0], _np(Pworld)[:, 1], marker="o", alpha=0.55, label="final control polygon")
    a.set_aspect("equal")
    a.grid(alpha=0.25)
    a.legend(fontsize=7)
    a.set_title(f"{task.mode}: world pose + intrinsic shape")

    a = ax[0, 1]
    a.plot(t, obs_err, label="world point-cloud Chamfer")
    sink_diag = np.concatenate([[np.nan], _np(ro["sinkhorn"])])
    a.plot(t, sink_diag, label="Sinkhorn divergence")
    a.plot(t, intrinsic_err, label="intrinsic spline error")
    a.plot(t, pose_t_err, label="SE(2) translation error")
    a.plot(t, pose_r_err, label="SE(2) rotation error")
    a.axvline(1.0, linestyle="--", alpha=0.5, label="command ends")
    a.axhline(task.representation_floor, linestyle=":", label="representation floor")
    a.set_title("Separated convergence diagnostics")
    a.set_xlabel("time")
    a.legend(fontsize=7)
    a.grid(alpha=0.25)

    a = ax[0, 2]
    a.plot(t[1:], _np(ro["gap"]), label="endpoint gap")
    a.plot(t[1:], _np(ro["tangent_gap"]), label="tangent mismatch")
    a.axvline(1.0, linestyle="--", alpha=0.5)
    a.set_title("Closure measurements")
    a.legend(fontsize=8)
    a.grid(alpha=0.25)

    a = ax[1, 0]
    a.plot(_np(ro["P"][:, 0, 0]), _np(ro["pP"][:, 0, 0]))
    a.set_xlabel("body P[0,x]")
    a.set_ylabel("p_P[0,x]")
    a.set_title("Intrinsic control-point phase space")
    a.grid(alpha=0.25)

    a = ax[1, 1]
    a.plot(_np(ro["pose"][:, 0]), _np(ro["pPose"][:, 0]), label="x")
    a.plot(_np(ro["pose"][:, 2]), _np(ro["pPose"][:, 2]), label="theta")
    a.set_xlabel("pose coordinate")
    a.set_ylabel("pose momentum")
    a.set_title("Explicit SE(2) phase space")
    a.legend(fontsize=8)
    a.grid(alpha=0.25)

    a = ax[1, 2]
    xs = np.arange(cfg.n_spans)
    a.plot(xs, spans0, marker="o", label="initial spans")
    a.plot(xs, spanst, marker="o", label="target spans")
    a.plot(xs, spansf, marker="o", label="final spans")
    a.set_title("Learned knot intervals")
    a.set_xlabel("span index")
    a.legend(fontsize=8)
    a.grid(alpha=0.25)

    a = ax[2, 0]
    labels = ["kinetic T", "intrinsic rest V", "pose rest V", "OT state V", "closure V", "H", "dissipation"]
    for j, l in enumerate(labels):
        a.plot(t[1:], energy[:, j], label=l)
    a.axvline(1.0, linestyle="--", alpha=0.5)
    a.set_title("Hamiltonian / dissipation")
    a.legend(fontsize=7)
    a.grid(alpha=0.25)

    a = ax[2, 1]
    a.plot(t, _np(ro["P"][:, 0, 0]), label="body P0x")
    pref = np.concatenate([_np(ro["P_ref"][:, 0, 0]), [_np(ro["P_ref"][-1, 0, 0])]])
    a.plot(t, pref, linestyle="--", label="P0x ref")
    a.plot(t, _np(ro["pose"][:, 0]), label="pose x")
    gref = np.concatenate([_np(ro["pose_ref"][:, 0]), [_np(ro["pose_ref"][-1, 0])]])
    a.plot(t, gref, linestyle="--", label="pose x ref")
    a.set_title("Intrinsic / pose states vs rest paths")
    a.legend(fontsize=8)
    a.grid(alpha=0.25)

    a = ax[2, 2]
    a.axis("off")
    params = task.target_params or {}
    final_trans, final_rot = pose_errors(ro["pose"][-1], task.pose_target)
    final_speed = torch.sqrt(ro["pP"][-1].pow(2).mean() + ro["pE"][-1].pow(2).mean() + ro["pPose"][-1].pow(2).mean())
    txt = (
        f"mode: {task.mode}\n"
        f"conditioning: {task.conditioning_case}\n"
        f"family: {task.target_family}\n"
        f"hard OOD: {task.hard_ood}\n"
        f"alpha: {task.target_alpha:.3f}\n"
        f"inner radius: {params.get('inner_radius', float('nan')):.3f}\n"
        f"offset: {params.get('offset', float('nan')):.3f}\n"
        f"representation floor: {task.representation_floor:.4e}\n"
        f"final world cloud error: {obs_err[-1]:.4e}\n"
        f"final Sinkhorn: {float(ro['sinkhorn'][-1].detach().cpu()):.4e}\n"
        f"final intrinsic error: {intrinsic_err[-1]:.4e}\n"
        f"final pose trans: {float(final_trans.detach().cpu()):.4e}\n"
        f"final pose rot: {float(final_rot.detach().cpu()):.4e}\n"
        f"final gap: {float(ro['gap'][-1].detach().cpu()):.4e}\n"
        f"final speed: {float(final_speed.detach().cpu()):.4e}\n"
        f"rollout max selfX: {max(rollout_selfx):d}\n"
        f"rollout min oriented area: {min(rollout_area):.4e}"
    )
    a.text(0.03, 0.97, txt, va="top", family="monospace")
    fig.suptitle("Spline-only pH deformation: Sinkhorn target potential + explicit SE(2) + settling" + (f" — {title_suffix}" if title_suffix else ""))
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def render_spline_deformation_gif(
    model: SplineCurvePHTemplate,
    task: SplineCurveTask,
    cfg: SplineCurveConfig,
    path: str | Path,
    *,
    horizon_multiplier: float = 2.0,
    fps: int = 12,
    max_frames: int = 80,
    title_suffix: str = "",
) -> None:
    """Render the actual pH rollout so target approach/overshoot is visible.

    The animation deliberately shows the empirical target measure, the current
    world-space spline, and the current control polygon.  It can be generated
    from an existing checkpoint with CHECKPOINT_MODE=evaluate.
    """
    with torch.no_grad():
        ro = rollout_spline_ph(model, task, cfg, horizon_multiplier=horizon_multiplier)
    n = ro["P"].shape[0]
    if n <= max_frames:
        frames = np.arange(n, dtype=int)
    else:
        frames = np.linspace(0, n - 1, max_frames).round().astype(int)
    obs = _np(task.observation_samples) if task.observation_samples is not None else None

    # Stable view limits from target, initial, and fitted target.
    clouds = []
    if obs is not None:
        clouds.append(obs)
    C0b, _ = evaluate_curve(ro["P"][0], ro["eta"][0], cfg)
    clouds.append(_np(apply_pose(C0b, ro["pose"][0])))
    Ctb, _ = evaluate_curve(task.P_target, task.eta_target, cfg)
    clouds.append(_np(apply_pose(Ctb, task.pose_target)))
    allxy = np.concatenate(clouds, axis=0)
    lo = allxy.min(axis=0); hi = allxy.max(axis=0)
    pad = 0.10 * max(float((hi - lo).max()), 1e-3)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    if obs is not None:
        ax.scatter(obs[:, 0], obs[:, 1], s=7, alpha=0.22, label="target measure")
    curve_line, = ax.plot([], [], linewidth=2.0, label="pH spline")
    ctrl_line, = ax.plot([], [], marker="o", linewidth=1.0, alpha=0.7, label="control polygon")
    target_line, = ax.plot(_np(apply_pose(Ctb, task.pose_target))[:, 0], _np(apply_pose(Ctb, task.pose_target))[:, 1], linestyle="--", linewidth=1.2, label="fitted target config")
    info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", family="monospace", fontsize=9)
    ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal"); ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="lower left")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    def update(frame_index):
        k = int(frame_index)
        Cb, _ = evaluate_curve(ro["P"][k], ro["eta"][k], cfg)
        Cw = apply_pose(Cb, ro["pose"][k])
        Pw = apply_pose(ro["P"][k], ro["pose"][k])
        c = _np(Cw); p = _np(Pw)
        curve_line.set_data(c[:, 0], c[:, 1])
        ctrl_line.set_data(p[:, 0], p[:, 1])
        if k == 0:
            sink = float("nan")
            gap = float(_np(ro["gap"][0])) if ro["gap"].numel() else float("nan")
        else:
            sink = float(_np(ro["sinkhorn"][k - 1]))
            gap = float(_np(ro["gap"][k - 1]))
        info.set_text(f"t={k/cfg.n_steps:.2f}\nSinkhorn={sink:.4f}\ngap={gap:.4f}")
        ax.set_title("Spline pH deformation" + (f" — {title_suffix}" if title_suffix else ""))
        return curve_line, ctrl_line, info

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / max(fps, 1), blit=False)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)



def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, eps: float = 1e-12) -> bool:
    def on_segment(p, q, r):
        return (min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps and min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps)
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
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


def polyline_self_intersections(points: torch.Tensor | np.ndarray, *, closed: bool = True) -> int:
    pts = _np(points)
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
            if j == i:
                continue
            # skip adjacent segments sharing an endpoint
            if j == i + 1:
                continue
            if closed and i == 0 and j == seg_count - 1:
                continue
            c = pts[j]
            d = pts[(j + 1) % n] if closed else pts[j + 1]
            if _segments_intersect(a, b, c, d):
                count += 1
    return count


def summarize_task_geometry(task: SplineCurveTask, cfg: SplineCurveConfig) -> dict[str, float | int]:
    spans = knot_spans_from_logits(task.eta_target, cfg)
    obs = task.observation_samples if task.observation_samples is not None else apply_pose(task.target_samples, task.pose_target)
    validation = validate_spline_task_target(task, cfg)
    return {
        "min_span": float(spans.min().detach().cpu()),
        "max_span": float(spans.max().detach().cpu()),
        "target_self_intersections": int(validation["curve_self_intersections"]),
        "teacher_max_self_intersections": int(validation["teacher_max_self_intersections"]),
        "obs_self_intersections": int(polyline_self_intersections(obs, closed=True)),
        "representation_floor": float(task.representation_floor),
    }


def plot_training_task_gallery(task_groups: dict[str, list[SplineCurveTask]], cfg: SplineCurveConfig, path: str | Path, *, n_per_group: int = 3, title: str = "") -> dict[str, dict[str, float | int]]:
    groups = [(name, tasks[:n_per_group]) for name, tasks in task_groups.items() if tasks]
    if not groups:
        return {}
    nrows = len(groups)
    ncols = max(len(ts) for _, ts in groups)
    fig, ax = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows), squeeze=False)
    summary: dict[str, dict[str, float | int]] = {}
    for r, (name, tasks) in enumerate(groups):
        mins, maxs, selfx, teacher_selfx = [], [], [], []
        for c in range(ncols):
            a = ax[r, c]
            if c >= len(tasks):
                a.axis('off')
                continue
            task = tasks[c]
            stats = summarize_task_geometry(task, cfg)
            mins.append(stats['min_span']); maxs.append(stats['max_span']); selfx.append(stats['target_self_intersections']); teacher_selfx.append(stats['teacher_max_self_intersections'])
            obs = task.observation_samples if task.observation_samples is not None else apply_pose(task.target_samples, task.pose_target)
            init_body, _ = evaluate_curve(task.P0, task.eta0, cfg)
            init_world = apply_pose(init_body, task.pose0)
            targ_world = apply_pose(task.target_samples, task.pose_target)
            ctrl_world = apply_pose(task.P_target, task.pose_target)
            teacher = teacher_trajectory(task, cfg)
            mid = teacher["P"].shape[0] // 2
            mid_body, _ = evaluate_curve(teacher["P"][mid], teacher["eta"][mid], cfg)
            mid_world = apply_pose(mid_body, teacher["pose"][mid])
            if obs is not None:
                oo = _np(obs); a.scatter(oo[:,0], oo[:,1], s=6, alpha=0.25, label='obs cloud')
            tw = _np(targ_world); iw = _np(init_world); cw = _np(ctrl_world)
            a.plot(tw[:,0], tw[:,1], linestyle='--', linewidth=1.8, label='target spline')
            a.plot(iw[:,0], iw[:,1], linewidth=1.2, label='initial circle')
            mw = _np(mid_world)
            a.plot(mw[:,0], mw[:,1], linestyle=':', linewidth=1.4, alpha=0.8, label='teacher midpoint')
            a.plot(cw[:,0], cw[:,1], marker='o', linewidth=1.0, alpha=0.75, label='target control poly')
            a.set_aspect('equal'); a.grid(alpha=0.25)
            a.set_title(f"{name} #{c}\nalpha={task.target_alpha:.2f}, floor={task.representation_floor:.3f}\ntarget selfX={stats['target_self_intersections']}, teacher max selfX={stats['teacher_max_self_intersections']}\nspan=[{stats['min_span']:.3f},{stats['max_span']:.3f}]")
            if r == 0 and c == 0:
                a.legend(fontsize=7, loc='lower left')
        summary[name] = {
            'count': len(tasks),
            'mean_min_span': float(np.mean(mins)) if mins else float('nan'),
            'mean_max_span': float(np.mean(maxs)) if maxs else float('nan'),
            'max_self_intersections': int(max(selfx)) if selfx else 0,
            'num_self_intersecting': int(sum(v > 0 for v in selfx)),
            'max_teacher_self_intersections': int(max(teacher_selfx)) if teacher_selfx else 0,
            'num_teacher_paths_self_intersecting': int(sum(v > 0 for v in teacher_selfx)),
        }
    fig.suptitle(title or 'Generated spline training targets')
    fig.tight_layout()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return summary


def _sample_closed_vertices_from_target(task: SplineCurveTask, n_links: int) -> torch.Tensor:
    """Equal-arclength samples of the closed target curve.

    The previous index-based sampling distorted the rigid baseline whenever the
    adaptive spline knots were nonuniform.  Equal arclength makes all physical
    links comparable and removes knot-density artifacts from the baseline.
    """
    pts = task.target_samples
    if torch.linalg.norm(pts[0] - pts[-1]) < 1e-9:
        pts = pts[:-1]
    closed = torch.cat([pts, pts[:1]], dim=0)
    seg = closed[1:] - closed[:-1]
    seglen = torch.linalg.norm(seg, dim=-1)
    cumulative = torch.cat([torch.zeros(1, dtype=pts.dtype, device=pts.device), torch.cumsum(seglen, dim=0)], dim=0)
    perimeter = cumulative[-1].clamp_min(1e-12)
    svals = torch.arange(n_links, dtype=pts.dtype, device=pts.device) * (perimeter / n_links)
    verts = []
    for s in svals:
        j = torch.searchsorted(cumulative, s, right=True).item() - 1
        j = max(0, min(j, seg.shape[0] - 1))
        w = (s - cumulative[j]) / seglen[j].clamp_min(1e-12)
        verts.append(closed[j] + w * seg[j])
    verts = torch.stack(verts, dim=0)
    return torch.cat([verts, verts[:1]], dim=0)


def _canonicalize_vertices_for_chain(vertices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove body-frame SE(2) gauge expected by ``PlanarMultiLinkChain``.

    The chain kinematics fixes its first joint at the origin and its first link
    along +x.  Fitting absolute target centers without this gauge removal causes
    an apparent pose error even when the internal joint angles are reasonable.
    """
    p0 = vertices[0]
    e0 = vertices[1] - vertices[0]
    phi = torch.atan2(e0[1], e0[0])
    c, s = torch.cos(-phi), torch.sin(-phi)
    Rminus = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
    canonical = (vertices - p0) @ Rminus.T
    return canonical, p0, phi


def _restore_vertices_from_chain_gauge(vertices: torch.Tensor, p0: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(phi), torch.sin(phi)
    R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
    return vertices @ R.T + p0


def rigid_loop_geometry_baseline(task: SplineCurveTask, cfg: SplineCurveConfig, *, n_links: int = 10, joint_limit: float = 2.8, fit_iterations: int = 300) -> dict[str, object]:
    target_vertices_body = _sample_closed_vertices_from_target(task, n_links)
    target_vertices, body_origin, body_heading = _canonicalize_vertices_for_chain(target_vertices_body)
    centers, tangents = link_measurements_from_vertices(target_vertices)
    edge = target_vertices[1:] - target_vertices[:-1]
    perimeter = float(torch.linalg.norm(edge, dim=-1).sum().detach().cpu())
    chain = PlanarMultiLinkChain(ChainParams(n_links=n_links, total_length=perimeter, joint_limit=joint_limit), dtype=task.P_target.dtype, device=task.P_target.device)
    rcfg = RigidLoopConfig(target_fit_iterations=fit_iterations, target_joint_margin=0.985, strict_target_validation=False)
    q = fit_rigid_crescent_joint_angles(
        chain, cfg=rcfg, closure=LoopClosureModel(mode='hard'),
        target_link_centers=centers.detach().cpu().numpy(),
        target_link_tangents=tangents.detach().cpu().numpy(),
    )
    verts_chain = chain.kinematics(q)['joint_positions']
    total_rmse, pos_rmse, tan_rmse = rigid_target_measurement_errors(chain, q, centers, tangents, tangent_weight=rcfg.target_tangent_weight)
    verts_body = _restore_vertices_from_chain_gauge(verts_chain, body_origin, body_heading)
    verts_world = apply_pose(verts_body, task.pose_target)
    obs_err = float(point_cloud_chamfer(verts_world[:-1], task.observation_samples).detach().cpu()) if task.observation_samples is not None else float('nan')
    return {
        'vertices_body': verts_body.detach(),
        'vertices_world': verts_world.detach(),
        'q_target': q.detach(),
        'measurement_rmse': float(total_rmse.detach().cpu()),
        'position_rmse': float(pos_rmse.detach().cpu()),
        'tangent_rmse': float(tan_rmse.detach().cpu()),
        'observation_chamfer': obs_err,
        'perimeter': perimeter,
        'n_links': n_links,
        'body_alignment_origin_x': float(body_origin[0].detach().cpu()),
        'body_alignment_origin_y': float(body_origin[1].detach().cpu()),
        'body_alignment_heading': float(body_heading.detach().cpu()),
    }


def plot_rigid_baseline_comparison(task: SplineCurveTask, cfg: SplineCurveConfig, path: str | Path, *, n_links: int = 10, joint_limit: float = 2.8, fit_iterations: int = 300, title_suffix: str = "") -> dict[str, float]:
    baseline = rigid_loop_geometry_baseline(task, cfg, n_links=n_links, joint_limit=joint_limit, fit_iterations=fit_iterations)
    obs = task.observation_samples if task.observation_samples is not None else apply_pose(task.target_samples, task.pose_target)
    targ = apply_pose(task.target_samples, task.pose_target)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    a = ax[0]
    oo = _np(obs); tt = _np(targ); vv = _np(baseline['vertices_world'])
    a.scatter(oo[:,0], oo[:,1], s=6, alpha=0.25, label='obs cloud')
    a.plot(tt[:,0], tt[:,1], linestyle='--', linewidth=1.8, label='target spline')
    a.plot(vv[:,0], vv[:,1], marker='o', linewidth=1.4, label=f'closed rigid-loop ({n_links} links)')
    a.set_aspect('equal'); a.grid(alpha=0.25); a.legend(fontsize=8)
    a.set_title('Geometry-only baseline fit')
    b = ax[1]
    b.axis('off')
    txt=(
        f"conditioning: {task.conditioning_case}\n"
        f"mode: {task.mode}\n"
        f"alpha: {task.target_alpha:.3f}\n"
        f"representation floor (spline fit): {task.representation_floor:.4e}\n"
        f"rigid measurement rmse: {baseline['measurement_rmse']:.4e}\n"
        f"rigid position rmse: {baseline['position_rmse']:.4e}\n"
        f"rigid tangent rmse: {baseline['tangent_rmse']:.4e}\n"
        f"rigid world Chamfer: {baseline['observation_chamfer']:.4e}\n"
        f"rigid links: {baseline['n_links']}\n"
        f"perimeter: {baseline['perimeter']:.4f}\n"
        f"body gauge heading: {baseline['body_alignment_heading']:.3f}"
    )
    b.text(0.03, 0.97, txt, va='top', family='monospace')
    fig.suptitle('Closed rigid-loop comparison on geometry/point-cloud target' + (f" — {title_suffix}" if title_suffix else ""))
    fig.tight_layout()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {
        'rigid_measurement_rmse': float(baseline['measurement_rmse']),
        'rigid_position_rmse': float(baseline['position_rmse']),
        'rigid_tangent_rmse': float(baseline['tangent_rmse']),
        'rigid_observation_chamfer': float(baseline['observation_chamfer']),
        'rigid_perimeter': float(baseline['perimeter']),
    }


def plot_ablation_summary(rows: list[dict], path: str | Path) -> None:
    modes = []
    for r in rows:
        if r["label"] not in modes:
            modes.append(r["label"])
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.8))
    x = np.arange(len(modes))
    rep, zero, transfer, gap, iid, pose = [], [], [], [], [], []
    for m in modes:
        rr = [r for r in rows if r["label"] == m]
        rep.append(np.mean([r.get("ood_representation_floor", np.nan) for r in rr]))
        zero.append(np.mean([r.get("ood_zero_shape", np.nan) for r in rr]))
        transfer.append(np.mean([r.get("ood_transfer_shape", np.nan) for r in rr]))
        gap.append(np.mean([r.get("ood_transfer_gap", np.nan) for r in rr]))
        iid.append(np.mean([r.get("iid_shape", np.nan) for r in rr]))
        pose.append(np.mean([r.get("iid_pose_translation", np.nan) for r in rr]))
    w = 0.25
    ax[0].bar(x - w, rep, w, label="representation floor")
    ax[0].bar(x, zero, w, label="zero-shot")
    ax[0].bar(x + w, transfer, w, label="teacher-free transfer")
    ax[0].set_xticks(x, modes, rotation=28, ha="right")
    ax[0].set_ylabel("world target discrepancy")
    ax[0].set_title("Hard OOD legacy crescent")
    ax[0].legend(fontsize=8)
    ax[0].grid(axis="y", alpha=0.2)

    ax[1].bar(x, gap)
    ax[1].set_xticks(x, modes, rotation=28, ha="right")
    ax[1].set_ylabel("endpoint gap")
    ax[1].set_title("Closure ablation")
    ax[1].grid(axis="y", alpha=0.2)

    ax[2].bar(x, iid)
    ax[2].set_xticks(x, modes, rotation=28, ha="right")
    ax[2].set_ylabel("held-out target error")
    ax[2].set_title("Feasible curriculum validation")
    ax[2].grid(axis="y", alpha=0.2)

    ax[3].bar(x, pose)
    ax[3].set_xticks(x, modes, rotation=28, ha="right")
    ax[3].set_ylabel("translation error")
    ax[3].set_title("Explicit SE(2) validation")
    ax[3].grid(axis="y", alpha=0.2)

    fig.suptitle("Spline-only ablation after pose/shape factorization and settling curriculum")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
