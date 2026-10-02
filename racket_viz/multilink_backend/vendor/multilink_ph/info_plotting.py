from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation

from .chain import PlanarMultiLinkChain
from .tasks import LocomotionTask
from .rigid_loop import cyclic_local_shape_loss, rigid_link_measurement_loss
from .geometry import se2_endpoint_error


def _world_vertices(chain: PlanarMultiLinkChain, q: torch.Tensor, pose: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        q = torch.as_tensor(q)
        pose = torch.as_tensor(pose, dtype=q.dtype, device=q.device)
        kin = chain.kinematics(q)
        body = kin["joint_positions"]
        c, s = torch.cos(pose[2]), torch.sin(pose[2])
        R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
        world = (R @ body.T).T + pose[:2]
    return world.detach().cpu().numpy()


def _world_com(chain: PlanarMultiLinkChain, q: torch.Tensor, pose: torch.Tensor) -> np.ndarray:
    return _world_vertices(chain, q, pose).mean(axis=0)


def _finite(x):
    return np.asarray(torch.as_tensor(x).detach().cpu(), dtype=float)


def _ground_y_from_task(task: LocomotionTask) -> float:
    return float((task.metadata or {}).get("ground_y", 0.0))


def _frame_target_from_task(task: LocomotionTask):
    md = task.metadata or {}
    if "frame_target_world" not in md:
        return None
    return np.asarray(md["frame_target_world"], dtype=float)

def _rigid_loop_target_body(task: LocomotionTask):
    md = task.metadata or {}
    key = "rigid_loop_target_rigid_vertices_body" if "rigid_loop_target_rigid_vertices_body" in md else "rigid_loop_target_vertices_body"
    if key not in md:
        return None
    return np.asarray(md[key], dtype=float)

def _rigid_loop_sampled_crescent_body(task: LocomotionTask):
    md = task.metadata or {}
    # v18 prefers the dense continuous reference so the visualization does not
    # make a smooth crescent look like another coarse rigid polygon.
    if "rigid_loop_target_dense_crescent_body" in md:
        return np.asarray(md["rigid_loop_target_dense_crescent_body"], dtype=float)
    if "rigid_loop_target_sampled_crescent_body" not in md:
        return None
    return np.asarray(md["rigid_loop_target_sampled_crescent_body"], dtype=float)

def _loop_local_error_series(task: LocomotionTask, chain: PlanarMultiLinkChain, qhist) -> np.ndarray:
    md = task.metadata or {}
    vals=[]
    if "rigid_loop_target_link_centers_body" in md:
        for q in qhist:
            qq=torch.as_tensor(q, dtype=torch.float64)
            centers = torch.as_tensor(md["rigid_loop_target_link_centers_body"], dtype=qq.dtype, device=qq.device)
            tangents = torch.as_tensor(md["rigid_loop_target_link_tangents_body"], dtype=qq.dtype, device=qq.device)
            pred = chain.kinematics(qq)["joint_positions"]
            loss,_,_=rigid_link_measurement_loss(
                pred, centers, tangents, perimeter=float(md.get("rigid_loop_perimeter",chain.params.total_length)),
                tangent_weight=float(md.get("rigid_loop_target_tangent_weight",0.06)),
            )
            vals.append(float(torch.sqrt(loss+1e-16)))
        return np.asarray(vals)
    target = _rigid_loop_target_body(task)
    if target is None:
        return np.full(len(qhist), np.nan)
    for q in qhist:
        qq=torch.as_tensor(q, dtype=torch.float64)
        target_t = torch.as_tensor(target, dtype=qq.dtype, device=qq.device)
        from .rigid_loop import loop_body_vertices
        pred=loop_body_vertices(chain,qq,unique=True)
        loss,_=cyclic_local_shape_loss(pred,target_t,allow_cyclic_shift=bool(md.get("rigid_loop_allow_cyclic_seam",True)))
        vals.append(float(torch.sqrt(loss+1e-16)))
    return np.asarray(vals)


def _transform_body_vertices(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    c, ss = np.cos(pose[2]), np.sin(pose[2])
    R = np.array([[c, -ss], [ss, c]])
    return points @ R.T + pose[:2]


def _draw_external_world_context(ax, task: LocomotionTask, chain: PlanarMultiLinkChain, ph: dict, snapshot_ids: np.ndarray) -> None:
    case = ph.get("motion_force_case", "self_propulsion")
    if case == "frame_target":
        pos = _finite(ph["tracked_frame_position_world"])
        target = _frame_target_from_task(task)
        if len(pos) and np.isfinite(pos).any():
            ax.plot(pos[:, 0], pos[:, 1], linewidth=2.0, label="tracked frame path")
            ax.scatter([pos[0, 0]], [pos[0, 1]], marker="o", s=55, label="frame start")
        if target is not None:
            ax.scatter([target[0]], [target[1]], marker="*", s=100, label="frame target")
        forces = _finite(ph["tracked_frame_force_world"])
        for idx in snapshot_ids:
            if idx >= len(pos) or idx >= len(forces) or not np.isfinite(pos[idx]).all():
                continue
            f = forces[idx]
            scale = 0.06 / max(np.linalg.norm(f), 1e-8)
            ax.arrow(pos[idx,0], pos[idx,1], scale*f[0], scale*f[1], width=0.002, head_width=0.018, length_includes_head=True, alpha=.7)
    elif case == "rigid_loop":
        target_body = _rigid_loop_target_body(task)
        target_pose = task.pose_target.detach().cpu().numpy()
        if target_body is not None:
            target_world = _transform_body_vertices(target_body, target_pose)
            ax.plot(target_world[:,0], target_world[:,1], linestyle="--", linewidth=2.2, label="rigid target polygon")
        sampled = _rigid_loop_sampled_crescent_body(task)
        if sampled is not None:
            sampled_world = _transform_body_vertices(sampled, target_pose)
            ax.plot(sampled_world[:,0], sampled_world[:,1], linestyle=":", linewidth=1.3, alpha=.8, label="sampled crescent (geometry reference)")
        init_body = np.asarray((task.metadata or {}).get("rigid_loop_initial_vertices_body", []), dtype=float)
        if init_body.size:
            init_world = _transform_body_vertices(init_body, task.initial_pose.detach().cpu().numpy())
            ax.plot(init_world[:,0], init_world[:,1], linestyle=":", alpha=.7, label="initial circle")
        # Explicitly draw the material seam.  For measurement/open closure this
        # makes the allowed discontinuity visible instead of hiding it in a scalar.
        try:
            vf = _world_vertices(chain, ph["q"][-1], ph["pose"][-1])
            gap = float(np.linalg.norm(vf[-1] - vf[0]))
            ax.plot([vf[-1,0], vf[0,0]], [vf[-1,1], vf[0,1]], linestyle="--", linewidth=1.8, label=f"terminal seam gap={gap:.3f}")
        except Exception:
            pass
    elif case == "gravity_ground":
        gy = _ground_y_from_task(task)
        ax.axhline(gy, linewidth=2.0, label="ground y=0")
        gp = _finite(ph["ground_contact_positions_world"])
        ga = _finite(ph["ground_active"])
        for idx in snapshot_ids:
            if idx >= len(gp):
                continue
            active = ga[idx] > 0.5
            if np.any(active):
                ax.scatter(gp[idx, active, 0], gp[idx, active, 1], s=28, marker="s", label="active contact" if idx == snapshot_ids[0] else None)
        # Gravity direction indicator in world coordinates.
        q0, p0 = ph["q"][0], ph["pose"][0]
        com0 = _world_com(chain, q0, p0)
        ax.arrow(com0[0], com0[1]+0.12, 0.0, -0.09, width=0.003, head_width=0.022, length_includes_head=True)
        ax.text(com0[0]+0.02, com0[1]+0.08, "gravity", fontsize=8)


def plot_world_pose_diagnostics(
    task: LocomotionTask,
    chain: PlanarMultiLinkChain,
    ph: dict[str, torch.Tensor],
    teacher: dict[str, torch.Tensor],
    path: str | Path,
    dt: float,
) -> None:
    """Case-aware report: locomotion, external task geometry, phase space and passivity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 3, figsize=(15, 15))
    case = ph.get("motion_force_case", "self_propulsion")

    # 1) World geometry: explicitly reveal what the external task is doing.
    ax = axes[0, 0]
    q = ph["q"]; pose = ph["pose"]
    com = np.asarray([_world_com(chain, q[k], pose[k]) for k in range(len(q))])
    ax.plot(com[:, 0], com[:, 1], label="pH CoM", linewidth=1.8)
    ids = np.unique(np.linspace(0, len(q) - 1, 6).round().astype(int))
    for idx in ids:
        v = _world_vertices(chain, q[idx], pose[idx])
        ax.plot(v[:, 0], v[:, 1], marker="o", markersize=2.5, alpha=0.42)
    # Teacher is an intrinsic/no-force reference in externally forced cases.
    tq, tp = teacher["q"], teacher["pose"]
    tcom = np.asarray([_world_com(chain, tq[k], tp[k]) for k in range(len(tq))])
    teacher_label = "variational teacher" if case == "self_propulsion" else "intrinsic variational reference (unforced)"
    ax.plot(tcom[:, 0], tcom[:, 1], linestyle="--", alpha=.65, label=teacher_label)
    p0 = task.initial_pose.detach().cpu().numpy()
    ax.scatter([p0[0]], [p0[1]], marker="o", s=42, label="initial base")
    if case in {"self_propulsion", "rigid_loop"}:
        pt = task.pose_target.detach().cpu().numpy()
        ax.scatter([pt[0]], [pt[1]], marker="x", s=65, label="target pose origin")
    _draw_external_world_context(ax, task, chain, ph, ids[:-1])
    ax.set_title({
        "self_propulsion":"World locomotion: intrinsic propulsion",
        "frame_target":"World motion: force applied at one link frame",
        "gravity_ground":"World motion: gravity, impact/contact with y=0",
        "rigid_loop":f"Rigid-loop deformation ({str((task.metadata or {}).get('rigid_loop_closure_mode','hard'))} closure): circle to crescent",
    }.get(case, f"World motion: {case}"))
    ax.axis("equal"); ax.grid(True, alpha=0.25); ax.legend(fontsize=7)

    # 2) Primary physical task coordinate.
    ax = axes[0, 1]
    if case == "frame_target":
        pos = _finite(ph["tracked_frame_position_world"])
        target = _frame_target_from_task(task)
        tt = np.arange(len(pos)) * dt
        ax.plot(tt, pos[:,0], label="frame x")
        ax.plot(tt, pos[:,1], label="frame y")
        if target is not None:
            ax.axhline(target[0], linestyle="--", label="target x")
            ax.axhline(target[1], linestyle=":", label="target y")
            err = np.linalg.norm(pos - target[None,:], axis=1)
            ax.plot(tt, err, linewidth=2, label="frame distance error")
        ax.set_title("Tracked frame center: $(x,y)\\rightarrow(x',y')$")
    elif case == "gravity_ground":
        gaps = _finite(ph["min_ground_gap"])
        counts = _finite(ph["contact_count"])
        tt = np.arange(len(gaps))*dt
        ax.plot(tt, gaps, label="minimum ground gap")
        ax.axhline(0.0, linewidth=.8)
        ax.plot(tt, counts / max(chain.n_links+1,1), label="contact fraction")
        com_y = com[:-1,1] if len(com) > len(gaps) else com[:len(gaps),1]
        ax.plot(tt[:len(com_y)], com_y, label="CoM y", alpha=.8)
        ax.set_title("Ground-contact event: gap $\\to0$ under gravity")
    elif case == "rigid_loop":
        qT = np.asarray((task.metadata or {}).get("rigid_loop_q_target"), dtype=float)
        qcur = _finite(ph["q"])
        tt = np.arange(len(qcur)) * dt
        qerr = np.sqrt(np.mean((qcur - qT[None,:])**2, axis=1))
        local_err = _loop_local_error_series(task, chain, qcur)
        pose_err=[]
        for pp in ph["pose"]:
            pose_err.append(float(torch.linalg.norm(se2_endpoint_error(pp, task.pose_target)).detach()))
        pose_err=np.asarray(pose_err)
        cerr = _finite(ph["loop_closure_error"])
        perr = _finite(ph["loop_perimeter_error"])
        ax.plot(tt, local_err, linewidth=2.0, label="all-link crescent measurement RMS")
        ax.plot(tt, qerr, label="joint-coordinate RMS", alpha=.85)
        ax.plot(np.arange(len(pose_err))*dt, pose_err, label="SE(2) pose error", alpha=.85)
        ax.plot(np.arange(len(cerr))*dt, cerr, label="endpoint seam gap")
        ax.plot(np.arange(len(perr))*dt, perr, label="perimeter error")
        floor=float((task.metadata or {}).get("rigid_loop_representation_error",np.nan))
        if np.isfinite(floor): ax.axhline(floor, linestyle=":", linewidth=1.2, label="best rigid-link measurement floor")
        mode=str((task.metadata or {}).get("rigid_loop_closure_mode", ph.get("loop_closure_mode", "hard")))
        if mode == "measurement":
            tol=float((task.metadata or {}).get("rigid_loop_closure_tolerance_rel",0.05))*float((task.metadata or {}).get("rigid_loop_perimeter",chain.params.total_length))
            ax.axhline(tol, linestyle="--", linewidth=1.1, label=f"measurement tolerance={tol:.3f}")
        ax.set_title(f"Decomposed target: shape + pose + closure ({mode})")
    else:
        pose_ph = _finite(ph["pose"]); pose_t = _finite(teacher["pose"])
        t_ph = np.arange(len(pose_ph)) * dt
        t_t = np.linspace(0.0, dt * (len(pose_ph) - 1), len(pose_t))
        for j, lab in enumerate(["x", "y", "theta"]):
            ax.plot(t_ph, pose_ph[:, j], label=f"pH {lab}")
            ax.plot(t_t, pose_t[:, j], linestyle="--", alpha=0.7, label=f"teacher {lab}")
        ax.set_title("Global pose $g(t)\\in SE(2)$")
    ax.set_xlabel("time"); ax.grid(True, alpha=0.25); ax.legend(fontsize=7, ncol=2)

    # 3) Intrinsic kinematics remains active in all three cases.
    ax = axes[0, 2]
    qph = _finite(ph["q"]); qref = _finite(ph["q_ref"])
    tqv = np.arange(len(qph)) * dt
    nshow = min(3, qph.shape[1])
    for j in range(nshow):
        ax.plot(tqv, qph[:, j], label=f"q{j+1}")
        tr = np.arange(len(qref)) * dt
        ax.plot(tr, qref[:, j], linestyle="--", alpha=.75, label=f"qref{j+1}")
    ax.set_title("Intrinsic motion: realized joints vs spline rest shape")
    ax.set_xlabel("time"); ax.grid(True, alpha=.25); ax.legend(fontsize=7, ncol=2)

    # 4) phase portrait
    ax=axes[1,0]; pph=_finite(ph["pi"]); ptch=_finite(teacher["pi"]); qt=_finite(teacher["q"])
    ax.plot(qph[:,0],pph[:,3],label="pH",linewidth=1.8); ax.plot(qt[:,0],ptch[:,3],linestyle="--",label="intrinsic teacher")
    ax.scatter([qph[0,0]],[pph[0,3]],marker="o",s=30); ax.scatter([qph[-1,0]],[pph[-1,3]],marker="x",s=45)
    ax.set_xlabel("q1"); ax.set_ylabel("p_q1"); ax.set_title("Internal phase-space portrait"); ax.grid(True,alpha=.25); ax.legend(fontsize=8)

    # 5) global momentum map
    ax=axes[1,1]; Jp=_finite(ph["momentum_map"]); tm=np.arange(len(Jp))*dt
    for j,lab in enumerate(["Lz","Px","Py"]): ax.plot(tm,Jp[:,j],label=lab)
    ax.set_title("SE(2) momentum map under actual force law"); ax.set_xlabel("time"); ax.grid(True,alpha=.25); ax.legend(fontsize=8)

    # 6) Hamiltonian + external potential
    ax=axes[1,2]; H=_finite(ph["hamiltonian"]); T=_finite(ph["kinetic_energy"]); V=_finite(ph["potential_energy"]); th=np.arange(len(H))*dt
    ax.plot(th,H,label="mechanical H=T+V",linewidth=2); ax.plot(th,T,label="kinetic T"); ax.plot(th,V,label="stored potential V")
    Ue=_finite(ph["external_potential"])
    if np.max(np.abs(Ue))>1e-12: ax.plot(np.arange(len(Ue))*dt,Ue,linestyle="--",label="external/contact potential")
    ax.set_title("Hamiltonian and external potential monitor"); ax.set_xlabel("time"); ax.grid(True,alpha=.25); ax.legend(fontsize=8)

    # 7) generalized force decomposition
    ax=axes[2,0]; tf=np.arange(ph["spring_force_joint"].shape[0])*dt
    for key,lab in [("spring_force_joint","intrinsic spring"),("fim_force_joint","FIM damping"),("environment_force_joint","environment drag"),("external_joint_force","external force pullback"),("total_joint_force","total joint")]:
        arr=_finite(ph[key]); ax.plot(tf,np.linalg.norm(arr,axis=1),label=lab)
    ax.set_title("Intrinsic/generalized force norms"); ax.set_xlabel("time"); ax.grid(True,alpha=.25); ax.legend(fontsize=7)

    # 8) External force/contact physics itself.
    ax=axes[2,1]
    if case == "frame_target":
        ff=_finite(ph["tracked_frame_force_world"]); tt=np.arange(len(ff))*dt
        ax.plot(tt,ff[:,0],label="frame force Fx"); ax.plot(tt,ff[:,1],label="frame force Fy"); ax.plot(tt,np.linalg.norm(ff,axis=1),label="|F_frame|")
        ax.set_title("Direct force applied to selected link frame")
    elif case == "gravity_ground":
        gf=_finite(ph["ground_contact_forces_world"]); gaps=_finite(ph["min_ground_gap"]); tt=np.arange(len(gf))*dt
        ax.plot(tt,np.linalg.norm(gf,axis=2).sum(axis=1),label="total ground reaction")
        ax.plot(tt,gaps,label="min gap y-ground")
        ax.plot(tt,_finite(ph["contact_count"]),label="active contacts")
        ax.axhline(0,linewidth=.8)
        ax.set_title("Ground reaction / penetration/contact onset")
    elif case == "rigid_loop":
        mode = str(ph.get("loop_closure_mode", (task.metadata or {}).get("rigid_loop_closure_mode", "hard")))
        lf = _finite(ph["loop_constraint_force"]); cp = _finite(ph["loop_constraint_power"]); tt=np.arange(len(lf))*dt
        if mode == "hard":
            ax.plot(tt, np.linalg.norm(lf,axis=1), label="closure reaction |Q_c|")
            ax.plot(tt, cp, label="closure reaction power")
            if "loop_projection_energy_jump" in ph:
                ej=_finite(ph["loop_projection_energy_jump"])
                ax.plot(np.arange(len(ej))*dt, ej/max(dt,1e-12), linestyle="--", label="position projection ΔH/dt")
            ax.set_title("Hard closure: reaction and projection work")
        elif mode == "measurement":
            fs=_finite(ph["loop_seam_spring_force"]); fd=_finite(ph["loop_seam_damping_force"]); gap=_finite(ph["loop_closure_error"]); vp=_finite(ph["loop_seam_potential"])
            ax.plot(tt,np.linalg.norm(fs,axis=1),label="measurement seam spring")
            ax.plot(tt,np.linalg.norm(fd,axis=1),label="measurement seam damping")
            ax.plot(tt,gap,label="endpoint gap")
            ax.plot(tt,vp,label="seam potential")
            ax.set_title("Measurement-driven compliant seam")
        else:
            gap=_finite(ph["loop_closure_error"]); ax.plot(tt,gap,label="endpoint gap (unconstrained)")
            ax.set_title("Open seam: no closure force")
        ax.axhline(0,linewidth=.8)
    else:
        wrench=_finite(ph["environment_wrench_global"])
        for j,lab in enumerate(["Fx","Fy","tau_z"]): ax.plot(tf,wrench[:,j],label=lab)
        ax.set_title("Environmental generalized wrench")
    ax.set_xlabel("time"); ax.grid(True,alpha=.25); ax.legend(fontsize=7)

    # 9) Power balance with external port.
    ax=axes[2,2]; pref=_finite(ph["reference_power"]); pd=_finite(ph["dissipation_power"]); pext=_finite(ph["external_power"]); res=_finite(ph["power_balance_residual"])
    ax.plot(tf,pref,label="intrinsic reference supply")
    if np.max(np.abs(pext))>1e-12: ax.plot(tf,pext,label="external/contact power")
    ax.plot(tf,-pd,label="-dissipation"); ax.plot(tf,res,label="balance residual",alpha=.7); ax.axhline(0,linewidth=.8)
    ax.set_title("Power ports: intrinsic + external + dissipation"); ax.set_xlabel("time"); ax.grid(True,alpha=.25); ax.legend(fontsize=7)

    # 10) FIM
    ax=axes[3,0]; fe=_finite(ph["fim_eigs"]); ti=np.arange(len(fe))*dt
    for j in range(fe.shape[1]): ax.plot(ti,fe[:,j],linewidth=1)
    ax.set_title("Configuration FIM eigenvalues"); ax.set_xlabel("time"); ax.grid(True,alpha=.25)

    # 11) SF
    ax=axes[3,1]; se=_finite(ph["sf_eigs"])
    if se.ndim == 2 and se.shape[0] > 0:
        ti_sf=np.arange(len(se))*dt
        for j in range(se.shape[1]): ax.plot(ti_sf,se[:,j],linewidth=1.2,label=f"lambda{j+1}")
        ax.legend(fontsize=8)
    else:
        ax.text(.5,.5,"SF diagnostics disabled",ha="center",va="center",transform=ax.transAxes,fontsize=9)
    ax.set_title("Empirical Souriau--Fisher spectrum"); ax.set_xlabel("time"); ax.grid(True,alpha=.25)

    # 12) passivity
    ax=axes[3,2]; margin=_finite(ph["passivity_margin"]); balance=_finite(ph["energy_balance_residual_cum"]); tw=np.arange(len(margin))*dt
    ax.plot(tw,margin,label="supply work - Delta H"); ax.plot(tw,_finite(ph["dissipated_work"]),label="accum. dissipation"); ax.plot(tw,balance,label="balance residual"); ax.axhline(0,linewidth=.8)
    ax.set_title("Passivity / cumulative energy balance"); ax.set_xlabel("time"); ax.grid(True,alpha=.25); ax.legend(fontsize=7)

    if case == "frame_target":
        task_desc = f"tracked link={int((task.metadata or {}).get('force_link_index',-1))}, target={_frame_target_from_task(task)}"
    elif case == "gravity_ground":
        task_desc = f"ground y={_ground_y_from_task(task):.2f}, initial clearance={(task.metadata or {}).get('initial_clearance',float('nan')):.3f}"
    elif case == "rigid_loop":
        task_desc = f"fixed perimeter={float((task.metadata or {}).get('rigid_loop_perimeter',chain.params.total_length)):.3f}, closure={str((task.metadata or {}).get('rigid_loop_closure_mode','hard'))}, circle→crescent"
    else:
        task_desc = "positioned-shape SE(2) terminal task"
    # Mark the end of the learned reference actuation.  Extended evaluation
    # continues to the right of this line to reveal whether momentum/contact
    # dynamics actually settle.
    control_horizon = float(torch.as_tensor(ph.get("control_horizon", 1.0)).detach().cpu())
    for _ax in [axes[0,1], axes[0,2], axes[1,1], axes[1,2], axes[2,0], axes[2,1], axes[2,2], axes[3,0], axes[3,1], axes[3,2]]:
        _ax.axvline(control_horizon, linestyle="--", linewidth=0.9, alpha=0.65)

    fig.suptitle(f"Articulated intrinsic + external-force task\ncase={case}: {task_desc}")
    fig.tight_layout(rect=[0,0,1,.965]); fig.savefig(path,dpi=175); plt.close(fig)

def animate_world_pose_comparison(
    task: LocomotionTask,
    chain: PlanarMultiLinkChain,
    ph: dict[str, torch.Tensor],
    teacher: dict[str, torch.Tensor],
    path: str | Path,
    fps: int = 8,
) -> None:
    """Animation that makes the actual external-force mechanism visible."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    case = ph.get("motion_force_case", "self_propulsion")
    q1,p1=ph["q"],ph["pose"]; q2,p2=teacher["q"],teacher["pose"]; N=max(len(q1),len(q2))
    all_pts=[]
    for q,p in [(q1,p1),(q2,p2)]:
        for k in np.linspace(0,len(q)-1,min(len(q),12)).round().astype(int): all_pts.append(_world_vertices(chain,q[k],p[k]))
    if case == "frame_target":
        target=_frame_target_from_task(task)
        if target is not None: all_pts.append(np.asarray([target]))
    if case == "gravity_ground":
        gy=_ground_y_from_task(task)
        pts0=np.concatenate(all_pts,axis=0); all_pts.append(np.asarray([[pts0[:,0].min(),gy],[pts0[:,0].max(),gy]]))
    if case == "rigid_loop":
        tb=_rigid_loop_target_body(task)
        if tb is not None:
            tw=_transform_body_vertices(tb,task.pose_target.detach().cpu().numpy()); all_pts.append(tw)
    pts=np.concatenate(all_pts,axis=0); xmin,ymin=pts.min(axis=0); xmax,ymax=pts.max(axis=0); pad=.12*max(xmax-xmin,ymax-ymin,1e-2)
    fig,axes=plt.subplots(1,2,figsize=(11,5)); lines=[]; trails=[]; force_arrows=[None,None]; frame_trails=[]
    titles=[f"pH + FIM/SF ({case})","intrinsic variational reference"]
    for ai,(ax,title) in enumerate(zip(axes,titles)):
        body,=ax.plot([],[],marker='o',linewidth=2.2); trail,=ax.plot([],[],linewidth=1.2,alpha=.7); lines.append(body); trails.append(trail)
        if ai==0 and case=="frame_target":
            target=_frame_target_from_task(task); start=np.asarray((task.metadata or {}).get('frame_start_world',[np.nan,np.nan]),dtype=float)
            if target is not None: ax.scatter([target[0]],[target[1]],marker='*',s=100,label='frame target')
            if np.isfinite(start).all(): ax.scatter([start[0]],[start[1]],marker='o',s=45,label='frame start')
            ft,=ax.plot([],[],linewidth=1.8,label='tracked frame'); frame_trails.append(ft)
        else: frame_trails.append(None)
        if ai==0 and case=="gravity_ground": ax.axhline(_ground_y_from_task(task),linewidth=2,label='ground y=0')
        if ai==0 and case=="self_propulsion":
            pt=task.pose_target.detach().cpu().numpy(); ax.scatter([pt[0]],[pt[1]],marker='x',s=65,label='target base')
        if ai==0 and case=="rigid_loop":
            tb=_rigid_loop_target_body(task)
            target_pose=task.pose_target.detach().cpu().numpy()
            if tb is not None:
                tw=_transform_body_vertices(tb,target_pose); ax.plot(tw[:,0],tw[:,1],linestyle="--",linewidth=2.2,label="rigid target polygon")
            sampled=_rigid_loop_sampled_crescent_body(task)
            if sampled is not None:
                sw=_transform_body_vertices(sampled,target_pose); ax.plot(sw[:,0],sw[:,1],linestyle=":",linewidth=1.2,label="sampled crescent")
            ax.scatter([target_pose[0]],[target_pose[1]],marker="x",s=60,label="target pose")
        ax.set_title(title); ax.set_xlim(xmin-pad,xmax+pad); ax.set_ylim(ymin-pad,ymax+pad); ax.set_aspect('equal'); ax.grid(True,alpha=.25)
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=7)
    comhist=[[],[]]; framehist=[]
    def update(i):
        artists=[]
        for a,(q,p) in enumerate([(q1,p1),(q2,p2)]):
            idx=min(i,len(q)-1); v=_world_vertices(chain,q[idx],p[idx]); c=v.mean(axis=0); comhist[a].append(c); cc=np.asarray(comhist[a]); lines[a].set_data(v[:,0],v[:,1]); trails[a].set_data(cc[:,0],cc[:,1]); artists += [lines[a],trails[a]]
        if case=="frame_target":
            pos=_finite(ph['tracked_frame_position_world']); frc=_finite(ph['tracked_frame_force_world']); idx=min(i,len(pos)-1)
            if np.isfinite(pos[idx]).all():
                framehist.append(pos[idx]); fh=np.asarray(framehist); frame_trails[0].set_data(fh[:,0],fh[:,1]); artists.append(frame_trails[0])
                # Replace force arrow each frame.
                if force_arrows[0] is not None: force_arrows[0].remove()
                f=frc[idx]; sc=.08/max(np.linalg.norm(f),1e-8); force_arrows[0]=axes[0].arrow(pos[idx,0],pos[idx,1],sc*f[0],sc*f[1],width=.002,head_width=.018,length_includes_head=True); artists.append(force_arrows[0])
        elif case=="gravity_ground":
            gp=_finite(ph['ground_contact_positions_world']); ga=_finite(ph['ground_active']); idx=min(i,len(gp)-1)
            # Draw current active contacts as a fresh scatter.
            if hasattr(update,'contact_scatter') and update.contact_scatter is not None: update.contact_scatter.remove()
            act=ga[idx]>.5
            update.contact_scatter=axes[0].scatter(gp[idx,act,0],gp[idx,act,1],marker='s',s=35) if np.any(act) else None
            if update.contact_scatter is not None: artists.append(update.contact_scatter)
        return artists
    update.contact_scatter=None
    ani=animation.FuncAnimation(fig,update,frames=N,interval=1000/max(fps,1),blit=False)
    ani.save(path,writer=animation.PillowWriter(fps=fps),dpi=130); plt.close(fig)

def plot_pinocchio_transfer_diagnostics(
    task: LocomotionTask,
    chain: PlanarMultiLinkChain,
    result: dict,
    path: str | Path,
    dt: float,
) -> None:
    """Focused diagnostic panel for optional Pinocchio+anisotropic-contact transfer."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pose = np.asarray(result["pose"], dtype=float)
    joints = np.asarray(result["joints"], dtype=float)
    qref = np.asarray(result.get("q_ref", np.zeros((max(len(joints)-1,1), joints.shape[1]))), dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    ax=axes[0,0]
    com=[]
    for k in range(len(joints)):
        q=torch.as_tensor(joints[k],dtype=chain.dtype); p=torch.as_tensor(pose[k],dtype=chain.dtype)
        v=_world_vertices(chain,q,p); com.append(v.mean(axis=0))
    com=np.asarray(com); ax.plot(com[:,0],com[:,1],linewidth=1.8)
    for idx in np.unique(np.linspace(0,len(joints)-1,6).round().astype(int)):
        q=torch.as_tensor(joints[idx],dtype=chain.dtype); p=torch.as_tensor(pose[idx],dtype=chain.dtype)
        v=_world_vertices(chain,q,p); ax.plot(v[:,0],v[:,1],marker='o',markersize=2,alpha=.5)
    ax.scatter([task.pose_target[0].item()],[task.pose_target[1].item()],marker='x',s=60,color='black')
    ax.set_title('Pinocchio world locomotion'); ax.axis('equal'); ax.grid(True,alpha=.25)

    ax=axes[0,1]
    t=np.arange(len(joints))*dt
    for j in range(min(3,joints.shape[1])):
        ax.plot(t,joints[:,j],label=f'q{j+1}')
        if len(qref): ax.plot(np.arange(len(qref))*dt,qref[:,j],linestyle='--',alpha=.7,label=f'qref{j+1}')
    ax.set_title('Joint kinematics / spline reference'); ax.grid(True,alpha=.25); ax.legend(fontsize=7,ncol=2)

    ax=axes[0,2]
    H=np.asarray(result.get('hamiltonian',[]),dtype=float)
    if len(H): ax.plot(np.arange(len(H))*dt,H,label='H')
    if len(result.get('kinetic_energy',[])): ax.plot(np.arange(len(result['kinetic_energy']))*dt,result['kinetic_energy'],label='T')
    if len(result.get('spring_energy',[])): ax.plot(np.arange(len(result['spring_energy']))*dt,result['spring_energy'],label='V')
    ax.set_title('Pinocchio Hamiltonian diagnostic'); ax.grid(True,alpha=.25); ax.legend(fontsize=8)

    ax=axes[1,0]
    tq=np.arange(len(result.get('joint_torque',[])))*dt
    if len(tq): ax.plot(tq,np.linalg.norm(np.asarray(result['joint_torque']),axis=1),label='joint torque')
    if len(result.get('fim_force_joint',[])): ax.plot(tq,np.linalg.norm(np.asarray(result['fim_force_joint']),axis=1),label='FIM damping')
    ax.set_title('Actuation / FIM force'); ax.grid(True,alpha=.25); ax.legend(fontsize=8)

    ax=axes[1,1]
    if len(result.get('fim_eigs',[])):
        F=np.asarray(result['fim_eigs']); ti=np.arange(len(F))*dt
        for j in range(F.shape[1]): ax.plot(ti,F[:,j],linewidth=1)
    if len(result.get('sf_eigs',[])):
        S=np.asarray(result['sf_eigs']); ts=np.arange(len(S))*dt
        for j in range(S.shape[1]): ax.plot(ts,S[:,j],linestyle='--',linewidth=1)
    ax.set_title('FIM (solid) / SF (dashed) spectra'); ax.grid(True,alpha=.25)

    ax=axes[1,2]
    tp=np.arange(len(result.get('contact_power',[])))*dt
    if len(tp): ax.plot(tp,result['contact_power'],label='contact/friction power')
    if len(result.get('reference_power',[])): ax.plot(tp,result['reference_power'],label='reference power')
    if len(result.get('joint_damping_power',[])): ax.plot(tp,result['joint_damping_power'],label='joint+FIM damping')
    ax.axhline(0,color='black',linewidth=.7); ax.set_title('Power ports / passivity signs'); ax.grid(True,alpha=.25); ax.legend(fontsize=7)

    fig.suptitle('Optional Pinocchio articulated-transfer check')
    fig.tight_layout(rect=[0,0,1,.96]); fig.savefig(path,dpi=170); plt.close(fig)


def plot_three_motion_case_overview(
    tasks: dict[str, LocomotionTask] | LocomotionTask,
    chain: PlanarMultiLinkChain,
    results: dict[str, dict[str, torch.Tensor]],
    teacher: dict[str, torch.Tensor],
    path: str | Path,
) -> None:
    """Three distinct physical tasks, not three force laws on one pose target."""
    if isinstance(tasks, LocomotionTask):
        task_map={c:tasks for c in ["self_propulsion","frame_target","gravity_ground"]}
    else:
        task_map=tasks
    cases=["self_propulsion","frame_target","gravity_ground"]
    labels={"self_propulsion":"SE(2) intrinsic locomotion","frame_target":"Selected frame: (x,y) -> (x',y')","gravity_ground":"Gravity -> contact with ground y=0"}
    fig,axes=plt.subplots(1,3,figsize=(16,5.2),squeeze=False)
    all_pts=[]
    for c in cases:
        r=results[c]
        for k in np.unique(np.linspace(0,len(r['q'])-1,8).round().astype(int)): all_pts.append(_world_vertices(chain,r['q'][k],r['pose'][k]))
        if c=='frame_target':
            tgt=_frame_target_from_task(task_map[c]);
            if tgt is not None: all_pts.append(np.asarray([tgt]))
    pts=np.concatenate(all_pts,axis=0); xmin,ymin=pts.min(axis=0); xmax,ymax=pts.max(axis=0); pad=.12*max(xmax-xmin,ymax-ymin,1e-2)
    for ax,c in zip(axes.flat,cases):
        r=results[c]; task=task_map[c]; q=r['q']; pose=r['pose']; com=np.asarray([_world_com(chain,q[k],pose[k]) for k in range(len(q))])
        ax.plot(com[:,0],com[:,1],linewidth=2,label='pH CoM')
        for idx in np.unique(np.linspace(0,len(q)-1,6).round().astype(int)):
            v=_world_vertices(chain,q[idx],pose[idx]); ax.plot(v[:,0],v[:,1],marker='o',markersize=2,alpha=.38)
        if c=='self_propulsion':
            pt=task.pose_target.detach().cpu().numpy(); ax.scatter([pt[0]],[pt[1]],marker='x',s=65,label='target base pose')
        elif c=='frame_target':
            pos=_finite(r['tracked_frame_position_world']); tgt=_frame_target_from_task(task); ax.plot(pos[:,0],pos[:,1],linewidth=2,label='tracked link frame')
            if tgt is not None: ax.scatter([tgt[0]],[tgt[1]],marker='*',s=100,label='frame target')
            start=np.asarray((task.metadata or {}).get('frame_start_world',[np.nan,np.nan]),dtype=float)
            if np.isfinite(start).all(): ax.scatter([start[0]],[start[1]],marker='o',s=45,label='frame start')
        elif c=='gravity_ground':
            gy=_ground_y_from_task(task); ax.axhline(gy,linewidth=2,label='ground y=0'); gp=_finite(r['ground_contact_positions_world']); ga=_finite(r['ground_active']);
            active_any=ga[-1]>.5
            if np.any(active_any): ax.scatter(gp[-1,active_any,0],gp[-1,active_any,1],marker='s',s=35,label='terminal contacts')
        ax.set_title(labels[c]); ax.set_xlim(xmin-pad,xmax+pad); ax.set_ylim(min(ymin-pad,_ground_y_from_task(task)-.05) if c=='gravity_ground' else ymin-pad,ymax+pad); ax.set_aspect('equal'); ax.grid(True,alpha=.25); ax.legend(fontsize=7)
    fig.suptitle('Three articulated tasks: intrinsic world pose, direct frame force, and gravity-ground contact')
    fig.tight_layout(rect=[0,0,1,.95]); fig.savefig(path,dpi=175); plt.close(fig)


def animate_three_motion_cases(
    tasks: dict[str, LocomotionTask] | LocomotionTask,
    chain: PlanarMultiLinkChain,
    results: dict[str, dict[str, torch.Tensor]],
    path: str | Path,
    fps: int = 7,
) -> None:
    if isinstance(tasks, LocomotionTask): task_map={c:tasks for c in ["self_propulsion","frame_target","gravity_ground"]}
    else: task_map=tasks
    cases=["self_propulsion","frame_target","gravity_ground"]
    labels={"self_propulsion":"SE(2) intrinsic locomotion","frame_target":"One link frame pulled to target","gravity_ground":"Gravity + y=0 contact"}
    N=max(len(results[c]['q']) for c in cases); all_pts=[]
    for c in cases:
        r=results[c]
        for k in np.unique(np.linspace(0,len(r['q'])-1,10).round().astype(int)): all_pts.append(_world_vertices(chain,r['q'][k],r['pose'][k]))
    pts=np.concatenate(all_pts,axis=0); xmin,ymin=pts.min(axis=0); xmax,ymax=pts.max(axis=0); pad=.12*max(xmax-xmin,ymax-ymin,1e-2)
    fig,axes=plt.subplots(1,3,figsize=(16,5.2)); lines=[]; trails=[]; frame_line=None; frame_hist=[]; contact_scatter=None
    for ax,c in zip(axes,cases):
        line,=ax.plot([],[],marker='o',linewidth=2.2); trail,=ax.plot([],[],linewidth=1.2,alpha=.7); lines.append(line); trails.append(trail); task=task_map[c]
        if c=='self_propulsion':
            pt=task.pose_target.detach().cpu().numpy(); ax.scatter([pt[0]],[pt[1]],marker='x',s=65)
        elif c=='frame_target':
            tgt=_frame_target_from_task(task); start=np.asarray((task.metadata or {}).get('frame_start_world',[np.nan,np.nan]),dtype=float)
            if tgt is not None: ax.scatter([tgt[0]],[tgt[1]],marker='*',s=100)
            if np.isfinite(start).all(): ax.scatter([start[0]],[start[1]],marker='o',s=45)
            frame_line,=ax.plot([],[],linewidth=1.8)
        elif c=='gravity_ground': ax.axhline(_ground_y_from_task(task),linewidth=2)
        ax.set_title(labels[c]); ax.set_xlim(xmin-pad,xmax+pad); ax.set_ylim(min(ymin-pad,-.05),ymax+pad); ax.set_aspect('equal'); ax.grid(True,alpha=.25)
    comhist=[[],[],[]]
    def update(i):
        nonlocal contact_scatter
        artists=[]
        for j,c in enumerate(cases):
            r=results[c]; idx=min(i,len(r['q'])-1); v=_world_vertices(chain,r['q'][idx],r['pose'][idx]); cc=v.mean(axis=0); comhist[j].append(cc); hist=np.asarray(comhist[j]); lines[j].set_data(v[:,0],v[:,1]); trails[j].set_data(hist[:,0],hist[:,1]); artists += [lines[j],trails[j]]
        r=results['frame_target']; idx=min(i,len(r['tracked_frame_position_world'])-1); pos=_finite(r['tracked_frame_position_world'])[idx]
        if frame_line is not None and np.isfinite(pos).all():
            frame_hist.append(pos); fh=np.asarray(frame_hist); frame_line.set_data(fh[:,0],fh[:,1]); artists.append(frame_line)
        r=results['gravity_ground']; idx=min(i,len(r['ground_contact_positions_world'])-1); gp=_finite(r['ground_contact_positions_world'])[idx]; ga=_finite(r['ground_active'])[idx]>.5
        if contact_scatter is not None: contact_scatter.remove(); contact_scatter=None
        if np.any(ga): contact_scatter=axes[2].scatter(gp[ga,0],gp[ga,1],marker='s',s=35); artists.append(contact_scatter)
        return artists
    ani=animation.FuncAnimation(fig,update,frames=N,interval=1000/max(fps,1),blit=False); ani.save(path,writer=animation.PillowWriter(fps=fps),dpi=130); plt.close(fig)



def _primary_error_series(task: LocomotionTask, rollout: dict[str, torch.Tensor]) -> np.ndarray:
    """Task-specific error over the *actual* physical rollout."""
    from .geometry import se2_endpoint_error
    case = str(rollout.get("motion_force_case", "self_propulsion"))
    if case == "frame_target":
        pos = _finite(rollout["tracked_frame_position_world"])
        tgt = _frame_target_from_task(task)
        if tgt is None:
            return np.full(len(pos), np.nan)
        return np.linalg.norm(pos - tgt[None, :], axis=1)
    if case == "gravity_ground":
        return np.abs(_finite(rollout["min_ground_gap"]))
    if case == "rigid_loop":
        q = _finite(rollout["q"]); qT = np.asarray((task.metadata or {})["rigid_loop_q_target"], dtype=float)
        return np.sqrt(np.mean((q - qT[None,:])**2, axis=1))
    vals = []
    for p in rollout["pose"]:
        vals.append(float(torch.linalg.norm(se2_endpoint_error(p, task.pose_target)).detach()))
    return np.asarray(vals, dtype=float)


def plot_information_metric_ablation(
    tasks: dict[str, LocomotionTask],
    results: dict[str, dict[str, dict[str, torch.Tensor]]],
    path: str | Path,
    dt: float,
) -> None:
    """Focused FIM/SF necessity plot over the extended settling horizon.

    ``results[case][variant]`` uses variants ``none``, ``fim_only``,
    ``sf_only`` and ``fim_sf``.  Left column shows the physical task error;
    right column shows generalized-speed decay.  A vertical line marks the end
    of the learned spline/reference actuation; everything to its right is the
    free/contact settling tail.
    """
    cases = [c for c in ["self_propulsion", "frame_target", "gravity_ground"] if c in results]
    labels = {
        "none": "no FIM / no SF",
        "fim_only": "FIM only",
        "sf_only": "SF only",
        "fim_sf": "FIM + SF",
    }
    fig, axes = plt.subplots(len(cases), 2, figsize=(13, 4.0 * len(cases)), squeeze=False)
    for row, case in enumerate(cases):
        task = tasks[case]
        for variant in ["none", "fim_only", "sf_only", "fim_sf"]:
            if variant not in results[case]:
                continue
            r = results[case][variant]
            err = _primary_error_series(task, r)
            t_err = np.arange(len(err)) * dt
            line0, = axes[row, 0].plot(t_err, err, label=labels[variant], linewidth=1.7)
            nu = _finite(r["nu"])
            speed = np.linalg.norm(nu, axis=1)
            t_speed = np.arange(len(speed)) * dt
            line1, = axes[row, 1].plot(t_speed, speed, label=labels[variant], linewidth=1.7)
            if bool(float(torch.as_tensor(r.get("diverged", 0.0)).detach().cpu()) > 0.5):
                if len(err): axes[row, 0].scatter([t_err[-1]], [err[-1]], marker="x", s=38, color=line0.get_color())
                if len(speed): axes[row, 1].scatter([t_speed[-1]], [speed[-1]], marker="x", s=38, color=line1.get_color())
        sample = next(iter(results[case].values()))
        tc = float(torch.as_tensor(sample.get("control_horizon", 1.0)).detach().cpu())
        for col in range(2):
            axes[row, col].axvline(tc, linestyle="--", linewidth=1.0, alpha=0.75, label="actuation ends" if row == 0 else None)
            axes[row, col].grid(True, alpha=0.25)
            axes[row, col].set_xlabel("time")
        axes[row, 0].set_ylabel("task error")
        axes[row, 1].set_ylabel(r"$\|\nu\|$")
        axes[row, 1].set_yscale("symlog", linthresh=1e-3)
        axes[row, 0].set_title(f"{case}: physical task convergence")
        axes[row, 1].set_title(f"{case}: velocity / steady-state convergence")
        axes[row, 0].legend(fontsize=8)
        axes[row, 1].legend(fontsize=8)
    fig.suptitle("Necessity of configuration FIM and Souriau–Fisher metric over extended evaluation")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def animate_information_metric_variants(
    task: LocomotionTask,
    chain: PlanarMultiLinkChain,
    results: dict[str, dict[str, torch.Tensor]],
    path: str | Path,
    dt: float,
    fps: int = 7,
) -> None:
    """Animate the same random physical task under all available FIM/SF checkpoints.

    The four canonical panels are ``none``, ``fim_only``, ``sf_only`` and
    ``fim_sf``. Missing variants are shown as unavailable.  Each panel draws the
    articulated body, its center-of-mass trail, and the physical task geometry
    (SE(2) target, selected frame target, or ground/contact state).
    """
    variants = ["none", "fim_only", "sf_only", "fim_sf"]
    labels = {
        "none": "no FIM / no SF",
        "fim_only": "FIM only",
        "sf_only": "SF only",
        "fim_sf": "FIM + SF",
    }
    available = [v for v in variants if v in results]
    if not available:
        raise ValueError("results contains no recognized information-metric variants")

    # Collect finite geometry over all rollouts to keep a common world scale.
    all_pts = []
    for variant in available:
        r = results[variant]
        for k in range(len(r["q"])):
            v = _world_vertices(chain, r["q"][k], r["pose"][k])
            if np.isfinite(v).all():
                all_pts.append(v)
        if r.get("motion_force_case", "") == "frame_target":
            tgt = _frame_target_from_task(task)
            if tgt is not None:
                all_pts.append(np.asarray([tgt], dtype=float))
        if r.get("motion_force_case", "") == "rigid_loop":
            tb = _rigid_loop_target_body(task)
            if tb is not None:
                all_pts.append(_transform_body_vertices(tb, task.initial_pose.detach().cpu().numpy()))
    pts = np.concatenate(all_pts, axis=0)
    xmin, ymin = pts.min(axis=0); xmax, ymax = pts.max(axis=0)
    span = max(xmax - xmin, ymax - ymin, 1e-2); pad = 0.12 * span
    case = str(results[available[0]].get("motion_force_case", "self_propulsion"))
    if case == "gravity_ground":
        ymin = min(ymin, _ground_y_from_task(task) - 0.05)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), squeeze=False)
    bodies = {}; trails = {}; frame_trails = {}; contact_scatters = {}; histories = {}
    target_pose = task.pose_target.detach().cpu().numpy()
    frame_target = _frame_target_from_task(task)

    for ax, variant in zip(axes.flat, variants):
        if variant not in results:
            ax.text(0.5, 0.5, f"{labels[variant]}\ncheckpoint unavailable", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off(); continue
        color = {"none":"tab:blue", "fim_only":"tab:orange", "sf_only":"tab:green", "fim_sf":"tab:red"}[variant]
        body, = ax.plot([], [], marker="o", markersize=4, linewidth=2, color=color)
        trail, = ax.plot([], [], linewidth=1.5, alpha=.8, color=color, label="CoM trail")
        bodies[variant] = body; trails[variant] = trail; histories[variant] = []
        frame_trails[variant] = None; contact_scatters[variant] = None

        if case == "self_propulsion":
            ax.scatter([target_pose[0]], [target_pose[1]], marker="x", s=65, color="black", label="target base")
        elif case == "frame_target":
            if frame_target is not None:
                ax.scatter([frame_target[0]], [frame_target[1]], marker="*", s=105, color="black", label="frame target")
            ft, = ax.plot([], [], linewidth=1.4, linestyle="--", color=color, label="tracked frame")
            frame_trails[variant] = ft
        elif case == "gravity_ground":
            ax.axhline(_ground_y_from_task(task), linewidth=2, color="black", label="ground y=0")
        elif case == "rigid_loop":
            tb = _rigid_loop_target_body(task)
            if tb is not None:
                tw = _transform_body_vertices(tb, task.initial_pose.detach().cpu().numpy())
                ax.plot(tw[:,0], tw[:,1], linestyle="--", linewidth=1.7, color="black", label="target crescent")

        ax.set_xlim(xmin - pad, xmax + pad); ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_aspect("equal", adjustable="box"); ax.grid(True, alpha=.25)
        ax.set_title(labels[variant]); ax.legend(fontsize=7, loc="best")

    max_frames = max(len(results[v]["q"]) for v in available)

    def update(i):
        artists = []
        for variant in available:
            r = results[variant]
            idx = min(i, len(r["q"]) - 1)
            verts = _world_vertices(chain, r["q"][idx], r["pose"][idx])
            bodies[variant].set_data(verts[:, 0], verts[:, 1]); artists.append(bodies[variant])
            c = verts.mean(axis=0); histories[variant].append(c)
            cc = np.asarray(histories[variant]); trails[variant].set_data(cc[:, 0], cc[:, 1]); artists.append(trails[variant])

            if case == "frame_target" and frame_trails[variant] is not None:
                pos = _finite(r["tracked_frame_position_world"])
                j = min(idx, len(pos)-1)
                frame_trails[variant].set_data(pos[:j+1,0], pos[:j+1,1]); artists.append(frame_trails[variant])
            elif case == "gravity_ground":
                if contact_scatters[variant] is not None:
                    contact_scatters[variant].remove(); contact_scatters[variant] = None
                gp = _finite(r["ground_contact_positions_world"]); ga = _finite(r["ground_active"])
                j = min(idx, len(gp)-1); active = ga[j] > 0.5
                if np.any(active):
                    ax = axes.flat[variants.index(variant)]
                    contact_scatters[variant] = ax.scatter(gp[j,active,0], gp[j,active,1], marker="s", s=35, color="black")
                    artists.append(contact_scatters[variant])

            err = _primary_error_series(task, r)
            ej = err[min(idx, len(err)-1)] if len(err) else np.nan
            speed = np.linalg.norm(_finite(r["nu"])[min(idx, len(r["nu"])-1)])
            ax = axes.flat[variants.index(variant)]
            ax.set_title(f"{labels[variant]}\nerr={ej:.3g}, ||nu||={speed:.3g}, t={idx*dt:.2f}")
        return artists

    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=1000/max(fps,1), blit=False)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(path, writer=animation.PillowWriter(fps=fps), dpi=125)
    plt.close(fig)


def plot_four_motion_case_overview(tasks, chain: PlanarMultiLinkChain, results: dict[str, dict], teachers, path: str | Path) -> None:
    """2x2 overview for the three locomotion/force tasks plus rigid circle-to-crescent loop."""
    if isinstance(tasks, LocomotionTask):
        task_map={c:tasks for c in ["self_propulsion","frame_target","gravity_ground","rigid_loop"]}
    else:
        task_map=tasks
    cases=[c for c in ["self_propulsion","frame_target","gravity_ground","rigid_loop"] if c in results]
    labels={
        "self_propulsion":"SE(2) intrinsic locomotion",
        "frame_target":"Selected frame: (x,y) → (x',y')",
        "gravity_ground":"Gravity → ground y=0",
        "rigid_loop":"Rigid loop: circle → crescent",
    }
    fig,axes=plt.subplots(2,2,figsize=(12,10)); axes=axes.flat
    for ax,c in zip(axes,cases):
        task=task_map[c]; r=results[c]; q=r["q"]; pose=r["pose"]
        ids=np.unique(np.linspace(0,len(q)-1,6).round().astype(int))
        com=np.asarray([_world_com(chain,q[k],pose[k]) for k in range(len(q))])
        ax.plot(com[:,0],com[:,1],linewidth=1.4,label="CoM")
        for idx in ids:
            v=_world_vertices(chain,q[idx],pose[idx]); ax.plot(v[:,0],v[:,1],marker="o",markersize=2.4,alpha=.35)
        _draw_external_world_context(ax,task,chain,r,ids[:-1])
        if c=="rigid_loop":
            target=_rigid_loop_target_body(task)
            if target is not None:
                tw=_transform_body_vertices(target,task.initial_pose.detach().cpu().numpy())
                ax.plot(tw[:,0],tw[:,1],linestyle="--",linewidth=2,label="target crescent")
        ax.set_title(labels[c]); ax.set_aspect("equal",adjustable="box"); ax.grid(True,alpha=.25); ax.legend(fontsize=7)
    for ax in list(axes)[len(cases):]: ax.set_axis_off()
    fig.suptitle("Four articulated tasks under a common spline / pH representation")
    fig.tight_layout(); fig.savefig(path,dpi=175); plt.close(fig)


def animate_four_motion_cases(tasks, chain: PlanarMultiLinkChain, results: dict[str, dict], path: str | Path, fps: int = 7) -> None:
    """2x2 animation including the rigid-loop circle-to-crescent task."""
    if isinstance(tasks, LocomotionTask):
        task_map={c:tasks for c in ["self_propulsion","frame_target","gravity_ground","rigid_loop"]}
    else:
        task_map=tasks
    cases=[c for c in ["self_propulsion","frame_target","gravity_ground","rigid_loop"] if c in results]
    labels={"self_propulsion":"Intrinsic locomotion","frame_target":"One frame pulled","gravity_ground":"Gravity + ground","rigid_loop":"Rigid circle → crescent"}
    allpts=[]
    for c in cases:
        r=results[c]
        for k in np.linspace(0,len(r["q"])-1,min(len(r["q"]),14)).round().astype(int): allpts.append(_world_vertices(chain,r["q"][k],r["pose"][k]))
    P=np.concatenate(allpts,axis=0); xmin,ymin=P.min(axis=0); xmax,ymax=P.max(axis=0); pad=.12*max(xmax-xmin,ymax-ymin,1e-3)
    fig,axs=plt.subplots(2,2,figsize=(11,9)); axs=axs.flat; bodies={}; trails={}
    for ax,c in zip(axs,cases):
        task=task_map[c]; ax.set_xlim(xmin-pad,xmax+pad); ax.set_ylim(ymin-pad,ymax+pad); ax.set_aspect("equal"); ax.grid(True,alpha=.25); ax.set_title(labels[c])
        body,=ax.plot([],[],marker="o",markersize=4,linewidth=2); trail,=ax.plot([],[],linewidth=1.2,alpha=.7); bodies[c]=body; trails[c]=trail
        if c=="gravity_ground": ax.axhline(_ground_y_from_task(task),linewidth=1.5)
        if c=="frame_target":
            tgt=_frame_target_from_task(task); ax.scatter([tgt[0]],[tgt[1]],marker="*",s=100,color="black")
        if c=="rigid_loop":
            tb=_rigid_loop_target_body(task)
            if tb is not None:
                tw=_transform_body_vertices(tb,task.initial_pose.detach().cpu().numpy()); ax.plot(tw[:,0],tw[:,1],linestyle="--",linewidth=1.7)
    N=max(len(results[c]["q"]) for c in cases)
    def update(i):
        arts=[]
        for c in cases:
            r=results[c]; idx=min(i,len(r["q"])-1); v=_world_vertices(chain,r["q"][idx],r["pose"][idx]); bodies[c].set_data(v[:,0],v[:,1])
            com=np.asarray([_world_com(chain,r["q"][k],r["pose"][k]) for k in range(idx+1)]); trails[c].set_data(com[:,0],com[:,1]); arts.extend([bodies[c],trails[c]])
        return arts
    anim=animation.FuncAnimation(fig,update,frames=N,interval=1000/max(fps,1),blit=False); path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); anim.save(path,writer=animation.PillowWriter(fps=fps),dpi=125); plt.close(fig)


def plot_rigid_loop_decomposition(
    task: LocomotionTask,
    chain: PlanarMultiLinkChain,
    ph: dict[str, torch.Tensor],
    teacher: dict[str, torch.Tensor],
    path: str | Path,
    dt: float,
) -> None:
    """Focused diagnostic for the circle->crescent factorization.

    Left: quotient/local geometry in a material body frame.
    Middle: positioned geometry in world SE(2).
    Right: local-shape, joint-coordinate and SE(2) errors over time.
    """
    from .rigid_loop import canonicalize_loop_vertices, cyclic_local_shape_loss

    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    md = task.metadata or {}
    qhist = ph["q"]
    qT = torch.as_tensor(md["rigid_loop_q_target"], dtype=qhist.dtype, device=qhist.device)
    target = torch.as_tensor(md.get("rigid_loop_target_rigid_vertices_body", md["rigid_loop_target_vertices_body"]), dtype=qhist.dtype, device=qhist.device)
    sampled_np = _rigid_loop_sampled_crescent_body(task)

    pred_body = chain.kinematics(qhist[-1])["joint_positions"].detach().cpu()
    target_body = target.detach().cpu()
    pred_can = canonicalize_loop_vertices(pred_body).detach().cpu().numpy()
    # Select the same seam gauge used by the local loss.
    _, seam = cyclic_local_shape_loss(pred_body, target_body, allow_cyclic_shift=bool(md.get("rigid_loop_allow_cyclic_seam", True)))
    tgt_unique = target_body[:-1] if target_body.shape[0] == pred_can.shape[0] + 1 else target_body
    tgt_shift = torch.roll(tgt_unique, shifts=-int(seam), dims=-2)
    tgt_can = canonicalize_loop_vertices(tgt_shift).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    ax = axes[0]
    ax.plot(np.r_[pred_can[:,0], pred_can[0,0]], np.r_[pred_can[:,1], pred_can[0,1]], marker="o", label="pH final local shape")
    ax.plot(np.r_[tgt_can[:,0], tgt_can[0,0]], np.r_[tgt_can[:,1], tgt_can[0,1]], marker="o", linestyle="--", label="rigid crescent target")
    if sampled_np is not None:
        samp = torch.as_tensor(sampled_np, dtype=qhist.dtype, device=qhist.device)
        samp_can = canonicalize_loop_vertices(samp).detach().cpu().numpy()
        ax.plot(np.r_[samp_can[:,0], samp_can[0,0]], np.r_[samp_can[:,1], samp_can[0,1]], linestyle=":", label="sampled crescent reference")
    ax.set_title("Local deformation modulo $SE(2)$")
    ax.axis("equal"); ax.grid(True, alpha=.25); ax.legend(fontsize=7)

    ax = axes[1]
    p_final = ph["pose"][-1].detach().cpu().numpy()
    w_final = _world_vertices(chain, ph["q"][-1], ph["pose"][-1])
    target_np = _rigid_loop_target_body(task)
    target_world = _transform_body_vertices(target_np, task.pose_target.detach().cpu().numpy()) if target_np is not None else None
    teacher_world = _world_vertices(chain, teacher["q"][-1], teacher["pose"][-1])
    ax.plot(w_final[:,0], w_final[:,1], marker="o", label="pH final positioned loop")
    if target_world is not None: ax.plot(target_world[:,0], target_world[:,1], marker="o", linestyle="--", label="positioned target")
    ax.plot(teacher_world[:,0], teacher_world[:,1], linestyle=":", label="geometric teacher final")
    ax.scatter([task.pose_target[0].item()], [task.pose_target[1].item()], marker="x", s=55, label="target pose origin")
    ax.set_title("Global $SE(2)$ placement")
    ax.axis("equal"); ax.grid(True, alpha=.25); ax.legend(fontsize=7)

    ax = axes[2]
    local = _loop_local_error_series(task, chain, _finite(ph["q"]))
    qnp = _finite(ph["q"]); qTnp = qT.detach().cpu().numpy()
    qerr = np.sqrt(np.mean((qnp - qTnp[None,:])**2, axis=1))
    perr = np.asarray([float(torch.linalg.norm(se2_endpoint_error(p, task.pose_target)).detach()) for p in ph["pose"]])
    t = np.arange(len(qnp))*dt
    ax.plot(t, local, linewidth=2, label="local shape RMS")
    ax.plot(t, qerr, label="joint RMS")
    ax.plot(np.arange(len(perr))*dt, perr, label="$SE(2)$ pose error")
    floor = float(md.get("rigid_loop_representation_error", np.nan))
    if np.isfinite(floor): ax.axhline(floor, linestyle=":", label="best rigid-link measurement floor")
    ctrl = float(torch.as_tensor(ph.get("control_horizon", 1.0)).detach().cpu())
    ax.axvline(ctrl, linestyle="--", alpha=.7, label="actuation ends")
    ax.set_title("Separated convergence diagnostics")
    ax.set_xlabel("time"); ax.grid(True, alpha=.25); ax.legend(fontsize=7)

    fig.suptitle("Rigid-loop circle→crescent: local shape and global pose are trained as distinct factors")
    fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(path, dpi=180); plt.close(fig)
