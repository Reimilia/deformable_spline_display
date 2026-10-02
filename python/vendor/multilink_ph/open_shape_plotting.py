from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation

from .chain import PlanarMultiLinkChain
from .tasks import LocomotionTask
from .geometry import se2_endpoint_error
from .open_shape import canonicalize_open_vertices, open_local_shape_loss


def _world_vertices(chain: PlanarMultiLinkChain, q: torch.Tensor, pose: torch.Tensor) -> np.ndarray:
    q = torch.as_tensor(q)
    pose = torch.as_tensor(pose, dtype=q.dtype, device=q.device)
    pts = chain.kinematics(q)["joint_positions"]
    c, s = torch.cos(pose[2]), torch.sin(pose[2])
    R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
    return ((R @ pts.T).T + pose[:2]).detach().cpu().numpy()


def _transform(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    c, s = np.cos(pose[2]), np.sin(pose[2])
    R = np.array([[c, -s], [s, c]])
    return points @ R.T + pose[:2]


def _local_series(task: LocomotionTask, chain: PlanarMultiLinkChain, qhist: torch.Tensor) -> np.ndarray:
    target = torch.as_tensor((task.metadata or {})["open_shape_target_rigid_vertices_body"], dtype=qhist.dtype, device=qhist.device)
    out=[]
    for q in qhist:
        pred=chain.kinematics(q)["joint_positions"]
        out.append(float(torch.sqrt(open_local_shape_loss(pred,target)+1e-16).detach()))
    return np.asarray(out)


def plot_open_shape_diagnostics(
    task: LocomotionTask,
    chain: PlanarMultiLinkChain,
    ph: dict[str, torch.Tensor],
    teacher: dict[str, torch.Tensor],
    path: str | Path,
    dt: float,
) -> None:
    """Nine-panel open-chain control experiment for isolating closure effects."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    md=task.metadata or {}
    fig,axes=plt.subplots(3,3,figsize=(15,13))
    q=ph["q"]; pose=ph["pose"]
    target_body=np.asarray(md["open_shape_target_rigid_vertices_body"],dtype=float)
    sampled=np.asarray(md["open_shape_target_sampled_crescent_body"],dtype=float)
    target_pose=task.pose_target.detach().cpu().numpy()

    # 1. World motion and target alignment.
    ax=axes[0,0]
    ids=np.unique(np.linspace(0,len(q)-1,6).round().astype(int))
    com=[]
    for k in range(len(q)):
        w=_world_vertices(chain,q[k],pose[k]); com.append(w.mean(axis=0))
    com=np.asarray(com); ax.plot(com[:,0],com[:,1],linewidth=1.8,label="pH CoM")
    for k in ids:
        w=_world_vertices(chain,q[k],pose[k]); ax.plot(w[:,0],w[:,1],marker="o",markersize=2.5,alpha=.38)
    tw=_transform(target_body,target_pose); sw=_transform(sampled,target_pose)
    ax.plot(tw[:,0],tw[:,1],linestyle="--",linewidth=2.2,label="open rigid target")
    ax.plot(sw[:,0],sw[:,1],linestyle=":",linewidth=1.2,label="sampled open crescent")
    ax.scatter([target_pose[0]],[target_pose[1]],marker="x",s=55,label="target pose")
    ax.set_title("World alignment: no loop-closure constraint"); ax.axis("equal"); ax.grid(True,alpha=.25); ax.legend(fontsize=7)

    # 2. Local shape modulo SE2.
    ax=axes[0,1]
    pred=canonicalize_open_vertices(chain.kinematics(q[-1])["joint_positions"]).detach().cpu().numpy()
    tgt=canonicalize_open_vertices(torch.as_tensor(target_body,dtype=q.dtype,device=q.device)).detach().cpu().numpy()
    sam=canonicalize_open_vertices(torch.as_tensor(sampled,dtype=q.dtype,device=q.device)).detach().cpu().numpy()
    ax.plot(pred[:,0],pred[:,1],marker="o",label="pH final local")
    ax.plot(tgt[:,0],tgt[:,1],marker="o",linestyle="--",label="open rigid target")
    ax.plot(sam[:,0],sam[:,1],linestyle=":",label="sampled reference")
    ax.set_title("Local shape after removing $SE(2)$"); ax.axis("equal"); ax.grid(True,alpha=.25); ax.legend(fontsize=7)

    # 3. Separate errors and free endpoint gap.
    ax=axes[0,2]; t=np.arange(len(q))*dt
    qT=np.asarray(md["open_shape_q_target"],dtype=float); qnp=q.detach().cpu().numpy(); qerr=np.sqrt(np.mean((qnp-qT[None,:])**2,axis=1))
    local=_local_series(task,chain,q)
    perr=np.asarray([float(torch.linalg.norm(se2_endpoint_error(pp,task.pose_target)).detach()) for pp in pose])
    gaps=np.asarray([np.linalg.norm(chain.kinematics(qq)["joint_positions"].detach().cpu().numpy()[-1]-chain.kinematics(qq)["joint_positions"].detach().cpu().numpy()[0]) for qq in q])
    ax.plot(t,local,linewidth=2,label="local shape RMS"); ax.plot(t,qerr,label="joint RMS"); ax.plot(np.arange(len(perr))*dt,perr,label="$SE(2)$ pose error"); ax.plot(t,gaps,label="endpoint gap")
    floor=float(md.get("open_shape_representation_error",np.nan));
    if np.isfinite(floor): ax.axhline(floor,linestyle=":",label="representation floor")
    ax.set_title("Shape/pose alignment with free endpoints"); ax.set_xlabel("time"); ax.grid(True,alpha=.25); ax.legend(fontsize=7)

    # 4. q tracking.
    ax=axes[1,0]; qref=ph["q_ref"].detach().cpu().numpy();
    for j in range(min(4,qnp.shape[1])):
        ax.plot(t,qnp[:,j],label=f"q{j+1}"); ax.plot(np.arange(len(qref))*dt,qref[:,j],linestyle="--",alpha=.7,label=f"qref{j+1}")
    ax.set_title("Realized joints vs spline rest shape"); ax.set_xlabel("time"); ax.grid(True,alpha=.25); ax.legend(fontsize=6,ncol=2)

    # 5. Phase portrait.
    ax=axes[1,1]; pi=ph["pi"].detach().cpu().numpy(); tq=teacher["q"].detach().cpu().numpy(); tp=teacher["pi"].detach().cpu().numpy()
    ax.plot(qnp[:,0],pi[:,3],label="pH"); ax.plot(tq[:,0],tp[:,3],linestyle="--",label="teacher"); ax.set_xlabel("q1"); ax.set_ylabel("p_q1"); ax.set_title("Internal phase-space portrait"); ax.grid(True,alpha=.25); ax.legend(fontsize=7)

    # 6. Hamiltonian.
    ax=axes[1,2]; H=ph["hamiltonian"].detach().cpu().numpy(); T=ph["kinetic_energy"].detach().cpu().numpy(); V=ph["potential_energy"].detach().cpu().numpy(); th=np.arange(len(H))*dt
    ax.plot(th,H,label="H"); ax.plot(th,T,label="T"); ax.plot(th,V,label="V"); ax.set_title("Hamiltonian"); ax.set_xlabel("time"); ax.grid(True,alpha=.25); ax.legend(fontsize=7)

    # 7. Forces: specifically show absence of closure reaction.
    ax=axes[2,0]; tf=np.arange(ph["spring_force_joint"].shape[0])*dt
    for key,lab in [("spring_force_joint","spring"),("fim_force_joint","FIM damping"),("environment_force_joint","environment drag"),("total_joint_force","total joint")]:
        arr=ph[key].detach().cpu().numpy(); ax.plot(tf,np.linalg.norm(arr,axis=1),label=lab)
    lf=ph["loop_constraint_force"].detach().cpu().numpy(); ax.plot(tf,np.linalg.norm(lf,axis=1),linestyle="--",label="closure reaction (must be 0)")
    ax.set_title("Force decomposition / closure control"); ax.set_xlabel("time"); ax.grid(True,alpha=.25); ax.legend(fontsize=7)

    # 8. Momentum and velocity.
    ax=axes[2,1]; J=ph["momentum_map"].detach().cpu().numpy(); tm=np.arange(len(J))*dt
    for j,lab in enumerate(["Lz","Px","Py"]): ax.plot(tm,J[:,j],label=lab)
    nu=ph["nu"].detach().cpu().numpy(); ax.plot(np.arange(len(nu))*dt,np.linalg.norm(nu,axis=1),linestyle="--",label="|nu|")
    ax.set_title("Global momentum / settling"); ax.set_xlabel("time"); ax.grid(True,alpha=.25); ax.legend(fontsize=7)

    # 9. Passivity.
    ax=axes[2,2]; margin=ph["passivity_margin"].detach().cpu().numpy(); balance=ph["energy_balance_residual_cum"].detach().cpu().numpy(); diss=ph["dissipated_work"].detach().cpu().numpy(); tw=np.arange(len(margin))*dt
    ax.plot(tw,margin,label="supply work - Delta H"); ax.plot(tw,diss,label="dissipated work"); ax.plot(tw,balance,label="balance residual"); ax.axhline(0,linewidth=.8); ax.set_title("Passivity / energy balance"); ax.set_xlabel("time"); ax.grid(True,alpha=.25); ax.legend(fontsize=7)

    ctrl=float(torch.as_tensor(ph.get("control_horizon",1.0)).detach().cpu())
    for ax in axes.flat:
        if ax in [axes[0,0],axes[0,1]]: continue
        ax.axvline(ctrl,linestyle="--",linewidth=.8,alpha=.6)
    fig.suptitle("Open-chain closure-control experiment: circle arc -> crescent arc")
    fig.tight_layout(rect=[0,0,1,.96]); fig.savefig(path,dpi=175); plt.close(fig)


def animate_open_shape_comparison(
    task: LocomotionTask,
    chain: PlanarMultiLinkChain,
    ph: dict[str, torch.Tensor],
    teacher: dict[str, torch.Tensor],
    path: str | Path,
    fps: int=7,
) -> None:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    q1,p1=ph["q"],ph["pose"]; q2,p2=teacher["q"],teacher["pose"]; N=max(len(q1),len(q2))
    target=np.asarray((task.metadata or {})["open_shape_target_rigid_vertices_body"],dtype=float); tw=_transform(target,task.pose_target.detach().cpu().numpy())
    allp=[tw]
    for q,p in [(q1,p1),(q2,p2)]:
        for k in np.linspace(0,len(q)-1,min(10,len(q))).round().astype(int): allp.append(_world_vertices(chain,q[k],p[k]))
    pts=np.concatenate(allp,axis=0); xmin,ymin=pts.min(axis=0); xmax,ymax=pts.max(axis=0); pad=.12*max(xmax-xmin,ymax-ymin,1e-2)
    fig,axes=plt.subplots(1,2,figsize=(11,5)); lines=[]; trails=[]; comhist=[[],[]]
    for ax,title in zip(axes,["pH open-chain control","intrinsic geometric teacher"]):
        b,=ax.plot([],[],marker="o",linewidth=2.2); tr,=ax.plot([],[],linewidth=1.2,alpha=.7); lines.append(b); trails.append(tr); ax.plot(tw[:,0],tw[:,1],linestyle="--",linewidth=2,label="positioned open target"); ax.set_title(title); ax.set_xlim(xmin-pad,xmax+pad); ax.set_ylim(ymin-pad,ymax+pad); ax.set_aspect("equal"); ax.grid(True,alpha=.25); ax.legend(fontsize=7)
    def upd(i):
        arts=[]
        for a,(q,p) in enumerate([(q1,p1),(q2,p2)]):
            k=min(i,len(q)-1); w=_world_vertices(chain,q[k],p[k]); c=w.mean(axis=0); comhist[a].append(c); cc=np.asarray(comhist[a]); lines[a].set_data(w[:,0],w[:,1]); trails[a].set_data(cc[:,0],cc[:,1]); arts.extend([lines[a],trails[a]])
        return arts
    ani=animation.FuncAnimation(fig,upd,frames=N,interval=1000/max(fps,1),blit=False); ani.save(path,writer=animation.PillowWriter(fps=fps),dpi=120); plt.close(fig)
