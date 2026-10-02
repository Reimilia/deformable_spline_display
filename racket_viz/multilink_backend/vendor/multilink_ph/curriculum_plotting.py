from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from .chain import PlanarMultiLinkChain
from .geometry import se2_endpoint_error
from .rigid_loop import loop_body_vertices, cyclic_local_shape_loss
from .open_shape import open_body_vertices, open_local_shape_loss


def _world_vertices(chain: PlanarMultiLinkChain, q: torch.Tensor, pose: torch.Tensor, *, closed: bool) -> np.ndarray:
    q = torch.as_tensor(q)
    pose = torch.as_tensor(pose, dtype=q.dtype, device=q.device)
    body = chain.kinematics(q)["joint_positions"]
    c, s = torch.cos(pose[2]), torch.sin(pose[2])
    R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
    world = (R @ body.T).T + pose[:2]
    if closed and torch.linalg.norm(world[-1] - world[0]) > 1e-9:
        world = torch.cat([world, world[:1]], dim=0)
    return world.detach().cpu().numpy()


def _errors(task, chain, rollout, case: str):
    dtype = rollout["q"].dtype
    device = rollout["q"].device
    md = task.metadata or {}
    local, joint, pose = [], [], []
    if case == "rigid_loop":
        target = torch.as_tensor(md["rigid_loop_target_rigid_vertices_body"], dtype=dtype, device=device)
        qT = torch.as_tensor(md["rigid_loop_q_target"], dtype=dtype, device=device)
    else:
        target = torch.as_tensor(md["open_shape_target_rigid_vertices_body"], dtype=dtype, device=device)
        qT = torch.as_tensor(md["open_shape_q_target"], dtype=dtype, device=device)
    for k in range(rollout["q"].shape[0]):
        q = rollout["q"][k]
        if case == "rigid_loop":
            pred = loop_body_vertices(chain, q, unique=True)
            l, _ = cyclic_local_shape_loss(pred, target, allow_cyclic_shift=bool(md.get("rigid_loop_allow_cyclic_seam", False)))
        else:
            pred = open_body_vertices(chain, q)
            l = open_local_shape_loss(pred, target)
        local.append(torch.sqrt(l + 1e-16))
        joint.append(torch.sqrt(torch.mean((q - qT) ** 2) + 1e-16))
        pose.append(torch.linalg.norm(se2_endpoint_error(rollout["pose"][k], task.pose_target)))
    return torch.stack(local), torch.stack(joint), torch.stack(pose)


def plot_two_stage_curriculum(task, chain: PlanarMultiLinkChain, shape_rollout: dict, positioned_rollout: dict,
                              path: str | Path, dt: float, control_steps: int, case: str) -> None:
    """Show whether Phase II improves pose without destroying the Phase-I shape solution."""
    ls, js, ps = _errors(task, chain, shape_rollout, case)
    lp, jp, pp = _errors(task, chain, positioned_rollout, case)
    ts = np.arange(len(ls)) * dt
    tp = np.arange(len(lp)) * dt
    t_control = control_steps * dt

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    # Positioned world geometry at final time.
    closed = case == "rigid_loop"
    Xs = _world_vertices(chain, shape_rollout["q"][-1], shape_rollout["pose"][-1], closed=closed)
    Xp = _world_vertices(chain, positioned_rollout["q"][-1], positioned_rollout["pose"][-1], closed=closed)
    if case == "rigid_loop":
        tb = torch.as_tensor((task.metadata or {})["rigid_loop_target_rigid_vertices_body"], dtype=positioned_rollout["q"].dtype, device=positioned_rollout["q"].device)
    else:
        tb = torch.as_tensor((task.metadata or {})["open_shape_target_rigid_vertices_body"], dtype=positioned_rollout["q"].dtype, device=positioned_rollout["q"].device)
    poseT = task.pose_target
    c, s = torch.cos(poseT[2]), torch.sin(poseT[2])
    R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
    Xt = ((R @ tb.T).T + poseT[:2]).detach().cpu().numpy()
    if closed and np.linalg.norm(Xt[-1] - Xt[0]) > 1e-9:
        Xt = np.vstack([Xt, Xt[0]])
    ax[0,0].plot(Xt[:,0], Xt[:,1], "k--", label="positioned target")
    ax[0,0].plot(Xs[:,0], Xs[:,1], marker="o", alpha=.8, label="Phase I shape model")
    ax[0,0].plot(Xp[:,0], Xp[:,1], marker="o", alpha=.8, label="Phase II positioned model")
    ax[0,0].axis("equal"); ax[0,0].grid(True, alpha=.25); ax[0,0].legend(fontsize=8)
    ax[0,0].set_title("Final world configuration")

    ax[0,1].semilogy(ts, ls.detach().cpu().numpy()+1e-12, label="Phase I local")
    ax[0,1].semilogy(tp, lp.detach().cpu().numpy()+1e-12, label="Phase II local")
    ax[0,1].semilogy(ts, js.detach().cpu().numpy()+1e-12, "--", label="Phase I joint")
    ax[0,1].semilogy(tp, jp.detach().cpu().numpy()+1e-12, "--", label="Phase II joint")
    ax[0,1].axvline(t_control, linestyle=":", color="k", linewidth=1, label="command ends")
    ax[0,1].set_title("Intrinsic shape preservation"); ax[0,1].set_xlabel("time"); ax[0,1].grid(True, alpha=.25); ax[0,1].legend(fontsize=8)

    ax[1,0].semilogy(ts, ps.detach().cpu().numpy()+1e-12, label="Phase I")
    ax[1,0].semilogy(tp, pp.detach().cpu().numpy()+1e-12, label="Phase II")
    ax[1,0].axvline(t_control, linestyle=":", color="k", linewidth=1)
    ax[1,0].set_title("Global SE(2) pose error"); ax[1,0].set_xlabel("time"); ax[1,0].grid(True, alpha=.25); ax[1,0].legend(fontsize=8)

    for label, r, tt in [("Phase I", shape_rollout, ts[:-1]), ("Phase II", positioned_rollout, tp[:-1])]:
        sf = torch.linalg.norm(r["spring_force_joint"], dim=-1).detach().cpu().numpy()
        V = r["potential_energy"][1:].detach().cpu().numpy() if r["potential_energy"].shape[0] == len(sf)+1 else r["potential_energy"][:len(sf)].detach().cpu().numpy()
        ax[1,1].plot(tt[:len(sf)], sf, label=f"{label} |spring force|")
        ax[1,1].plot(tt[:len(V)], V, "--", label=f"{label} shaping V")
    ax[1,1].axvline(t_control, linestyle=":", color="k", linewidth=1)
    ax[1,1].set_title("Does intrinsic shaping remain active?"); ax[1,1].set_xlabel("time"); ax[1,1].grid(True, alpha=.25); ax[1,1].legend(fontsize=8)

    fig.suptitle("Two-stage curriculum check: preserve local deformation while adding global pose")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
