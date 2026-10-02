from __future__ import annotations

from dataclasses import replace
import math
import numpy as np
import torch

from .chain import PlanarMultiLinkChain
from .tasks import LocomotionTask
from .rigid_loop import RigidLoopConfig, make_rigid_loop_tasks


def _rotation(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


def world_link_center(chain: PlanarMultiLinkChain, q: torch.Tensor, pose: torch.Tensor, link_index: int) -> torch.Tensor:
    kin = chain.kinematics(q)
    R = _rotation(pose[2])
    return pose[:2] + R @ kin["centers"][link_index]


def world_joint_positions(chain: PlanarMultiLinkChain, q: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    kin = chain.kinematics(q)
    R = _rotation(pose[2])
    return pose[:2][None, :] + (R @ kin["joint_positions"].T).T


def decorate_frame_target_task(
    base: LocomotionTask,
    chain: PlanarMultiLinkChain,
    Ksp: torch.Tensor,
    rng: np.random.Generator,
    *,
    displacement_fraction_range: tuple[float, float] = (0.12, 0.28),
    link_index: int | None = None,
    vertical_fraction_limit: float = 0.22,
) -> LocomotionTask:
    """Attach a random Cartesian target to one physical link-center frame.

    The intrinsic BVP shape target is retained.  The world-space task, however,
    is now the selected frame-center displacement, not a global base-pose target.
    """
    q0 = Ksp @ base.c0
    pose0 = base.initial_pose.clone()
    if link_index is None:
        candidates = list(range(1, max(chain.n_links - 1, 2)))
        link_index = int(rng.choice(candidates)) if candidates else chain.n_links // 2
        link_index = min(link_index, chain.n_links - 1)
    start = world_link_center(chain, q0, pose0, link_index)
    mag = float(rng.uniform(*displacement_fraction_range)) * chain.params.total_length
    # Bias toward horizontal/diagonal motion so the frame visibly travels in world coordinates.
    ang = float(rng.uniform(-0.65, 0.65))
    delta = mag * torch.tensor([math.cos(ang), math.sin(ang)], dtype=q0.dtype, device=q0.device)
    max_dy = vertical_fraction_limit * chain.params.total_length
    delta[1] = delta[1].clamp(-max_dy, max_dy)
    target = start + delta
    md = dict(base.metadata or {})
    md.update({
        "external_task": "frame_target",
        "force_link_index": int(link_index),
        "frame_start_world": start.detach().cpu().tolist(),
        "frame_target_world": target.detach().cpu().tolist(),
        "frame_target_delta": delta.detach().cpu().tolist(),
        "frame_target_distance": float(torch.linalg.norm(delta)),
    })
    return replace(base, metadata=md)


def decorate_gravity_ground_task(
    base: LocomotionTask,
    chain: PlanarMultiLinkChain,
    Ksp: torch.Tensor,
    rng: np.random.Generator,
    *,
    ground_y: float = 0.0,
    clearance_fraction_range: tuple[float, float] = (0.14, 0.30),
) -> LocomotionTask:
    """Lift the articulated chain above y=ground_y so gravity causes contact.

    The initial and target intrinsic shapes are unchanged.  The physical terminal
    objective becomes ground contact with little/no penetration while the pH
    template continues to execute the intrinsic deformation.
    """
    q0 = Ksp @ base.c0
    pose0 = base.initial_pose.clone()
    joints = world_joint_positions(chain, q0, pose0)
    clearance = float(rng.uniform(*clearance_fraction_range)) * chain.params.total_length
    desired_min_y = ground_y + clearance
    pose0[1] = pose0[1] + (desired_min_y - joints[:, 1].min())
    joints_new = world_joint_positions(chain, q0, pose0)
    md = dict(base.metadata or {})
    md.update({
        "external_task": "gravity_ground",
        "ground_y": float(ground_y),
        "initial_clearance": float(joints_new[:, 1].min() - ground_y),
        "initial_min_joint_y": float(joints_new[:, 1].min()),
        "ground_contact_target": True,
    })
    # pose_target is kept as a bookkeeping/teacher field but is not part of the
    # gravity-ground terminal objective in world_pose_info_loss.
    return replace(base, pose0=pose0, metadata=md)


def make_case_task_sets(
    base_tasks: list[LocomotionTask],
    chain: PlanarMultiLinkChain,
    Ksp: torch.Tensor,
    *,
    seed: int,
    frame_displacement_fraction_range: tuple[float, float] = (0.12, 0.28),
    ground_y: float = 0.0,
    clearance_fraction_range: tuple[float, float] = (0.14, 0.30),
    n_temporal_ctrl: int | None = None,
    rigid_loop_cfg: RigidLoopConfig | None = None,
) -> dict[str, list[LocomotionTask]]:
    """Create physical task interpretations from common intrinsic tasks.

    When ``n_temporal_ctrl`` is provided a fourth, independently sampled rigid-loop
    circle-to-crescent task set is added. The loop shares the same chain dimensions
    and pH template class, but imposes two closure constraints and fixed edge lengths.
    """
    rng = np.random.default_rng(seed)
    self_tasks = []
    frame_tasks = []
    gravity_tasks = []
    for base in base_tasks:
        md = dict(base.metadata or {})
        md["external_task"] = "self_propulsion"
        self_tasks.append(replace(base, metadata=md))
        frame_tasks.append(decorate_frame_target_task(
            base, chain, Ksp, rng,
            displacement_fraction_range=frame_displacement_fraction_range,
        ))
        gravity_tasks.append(decorate_gravity_ground_task(
            base, chain, Ksp, rng,
            ground_y=ground_y,
            clearance_fraction_range=clearance_fraction_range,
        ))
    out = {
        "self_propulsion": self_tasks,
        "frame_target": frame_tasks,
        "gravity_ground": gravity_tasks,
    }
    if n_temporal_ctrl is not None:
        out["rigid_loop"] = make_rigid_loop_tasks(
            len(base_tasks), chain, Ksp, n_temporal_ctrl, seed=seed + 17011, cfg=rigid_loop_cfg
        )
    return out
