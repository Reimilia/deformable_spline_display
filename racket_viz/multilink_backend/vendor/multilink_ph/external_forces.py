from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import math
import torch

from .chain import PlanarMultiLinkChain
from .tasks import LocomotionTask

MotionForceCase = Literal[
    "self_propulsion",
    "frame_target",
    "gravity_ground",
    "rigid_loop",
    "open_shape",
    # Legacy aliases retained so older scripts/checkpoints do not break.
    "target_attractor",
    "projected_gravity",
]


@dataclass(frozen=True)
class ExternalForceConfig:
    """External-force models for focused articulated-motion experiments.

    The physically distinct cases are:

    ``self_propulsion``
        No additional generalized force. World motion arises from intrinsic
        shape actuation and anisotropic environment interaction.

    ``frame_target``
        A Cartesian spring/damper is attached to *one selected link-center
        frame*. Its center is pulled from its initial world position toward a
        random target point. The force is mapped through that frame's
        articulated Jacobian, so it drives both floating pose and joint motion.

    ``gravity_ground``
        Gravity acts on every link mass and a unilateral penalty contact acts at
        all M+1 chain joints against a world ground line y=ground_y. This case
        therefore contains free fall, contact onset, normal reaction, tangential
        friction, and intrinsic pH actuation at the same time.

    ``rigid_loop``
        No added world force. The serial coordinates are constrained into a
        closed, fixed-edge polygon by a loop-closure reaction in the pH rollout.
        This is the circle-to-crescent rigid-perimeter task.

    ``open_shape``
        No added world force and no closure reaction. The same fixed-edge chain
        follows an open circle-arc to open crescent-arc target. This is a control
        experiment for separating target/alignment errors from closure-manifold
        effects.

    ``target_attractor`` and ``projected_gravity`` are old v7 aliases and map to
    the previous distributed-attractor / free projected-gravity behavior.
    """

    case: MotionForceCase = "self_propulsion"

    # One-frame target task.
    frame_stiffness: float = 1.4
    frame_damping: float = 0.30
    target_link_index: int | None = None

    # Legacy distributed attractor.
    attractor_stiffness: float = 0.65
    attractor_damping: float = 0.12

    # Gravity / ground.
    gravity_acceleration: float = 0.9
    gravity_angle: float = -math.pi / 2.0
    ground_y: float = 0.0
    ground_stiffness: float = 45.0
    ground_damping: float = 1.8
    ground_friction: float = 0.35
    ground_friction_velocity: float = 0.08

    @property
    def label(self) -> str:
        return {
            "self_propulsion": "intrinsic locomotion only",
            "frame_target": "single-frame target force",
            "gravity_ground": "gravity + unilateral ground",
            "rigid_loop": "closed rigid loop: circle to crescent",
            "open_shape": "open rigid chain: circle arc to crescent arc",
            "target_attractor": "legacy distributed target-attractor",
            "projected_gravity": "legacy free projected gravity",
        }[self.case]


def _rotation(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


def _point_velocity_jacobian(
    chain: PlanarMultiLinkChain,
    point_body: torch.Tensor,
    Jshape_point: torch.Tensor,
) -> torch.Tensor:
    """Body generalized point Jacobian H: u_point^b = H [xi,qdot]."""
    n = 3 + chain.n_joints
    H = torch.zeros(2, n, dtype=point_body.dtype, device=point_body.device)
    H[:, :2] = torch.eye(2, dtype=point_body.dtype, device=point_body.device)
    H[:, 2] = torch.stack([-point_body[1], point_body[0]])
    H[:, 3:] = Jshape_point
    return H


def _joint_position_shape_jacobians(chain: PlanarMultiLinkChain, kin: dict[str, torch.Tensor]) -> torch.Tensor:
    """Jacobian of all M+1 body-frame joint positions wrt M-1 joint angles."""
    tangents = kin["tangents"]
    joints = kin["joint_positions"]
    M = chain.n_links
    Jp = torch.zeros(M + 1, 2, chain.n_joints, dtype=joints.dtype, device=joints.device)
    Jt = chain._Jvec(tangents)
    # point k is the end of links 0,...,k-1. q_j rotates links >= j+1.
    for k in range(1, M + 1):
        for j in range(min(k - 1, chain.n_joints)):
            val = torch.zeros(2, dtype=joints.dtype, device=joints.device)
            for a in range(j + 1, k):
                val = val + chain.lengths[a] * Jt[a]
            Jp[k, :, j] = val
    return Jp


def _empty_result(chain: PlanarMultiLinkChain, dtype: torch.dtype, device: torch.device) -> dict[str, torch.Tensor]:
    n = 3 + chain.n_joints
    z3 = torch.zeros(3, dtype=dtype, device=device)
    zj = torch.zeros(chain.n_joints, dtype=dtype, device=device)
    zm_pos = torch.zeros(chain.n_links, 2, dtype=dtype, device=device)
    zm_force = torch.zeros(chain.n_links, 2, dtype=dtype, device=device)
    zg_pos = torch.zeros(chain.n_links + 1, 2, dtype=dtype, device=device)
    zg_force = torch.zeros(chain.n_links + 1, 2, dtype=dtype, device=device)
    zg_vel = torch.zeros(chain.n_links + 1, 2, dtype=dtype, device=device)
    return {
        "generalized_force": torch.zeros(n, dtype=dtype, device=device),
        "wrench_body": z3,
        "wrench_world": z3,
        "joint_force": zj,
        "link_forces_world": zm_force,
        "link_positions_world": zm_pos,
        "target_positions_world": torch.full_like(zm_pos, float("nan")),
        "power": torch.zeros((), dtype=dtype, device=device),
        "conservative_power": torch.zeros((), dtype=dtype, device=device),
        "damping_power": torch.zeros((), dtype=dtype, device=device),
        "potential": torch.zeros((), dtype=dtype, device=device),
        "tracked_link_index": torch.tensor(-1, dtype=torch.long, device=device),
        "tracked_frame_position_world": torch.full((2,), float("nan"), dtype=dtype, device=device),
        "tracked_frame_target_world": torch.full((2,), float("nan"), dtype=dtype, device=device),
        "tracked_frame_force_world": torch.zeros(2, dtype=dtype, device=device),
        "ground_contact_positions_world": zg_pos,
        "ground_contact_forces_world": zg_force,
        "ground_contact_velocities_world": zg_vel,
        "ground_gap": torch.full((chain.n_links + 1,), float("nan"), dtype=dtype, device=device),
        "ground_active": torch.zeros(chain.n_links + 1, dtype=dtype, device=device),
        "min_ground_gap": torch.tensor(float("nan"), dtype=dtype, device=device),
        "contact_count": torch.zeros((), dtype=dtype, device=device),
    }


def _metadata_tensor(task: LocomotionTask, key: str, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
    if task.metadata is None or key not in task.metadata:
        return None
    return torch.as_tensor(task.metadata[key], dtype=dtype, device=device)


def external_generalized_force(
    cfg: ExternalForceConfig | None,
    chain: PlanarMultiLinkChain,
    q: torch.Tensor,
    pose: torch.Tensor,
    nu: torch.Tensor,
    task: LocomotionTask,
    Ksp: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Map world forces into generalized forces without changing kinematics.

    ``nu=[v_x^b,v_y^b,omega,qdot]``. Every applied world force is pulled back by
    a point/frame Jacobian. No external case directly overwrites ``pose``.
    """
    if cfg is None:
        cfg = ExternalForceConfig()
    dtype, device = q.dtype, q.device
    out = _empty_result(chain, dtype, device)
    if cfg.case in {"self_propulsion", "rigid_loop", "open_shape"}:
        return out

    kin = chain.kinematics(q)
    centers = kin["centers"]
    Jshape = kin["Jshape"]
    R = _rotation(pose[2])
    Rt = R.transpose(0, 1)
    Q = torch.zeros(3 + chain.n_joints, dtype=dtype, device=device)
    conservative_power = torch.zeros((), dtype=dtype, device=device)
    damping_power = torch.zeros((), dtype=dtype, device=device)
    potential = torch.zeros((), dtype=dtype, device=device)

    # Cache link-center world positions/velocities.
    link_vel_world: list[torch.Tensor] = []
    for i in range(chain.n_links):
        H = _point_velocity_jacobian(chain, centers[i], Jshape[i])
        u_body = H @ nu
        out["link_positions_world"][i] = pose[:2] + R @ centers[i]
        link_vel_world.append(R @ u_body)

    # ------------------------------------------------------------------
    # New physical case 1: force applied to one selected link-center frame.
    # ------------------------------------------------------------------
    if cfg.case == "frame_target":
        idx_meta = None if task.metadata is None else task.metadata.get("force_link_index")
        idx = int(cfg.target_link_index if cfg.target_link_index is not None else (idx_meta if idx_meta is not None else chain.n_links // 2))
        idx = max(0, min(chain.n_links - 1, idx))
        target = _metadata_tensor(task, "frame_target_world", dtype=dtype, device=device)
        if target is None:
            raise ValueError("frame_target case requires task.metadata['frame_target_world']")
        H = _point_velocity_jacobian(chain, centers[idx], Jshape[idx])
        pos = out["link_positions_world"][idx]
        vel = link_vel_world[idx]
        err = target - pos
        f_cons = cfg.frame_stiffness * err
        f_damp = -cfg.frame_damping * vel
        f_world = f_cons + f_damp
        f_body = Rt @ f_world
        Q = Q + H.transpose(0, 1) @ f_body
        out["link_forces_world"][idx] = f_world
        out["target_positions_world"][idx] = target
        conservative_power = torch.dot(f_cons, vel)
        damping_power = torch.dot(f_damp, vel)
        potential = 0.5 * cfg.frame_stiffness * torch.dot(err, err)
        out["tracked_link_index"] = torch.tensor(idx, dtype=torch.long, device=device)
        out["tracked_frame_position_world"] = pos
        out["tracked_frame_target_world"] = target
        out["tracked_frame_force_world"] = f_world

    # ------------------------------------------------------------------
    # New physical case 2: gravity in world -y + unilateral ground y=0.
    # ------------------------------------------------------------------
    elif cfg.case == "gravity_ground":
        g_world = cfg.gravity_acceleration * torch.tensor(
            [math.cos(cfg.gravity_angle), math.sin(cfg.gravity_angle)], dtype=dtype, device=device
        )
        # Gravity acts at link centers.
        for i in range(chain.n_links):
            H = _point_velocity_jacobian(chain, centers[i], Jshape[i])
            pos = out["link_positions_world"][i]
            vel = link_vel_world[i]
            f_world = chain.masses[i] * g_world
            Q = Q + H.transpose(0, 1) @ (Rt @ f_world)
            out["link_forces_world"][i] = out["link_forces_world"][i] + f_world
            conservative_power = conservative_power + torch.dot(f_world, vel)
            potential = potential - chain.masses[i] * torch.dot(g_world, pos)

        # Ground contact acts at all M+1 articulated joint points.
        Jpoints = _joint_position_shape_jacobians(chain, kin)
        joints_body = kin["joint_positions"]
        ground_pos = []
        ground_force = []
        ground_vel = []
        gaps = []
        active = []
        for k in range(chain.n_links + 1):
            H = _point_velocity_jacobian(chain, joints_body[k], Jpoints[k])
            vel_w = R @ (H @ nu)
            pos_w = pose[:2] + R @ joints_body[k]
            gap = pos_w[1] - cfg.ground_y
            penetration = torch.relu(-gap)
            is_active = (penetration > 0).to(dtype)
            # Upward spring + damping only opposes downward velocity while in contact.
            fn_spring = cfg.ground_stiffness * penetration
            fn_damp = cfg.ground_damping * torch.relu(-vel_w[1]) * is_active
            fn = fn_spring + fn_damp
            # Smooth Coulomb-like tangential friction, bounded by mu*Fn.
            ft = -cfg.ground_friction * fn * torch.tanh(vel_w[0] / max(cfg.ground_friction_velocity, 1e-6))
            f_ground = torch.stack([ft, fn])
            Q = Q + H.transpose(0, 1) @ (Rt @ f_ground)
            # Spring contact is conservative; normal damping + friction dissipate.
            potential = potential + 0.5 * cfg.ground_stiffness * penetration**2
            conservative_power = conservative_power + fn_spring * vel_w[1]
            damping_power = damping_power + fn_damp * vel_w[1] + ft * vel_w[0]
            ground_pos.append(pos_w)
            ground_force.append(f_ground)
            ground_vel.append(vel_w)
            gaps.append(gap)
            active.append(is_active)
        out["ground_contact_positions_world"] = torch.stack(ground_pos)
        out["ground_contact_forces_world"] = torch.stack(ground_force)
        out["ground_contact_velocities_world"] = torch.stack(ground_vel)
        out["ground_gap"] = torch.stack(gaps)
        out["ground_active"] = torch.stack(active)
        out["min_ground_gap"] = out["ground_gap"].min()
        out["contact_count"] = out["ground_active"].sum()

    # --------------------------------------------------------
    # Legacy v7 cases retained for backward compatibility.
    # --------------------------------------------------------
    elif cfg.case == "target_attractor":
        q_target = Ksp @ task.c_target if task.kind == "bvp" else Ksp @ task.c0
        target_centers = chain.kinematics(q_target)["centers"]
        Rtar = _rotation(task.pose_target[2])
        for i in range(chain.n_links):
            H = _point_velocity_jacobian(chain, centers[i], Jshape[i])
            pos = out["link_positions_world"][i]
            vel = link_vel_world[i]
            target = task.pose_target[:2] + Rtar @ target_centers[i]
            out["target_positions_world"][i] = target
            err = target - pos
            k = cfg.attractor_stiffness / chain.n_links
            d = cfg.attractor_damping / chain.n_links
            f_cons = k * err
            f_damp = -d * vel
            f_world = f_cons + f_damp
            Q = Q + H.transpose(0, 1) @ (Rt @ f_world)
            out["link_forces_world"][i] = f_world
            potential = potential + 0.5 * k * torch.dot(err, err)
            conservative_power = conservative_power + torch.dot(f_cons, vel)
            damping_power = damping_power + torch.dot(f_damp, vel)

    elif cfg.case == "projected_gravity":
        g_world = cfg.gravity_acceleration * torch.tensor(
            [math.cos(cfg.gravity_angle), math.sin(cfg.gravity_angle)], dtype=dtype, device=device
        )
        for i in range(chain.n_links):
            H = _point_velocity_jacobian(chain, centers[i], Jshape[i])
            pos = out["link_positions_world"][i]
            vel = link_vel_world[i]
            f_world = chain.masses[i] * g_world
            Q = Q + H.transpose(0, 1) @ (Rt @ f_world)
            out["link_forces_world"][i] = f_world
            potential = potential - chain.masses[i] * torch.dot(g_world, pos)
            conservative_power = conservative_power + torch.dot(f_world, vel)
    else:
        raise ValueError(f"unknown external force case: {cfg.case}")

    force_world = R @ Q[:2]
    wrench_world = torch.stack([force_world[0], force_world[1], Q[2]])
    out["generalized_force"] = Q
    out["wrench_body"] = Q[:3]
    out["wrench_world"] = wrench_world
    out["joint_force"] = Q[3:]
    out["power"] = conservative_power + damping_power
    out["conservative_power"] = conservative_power
    out["damping_power"] = damping_power
    out["potential"] = potential
    return out
