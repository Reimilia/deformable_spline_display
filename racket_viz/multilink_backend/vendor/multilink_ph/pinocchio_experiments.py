from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import math
import numpy as np
import torch

from .bspline import make_temporal_basis, spatial_curvature_to_joint_matrix
from .chain import ChainParams, PlanarMultiLinkChain
from .geometry import se2_endpoint_error
from .ph_model import TaskConditionedPHTemplate, rollout_ph
from .pinocchio_backend import (
    FrameFrictionSpec,
    PinocchioRigidBodyBackend,
    PinocchioUnavailableError,
)
from .pinocchio_ph import (
    PinocchioPHRolloutConfig,
    alternating_endpoint_contact_schedule,
    fixed_frame_contact_schedule,
    multiphase_endpoint_contact_schedule,
    rollout_pinocchio_ph_single,
)
from .tasks import LocomotionTask, generate_tasks, stack_task_features
from .urdf_chain import write_planar_chain_urdf

BackendCase = Literal["A", "B", "C", "D"]

CASE_LABELS = {
    "A": "Synthetic pH: hand-built M,A,B,C",
    "B": "Pinocchio ABA: articulated mechanics, no contact",
    "C": "Pinocchio ABA + anisotropic frame friction",
    "D": "Pinocchio CRBA/Jacobian planar KKT + alternating rigid anchors",
}


@dataclass
class LoadedExperiment:
    model: TaskConditionedPHTemplate
    chain: PlanarMultiLinkChain
    Ksp: torch.Tensor
    n_steps: int
    dt: float
    n_temporal_ctrl: int
    checkpoint_args: dict


def _wrap_angle(x: float) -> float:
    return math.atan2(math.sin(x), math.cos(x))


def _pose_error_np(pose: np.ndarray, target: np.ndarray) -> float:
    d = np.asarray(pose, dtype=float) - np.asarray(target, dtype=float)
    d[2] = _wrap_angle(float(d[2]))
    return float(np.linalg.norm(d))


def load_checkpoint_experiment(checkpoint: str | Path, dtype: torch.dtype = torch.float64) -> LoadedExperiment:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = dict(ckpt.get("args", {}))
    cp = dict(ckpt.get("chain_params", {}))
    params = ChainParams(**cp) if cp else ChainParams(n_links=int(args.get("n_links", 5)))
    chain = PlanarMultiLinkChain(params, dtype=dtype)
    if "Ksp" in ckpt:
        Ksp = ckpt["Ksp"].to(dtype=dtype)
    else:
        Ksp = torch.as_tensor(
            spatial_curvature_to_joint_matrix(params.n_links, int(args.get("n_modes", 4)), degree=3),
            dtype=dtype,
        )
    n_modes = int(Ksp.shape[1])
    model_cfg = dict(ckpt.get("model_config", {}))
    hidden = int(model_cfg.get("hidden", 64 if bool(args.get("smoke", False)) else args.get("hidden", 96)))
    model = TaskConditionedPHTemplate(
        n_modes=n_modes,
        n_joints=chain.n_joints,
        hidden=hidden,
        ref_scale=float(model_cfg.get("ref_scale", 2.2)),
        stiffness_init=float(model_cfg.get("stiffness_init", params.joint_stiffness)),
        extra_damping_init=float(model_cfg.get("extra_damping_init", 0.02)),
        dtype=dtype,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    n_steps = int(args.get("n_steps", 12))
    return LoadedExperiment(
        model=model,
        chain=chain,
        Ksp=Ksp,
        n_steps=n_steps,
        dt=1.0 / n_steps,
        n_temporal_ctrl=int(args.get("n_temporal_ctrl", 6)),
        checkpoint_args=args,
    )


def make_transfer_tasks(exp: LoadedExperiment, per_type: int = 1, seed: int = 64, amplitude: float | None = None) -> list[LocomotionTask]:
    amp = float(amplitude if amplitude is not None else exp.checkpoint_args.get("amplitude", 1.6))
    basis_open = make_temporal_basis(exp.n_steps, exp.n_temporal_ctrl, periodic=False, dtype=exp.Ksp.dtype)
    basis_periodic = make_temporal_basis(exp.n_steps, exp.n_temporal_ctrl, periodic=True, dtype=exp.Ksp.dtype)
    tasks: list[LocomotionTask] = []
    for j, kind in enumerate(["bvp", "iso", "periodic"]):
        tasks += generate_tasks(
            kind,
            per_type,
            exp.chain,
            exp.Ksp,
            basis_open,
            basis_periodic,
            exp.dt,
            seed=seed + 100 * j,
            amplitude=amp,
        )
    return tasks


def _synthetic_case(exp: LoadedExperiment, task: LocomotionTask) -> dict:
    features = stack_task_features([task])
    with torch.no_grad():
        r = rollout_ph(exp.model, features, exp.chain, exp.Ksp, exp.n_steps, exp.dt)
    pose = r["pose"][0].cpu().numpy()
    joints = r["q"][0].cpu().numpy()
    qref = r["q_ref"][0].cpu().numpy()
    nu = r["nu"][0].cpu().numpy()

    # Split synthetic pH diagnostics using the same moving-potential convention
    # used by the Pinocchio bridge.
    K = exp.model.stiffness.detach().cpu().numpy()
    Dextra = exp.model.extra_damping.detach().cpu().numpy()
    p_act, p_ref, p_damp, p_contact = [], [], [], []
    H = []
    for k in range(exp.n_steps):
        qk = joints[k]
        qrefk = qref[k]
        qdot = nu[k, 3:]
        effort = K * (qk - qrefk)
        uj = -effort - Dextra * qdot
        qref_dot = np.zeros_like(qrefk) if k == 0 else (qref[k] - qref[k-1]) / exp.dt
        p_act.append(float(np.dot(uj, qdot)))
        p_ref.append(-float(np.dot(effort, qref_dot)))
        p_damp.append(-float(np.dot(Dextra * qdot, qdot)))
        # Environment block includes substrate and nominal joint damping.  For
        # this diagnostic it is grouped as the contact/dissipation port.
        ops = exp.chain.assemble(torch.as_tensor(qk, dtype=exp.Ksp.dtype))
        Denv = ops["D"].detach().cpu().numpy()
        p_contact.append(-float(nu[k] @ Denv @ nu[k]))
        M = ops["M"].detach().cpu().numpy()
        ke = 0.5 * float(nu[k] @ M @ nu[k])
        V = 0.5 * float(np.dot(K * (qk - qrefk), qk - qrefk))
        H.append(ke + V)
    # Terminal H using last q and last available velocity/moving reference.
    qlast = joints[-1]
    nulast = nu[-1]
    ops = exp.chain.assemble(torch.as_tensor(qlast, dtype=exp.Ksp.dtype))
    M = ops["M"].detach().cpu().numpy()
    H.append(0.5 * float(nulast @ M @ nulast) + 0.5 * float(np.dot(K * (qlast - qref[-1]), qlast - qref[-1])))
    H = np.asarray(H)
    p_ref = np.asarray(p_ref); p_damp = np.asarray(p_damp); p_contact = np.asarray(p_contact)
    energy_resid = float(H[-1] - H[0] - exp.dt * np.sum(p_ref + p_damp + p_contact))
    return {
        "case": "A",
        "pose": pose,
        "joints": joints,
        "q_ref": qref,
        "joint_torque": np.zeros((exp.n_steps, exp.chain.n_joints)),
        "actuator_power": np.asarray(p_act),
        "contact_power": p_contact,
        "hamiltonian": H,
        "active_contacts": np.asarray([""] * exp.n_steps, dtype=object),
        "actuator_energy_abs": float(exp.dt * np.sum(np.abs(p_act))),
        "friction_energy": float(-exp.dt * np.sum(np.minimum(p_contact, 0.0))),
        "max_lateral_slip": 0.0,
        "max_rigid_contact_velocity": 0.0,
        "energy_balance_residual": energy_resid,
    }


def _pinocchio_common_result(
    case: BackendCase,
    backend: PinocchioRigidBodyBackend,
    r: dict[str, np.ndarray],
) -> dict:
    pose = np.asarray([backend.planar_pose(q) for q in r["q_pin"]])
    joints = np.asarray([backend.joint_configuration(q) for q in r["q_pin"]])
    out = dict(r)
    out.update(case=case, pose=pose, joints=joints)
    return out


def run_transfer_case(
    case: BackendCase,
    exp: LoadedExperiment,
    task: LocomotionTask,
    backend: PinocchioRigidBodyBackend | None,
    friction_scale: float | None = None,
    rigid_schedule_mode: str = "head_tail_2",
) -> dict:
    if case == "A":
        return _synthetic_case(exp, task)
    if backend is None:
        raise PinocchioUnavailableError("Pinocchio backend is required for cases B/C/D")
    features = stack_task_features([task])
    cfg = PinocchioPHRolloutConfig(
        n_steps=exp.n_steps,
        dt=exp.dt,
        joint_viscous_damping=exp.chain.params.joint_damping,
    )
    friction_specs: tuple[FrameFrictionSpec, ...] = ()
    rigid = False
    schedule = None
    if case == "C":
        # Match the synthetic drag coefficients as closely as possible: each
        # contact frame represents one link center.
        w = float(friction_scale if friction_scale is not None else exp.chain.params.drag_scale / exp.chain.n_links)
        friction_specs = tuple(
            FrameFrictionSpec(
                f"contact_{i}",
                c_tangent=exp.chain.params.c_parallel,
                c_normal=exp.chain.params.c_perp,
                weight=w,
            )
            for i in range(exp.chain.n_links)
        )
    elif case == "D":
        rigid = True
        head = "contact_0"
        tail = f"contact_{exp.chain.n_links-1}"
        mode = rigid_schedule_mode.lower()
        if mode == "head":
            schedule = fixed_frame_contact_schedule(head)
        elif mode == "tail":
            schedule = fixed_frame_contact_schedule(tail)
        elif mode in {"head_tail", "head_tail_2"}:
            schedule = alternating_endpoint_contact_schedule(head, tail)
        elif mode == "tail_head_2":
            schedule = multiphase_endpoint_contact_schedule(head, tail, n_phases=2, start_with_a=False)
        elif mode == "head_tail_4":
            schedule = multiphase_endpoint_contact_schedule(head, tail, n_phases=4, start_with_a=True)
        elif mode == "tail_head_4":
            schedule = multiphase_endpoint_contact_schedule(head, tail, n_phases=4, start_with_a=False)
        else:
            raise ValueError(
                "unknown rigid_schedule_mode. Expected one of "
                "head, tail, head_tail_2, tail_head_2, head_tail_4, tail_head_4"
            )
    r = rollout_pinocchio_ph_single(
        exp.model,
        features,
        backend,
        exp.Ksp,
        cfg,
        friction_specs=friction_specs,
        rigid_contact=rigid,
        contact_schedule=schedule,
    )
    return _pinocchio_common_result(case, backend, r)


def metrics_for_result(task: LocomotionTask, result: dict, exp: LoadedExperiment) -> dict[str, float | str]:
    pose_target = task.pose_target.detach().cpu().numpy()
    pose_error = _pose_error_np(result["pose"][-1], pose_target)
    joints = np.asarray(result["joints"], dtype=float)
    if task.kind == "bvp":
        qtarget = (exp.Ksp @ task.c_target).detach().cpu().numpy()
        shape_error = float(np.linalg.norm(joints[-1] - qtarget))
    elif task.kind == "periodic":
        shape_error = float(np.linalg.norm(joints[-1] - joints[0]))
    else:
        shape_error = 0.0
    return {
        "task": task.kind,
        "case": str(result["case"]),
        "backend": CASE_LABELS[str(result["case"])],
        "pose_error": pose_error,
        "shape_or_cycle_error": shape_error,
        "actuator_energy": float(np.asarray(result.get("actuator_energy_abs", np.nan))),
        "friction_energy": float(np.asarray(result.get("friction_energy", np.nan))),
        "max_slip": float(np.asarray(result.get("max_lateral_slip", np.nan))),
        "max_rigid_contact_velocity": float(np.asarray(result.get("max_rigid_contact_velocity", np.nan))),
        "energy_balance_residual": float(np.asarray(result.get("energy_balance_residual", np.nan))),
        "terminal_speed": float(np.linalg.norm(result.get("v_pin", np.zeros((1, 1)))[-1])) if "v_pin" in result else float(np.linalg.norm(result.get("joints", [[0]])[-1] - result.get("joints", [[0]])[-2])) / exp.dt,
    }


def build_pinocchio_backend(exp: LoadedExperiment, output_dir: str | Path) -> tuple[PinocchioRigidBodyBackend, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    urdf = write_planar_chain_urdf(
        output_dir / "generated_chain.urdf",
        n_links=exp.chain.n_links,
        total_length=exp.chain.params.total_length,
        total_mass=exp.chain.params.total_mass,
        include_contact_frames=True,
    )
    backend = PinocchioRigidBodyBackend.from_urdf(urdf, root_joint="planar")
    return backend, urdf
