from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np
import torch

from .ph_model import TaskConditionedPHTemplate
from .pinocchio_backend import FrameFrictionSpec, PinocchioRigidBodyBackend, RigidWorldContactSpec
from .pinocchio_info_geometry import pinocchio_configuration_fim, pinocchio_empirical_souriau_fisher


@dataclass
class PinocchioPHRolloutConfig:
    n_steps: int = 50
    dt: float = 0.02
    joint_viscous_damping: float = 0.08
    project_velocity_on_contact_switch: bool = True
    rigid_contact_solver: str = "planar_kkt"


ContactSchedule = Callable[[int, int], Sequence[RigidWorldContactSpec]]


def alternating_endpoint_contact_schedule(frame_a: str, frame_b: str) -> ContactSchedule:
    """Return a two-mode concertina-like rigid-contact schedule.

    The first endpoint is anchored during the first half of a rollout and the
    second endpoint during the second half.  Each new anchor is attached at its
    *current* world position, so the constraint is a fixed active-mode model,
    not a teleportation of the robot.
    """

    def schedule(k: int, n_steps: int) -> Sequence[RigidWorldContactSpec]:
        frame = frame_a if k < n_steps // 2 else frame_b
        return (RigidWorldContactSpec(frame, "3D"),)

    return schedule




def fixed_frame_contact_schedule(frame: str) -> ContactSchedule:
    """Keep one named frame rigidly anchored for the entire rollout."""

    def schedule(k: int, n_steps: int) -> Sequence[RigidWorldContactSpec]:
        return (RigidWorldContactSpec(frame, "3D"),)

    return schedule


def multiphase_endpoint_contact_schedule(
    frame_a: str,
    frame_b: str,
    n_phases: int = 2,
    start_with_a: bool = True,
) -> ContactSchedule:
    """Alternate endpoint anchors over ``n_phases`` equal temporal phases.

    ``n_phases=2`` recovers the original half-cycle concertina schedule.  Larger
    values are useful in the full-scale contact ablation because they test
    whether the learned pH template depends on a particular hand-coded switching
    frequency.
    """
    if n_phases < 1:
        raise ValueError("n_phases must be >= 1")

    def schedule(k: int, n_steps: int) -> Sequence[RigidWorldContactSpec]:
        phase = min(n_phases - 1, int((k * n_phases) / max(n_steps, 1)))
        use_a = (phase % 2 == 0) if start_with_a else (phase % 2 == 1)
        frame = frame_a if use_a else frame_b
        return (RigidWorldContactSpec(frame, "3D"),)

    return schedule

def _finite_difference_reference_velocity(refs: list[np.ndarray], qref: np.ndarray, dt: float) -> np.ndarray:
    if not refs:
        return np.zeros_like(qref)
    return (qref - refs[-1]) / dt


def rollout_pinocchio_ph_single(
    model: TaskConditionedPHTemplate,
    feature_single: dict[str, torch.Tensor],
    backend: PinocchioRigidBodyBackend,
    Ksp: torch.Tensor,
    cfg: PinocchioPHRolloutConfig,
    friction_specs: Sequence[FrameFrictionSpec] = (),
    rigid_contact: bool = False,
    contact_schedule: ContactSchedule | None = None,
    fim_frame_names: Sequence[str] = (),
    compute_information_metrics: bool = True,
) -> dict[str, np.ndarray]:
    """Evaluate the same learned Hamiltonian shaper on Pinocchio mechanics.

    This is an evaluation/transfer bridge.  Pinocchio calls are NumPy/C++ and
    therefore intentionally sit outside PyTorch autograd.  The backend exposes
    ABA/RNEA/contact derivatives separately for a future custom backward.

    Diagnostics make the pH/contact separation explicit:

    * ``actuator_power`` is generalized joint torque power ``u^T v_j``;
    * ``reference_power`` is the explicit time-dependent potential supply
      ``partial_t V_theta = -K(q-q_ref)^T qref_dot``;
    * ``joint_damping_power`` is non-positive;
    * ``contact_power`` is non-positive for smooth anisotropic friction and is
      taken as zero for ideal rigid constraints;
    * ``energy_balance_residual`` checks the discrete version of
      ``Delta H - integral(P_ref + P_contact + P_damping) dt``.
    """
    if feature_single["c0"].shape[0] != 1:
        raise ValueError("rollout_pinocchio_ph_single expects batch size 1")
    if rigid_contact and contact_schedule is None:
        raise ValueError("rigid_contact=True requires an explicit contact_schedule")

    dtype = Ksp.dtype
    device = Ksp.device
    with torch.no_grad():
        qj0_t = torch.einsum("jm,bm->bj", Ksp, feature_single["c0"])[0]
    qj = qj0_t.cpu().numpy().astype(float)
    pose0_t = feature_single.get("pose0")
    pose0 = np.zeros(3) if pose0_t is None else pose0_t[0].detach().cpu().numpy().astype(float)
    q = backend.planar_configuration(pose0, qj)
    v = np.zeros(backend.nv, dtype=float)

    qs = [q.copy()]
    vs = [v.copy()]
    refs: list[np.ndarray] = []
    torques = []
    actuator_power = []
    actuator_abs_power = []
    reference_power = []
    joint_damping_power = []
    contact_power = []
    max_lateral_slip_step = []
    contact_velocity_norm = []
    fim_force_joint = []
    fim_eigs = []
    sf_eigs = []
    momentum_map = []
    lambdas = []
    kinetic_energy = [backend.kinetic_energy(q, v)]
    spring_energy = []
    hamiltonian = []
    active_contacts: list[str] = []
    previous_contact_signature: tuple[tuple[str, str], ...] = ()

    # Initial potential uses the initial rest shape.
    with torch.no_grad():
        cref0 = model.reference(feature_single, torch.tensor([0.0], dtype=dtype, device=device))
        qref0 = torch.einsum("jm,bm->bj", Ksp, cref0)[0].cpu().numpy().astype(float)
        K_np0 = model.stiffness.cpu().numpy().astype(float)
    V0 = 0.5 * float(np.dot(K_np0 * (qj - qref0), qj - qref0))
    spring_energy.append(V0)
    hamiltonian.append(kinetic_energy[-1] + V0)

    for k in range(cfg.n_steps):
        tau_phase = torch.tensor([k / cfg.n_steps], dtype=dtype, device=device)
        with torch.no_grad():
            cref = model.reference(feature_single, tau_phase)
            qref = torch.einsum("jm,bm->bj", Ksp, cref)[0]
            stiffness = model.stiffness
            extra_damping = model.extra_damping
        qref_np = qref.cpu().numpy().astype(float)
        K_np = stiffness.cpu().numpy().astype(float)
        D_np = (cfg.joint_viscous_damping + extra_damping.cpu().numpy()).astype(float)
        qref_dot = _finite_difference_reference_velocity(refs, qref_np, cfg.dt)

        rho_f = float(getattr(model, "fim_gain", torch.tensor(0.0)).detach().cpu()) if hasattr(model, "fim_gain") else 0.0
        FIM = pinocchio_configuration_fim(backend, q, fim_frame_names) if (fim_frame_names and rho_f > 0.0) else np.zeros((backend.n_actuated, backend.n_actuated))

        # Included URDF has scalar internal joints starting after the planar root.
        qj_current = q[backend.base_nq :]
        vj = v[backend.base_nv :]
        spring_effort = K_np * (qj_current - qref_np)
        uj_spring = -spring_effort
        uj_damping = -D_np * vj
        uj_fim = -rho_f * (FIM @ vj)
        uj = uj_spring + uj_damping + uj_fim
        tau = backend.actuation_vector(uj)

        # The active rigid mode is chosen outside the learned pH model.  When a
        # mode changes we project velocity in the kinetic metric before the
        # acceleration-level constrained solve, avoiding an inconsistent Jv.
        if rigid_contact:
            assert contact_schedule is not None
            specs = tuple(contact_schedule(k, cfg.n_steps))
            sig = tuple((s.frame_name, s.contact_type.upper()) for s in specs)
            if sig != previous_contact_signature:
                if cfg.rigid_contact_solver == "planar_kkt":
                    backend.activate_planar_world_contacts(q, specs)
                    if cfg.project_velocity_on_contact_switch:
                        v = backend.project_velocity_to_planar_contacts(q, v, specs)
                else:
                    backend.activate_rigid_world_contacts(q, specs)
                    if cfg.project_velocity_on_contact_switch:
                        v = backend.project_velocity_to_rigid_contacts(q, v, specs)
                previous_contact_signature = sig
            active_contacts.append("+".join(s.frame_name for s in specs))
        else:
            active_contacts.append("")

        # Diagnostics are evaluated at the beginning of the step.
        kin_e = backend.kinetic_energy(q, v)
        V = 0.5 * float(np.dot(K_np * (qj_current - qref_np), qj_current - qref_np))
        p_act = float(np.dot(uj, vj))
        p_ref = -float(np.dot(spring_effort, qref_dot))
        p_damp = -float(np.dot(D_np * vj, vj)) + float(np.dot(uj_fim, vj))

        friction_diag = backend.anisotropic_frame_friction(q, v, friction_specs) if friction_specs else None
        p_contact = float(friction_diag["contact_power"]) if friction_diag is not None else 0.0
        lat_slip = float(friction_diag["max_lateral_speed"]) if friction_diag is not None else 0.0
        if rigid_contact and cfg.rigid_contact_solver == "planar_kkt":
            Jactive = backend.planar_rigid_contact_jacobian(q)
            rigid_vel = float(np.linalg.norm(Jactive @ v))
        else:
            rigid_vel = backend.active_rigid_contact_velocity_norm(q, v) if rigid_contact else 0.0

        if compute_information_metrics:
            momentum_map.append(backend.planar_centroidal_momentum(q, v))
            if fim_frame_names:
                fim_eigs.append(np.linalg.eigvalsh(FIM if rho_f > 0.0 else pinocchio_configuration_fim(backend, q, fim_frame_names)))
            SF, _ = pinocchio_empirical_souriau_fisher(backend, q, v)
            sf_eigs.append(np.linalg.eigvalsh(SF))

        step = backend.step_semi_implicit(
            q,
            v,
            tau,
            cfg.dt,
            friction_specs=friction_specs,
            rigid_contact=rigid_contact,
            rigid_contact_solver=cfg.rigid_contact_solver,
        )
        q = np.asarray(step["q"], dtype=float)
        v = np.asarray(step["v"], dtype=float)
        qs.append(q.copy())
        vs.append(v.copy())
        refs.append(qref_np)
        torques.append(uj.copy())
        actuator_power.append(p_act)
        actuator_abs_power.append(abs(p_act))
        reference_power.append(p_ref)
        joint_damping_power.append(p_damp)
        if rigid_contact:
            p_contact = float(step.get("contact_power", p_contact))
        contact_power.append(p_contact)
        max_lateral_slip_step.append(lat_slip)
        contact_velocity_norm.append(rigid_vel)
        fim_force_joint.append(uj_fim.copy())
        if "lambda" in step:
            lambdas.append(np.asarray(step["lambda"], dtype=float).copy())

        qj_next = q[backend.base_nq :]
        ke_next = backend.kinetic_energy(q, v)
        V_next = 0.5 * float(np.dot(K_np * (qj_next - qref_np), qj_next - qref_np))
        kinetic_energy.append(ke_next)
        spring_energy.append(V_next)
        hamiltonian.append(ke_next + V_next)

    # Discrete energy accounting uses left-endpoint powers.  It is a diagnostic,
    # not a claim that the semi-implicit step is an exact discrete-pH method.
    H = np.asarray(hamiltonian, dtype=float)
    p_ref_a = np.asarray(reference_power, dtype=float)
    p_contact_a = np.asarray(contact_power, dtype=float)
    p_damp_a = np.asarray(joint_damping_power, dtype=float)
    predicted_delta_h = cfg.dt * float(np.sum(p_ref_a + p_contact_a + p_damp_a))
    energy_balance_residual = float(H[-1] - H[0] - predicted_delta_h)

    out = {
        "q_pin": np.asarray(qs),
        "v_pin": np.asarray(vs),
        "q_ref": np.asarray(refs),
        "joint_torque": np.asarray(torques),
        "actuator_power": np.asarray(actuator_power),
        "actuator_abs_power": np.asarray(actuator_abs_power),
        "reference_power": p_ref_a,
        "joint_damping_power": p_damp_a,
        "contact_power": p_contact_a,
        "max_lateral_slip_step": np.asarray(max_lateral_slip_step),
        "contact_velocity_norm": np.asarray(contact_velocity_norm),
        "fim_force_joint": np.asarray(fim_force_joint),
        "fim_eigs": np.asarray(fim_eigs),
        "sf_eigs": np.asarray(sf_eigs),
        "momentum_map": np.asarray(momentum_map),
        "kinetic_energy": np.asarray(kinetic_energy),
        "spring_energy": np.asarray(spring_energy),
        "hamiltonian": H,
        "active_contacts": np.asarray(active_contacts, dtype=object),
        "energy_balance_residual": np.asarray(energy_balance_residual),
        "actuator_energy_abs": np.asarray(cfg.dt * float(np.sum(np.abs(actuator_power)))),
        "actuator_work_net": np.asarray(cfg.dt * float(np.sum(actuator_power))),
        "friction_energy": np.asarray(-cfg.dt * float(np.sum(np.minimum(p_contact_a, 0.0)))),
        "max_lateral_slip": np.asarray(float(np.max(max_lateral_slip_step)) if max_lateral_slip_step else 0.0),
        "max_rigid_contact_velocity": np.asarray(float(np.max(contact_velocity_norm)) if contact_velocity_norm else 0.0),
    }
    if lambdas:
        # Contact dimension can change if the schedule changes, but the bundled
        # endpoint schedule keeps the same 3D size.  Fall back to object dtype if needed.
        try:
            out["lambda"] = np.asarray(lambdas, dtype=float)
        except ValueError:
            out["lambda"] = np.asarray(lambdas, dtype=object)
    return out
