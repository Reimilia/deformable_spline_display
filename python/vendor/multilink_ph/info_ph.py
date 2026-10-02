from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from .chain import PlanarMultiLinkChain
from .geometry import se2_compose, se2_exp_increment, se2_coadjoint_term, se2_endpoint_error, se2_relative, se2_log
from .tasks import LocomotionTask, stack_task_features
from .ph_model import TaskConditionedPHTemplate
from .external_forces import ExternalForceConfig, external_generalized_force
from .rigid_loop import (
    loop_closure, project_loop_configuration, project_loop_velocity_mass_metric, loop_perimeter,
    loop_body_vertices, cyclic_local_shape_loss, rigid_link_measurement_loss,
    LoopClosureModel, closure_model_from_task, measurement_closure_terms
)
from .open_shape import open_body_vertices, open_local_shape_loss
from .info_geometry import (
    InformationGeometryConfig,
    configuration_fim,
    empirical_souriau_fisher_metric,
    souriau_momentum_map,
    generalized_momentum,
)




def _safe_eigvalsh_spd(A: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Stable spectrum for diagnostics, including deliberately unstable ablations."""
    A = torch.nan_to_num(A, nan=0.0, posinf=1e12, neginf=-1e12)
    A = 0.5 * (A + A.transpose(-1, -2))
    scale = A.detach().abs().max().clamp_min(1.0)
    B = A / scale
    eye = torch.eye(B.shape[-1], dtype=B.dtype, device=B.device)
    B = B + eps * eye
    try:
        return torch.linalg.eigvalsh(B) * scale
    except RuntimeError:
        # Diagnostic fallback; dynamics never use these eigenvalues.
        return torch.diagonal(B, dim1=-2, dim2=-1) * scale

class WorldPoseInformationPHTemplate(TaskConditionedPHTemplate):
    """pH template with a learnable FIM-shaped intrinsic damping gain."""

    def __init__(self, *args, fim_gain_init: float = 0.12, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.log_fim_gain = torch.nn.Parameter(
            torch.tensor(math.log(max(fim_gain_init, 1e-6)), dtype=self.log_stiffness.dtype)
        )

    @property
    def fim_gain(self) -> torch.Tensor:
        return torch.exp(self.log_fim_gain).clamp(1e-6, 10.0)


def stack_world_pose_features(
    task: LocomotionTask,
    *,
    include_pose_target: bool | None = None,
) -> dict[str, torch.Tensor]:
    """Build task features for the pH reference network.

    For ``rigid_loop`` and ``open_shape`` the curriculum deliberately removes
    pose information during Phase I, but Phase II must condition on the desired
    *relative* SE(2) displacement.  v13 always zeroed this slot for these two
    tasks, which made the positioned curriculum unable to represent different
    pose targets except indirectly through the shape endpoints.
    """
    features = stack_task_features([task])
    rel = se2_log(se2_relative(features["pose0"], features["pose_target"]))
    # Reuse the existing 3-D task-conditioning slot but feed the *physical task
    # coordinate* for external-force experiments. Separate models are trained
    # per force regime, so no extra case one-hot is required.
    md = task.metadata or {}
    ext = md.get("external_task", "self_propulsion")
    if ext == "frame_target" and "frame_target_delta" in md:
        d = torch.as_tensor(md["frame_target_delta"], dtype=rel.dtype, device=rel.device)
        rel = torch.stack([d[0], d[1], torch.zeros((), dtype=rel.dtype, device=rel.device)])[None, :]
    elif ext == "gravity_ground":
        clearance = float(md.get("initial_clearance", 0.0))
        rel = torch.tensor([[0.0, -clearance, 0.0]], dtype=rel.dtype, device=rel.device)
    elif ext in {"rigid_loop", "open_shape"}:
        # Phase I is quotient-space training, so pose is a nuisance variable.
        # Phase II explicitly receives the desired *relative* pose.  Using the
        # relative Lie-log coordinate preserves left-frame equivariance.
        if include_pose_target is None:
            include_pose_target = False
        if not include_pose_target:
            rel = torch.zeros_like(rel)
    features["pose_target_for_model"] = rel
    return features


@dataclass
class WorldPoseLossConfig:
    pose_weight: float = 500.0
    shape_weight: float = 100.0
    terminal_momentum_weight: float = 4.0
    reference_teacher_weight: float = 10.0
    phase_q_weight: float = 3.0
    phase_nu_weight: float = 0.8
    sf_momentum_weight: float = 0.25
    sf_terminal_momentum_weight: float = 0.25
    passivity_residual_weight: float = 0.1
    joint_limit_weight: float = 50.0
    # External-task terminal constraints.
    frame_target_weight: float = 600.0
    ground_touch_weight: float = 650.0
    ground_penetration_weight: float = 1500.0
    ground_terminal_speed_weight: float = 8.0
    # Rigid-loop target is decomposed into intrinsic geometry and optional SE(2) placement.
    # ``rigid_loop_target_weight`` is retained as the joint-coordinate auxiliary term
    # for backward compatibility; the geometric local-shape loss is primary.
    rigid_loop_target_weight: float = 80.0
    rigid_loop_local_shape_weight: float = 900.0
    rigid_loop_pose_weight: float = 0.0
    rigid_loop_closure_weight: float = 2500.0
    # Phase-II hold terms are evaluated after the moving reference has stopped.
    # They prevent a model from reaching the pose transiently while abandoning
    # the terminal intrinsic configuration during settling.
    rigid_loop_tail_joint_weight: float = 0.0
    rigid_loop_tail_pose_weight: float = 0.0
    # Open-chain closure-control experiment.
    open_shape_target_weight: float = 80.0
    open_shape_local_shape_weight: float = 900.0
    open_shape_pose_weight: float = 0.0
    open_shape_tail_joint_weight: float = 0.0
    open_shape_tail_pose_weight: float = 0.0


def _hamiltonian(chain: PlanarMultiLinkChain, q: torch.Tensor, pi: torch.Tensor, qref: torch.Tensor, stiffness: torch.Tensor, extra_potential: torch.Tensor | float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M = chain.assemble(q)["M"]
    nu = torch.linalg.solve(M, pi.unsqueeze(-1)).squeeze(-1)
    T = 0.5 * torch.dot(pi, nu)
    V_intrinsic = 0.5 * torch.sum(stiffness * (q - qref) ** 2)
    V_extra = torch.as_tensor(extra_potential, dtype=q.dtype, device=q.device)
    V = V_intrinsic + V_extra
    return T + V, T, V


def teacher_phase_data(
    chain: PlanarMultiLinkChain,
    teacher_traj: dict[str, torch.Tensor],
    info_cfg: InformationGeometryConfig | None = None,
    *,
    compute_sf: bool = True,
) -> dict[str, torch.Tensor]:
    """Lift a variational geometric trajectory into mechanical phase variables."""
    info_cfg = info_cfg or InformationGeometryConfig()
    q = teacher_traj["q"]
    qdot = teacher_traj["qdot"]
    xi = teacher_traj["xi"]
    if xi.shape[0] + 1 == q.shape[0]:
        xi_full = torch.cat([xi, xi[-1:]], dim=0)
    else:
        xi_full = xi[: q.shape[0]]
    nu = torch.cat([xi_full, qdot], dim=-1)
    pis = []
    sf = []
    sf_eigs = []
    fim = []
    fim_eigs = []
    Jmom = []
    for k in range(q.shape[0]):
        pi = generalized_momentum(chain, q[k], nu[k])
        F = configuration_fim(chain, q[k], info_cfg)
        if compute_sf:
            SF, _, _ = empirical_souriau_fisher_metric(chain, q[k], nu[k], info_cfg)
        else:
            SF = torch.eye(3, dtype=q.dtype, device=q.device)
        pis.append(pi)
        fim.append(F)
        fim_eigs.append(torch.linalg.eigvalsh(F))
        sf.append(SF)
        sf_eigs.append(torch.linalg.eigvalsh(SF))
        Jmom.append(souriau_momentum_map(pi))
    return {
        "q": q,
        "nu": nu,
        "pi": torch.stack(pis),
        "fim": torch.stack(fim),
        "fim_eigs": torch.stack(fim_eigs),
        "sf": torch.stack(sf),
        "sf_eigs": torch.stack(sf_eigs),
        "momentum_map": torch.stack(Jmom),
        "pose": teacher_traj["pose"],
        "c": teacher_traj["c"],
    }


def rollout_world_pose_info_ph(
    model: WorldPoseInformationPHTemplate,
    features: dict[str, torch.Tensor],
    chain: PlanarMultiLinkChain,
    Ksp: torch.Tensor,
    n_steps: int,
    dt: float,
    info_cfg: InformationGeometryConfig | None = None,
    compute_sf_diagnostics: bool = True,
    external_force_cfg: ExternalForceConfig | None = None,
    task: LocomotionTask | None = None,
    control_steps: int | None = None,
    use_fim_damping: bool = True,
    compute_terminal_sf: bool = True,
    max_state_norm: float = 1e6,
) -> dict[str, torch.Tensor]:
    """Single-task pH rollout with FIM damping and SF diagnostics.

    Global pose is never actuated directly.  The model produces only an
    intrinsic rest-shape potential.  The body SE(2) motion is generated through
    the articulated inertia and anisotropic environmental dissipation blocks.
    """
    info_cfg = info_cfg or InformationGeometryConfig()
    if features["c0"].shape[0] != 1:
        raise ValueError("focused world-pose rollout expects batch size 1")
    if external_force_cfg is not None and task is None:
        raise ValueError("task is required when external_force_cfg is provided")
    dtype, device = Ksp.dtype, Ksp.device
    control_steps = int(n_steps if control_steps is None else control_steps)
    if control_steps <= 0 or control_steps > n_steps:
        raise ValueError("control_steps must satisfy 0 < control_steps <= n_steps")
    motion_case = external_force_cfg.case if external_force_cfg is not None else str((task.metadata or {}).get("external_task", "self_propulsion") if task is not None else "self_propulsion")
    loop_model = closure_model_from_task(task) if (motion_case == "rigid_loop" and task is not None) else LoopClosureModel(mode="open")
    loop_mode = loop_model.normalized_mode()
    q = (Ksp @ features["c0"][0]).clone()
    if motion_case == "rigid_loop" and task is not None and task.metadata is not None:
        q = torch.as_tensor(task.metadata["rigid_loop_q_circle"], dtype=dtype, device=device).clone()
        if loop_mode == "hard":
            q = project_loop_configuration(chain, q, iterations=8, regularization=1e-8)
    elif motion_case == "open_shape" and task is not None and task.metadata is not None:
        q = torch.as_tensor(task.metadata["open_shape_q_initial"], dtype=dtype, device=device).clone()
    pose = features["pose0"][0].clone()
    n = 3 + chain.n_joints
    pi = torch.zeros(n, dtype=dtype, device=device)
    I = torch.eye(n, dtype=dtype, device=device)

    poses = [pose]
    qs = [q]
    pis = [pi]
    nus = []
    qrefs = []
    crefs = []
    fim_list = []
    sf_list = []
    fim_eigs = []
    sf_eigs = []
    momentum_maps = [souriau_momentum_map(pi)]
    spring_forces = []
    fim_forces = []
    environment_forces = []
    total_joint_forces = []
    rigid_environment_forces = []
    kinetic = []
    potential = []
    Hs = []
    p_ref_list = []
    p_diss_list = []
    p_fim_list = []
    p_env_list = []
    power_residual = []
    external_generalized = []
    external_wrench_body = []
    external_wrench_world = []
    external_joint_forces = []
    external_link_forces_world = []
    external_link_positions_world = []
    external_target_positions_world = []
    external_power = []
    external_conservative_power = []
    external_damping_power = []
    external_potential = []
    tracked_frame_positions = []
    tracked_frame_targets = []
    tracked_frame_forces = []
    tracked_link_indices = []
    ground_positions = []
    ground_forces = []
    ground_velocities = []
    ground_gaps = []
    ground_active = []
    min_ground_gaps = []
    contact_counts = []
    loop_closure_errors = []
    loop_perimeter_errors = []
    loop_constraint_forces = []
    loop_constraint_power = []
    loop_projection_energy_jump = []
    loop_closure_measurement_loss = []
    loop_seam_potential = []
    loop_seam_spring_force = []
    loop_seam_damping_force = []
    loop_closure_sigma = []
    loop_closure_stiffness = []
    loop_body_vertices_hist = [loop_body_vertices(chain, q, unique=False)] if motion_case == "rigid_loop" else []
    open_body_vertices_hist = [open_body_vertices(chain, q)] if motion_case == "open_shape" else []

    # Initial reference/energy.
    tau0 = torch.zeros(1, dtype=dtype, device=device)
    c_ref0 = model.reference(features, tau0)[0]
    q_ref0 = Ksp @ c_ref0
    zero_qdot0 = torch.zeros(chain.n_joints, dtype=dtype, device=device)
    seam0 = measurement_closure_terms(chain, q, zero_qdot0, loop_model) if motion_case == "rigid_loop" else None
    H0, T0, V0 = _hamiltonian(chain, q, pi, q_ref0, model.stiffness, 0.0 if seam0 is None else seam0["potential"])
    Hs.append(H0); kinetic.append(T0); potential.append(V0)

    diverged = False
    completed_steps = 0
    for k in range(n_steps):
        state_vec = torch.cat([pose.reshape(-1), q.reshape(-1), pi.reshape(-1)])
        if (not bool(torch.isfinite(state_vec).all())) or float(state_vec.detach().abs().max()) > max_state_norm:
            diverged = True
            break
        # The learned spline/potential is active over ``control_steps``.  A longer
        # evaluation rollout clamps tau=1 afterwards so the system can settle
        # under damping/contact without changing the terminal rest shape.
        tau_val = min(k / control_steps, 1.0)
        tau_next_val = min((k + 1) / control_steps, 1.0)
        tau = torch.tensor([tau_val], dtype=dtype, device=device)
        tau_next = torch.tensor([tau_next_val], dtype=dtype, device=device)
        c_ref = model.reference(features, tau)[0]
        c_ref_next = model.reference(features, tau_next)[0]
        q_ref = Ksp @ c_ref
        q_ref_next = Ksp @ c_ref_next
        qref_dot = (q_ref_next - q_ref) / dt

        ops = chain.assemble(q)
        M = ops["M"]
        Dbase = ops["D"]
        # The synthetic A/B/C drag is a substrate-contact surrogate.  In the
        # gravity-ground experiment the body starts *above* the floor, so that
        # drag must not act in free flight.  Keep only internal joint damping;
        # ground normal/friction forces are supplied by external_forces.py once
        # contact occurs.
        if motion_case == "gravity_ground":
            Dflight = torch.zeros_like(Dbase)
            Dflight[3:, 3:] = chain.params.joint_damping * torch.eye(chain.n_joints, dtype=dtype, device=device)
            Dbase = Dflight
        nu = torch.linalg.solve(M, pi.unsqueeze(-1)).squeeze(-1)
        FIM = configuration_fim(chain, q, info_cfg)
        Dfim = torch.zeros_like(Dbase)
        if use_fim_damping:
            Dfim[3:, 3:] = model.fim_gain * FIM
        Dextra = torch.zeros_like(Dbase)
        Dextra[3:, 3:] = torch.diag(model.extra_damping)
        seam = measurement_closure_terms(chain, q, nu[3:], loop_model) if motion_case == "rigid_loop" else None
        Dseam = torch.zeros_like(Dbase)
        seam_spring_joint = torch.zeros(chain.n_joints, dtype=dtype, device=device)
        if seam is not None and loop_mode == "measurement":
            Dseam[3:, 3:] = seam["damping_matrix_joint"]
            seam_spring_joint = seam["spring_force_joint"]
        Dtotal = Dbase + Dfim + Dextra + Dseam

        xi = nu[:3]
        mu = pi[:3]
        coad = se2_coadjoint_term(xi, mu)
        spring_joint = -model.stiffness * (q - q_ref)
        rhs = torch.cat([coad, spring_joint + seam_spring_joint], dim=-1)

        if external_force_cfg is not None:
            ext = external_generalized_force(external_force_cfg, chain, q, pose, nu, task, Ksp)
            rhs = rhs + ext["generalized_force"]
        else:
            z3 = torch.zeros(3, dtype=dtype, device=device)
            zj = torch.zeros(chain.n_joints, dtype=dtype, device=device)
            zm = torch.zeros(chain.n_links, 2, dtype=dtype, device=device)
            ext = {
                "generalized_force": torch.zeros(n, dtype=dtype, device=device),
                "wrench_body": z3, "wrench_world": z3, "joint_force": zj,
                "link_forces_world": zm, "link_positions_world": zm,
                "target_positions_world": torch.full_like(zm, float("nan")),
                "power": torch.zeros((), dtype=dtype, device=device),
                "conservative_power": torch.zeros((), dtype=dtype, device=device),
                "damping_power": torch.zeros((), dtype=dtype, device=device),
                "potential": torch.zeros((), dtype=dtype, device=device),
                "tracked_link_index": torch.tensor(-1, dtype=torch.long, device=device),
                "tracked_frame_position_world": torch.full((2,), float("nan"), dtype=dtype, device=device),
                "tracked_frame_target_world": torch.full((2,), float("nan"), dtype=dtype, device=device),
                "tracked_frame_force_world": torch.zeros(2, dtype=dtype, device=device),
                "ground_contact_positions_world": torch.zeros(chain.n_links + 1, 2, dtype=dtype, device=device),
                "ground_contact_forces_world": torch.zeros(chain.n_links + 1, 2, dtype=dtype, device=device),
                "ground_contact_velocities_world": torch.zeros(chain.n_links + 1, 2, dtype=dtype, device=device),
                "ground_gap": torch.full((chain.n_links + 1,), float("nan"), dtype=dtype, device=device),
                "ground_active": torch.zeros(chain.n_links + 1, dtype=dtype, device=device),
                "min_ground_gap": torch.tensor(float("nan"), dtype=dtype, device=device),
                "contact_count": torch.zeros((), dtype=dtype, device=device),
            }

        Minv = torch.linalg.solve(M, I)
        lhs = I + dt * (Dtotal @ Minv)
        pi_next = torch.linalg.solve(lhs, (pi + dt * rhs).unsqueeze(-1)).squeeze(-1)
        nu_free = torch.linalg.solve(M, pi_next.unsqueeze(-1)).squeeze(-1)
        loop_reaction = torch.zeros_like(pi_next)
        if motion_case == "rigid_loop" and loop_mode == "hard":
            nu_next, _ = project_loop_velocity_mass_metric(
                chain, q, M, nu_free, closure_position_gain=10.0, regularization=1e-8
            )
            pi_constrained = M @ nu_next
            loop_reaction = (pi_constrained - pi_next) / dt
            pi_next = pi_constrained
        else:
            nu_next = nu_free
        xi_next = nu_next[:3]
        qdot_next = nu_next[3:]

        # Forces evaluated with the implicit velocity used by the update.
        f_fim_joint = -(model.fim_gain * FIM) @ qdot_next if use_fim_damping else torch.zeros_like(qdot_next)
        f_env_gen = -(Dbase @ nu_next)
        f_extra_joint = -model.extra_damping * qdot_next
        f_seam_damp_joint = -(seam["damping_matrix_joint"] @ qdot_next) if (seam is not None and loop_mode == "measurement") else torch.zeros_like(qdot_next)
        total_joint = spring_joint + seam_spring_joint + f_seam_damp_joint + f_fim_joint + f_env_gen[3:] + f_extra_joint + ext["joint_force"] + loop_reaction[3:]

        # Supply rate from the explicitly moving potential, external-force port, and dissipation.
        p_ref = -torch.dot(model.stiffness * (q - q_ref), qref_dot)
        p_ext = torch.dot(ext["generalized_force"], nu_next)
        p_fim = torch.dot(qdot_next, f_fim_joint)  # <= 0
        p_env = torch.dot(nu_next, f_env_gen)      # <= 0
        p_extra = torch.dot(qdot_next, f_extra_joint)  # <= 0
        p_seam_damp = torch.dot(qdot_next, f_seam_damp_joint)  # <= 0
        p_diss = -(p_fim + p_env + p_extra + p_seam_damp)        # >= 0

        pose_next = se2_compose(pose, se2_exp_increment(xi_next, dt))
        q_trial = q + dt * qdot_next
        projection_jump = torch.zeros((), dtype=dtype, device=device)
        if motion_case == "rigid_loop" and loop_mode == "hard":
            # Measure the hidden energy change induced by the position-level
            # closure projection.  An ideal constraint reaction does no work, but
            # a separate Newton projection can still change the discrete energy.
            H_trial, _, _ = _hamiltonian(chain, q_trial, pi_next, q_ref_next, model.stiffness)
            q_next = project_loop_configuration(chain, q_trial, iterations=5, regularization=1e-8)
            # Re-identify momentum with the projected configuration's inertia.
            M_next = chain.assemble(q_next)["M"]
            pi_next = M_next @ nu_next
            H_proj, _, _ = _hamiltonian(chain, q_next, pi_next, q_ref_next, model.stiffness)
            projection_jump = H_proj - H_trial
        else:
            q_next = q_trial

        seam_next = measurement_closure_terms(chain, q_next, qdot_next, loop_model) if motion_case == "rigid_loop" else None
        H_next, T_next, V_next = _hamiltonian(
            chain, q_next, pi_next, q_ref_next, model.stiffness,
            0.0 if seam_next is None else seam_next["potential"],
        )
        dHdt = (H_next - Hs[-1]) / dt
        residual = dHdt - (p_ref + p_ext - p_diss)

        # Information metrics are monitors of the state actually traversed.
        fim_list.append(FIM)
        fim_eigs.append(_safe_eigvalsh_spd(FIM))
        if compute_sf_diagnostics:
            SF, _, _ = empirical_souriau_fisher_metric(chain, q, nu_next, info_cfg)
            sf_list.append(SF)
            sf_eigs.append(_safe_eigvalsh_spd(SF))
        momentum_maps.append(souriau_momentum_map(pi_next))
        spring_forces.append(spring_joint)
        fim_forces.append(f_fim_joint)
        environment_forces.append(f_env_gen[3:])
        rigid_environment_forces.append(f_env_gen[:3])
        total_joint_forces.append(total_joint)
        p_ref_list.append(p_ref)
        p_fim_list.append(p_fim)
        p_env_list.append(p_env + p_extra)
        p_diss_list.append(p_diss)
        power_residual.append(residual)
        external_generalized.append(ext["generalized_force"])
        external_wrench_body.append(ext["wrench_body"])
        external_wrench_world.append(ext["wrench_world"])
        external_joint_forces.append(ext["joint_force"])
        external_link_forces_world.append(ext["link_forces_world"])
        external_link_positions_world.append(ext["link_positions_world"])
        external_target_positions_world.append(ext["target_positions_world"])
        external_power.append(p_ext)
        external_conservative_power.append(ext["conservative_power"])
        external_damping_power.append(ext["damping_power"])
        external_potential.append(ext["potential"])
        tracked_frame_positions.append(ext["tracked_frame_position_world"])
        tracked_frame_targets.append(ext["tracked_frame_target_world"])
        tracked_frame_forces.append(ext["tracked_frame_force_world"])
        tracked_link_indices.append(ext["tracked_link_index"])
        ground_positions.append(ext["ground_contact_positions_world"])
        ground_forces.append(ext["ground_contact_forces_world"])
        ground_velocities.append(ext["ground_contact_velocities_world"])
        ground_gaps.append(ext["ground_gap"])
        ground_active.append(ext["ground_active"])
        min_ground_gaps.append(ext["min_ground_gap"])
        contact_counts.append(ext["contact_count"])
        if motion_case == "rigid_loop":
            cerr = torch.linalg.norm(loop_closure(chain, q_next))
            perr = torch.abs(loop_perimeter(chain, q_next) - float((task.metadata or {}).get("rigid_loop_perimeter", chain.params.total_length)))
            cpow = torch.dot(loop_reaction, nu_next)
            seam_diag = seam_next if seam_next is not None else measurement_closure_terms(chain, q_next, qdot_next, loop_model)
            seam_force_full = torch.zeros_like(pi_next)
            seam_damp_full = torch.zeros_like(pi_next)
            seam_force_full[3:] = seam_diag["spring_force_joint"]
            seam_damp_full[3:] = seam_diag["damping_force_joint"]
        else:
            cerr = torch.zeros((), dtype=dtype, device=device)
            perr = torch.zeros((), dtype=dtype, device=device)
            cpow = torch.zeros((), dtype=dtype, device=device)
            seam_diag = None
            seam_force_full = torch.zeros_like(pi_next)
            seam_damp_full = torch.zeros_like(pi_next)
        loop_closure_errors.append(cerr)
        loop_perimeter_errors.append(perr)
        loop_constraint_forces.append(loop_reaction)
        loop_constraint_power.append(cpow)
        loop_projection_energy_jump.append(projection_jump)
        loop_closure_measurement_loss.append(torch.zeros((), dtype=dtype, device=device) if seam_diag is None else seam_diag["measurement_loss"])
        loop_seam_potential.append(torch.zeros((), dtype=dtype, device=device) if seam_diag is None else seam_diag["potential"])
        loop_seam_spring_force.append(seam_force_full)
        loop_seam_damping_force.append(seam_damp_full)
        loop_closure_sigma.append(torch.zeros((), dtype=dtype, device=device) if seam_diag is None else seam_diag["sigma"])
        loop_closure_stiffness.append(torch.zeros((), dtype=dtype, device=device) if seam_diag is None else seam_diag["stiffness"])
        if motion_case == "rigid_loop":
            loop_body_vertices_hist.append(loop_body_vertices(chain, q_next, unique=False))
        elif motion_case == "open_shape":
            open_body_vertices_hist.append(open_body_vertices(chain, q_next))

        pose, q, pi = pose_next, q_next, pi_next
        poses.append(pose); qs.append(q); pis.append(pi); nus.append(nu_next)
        qrefs.append(q_ref); crefs.append(c_ref)
        Hs.append(H_next); kinetic.append(T_next); potential.append(V_next)
        completed_steps = k + 1

    # Terminal external-task geometry evaluated at the actual final state.
    if external_force_cfg is not None and task is not None:
        final_nu = nus[-1] if nus else torch.zeros(n, dtype=dtype, device=device)
        terminal_ext = external_generalized_force(external_force_cfg, chain, q, pose, final_nu, task, Ksp)
    else:
        terminal_ext = None

    H = torch.stack(Hs)
    pref = torch.stack(p_ref_list)
    pext = torch.stack(external_power) if external_power else torch.zeros_like(pref)
    pdiss = torch.stack(p_diss_list)
    supply_power = pref + pext
    input_work = torch.cat([torch.zeros(1, dtype=dtype, device=device), torch.cumsum(dt * supply_power, dim=0)])
    diss_work = torch.cat([torch.zeros(1, dtype=dtype, device=device), torch.cumsum(dt * pdiss, dim=0)])
    energy_delta = H - H[0]
    passivity_margin = input_work - energy_delta
    balance_residual_cum = energy_delta - input_work + diss_work

    # A single terminal SF metric is cheap enough to compute during training and
    # provides a group-momentum geometry for rest/steady-state regularization in
    # every force regime.  The metric itself is detached in the loss so the
    # controller cannot improve the objective by deforming the metric.
    if compute_terminal_sf:
        final_nu_sf = nus[-1] if nus else torch.zeros(n, dtype=dtype, device=device)
        terminal_sf, _, _ = empirical_souriau_fisher_metric(chain, q, final_nu_sf, info_cfg)
    else:
        terminal_sf = torch.eye(3, dtype=dtype, device=device)

    return {
        "pose": torch.stack(poses),
        "q": torch.stack(qs),
        "pi": torch.stack(pis),
        "nu": torch.stack(nus),
        "q_ref": torch.stack(qrefs),
        "c_ref": torch.stack(crefs),
        "fim": torch.stack(fim_list),
        "fim_eigs": torch.stack(fim_eigs),
        "sf": torch.stack(sf_list) if sf_list else torch.empty(0, 3, 3, dtype=dtype, device=device),
        "sf_eigs": torch.stack(sf_eigs) if sf_eigs else torch.empty(0, 3, dtype=dtype, device=device),
        "momentum_map": torch.stack(momentum_maps),
        "spring_force_joint": torch.stack(spring_forces),
        "fim_force_joint": torch.stack(fim_forces),
        "environment_force_joint": torch.stack(environment_forces),
        "environment_wrench_global": torch.stack(rigid_environment_forces),
        "total_joint_force": torch.stack(total_joint_forces),
        "external_generalized_force": torch.stack(external_generalized),
        "external_wrench_body": torch.stack(external_wrench_body),
        "external_wrench_world": torch.stack(external_wrench_world),
        "external_joint_force": torch.stack(external_joint_forces),
        "external_link_forces_world": torch.stack(external_link_forces_world),
        "external_link_positions_world": torch.stack(external_link_positions_world),
        "external_target_positions_world": torch.stack(external_target_positions_world),
        "external_power": pext,
        "external_conservative_power": torch.stack(external_conservative_power),
        "external_damping_power": torch.stack(external_damping_power),
        "external_potential": torch.stack(external_potential),
        "tracked_frame_position_world": torch.stack(tracked_frame_positions),
        "tracked_frame_target_world": torch.stack(tracked_frame_targets),
        "tracked_frame_force_world": torch.stack(tracked_frame_forces),
        "tracked_link_index": torch.stack(tracked_link_indices),
        "ground_contact_positions_world": torch.stack(ground_positions),
        "ground_contact_forces_world": torch.stack(ground_forces),
        "ground_contact_velocities_world": torch.stack(ground_velocities),
        "ground_gap": torch.stack(ground_gaps),
        "ground_active": torch.stack(ground_active),
        "min_ground_gap": torch.stack(min_ground_gaps),
        "contact_count": torch.stack(contact_counts),
        "loop_closure_error": torch.stack(loop_closure_errors),
        "loop_perimeter_error": torch.stack(loop_perimeter_errors),
        "loop_constraint_force": torch.stack(loop_constraint_forces),
        "loop_constraint_power": torch.stack(loop_constraint_power),
        "loop_projection_energy_jump": torch.stack(loop_projection_energy_jump),
        "loop_closure_measurement_loss": torch.stack(loop_closure_measurement_loss),
        "loop_seam_potential": torch.stack(loop_seam_potential),
        "loop_seam_spring_force": torch.stack(loop_seam_spring_force),
        "loop_seam_damping_force": torch.stack(loop_seam_damping_force),
        "loop_closure_sigma": torch.stack(loop_closure_sigma),
        "loop_closure_stiffness": torch.stack(loop_closure_stiffness),
        "loop_closure_mode": loop_mode,
        "loop_body_vertices": torch.stack(loop_body_vertices_hist) if loop_body_vertices_hist else torch.empty(0, chain.n_links + 1, 2, dtype=dtype, device=device),
        "open_body_vertices": torch.stack(open_body_vertices_hist) if open_body_vertices_hist else torch.empty(0, chain.n_links + 1, 2, dtype=dtype, device=device),
        "terminal_tracked_frame_position_world": terminal_ext["tracked_frame_position_world"] if terminal_ext is not None else torch.full((2,), float("nan"), dtype=dtype, device=device),
        "terminal_tracked_frame_target_world": terminal_ext["tracked_frame_target_world"] if terminal_ext is not None else torch.full((2,), float("nan"), dtype=dtype, device=device),
        "terminal_ground_gap": terminal_ext["ground_gap"] if terminal_ext is not None else torch.full((chain.n_links + 1,), float("nan"), dtype=dtype, device=device),
        "terminal_ground_velocity_world": terminal_ext["ground_contact_velocities_world"] if terminal_ext is not None else torch.zeros(chain.n_links + 1, 2, dtype=dtype, device=device),
        "motion_force_case": motion_case,
        "kinetic_energy": torch.stack(kinetic),
        "potential_energy": torch.stack(potential),
        "hamiltonian": H,
        "reference_power": pref,
        "supply_power": supply_power,
        "fim_power": torch.stack(p_fim_list),
        "environment_power": torch.stack(p_env_list),
        "dissipation_power": pdiss,
        "power_balance_residual": torch.stack(power_residual),
        "input_work": input_work,
        "dissipated_work": diss_work,
        "passivity_margin": passivity_margin,
        "energy_balance_residual_cum": balance_residual_cum,
        "fim_gain": model.fim_gain if use_fim_damping else torch.zeros_like(model.fim_gain),
        "fim_enabled": torch.tensor(1.0 if use_fim_damping else 0.0, dtype=dtype, device=device),
        "terminal_sf": terminal_sf,
        "terminal_sf_eigs": _safe_eigvalsh_spd(terminal_sf),
        "control_steps": torch.tensor(control_steps, dtype=torch.long, device=device),
        "control_horizon": torch.tensor(control_steps * dt, dtype=dtype, device=device),
        "evaluation_horizon": torch.tensor(n_steps * dt, dtype=dtype, device=device),
        "actual_horizon": torch.tensor(completed_steps * dt, dtype=dtype, device=device),
        "completed_steps": torch.tensor(completed_steps, dtype=torch.long, device=device),
        "diverged": torch.tensor(1.0 if diverged else 0.0, dtype=dtype, device=device),
    }


def world_pose_info_loss(
    model: WorldPoseInformationPHTemplate,
    rollout: dict[str, torch.Tensor],
    task: LocomotionTask,
    teacher: dict[str, torch.Tensor],
    Ksp: torch.Tensor,
    cfg: WorldPoseLossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Task loss with physically distinct terminal objectives.

    * self_propulsion: positioned-shape BVP in SE(2) x shape space.
    * frame_target: selected link-center frame must reach its Cartesian target;
      the old global pose target is not used as a competing terminal condition.
    * gravity_ground: chain must touch y=ground_y with little penetration and
      low terminal contact-point speed while still completing intrinsic shape.
    * rigid_loop: fixed-edge closed polygon must deform from circle to the
      perimeter-compatible crescent while satisfying loop closure.
    * open_shape: the same fixed-edge articulated chain deforms from a cut
      circle arc to a cut crescent arc with no loop-closure reaction. This is a
      diagnostic control for the closed-loop constraint manifold.

    The variational teacher is still used for intrinsic reference/shape
    supervision. Global momentum/SF matching is disabled for externally forced
    cases because the no-force teacher has a different momentum balance.
    """
    cfg = cfg or WorldPoseLossConfig()
    case = str(rollout.get("motion_force_case", "self_propulsion"))

    pose_err = se2_endpoint_error(rollout["pose"][-1], task.pose_target)
    pose_loss = torch.dot(pose_err, pose_err)
    q_target = Ksp @ task.c_target
    if task.kind == "bvp":
        shape_loss = torch.mean((rollout["q"][-1] - q_target) ** 2)
    elif task.kind == "periodic":
        shape_loss = torch.mean((rollout["q"][-1] - rollout["q"][0]) ** 2)
    else:
        shape_loss = torch.zeros((), dtype=Ksp.dtype, device=Ksp.device)
    terminal_mom = torch.mean(rollout["pi"][-1] ** 2)
    terminal_joint_mom = torch.mean(rollout["pi"][-1, 3:] ** 2)
    J_terminal = rollout["momentum_map"][-1]
    terminal_global_mom_euclidean = torch.mean(J_terminal ** 2)
    SF_terminal = rollout.get("terminal_sf", torch.eye(3, dtype=J_terminal.dtype, device=J_terminal.device)).detach()
    sf_terminal_mom = torch.dot(J_terminal, torch.linalg.solve(SF_terminal, J_terminal))
    # SF is used as the *metric choice* for terminal SE(2) momentum rather than
    # stacked on top of an identical Euclidean penalty.  The no-SF baselines
    # retain Euclidean global-momentum regularization, while all variants keep
    # the same Euclidean intrinsic joint-momentum regularization.
    if cfg.sf_terminal_momentum_weight > 0:
        terminal_global_metric_penalty = cfg.sf_terminal_momentum_weight * sf_terminal_mom
    else:
        terminal_global_metric_penalty = cfg.terminal_momentum_weight * terminal_global_mom_euclidean

    # External terminal objectives.
    zero = torch.zeros((), dtype=Ksp.dtype, device=Ksp.device)
    frame_target_loss = zero
    frame_target_error = zero
    ground_touch_loss = zero
    ground_touch_error = zero
    ground_penetration = zero
    ground_terminal_speed = zero
    rigid_loop_target_loss = zero
    rigid_loop_target_error = zero
    rigid_loop_local_shape_loss = zero
    rigid_loop_local_shape_error = zero
    rigid_loop_best_seam_shift = torch.zeros((), dtype=torch.long, device=Ksp.device)
    rigid_loop_pose_loss = zero
    rigid_loop_pose_error = zero
    rigid_loop_closure_loss = zero
    rigid_loop_closure_error = zero
    rigid_loop_closure_measurement_nll = zero
    rigid_loop_tail_joint_loss = zero
    rigid_loop_tail_pose_loss = zero
    open_shape_target_loss = zero
    open_shape_target_error = zero
    open_shape_local_shape_loss = zero
    open_shape_local_shape_error = zero
    open_shape_pose_loss = zero
    open_shape_pose_error = zero
    open_shape_tail_joint_loss = zero
    open_shape_tail_pose_loss = zero
    if case == "frame_target":
        p = rollout["terminal_tracked_frame_position_world"]
        pt = rollout["terminal_tracked_frame_target_world"]
        d = p - pt
        frame_target_loss = torch.dot(d, d)
        frame_target_error = torch.linalg.norm(d)
    elif case == "gravity_ground":
        gaps_T = rollout["terminal_ground_gap"]
        finite = torch.isfinite(gaps_T)
        gaps_T = torch.where(finite, gaps_T, torch.full_like(gaps_T, 1e3))
        min_gap_T = gaps_T.min()
        ground_touch_loss = min_gap_T**2
        ground_touch_error = torch.abs(min_gap_T)
        gaps_all = rollout["ground_gap"]
        gaps_all = torch.where(torch.isfinite(gaps_all), gaps_all, torch.zeros_like(gaps_all))
        penetration = torch.relu(-gaps_all)
        ground_penetration = torch.mean(penetration**2)
        vel_T = rollout["terminal_ground_velocity_world"]
        # Weight speeds of points near the ground; closest point always receives weight.
        weights = torch.softmax(-20.0 * gaps_T, dim=0)
        ground_terminal_speed = torch.sum(weights * torch.sum(vel_T**2, dim=-1))
    elif case == "rigid_loop":
        md = task.metadata or {}
        qT = torch.as_tensor(md["rigid_loop_q_target"], dtype=Ksp.dtype, device=Ksp.device)
        dq = rollout["q"][-1] - qT
        rigid_loop_target_loss = torch.mean(dq**2)
        rigid_loop_target_error = torch.sqrt(rigid_loop_target_loss + 1e-16)
        # v18 measurement protocol: all M physical links are observed through
        # midpoint+tangent measurements.  The seam endpoint x_M is not discarded;
        # it influences the final physical link.  Continuity x_M-x_0 is still a
        # separate closure measurement/constraint.
        if "rigid_loop_target_link_centers_body" in md:
            target_centers = torch.as_tensor(md["rigid_loop_target_link_centers_body"], dtype=Ksp.dtype, device=Ksp.device)
            target_tangents = torch.as_tensor(md["rigid_loop_target_link_tangents_body"], dtype=Ksp.dtype, device=Ksp.device)
            pred_vertices = rollout["loop_body_vertices"][-1]
            rigid_loop_local_shape_loss, _, _ = rigid_link_measurement_loss(
                pred_vertices, target_centers, target_tangents,
                perimeter=float(md.get("rigid_loop_perimeter", 1.0)),
                tangent_weight=float(md.get("rigid_loop_target_tangent_weight", 0.06)),
            )
            rigid_loop_best_seam_shift = torch.zeros((), dtype=torch.long, device=Ksp.device)
        else:
            target_vertices = torch.as_tensor(
                md.get("rigid_loop_target_rigid_vertices_body", md["rigid_loop_target_vertices_body"]),
                dtype=Ksp.dtype, device=Ksp.device,
            )
            pred_vertices = rollout["loop_body_vertices"][-1][:-1]
            rigid_loop_local_shape_loss, rigid_loop_best_seam_shift = cyclic_local_shape_loss(
                pred_vertices, target_vertices,
                allow_cyclic_shift=bool(md.get("rigid_loop_allow_cyclic_seam", True)),
            )
        rigid_loop_local_shape_error = torch.sqrt(rigid_loop_local_shape_loss + 1e-16)
        rigid_loop_pose_loss = pose_loss
        rigid_loop_pose_error = torch.linalg.norm(pose_err)
        rigid_loop_closure_error = rollout["loop_closure_error"][-1]
        closure_mode = str(md.get("rigid_loop_closure_mode", rollout.get("loop_closure_mode", "hard"))).lower()
        if closure_mode == "measurement":
            rigid_loop_closure_measurement_nll = rollout["loop_closure_measurement_loss"][-1]
            rigid_loop_closure_loss = rigid_loop_closure_measurement_nll
        elif closure_mode == "open":
            rigid_loop_closure_loss = torch.zeros_like(rigid_loop_closure_error)
        else:
            rigid_loop_closure_loss = rigid_loop_closure_error ** 2
        cs = int(rollout.get("control_steps", torch.tensor(rollout["q"].shape[0]-1)).detach().item())
        if rollout["q"].shape[0] > cs + 1:
            q_tail = rollout["q"][cs:]
            rigid_loop_tail_joint_loss = torch.mean((q_tail - qT[None, :]) ** 2)
            pe = [se2_endpoint_error(pp, task.pose_target) for pp in rollout["pose"][cs:]]
            rigid_loop_tail_pose_loss = torch.stack([torch.dot(e, e) for e in pe]).mean()
    elif case == "open_shape":
        md = task.metadata or {}
        qT = torch.as_tensor(md["open_shape_q_target"], dtype=Ksp.dtype, device=Ksp.device)
        dq = rollout["q"][-1] - qT
        open_shape_target_loss = torch.mean(dq**2)
        open_shape_target_error = torch.sqrt(open_shape_target_loss + 1e-16)
        target_vertices = torch.as_tensor(md["open_shape_target_rigid_vertices_body"], dtype=Ksp.dtype, device=Ksp.device)
        pred_vertices = rollout["open_body_vertices"][-1]
        open_shape_local_shape_loss = open_local_shape_loss(pred_vertices, target_vertices)
        open_shape_local_shape_error = torch.sqrt(open_shape_local_shape_loss + 1e-16)
        open_shape_pose_loss = pose_loss
        open_shape_pose_error = torch.linalg.norm(pose_err)
        cs = int(rollout.get("control_steps", torch.tensor(rollout["q"].shape[0]-1)).detach().item())
        if rollout["q"].shape[0] > cs + 1:
            q_tail = rollout["q"][cs:]
            open_shape_tail_joint_loss = torch.mean((q_tail - qT[None, :]) ** 2)
            pe = [se2_endpoint_error(pp, task.pose_target) for pp in rollout["pose"][cs:]]
            open_shape_tail_pose_loss = torch.stack([torch.dot(e, e) for e in pe]).mean()

    # Teacher conversion: retain intrinsic reference and shape path for all cases.
    Tref = min(rollout["c_ref"].shape[0], teacher["c"].shape[0] - 1)
    ref_loss = torch.mean((rollout["c_ref"][:Tref] - teacher["c"][:Tref]) ** 2)
    Tq = min(rollout["q"].shape[0], teacher["q"].shape[0])
    phase_q = torch.mean((rollout["q"][:Tq] - teacher["q"][:Tq]) ** 2)

    # Velocity/momentum teacher terms are meaningful only for the same unforced
    # physical system. External force cases intentionally change that balance.
    if case in {"self_propulsion", "rigid_loop", "open_shape"}:
        Tnu = min(rollout["nu"].shape[0], teacher["nu"].shape[0])
        phase_nu = torch.mean((rollout["nu"][:Tnu] - teacher["nu"][:Tnu]) ** 2)
        if cfg.sf_momentum_weight > 0:
            Tp = min(rollout["momentum_map"].shape[0], teacher["momentum_map"].shape[0])
            sf_terms = []
            for k in range(Tp):
                r = rollout["momentum_map"][k] - teacher["momentum_map"][k]
                SF = teacher["sf"][k]
                sf_terms.append(torch.dot(r, torch.linalg.solve(SF, r)))
            sf_loss = torch.stack(sf_terms).mean()
        else:
            sf_loss = zero
    else:
        phase_nu = zero
        sf_loss = zero

    passivity_res = torch.mean(rollout["power_balance_residual"] ** 2)
    limit_bound = float((task.metadata or {}).get("joint_limit", 1.0))
    limit = torch.relu(rollout["q"].abs() - limit_bound).pow(2).mean()

    if case == "self_propulsion":
        task_terminal = cfg.pose_weight * pose_loss
    elif case == "frame_target":
        task_terminal = cfg.frame_target_weight * frame_target_loss
    elif case == "gravity_ground":
        task_terminal = (
            cfg.ground_touch_weight * ground_touch_loss
            + cfg.ground_penetration_weight * ground_penetration
            + cfg.ground_terminal_speed_weight * ground_terminal_speed
        )
    elif case == "rigid_loop":
        task_terminal = (
            cfg.rigid_loop_local_shape_weight * rigid_loop_local_shape_loss
            + cfg.rigid_loop_target_weight * rigid_loop_target_loss
            + cfg.rigid_loop_pose_weight * rigid_loop_pose_loss
            + cfg.rigid_loop_closure_weight * rigid_loop_closure_loss
            + cfg.rigid_loop_tail_joint_weight * rigid_loop_tail_joint_loss
            + cfg.rigid_loop_tail_pose_weight * rigid_loop_tail_pose_loss
        )
    elif case == "open_shape":
        task_terminal = (
            cfg.open_shape_local_shape_weight * open_shape_local_shape_loss
            + cfg.open_shape_target_weight * open_shape_target_loss
            + cfg.open_shape_pose_weight * open_shape_pose_loss
            + cfg.open_shape_tail_joint_weight * open_shape_tail_joint_loss
            + cfg.open_shape_tail_pose_weight * open_shape_tail_pose_loss
        )
    else:
        # Legacy v7 cases keep the old pose objective.
        task_terminal = cfg.pose_weight * pose_loss

    total = (
        task_terminal
        + cfg.shape_weight * shape_loss
        + cfg.terminal_momentum_weight * terminal_joint_mom
        + terminal_global_metric_penalty
        + cfg.reference_teacher_weight * ref_loss
        + cfg.phase_q_weight * phase_q
        + cfg.phase_nu_weight * phase_nu
        + cfg.sf_momentum_weight * sf_loss
        # terminal_global_metric_penalty above already selects either the
        # Euclidean or Souriau--Fisher global-momentum metric.  Do not add the
        # SF terminal term a second time.
        + cfg.passivity_residual_weight * passivity_res
        + cfg.joint_limit_weight * limit
    )
    return total, {
        "pose_error": torch.linalg.norm(pose_err),
        "shape_error": torch.sqrt(shape_loss + 1e-16),
        "terminal_momentum": torch.sqrt(terminal_mom + 1e-16),
        "terminal_joint_momentum_mse": terminal_joint_mom,
        "terminal_global_momentum_euclidean_mse": terminal_global_mom_euclidean,
        "reference_mse": ref_loss,
        "phase_q_mse": phase_q,
        "phase_nu_mse": phase_nu,
        "sf_momentum_loss": sf_loss,
        "sf_terminal_momentum_loss": sf_terminal_mom,
        "passivity_residual_mse": passivity_res,
        "frame_target_error": frame_target_error,
        "ground_touch_error": ground_touch_error,
        "ground_penetration_mse": ground_penetration,
        "ground_terminal_speed_mse": ground_terminal_speed,
        "rigid_loop_target_error": rigid_loop_target_error,
        "rigid_loop_local_shape_error": rigid_loop_local_shape_error,
        "rigid_loop_pose_error": rigid_loop_pose_error,
        "rigid_loop_best_seam_shift": rigid_loop_best_seam_shift.to(dtype=Ksp.dtype),
        "rigid_loop_closure_error": rigid_loop_closure_error,
        "rigid_loop_closure_measurement_nll": rigid_loop_closure_measurement_nll,
        "rigid_loop_tail_joint_mse": rigid_loop_tail_joint_loss,
        "rigid_loop_tail_pose_mse": rigid_loop_tail_pose_loss,
        "open_shape_target_error": open_shape_target_error,
        "open_shape_local_shape_error": open_shape_local_shape_error,
        "open_shape_pose_error": open_shape_pose_error,
        "open_shape_tail_joint_mse": open_shape_tail_joint_loss,
        "open_shape_tail_pose_mse": open_shape_tail_pose_loss,
        "fim_gain": model.fim_gain,
    }

