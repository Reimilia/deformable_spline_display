from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F

from .chain import PlanarMultiLinkChain
from .geometry import se2_compose, se2_exp_increment, se2_coadjoint_term, se2_endpoint_error


class TaskConditionedPHTemplate(nn.Module):
    """Task-conditioned Hamiltonian shaping for the M-link spring chain.

    The network predicts a moving modal rest shape c_ref(t,task).  The physical
    Hamiltonian contains 1/2 (q-K_sp c_ref)^T K (q-K_sp c_ref), while the
    interconnection remains canonical/Lie-group mechanical and dissipation is PSD.
    This is intentionally the first, conservative learning tier: residual skew
    interconnection is not learned here.
    """

    def __init__(
        self,
        n_modes: int,
        n_joints: int,
        hidden: int = 96,
        ref_scale: float = 2.5,
        stiffness_init: float = 3.0,
        extra_damping_init: float = 0.02,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.n_modes = n_modes
        self.n_joints = n_joints
        self.ref_scale = ref_scale
        in_dim = 3 + n_modes + n_modes + 3 + 3  # type, c0, cT, target pose, phase
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, dtype=dtype), nn.Tanh(),
            nn.Linear(hidden, hidden, dtype=dtype), nn.Tanh(),
            nn.Linear(hidden, n_modes, dtype=dtype),
        )
        # Positive Hamiltonian stiffness and extra dissipative residual.
        self.log_stiffness = nn.Parameter(torch.full((n_joints,), math.log(stiffness_init), dtype=dtype))
        self.log_extra_damping = nn.Parameter(torch.full((n_joints,), math.log(extra_damping_init), dtype=dtype))

    @property
    def stiffness(self) -> torch.Tensor:
        return torch.exp(self.log_stiffness).clamp(0.05, 50.0)

    @property
    def extra_damping(self) -> torch.Tensor:
        return torch.exp(self.log_extra_damping).clamp(1e-5, 5.0)

    def reference(self, features: dict[str, torch.Tensor], tau: torch.Tensor) -> torch.Tensor:
        """Return modal rest shape c_ref for batch and normalized time tau in [0,1]."""
        B = features["c0"].shape[0]
        if tau.ndim == 0:
            tau = tau.expand(B)
        periodic_mask = (features["task_id"] == 2).to(features["c0"].dtype)
        phase = torch.stack([
            torch.sin(2.0 * math.pi * tau),
            torch.cos(2.0 * math.pi * tau),
            tau * (1.0 - periodic_mask),
        ], dim=-1)
        # Scale pose target into roughly O(1) feature range.
        pose_model = features.get("pose_target_for_model", features["pose_target"])
        pose_scaled = pose_model * torch.tensor(
            [20.0, 20.0, 2.0], dtype=features["c0"].dtype, device=features["c0"].device
        )
        x = torch.cat([
            features["task_onehot"], features["c0"], features["c_target"], pose_scaled, phase
        ], dim=-1)
        residual = self.ref_scale * torch.tanh(self.net(x))

        bvp_mask = (features["task_id"] == 0).to(features["c0"].dtype)[:, None]
        base_bvp = (1.0 - tau[:, None]) * features["c0"] + tau[:, None] * features["c_target"]
        base_other = features["c0"]
        base = bvp_mask * base_bvp + (1.0 - bvp_mask) * base_other
        # Envelope forces residual to vanish at endpoints for BVP.  For iso it
        # vanishes initially; for periodic it remains exactly periodic.
        iso_mask = (features["task_id"] == 1).to(features["c0"].dtype)[:, None]
        per_mask = (features["task_id"] == 2).to(features["c0"].dtype)[:, None]
        bvp_env = (4.0 * tau * (1.0 - tau))[:, None]
        iso_env = torch.sin(0.5 * math.pi * tau)[:, None]
        per_env = torch.ones_like(bvp_env)
        env = bvp_mask * bvp_env + iso_mask * iso_env + per_mask * per_env
        return base + env * residual


@dataclass
class PHLossConfig:
    pose_weight: float = 400.0
    bvp_shape_weight: float = 80.0
    terminal_momentum_weight: float = 2.0
    periodic_state_weight: float = 40.0
    dissipation_weight: float = 0.05
    reference_effort_weight: float = 0.01
    joint_limit_weight: float = 100.0


def rollout_ph(
    model: TaskConditionedPHTemplate,
    features: dict[str, torch.Tensor],
    chain: PlanarMultiLinkChain,
    Ksp: torch.Tensor,
    n_steps: int,
    dt: float,
) -> dict[str, torch.Tensor]:
    dtype = Ksp.dtype
    device = Ksp.device
    batch = features["c0"].shape[0]
    q = torch.einsum("jm,bm->bj", Ksp, features["c0"])
    pose = features.get("pose0")
    if pose is None:
        pose = torch.zeros(batch, 3, dtype=dtype, device=device)
    else:
        pose = pose.to(dtype=dtype, device=device).clone()
    pi = torch.zeros(batch, 3 + chain.n_joints, dtype=dtype, device=device)
    pi0 = pi.clone()
    q0 = q.clone()

    poses = [pose]
    qs = [q]
    pis = [pi]
    nus = []
    refs = []
    cref_list = []
    powers = []
    spring_energies = []

    I = torch.eye(3 + chain.n_joints, dtype=dtype, device=device).expand(batch, -1, -1)
    extra_diag = torch.diag(model.extra_damping).expand(batch, -1, -1)

    for k in range(n_steps):
        tau = torch.full((batch,), k / n_steps, dtype=dtype, device=device)
        c_ref = model.reference(features, tau)
        q_ref = torch.einsum("jm,bm->bj", Ksp, c_ref)
        ops = chain.assemble(q)
        M = ops["M"]
        D = ops["D"].clone()
        D[..., 3:, 3:] = D[..., 3:, 3:] + extra_diag
        nu = torch.linalg.solve(M, pi.unsqueeze(-1)).squeeze(-1)
        xi = nu[..., :3]
        mu = pi[..., :3]
        coad = se2_coadjoint_term(xi, mu)
        effort_q = model.stiffness * (q - q_ref)
        rhs = torch.cat([coad, -effort_q], dim=-1)

        Minv = torch.linalg.solve(M, I)
        lhs = I + dt * (D @ Minv)
        pi_next = torch.linalg.solve(lhs, (pi + dt * rhs).unsqueeze(-1)).squeeze(-1)
        nu_next = torch.linalg.solve(M, pi_next.unsqueeze(-1)).squeeze(-1)
        xi_next = nu_next[..., :3]
        qdot_next = nu_next[..., 3:]

        pose = se2_compose(pose, se2_exp_increment(xi_next, dt))
        q = q + dt * qdot_next
        pi = pi_next

        power = torch.einsum("bi,bij,bj->b", nu_next, D, nu_next)
        spring = 0.5 * ((q - q_ref) ** 2 * model.stiffness).sum(dim=-1)

        poses.append(pose)
        qs.append(q)
        pis.append(pi)
        nus.append(nu_next)
        refs.append(q_ref)
        cref_list.append(c_ref)
        powers.append(power)
        spring_energies.append(spring)

    # Endpoint reference at tau=1 for diagnostics.
    tau1 = torch.ones(batch, dtype=dtype, device=device)
    cref_final = model.reference(features, tau1)
    qref_final = torch.einsum("jm,bm->bj", Ksp, cref_final)

    return {
        "pose": torch.stack(poses, dim=1),
        "q": torch.stack(qs, dim=1),
        "pi": torch.stack(pis, dim=1),
        "nu": torch.stack(nus, dim=1),
        "q_ref": torch.stack(refs, dim=1),
        "c_ref": torch.stack(cref_list, dim=1),
        "qref_final": qref_final,
        "pi0": pi0,
        "q0": q0,
        "dissipation": 0.5 * dt * torch.stack(powers, dim=1).sum(dim=1),
        "spring_energy_mean": torch.stack(spring_energies, dim=1).mean(dim=1),
    }


def ph_task_loss(
    rollout: dict[str, torch.Tensor],
    features: dict[str, torch.Tensor],
    chain: PlanarMultiLinkChain,
    Ksp: torch.Tensor,
    cfg: PHLossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pose_err_vec = se2_endpoint_error(rollout["pose"][:, -1], features["pose_target"])
    pose_loss = (pose_err_vec**2).sum(dim=-1)
    ids = features["task_id"]
    bvp = (ids == 0).to(Ksp.dtype)
    iso = (ids == 1).to(Ksp.dtype)
    per = (ids == 2).to(Ksp.dtype)

    q_target = torch.einsum("jm,bm->bj", Ksp, features["c_target"])
    bvp_shape = ((rollout["q"][:, -1] - q_target) ** 2).mean(dim=-1)
    terminal_mom = (rollout["pi"][:, -1] ** 2).mean(dim=-1)
    periodic_q = ((rollout["q"][:, -1] - rollout["q"][:, 0]) ** 2).mean(dim=-1)
    periodic_pi = ((rollout["pi"][:, -1] - rollout["pi"][:, 0]) ** 2).mean(dim=-1)
    periodic_state = periodic_q + 0.2 * periodic_pi

    qabs = rollout["q"].abs()
    limit_pen = torch.relu(qabs - chain.params.joint_limit).pow(2).mean(dim=(1, 2))
    ref_effort = ((rollout["q_ref"] - rollout["q"][:, :-1]) ** 2).mean(dim=(1, 2))

    loss_each = (
        cfg.pose_weight * pose_loss
        + cfg.bvp_shape_weight * bvp * bvp_shape
        + cfg.terminal_momentum_weight * (bvp + iso) * terminal_mom
        + cfg.periodic_state_weight * per * periodic_state
        + cfg.dissipation_weight * rollout["dissipation"]
        + cfg.reference_effort_weight * ref_effort
        + cfg.joint_limit_weight * limit_pen
    )
    metrics = {
        "loss_each": loss_each,
        "pose_error": torch.linalg.norm(pose_err_vec, dim=-1),
        "pose_loss": pose_loss,
        "bvp_shape": bvp_shape,
        "terminal_momentum": terminal_mom,
        "periodic_state": periodic_state,
        "dissipation": rollout["dissipation"],
        "limit_pen": limit_pen,
        "ref_effort": ref_effort,
    }
    return loss_each.mean(), metrics
