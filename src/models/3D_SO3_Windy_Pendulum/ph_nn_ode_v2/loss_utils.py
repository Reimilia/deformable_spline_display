"""fp32-stable losses.

Mirrors src/utils/ode_utils.py but applies a strict-interior clamp before
torch.acos so the backward pass through arccos cannot produce NaN at cos = +/- 1
(where the analytic gradient -1/sqrt(1-cos^2) is infinite).
"""
import os, sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../utils')))
from ode_utils import (
    L2_loss,
    compute_rotation_matrix_from_unnormalized_rotmat,
)


# Strict-interior clamp epsilon for arccos.
# At cos = 1 - eps the gradient |d/dcos arccos| = 1/sqrt(2*eps - eps^2),
# so eps=1e-6 gives a max gradient of ~707, large but finite.
ACOS_EPS = 1e-6


def compute_geodesic_distance_from_two_matrices_safe(m1, m2):
    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))
    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2

    upper = (1.0 - ACOS_EPS) * torch.ones(batch, device=cos.device, dtype=cos.dtype)
    lower = (-1.0 + ACOS_EPS) * torch.ones(batch, device=cos.device, dtype=cos.dtype)
    cos = torch.min(cos, upper)
    cos = torch.max(cos, lower)

    theta = torch.acos(cos)
    return theta


def compute_geodesic_loss_safe(gt_r_matrix, out_r_matrix):
    theta = compute_geodesic_distance_from_two_matrices_safe(gt_r_matrix, out_r_matrix)
    theta = theta ** 2
    error = theta.mean()
    return error, theta


def rotmat_L2_geodesic_loss_safe(u, u_hat, split):
    # L2 is on q_dot (= ω) only — the control channel is identical between
    # target and prediction. Convention matches `ph_gp_sde` exactly:
    #   MSE_ω = mean over N of ‖Δω‖²    (sum over 3 components, mean over N)
    # so that the printed `test_l2_loss` here equals the GP_SDE trainer's
    # `MSE_ω` printed value on the same data.
    q_hat, q_dot_hat, _u_hat = torch.split(u_hat, split, dim=2)
    q,     q_dot,     _u     = torch.split(u,     split, dim=2)

    diff_omega = q_dot - q_dot_hat                                       # (T, B, 3)
    l2_loss = torch.mean(torch.sum(diff_omega * diff_omega, dim=-1))     # scalar

    q_hat = q_hat.flatten(start_dim=0, end_dim=1)
    q     = q.flatten(start_dim=0, end_dim=1)
    R_hat = compute_rotation_matrix_from_unnormalized_rotmat(q_hat)
    R     = compute_rotation_matrix_from_unnormalized_rotmat(q)
    geo_loss, _ = compute_geodesic_loss_safe(R, R_hat)
    return l2_loss + geo_loss, l2_loss, geo_loss


def _rotmat_L2_geodesic_diff_safe(u, u_hat, split):
    # Same change as above: drop the `u` channel from the L2 residual.
    q_hat, q_dot_hat, _u_hat = torch.split(u_hat, split, dim=1)
    q,     q_dot,     _u     = torch.split(u,     split, dim=1)
    l2_diff = torch.sum((q_dot - q_dot_hat) ** 2, dim=1)
    R_hat = compute_rotation_matrix_from_unnormalized_rotmat(q_hat)
    R     = compute_rotation_matrix_from_unnormalized_rotmat(q)
    _, geo_diff = compute_geodesic_loss_safe(R, R_hat)
    return l2_diff + geo_diff, l2_diff, geo_diff


def traj_rotmat_L2_geodesic_loss_safe(traj, traj_hat, split):
    total_loss = l2_loss = geo_loss = None
    for t in range(traj.shape[0]):
        u = traj[t, :, :]
        u_hat = traj_hat[t, :, :]
        if total_loss is None:
            total_loss, l2_loss, geo_loss = _rotmat_L2_geodesic_diff_safe(u, u_hat, split=split)
            total_loss = total_loss.unsqueeze(0)
            l2_loss = l2_loss.unsqueeze(0)
            geo_loss = geo_loss.unsqueeze(0)
        else:
            t_total, t_l2, t_geo = _rotmat_L2_geodesic_diff_safe(u, u_hat, split=split)
            total_loss = torch.cat((total_loss, t_total.unsqueeze(0)), dim=0)
            l2_loss = torch.cat((l2_loss, t_l2.unsqueeze(0)), dim=0)
            geo_loss = torch.cat((geo_loss, t_geo.unsqueeze(0)), dim=0)
    return total_loss, l2_loss, geo_loss


# ──────────────────────────────────────────────────────────────────────
# Physics-informed losses (Loss 1: power balance, Loss 2: consistency)
# ──────────────────────────────────────────────────────────────────────
#
# Conventions matching the network in this folder:
#   - state x = [q (9), ω (3), u (u_dim)]; ω is q_dot in trainer code
#   - inner.M_net(q) returns M⁻¹(q) ∈ R^{3×3} (so p = solve(M⁻¹, ω) = M ω)
#   - H(q, p) = ½ pᵀ M⁻¹ p + V(q)
#   - ∇_p H ≡ ω = M⁻¹ p
#
# Both losses use central differences on the interior of x_seq and drop
# the first / last frame.

def _inner_model(model):
    return model._orig_mod if hasattr(model, '_orig_mod') else model


def power_balance_loss(model, x_seq, dt):
    """Loss 1 — energy-flow consistency.

    Matches the central-difference derivative of H_θ along the trajectory
    against the analytic RHS uᵀ Bᵀ M⁻¹ p − pᵀ M⁻¹ D M⁻¹ p. The kinematic
    and gyroscopic terms cancel analytically, so this isolates the energy
    budget — i.e. mismatch between {M, V} (which set H) and {B, D} (which
    set its rate of change).

    Args:
        model: DissipativeSO3HamNODE (possibly torch.compile-wrapped).
        x_seq: (T, B, 9+3+u_dim) trajectory of [q, ω, u].
        dt:    scalar timestep between consecutive frames in x_seq.

    Returns:
        scalar mean-squared residual on the interior (T-2, B) frames.
    """
    inner = _inner_model(model)
    T, B, _ = x_seq.shape
    u_dim = inner.u_dim

    q_flat = x_seq[..., :9].reshape(T * B, 9)
    w_flat = x_seq[..., 9:12].reshape(T * B, 3)
    u_flat = x_seq[..., 12:12 + u_dim].reshape(T * B, u_dim)

    # M_net outputs M⁻¹(q); recover p = M ω.
    M_inv = inner.M_net(q_flat)                                          # (TB, 3, 3)
    p = torch.linalg.solve(M_inv, w_flat.unsqueeze(-1)).squeeze(-1)      # (TB, 3)

    # ∇_p H = M⁻¹ p ≡ ω
    dHdp = torch.matmul(M_inv, p.unsqueeze(-1)).squeeze(-1)              # (TB, 3)

    KE = 0.5 * (p * dHdp).sum(dim=-1)
    V_q = inner.V_net(q_flat).squeeze(-1)
    H = (KE + V_q).reshape(T, B)

    H_dot_lhs = (H[2:] - H[:-2]) / (2.0 * dt)                            # (T-2, B)

    g_q = inner.g_net(q_flat)
    if u_dim == 1:
        gw = (g_q * dHdp).sum(dim=-1, keepdim=True)                      # (TB, 1)
        power_in = (u_flat * gw).sum(dim=-1)
    else:
        gT_w = torch.matmul(g_q.transpose(1, 2),
                            dHdp.unsqueeze(-1)).squeeze(-1)              # (TB, u_dim)
        power_in = (u_flat * gT_w).sum(dim=-1)

    if inner.friction:
        Dw_q = inner.Dw_net(q_flat)
        D_w = torch.matmul(Dw_q, dHdp.unsqueeze(-1)).squeeze(-1)
        power_diss = (dHdp * D_w).sum(dim=-1)
    else:
        power_diss = torch.zeros_like(power_in)

    H_dot_rhs = (power_in - power_diss).reshape(T, B)[1:-1]              # (T-2, B)

    return ((H_dot_lhs - H_dot_rhs) ** 2).mean()


def consistency_subnet_losses(model, x_seq, dt):
    """Loss 2 (per-subnetwork back-solving) — returns (L_V, L_B, L_D).

    Rearranges the EoM to isolate one subnetwork at a time:

        L_V : ‖−(q^×)ᵀ∇_q V_θ − α‖²    (tangent-only form; see note below)
        L_B : ‖B_θ u − RHS_B‖²,  RHS_B = ṗ_data + (q^×)ᵀ∇_q H − p^× M⁻¹p + D M⁻¹p
        L_D : ‖D_θ M⁻¹ p − RHS_D‖², RHS_D = −ṗ_data − (q^×)ᵀ∇_q H + p^× M⁻¹p + B u

    On L_V: the math doc's pinv-based 9-dim form
        ‖∇_q V_θ − ∇_q V_implied‖² with ∇_q V_implied = −[(q^×)ᵀ]⁺ α
    has a structural floor from the SO(3)-kernel component of ∇_q V_θ
    that does not affect dynamics. We instead compare in the 3-dim
    tangent space — equivalent on-manifold but goes to zero with GT.
    Forward value coincides with L_B / L_D; gradient still routes only
    through V_θ and M_θ, preserving localisation.

    Per-component versions localize which subnetwork is most to blame on
    held-out data. Mirrors the network's cat-split trick to keep p_split
    independent of θ_M for the inner autograd.grad (fp32-stability).

    Args:
        model: DissipativeSO3HamNODE (possibly torch.compile-wrapped).
        x_seq: (T, B, 9+3+u_dim) trajectory of [q, ω, u].
        dt:    scalar timestep.

    Returns:
        (L_V, L_B, L_D) — three scalar tensors, mean over (T-2, B) interior frames.
    """
    inner = _inner_model(model)
    T, B, _ = x_seq.shape
    u_dim = inner.u_dim

    q_flat = x_seq[..., :9].reshape(T * B, 9)
    w_flat = x_seq[..., 9:12].reshape(T * B, 3)
    u_flat = x_seq[..., 12:12 + u_dim].reshape(T * B, u_dim)

    # Outer M_net call only feeds p; the cat-split below decouples it
    # from the inner autograd over (q, p).
    M_inv_for_p = inner.M_net(q_flat)
    p_raw = torch.linalg.solve(M_inv_for_p,
                               w_flat.unsqueeze(-1)).squeeze(-1)         # (TB, 3)

    qp = torch.cat([q_flat, p_raw], dim=1)
    q_split, p_split = torch.split(qp, [9, 3], dim=1)

    M_inv = inner.M_net(q_split)
    V_q = inner.V_net(q_split).squeeze(-1)
    g_q = inner.g_net(q_split)

    p_aug = p_split.unsqueeze(-1)
    KE = 0.5 * torch.matmul(p_aug.transpose(1, 2),
                            torch.matmul(M_inv, p_aug)).squeeze(-1).squeeze(-1)
    H = KE + V_q

    # ∇_qp H = [∇_q H, ∇_p H]
    dH = torch.autograd.grad(H.sum(), qp, create_graph=True, retain_graph=True)[0]
    dHdq, dHdp = torch.split(dH, [9, 3], dim=1)

    # ∇_q V alone (needed for L_V).
    dV_dq = torch.autograd.grad(V_q.sum(), qp, create_graph=True)[0][:, :9]

    q_3x3 = q_split.view(-1, 3, 3)
    grav_full = torch.linalg.cross(q_3x3, dHdq.view(-1, 3, 3), dim=2).sum(dim=1)
    # grav_V acts as the "V's contribution to dp" when removed from the bracket
    # in α (see L_V derivation: −½(q^×)ᵀ∇_q(pᵀM⁻¹p) folds to grav_KE = grav_full − grav_V).
    grav_V = torch.linalg.cross(q_3x3, dV_dq.view(-1, 3, 3), dim=2).sum(dim=1)

    gyro = torch.linalg.cross(p_split, dHdp, dim=1)

    if u_dim == 1:
        F = g_q * u_flat
    else:
        F = torch.matmul(g_q, u_flat.unsqueeze(-1)).squeeze(-1)

    if inner.friction:
        Dw_q = inner.Dw_net(q_split)
        D_dHdp = torch.matmul(Dw_q, dHdp.unsqueeze(-1)).squeeze(-1)
    else:
        D_dHdp = torch.zeros_like(dHdp)

    # Data-side ṗ via central diff (interior only); target → detach.
    p_traj = p_raw.reshape(T, B, 3).detach()
    dp_data = (p_traj[2:] - p_traj[:-2]) / (2.0 * dt)                    # (T-2, B, 3)

    def _interior(x):
        return x.reshape(T, B, *x.shape[1:])[1:-1]

    grav_full_i = _interior(grav_full)
    grav_V_i    = _interior(grav_V)
    gyro_i      = _interior(gyro)
    F_i         = _interior(F)
    D_dHdp_i    = _interior(D_dHdp)

    # ── L_V (tangent-only) ───────────────────────────────────────────
    # The doc's pinv-based 9-dim form has a structural floor from the
    # SO(3)-normal component of ∇_q V_θ that does not affect dynamics.
    # We instead compare on the dynamics-relevant 3-dim tangent space:
    #     L_V = ‖−(q^×)ᵀ ∇_q V_θ − α‖²   = ‖grav_V − α‖²
    # which goes to zero with GT subnets and clean data.
    # α = ṗ_data − [ grav_KE + gyro − D∇_pH + Bu ],  grav_KE = grav_full − grav_V.
    grav_KE_i = grav_full_i - grav_V_i
    alpha = dp_data - (grav_KE_i + gyro_i - D_dHdp_i + F_i)              # (T-2, B, 3)
    L_V = ((grav_V_i - alpha) ** 2).sum(dim=-1).mean()

    # ── L_B ──────────────────────────────────────────────────────────
    # RHS_B = ṗ_data − grav_full − gyro + D∇_pH       (since (q^×)ᵀ∇_qH = −grav_full)
    B_target = dp_data - grav_full_i - gyro_i + D_dHdp_i
    L_B = ((F_i - B_target) ** 2).sum(dim=-1).mean()

    # ── L_D ──────────────────────────────────────────────────────────
    # RHS_D = −ṗ_data + grav_full + gyro + Bu
    D_target = -dp_data + grav_full_i + gyro_i + F_i
    L_D = ((D_dHdp_i - D_target) ** 2).sum(dim=-1).mean()

    return L_V, L_B, L_D
