"""Sanity-check the physics-informed losses by replacing every subnetwork
with its analytic ground truth.

For the spherical pendulum (m = l = 1, g = 9.81):
    M⁻¹(q) = (1 / m·l²) I₃            (M_net is the inverse mass per network conv.)
    V(q)   = m·g·l · q[8]              (q[8] = R₃₃ ≈ height of bob / l)
    D(q)   = friction_coeff · I₃        (constant friction only)
    B(q)   = I₃                         (body-frame torque, identity for u_dim = 3)

With clean data the four losses should be within central-difference
truncation error (~O(Δt²)) of zero, plus any obs-noise / wind-noise
propagation through the data terms.

Usage example (same dataset as the training command):
    python verify_losses.py --obs_noise_std 0.01 --friction_coeff 0.5 \
        --external_force_std 0.0 --wind_force_std 0.0
"""
import argparse, os, sys
import numpy as np
import torch

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, '../../../..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src/utils'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'datasets'))
sys.path.insert(0, THIS_FILE_DIR)

from windy_pendulum_3d_datagen import get_dataset, arrange_data
from loss_utils import power_balance_loss, consistency_subnet_losses


class GTModel:
    """Stand-in for DissipativeSO3HamNODE that returns analytic GT physics.
    Same attribute surface that loss_utils relies on:
        .M_net(q) → (N, 3, 3)   (returns M⁻¹)
        .V_net(q) → (N, 1)
        .Dw_net(q) → (N, 3, 3)
        .g_net(q) → (N, 3, u_dim)
        .friction (bool), .u_dim (int)
    No `_orig_mod`, so loss_utils._inner_model returns this object directly.
    """
    def __init__(self, m=1.0, l=1.0, g=9.81,
                 friction_coeff=0.5, varying_friction=False,
                 u_dim=3, device='cpu', dtype=torch.float32):
        self.m, self.l, self.g = m, l, g
        self.varying_friction = varying_friction
        self.u_dim = u_dim
        self.friction = True
        self.device = device
        self.dtype = dtype
        self._I3 = torch.eye(3, device=device, dtype=dtype)
        fc = torch.as_tensor(friction_coeff, device=device, dtype=dtype)
        if fc.ndim == 0:
            fc = fc.expand(3)
        self._fc_diag = torch.diag(fc)

        if varying_friction:
            print("[warn] varying_friction GT depends on q_dot which the "
                  "Dw_net interface doesn't see; using constant approximation.")

    def M_net(self, q):
        N = q.shape[0]
        return (1.0 / (self.m * self.l ** 2)) * self._I3.expand(N, 3, 3)

    def V_net(self, q):
        return (self.m * self.g * self.l) * q[:, 8:9]

    def Dw_net(self, q):
        N = q.shape[0]
        return self._fc_diag.expand(N, 3, 3)

    def g_net(self, q):
        N = q.shape[0]
        return self._I3.expand(N, 3, 3)


def _residuals(gt, x_seq, dt):
    """Per-frame residual tensors before mean-reduction.
    Mirrors loss_utils.{power_balance,consistency_subnet}_losses.
    """
    T, B, _ = x_seq.shape
    u_dim = gt.u_dim
    q_flat = x_seq[..., :9].reshape(T * B, 9)
    w_flat = x_seq[..., 9:12].reshape(T * B, 3)
    u_flat = x_seq[..., 12:12 + u_dim].reshape(T * B, u_dim)

    M_inv_for_p = gt.M_net(q_flat)
    p_raw = torch.linalg.solve(M_inv_for_p, w_flat.unsqueeze(-1)).squeeze(-1)
    qp = torch.cat([q_flat, p_raw], dim=1)
    q_split, p_split = torch.split(qp, [9, 3], dim=1)

    M_inv = gt.M_net(q_split)
    V_q   = gt.V_net(q_split).squeeze(-1)
    g_q   = gt.g_net(q_split)
    Dw_q  = gt.Dw_net(q_split)

    p_aug = p_split.unsqueeze(-1)
    KE = 0.5 * torch.matmul(p_aug.transpose(1, 2),
                            torch.matmul(M_inv, p_aug)).squeeze(-1).squeeze(-1)
    H = KE + V_q

    dH = torch.autograd.grad(H.sum(), qp, create_graph=False, retain_graph=True)[0]
    dHdq, dHdp = torch.split(dH, [9, 3], dim=1)
    dV_dq = torch.autograd.grad(V_q.sum(), qp, create_graph=False)[0][:, :9]

    q_3x3 = q_split.view(-1, 3, 3)
    grav_full = torch.linalg.cross(q_3x3, dHdq.view(-1, 3, 3), dim=2).sum(dim=1)
    grav_V    = torch.linalg.cross(q_3x3, dV_dq.view(-1, 3, 3), dim=2).sum(dim=1)
    gyro = torch.linalg.cross(p_split, dHdp, dim=1)

    if u_dim == 1:
        F = g_q * u_flat
        gw = (g_q * dHdp).sum(dim=-1, keepdim=True)
        power_in = (u_flat * gw).sum(dim=-1)
    else:
        F = torch.matmul(g_q, u_flat.unsqueeze(-1)).squeeze(-1)
        gT_w = torch.matmul(g_q.transpose(1, 2), dHdp.unsqueeze(-1)).squeeze(-1)
        power_in = (u_flat * gT_w).sum(dim=-1)
    D_dHdp = torch.matmul(Dw_q, dHdp.unsqueeze(-1)).squeeze(-1)
    power_diss = (dHdp * D_dHdp).sum(dim=-1)

    p_traj = p_raw.reshape(T, B, 3).detach()
    dp_data = (p_traj[2:] - p_traj[:-2]) / (2.0 * dt)

    H_full = H.reshape(T, B).detach()
    Hdot_lhs = (H_full[2:] - H_full[:-2]) / (2.0 * dt)

    def _interior(x):
        return x.reshape(T, B, *x.shape[1:])[1:-1]

    grav_full_i = _interior(grav_full).detach()
    grav_V_i    = _interior(grav_V).detach()
    gyro_i      = _interior(gyro).detach()
    F_i         = _interior(F).detach()
    D_dHdp_i    = _interior(D_dHdp).detach()
    R_int       = _interior(q_3x3).detach()
    dV_dq_i     = _interior(dV_dq).detach()
    Hdot_rhs_i  = (power_in - power_diss).reshape(T, B)[1:-1].detach()

    grav_KE_i = grav_full_i - grav_V_i
    alpha = dp_data - (grav_KE_i + gyro_i - D_dHdp_i + F_i)
    alpha_b = alpha.unsqueeze(-2).expand(-1, -1, 3, -1)
    R_cross_alpha = torch.linalg.cross(R_int, alpha_b, dim=-1)
    dV_implied = -0.5 * R_cross_alpha.reshape(*alpha.shape[:-1], 9)

    res_power = Hdot_lhs - Hdot_rhs_i                 # (T-2, B)
    res_V     = dV_dq_i - dV_implied                  # (T-2, B, 9) — full pinv loss
    res_B     = F_i - (dp_data - grav_full_i - gyro_i + D_dHdp_i)        # (T-2, B, 3)
    res_D     = D_dHdp_i - (-dp_data + grav_full_i + gyro_i + F_i)       # (T-2, B, 3)

    # ṗ_data (central diff) vs ṗ_model (built from the EoM with GT subnets).
    # Both individually + their residual.
    dp_model = grav_full_i - D_dHdp_i + gyro_i + F_i                     # (T-2, B, 3)
    res_dp   = dp_data - dp_model                                        # (T-2, B, 3)

    # Dynamics-relevant V residual: project both ∇_q V_θ and ∇_q V_implied
    # through −(q^×)ᵀ (i.e. into the 3-dim tangent space). Equivalent to
    # comparing dp's V-contribution at each frame.
    #   −(q^×)ᵀ ∇_q V_θ      = grav_V_i
    #   −(q^×)ᵀ ∇_q V_implied = α                     (by construction)
    res_V_tangent = grav_V_i - alpha                  # (T-2, B, 3)

    # Kernel-only piece (the "wasted" 6-dim component of ∇_q V_θ that does
    # not affect dynamics). Equals ∇_q V_θ − P ∇_q V_θ, where
    # P = (q^×/2)(q^×)ᵀ is the projector onto the tangent space of SO(3).
    P_dV_b = grav_V_i.unsqueeze(-2).expand(-1, -1, 3, -1)        # (T-2, B, 3, 3)
    P_dV   = -0.5 * torch.linalg.cross(R_int, P_dV_b, dim=-1) \
                  .reshape(*grav_V_i.shape[:-1], 9)               # (T-2, B, 9)
    res_V_kernel = dV_dq_i - P_dV

    return {
        'power': res_power, 'V': res_V, 'B': res_B, 'D': res_D,
        'V_tangent': res_V_tangent, 'V_kernel': res_V_kernel,
        'dp_data': dp_data, 'dp_model': dp_model, 'dp_diff': res_dp,
    }


def _fmt(name, t, vector=True):
    """One-line stats for a residual tensor.
    If vector, treats last dim as a vector (reports ‖·‖² mean to match loss).
    """
    flat = t.reshape(-1) if not vector else t.reshape(-1, t.shape[-1])
    if vector:
        sq = (flat * flat).sum(dim=-1)
        mse = sq.mean().item()
    else:
        mse = (flat * flat).mean().item()
    return (f"  {name:<7s} mean‖·‖²={mse:.4e}  "
            f"max|·|={t.abs().max().item():.4e}  "
            f"min|·|={t.abs().min().item():.4e}  "
            f"std|·|={t.abs().std().item():.4e}")


def main():
    parser = argparse.ArgumentParser(description="Verify PH losses with GT subnets.")
    parser.add_argument('--obs_noise_std', type=float, default=0.0)
    parser.add_argument('--friction_coeff', type=float, default=0.5)
    parser.add_argument('--external_force_std', type=float, default=0.0)
    parser.add_argument('--wind_force_std', type=float, default=0.0)
    parser.add_argument('--external_force_type', type=str, default='sine',
                        choices=['sine', 'square', 'random', 'constant'])
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--timesteps', type=int, default=20)
    parser.add_argument('--num_points', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(PROJECT_ROOT,
                                             'datasets/data/windy_pendulum_3d'))
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU index, or -1 for CPU')
    parser.add_argument('--varying_friction', action='store_true')
    parser.add_argument('--random_u', action='store_true')
    parser.add_argument('--m', type=float, default=1.0)
    parser.add_argument('--l', type=float, default=1.0)
    parser.add_argument('--gravity', type=float, default=9.81)
    args = parser.parse_args()

    device = torch.device(
        f'cuda:{args.gpu}' if (args.gpu >= 0 and torch.cuda.is_available()) else 'cpu')
    dtype = torch.float32

    us = ((0.0,) * 3, (-1.0,) * 3, (1.0,) * 3, (-2.0,) * 3, (2.0,) * 3)
    data, _ = get_dataset(
        seed=args.seed, samples=args.samples, timesteps=args.timesteps,
        save_dir=args.data_dir, us=us, ori_rep="rotmat",
        friction_coeff=args.friction_coeff,
        varying_friction=args.varying_friction,
        external_force_type=args.external_force_type,
        external_force_std=args.external_force_std,
        wind_force_std=args.wind_force_std,
        obs_noise_std=args.obs_noise_std,
        random_u=args.random_u,
    )

    train_x, t_eval = arrange_data(data['x'], data['t'], num_points=args.num_points)
    train_x_cat = torch.tensor(np.concatenate(train_x, axis=1),
                               dtype=dtype, device=device, requires_grad=True)
    t_eval = torch.tensor(t_eval, dtype=dtype, device=device)
    dt = (t_eval[1] - t_eval[0]).item()

    print(f"\n=== Verifying losses with GT subnets ===")
    print(f"  data shape  : {tuple(train_x_cat.shape)}  (T, B, 15)")
    print(f"  dt          : {dt:.4e}    (Δt² = {dt * dt:.4e})")
    print(f"  obs noise σ : {args.obs_noise_std}")
    print(f"  friction c  : {args.friction_coeff}    varying={args.varying_friction}")
    print(f"  external σ  : {args.external_force_std}    type={args.external_force_type}")
    print(f"  wind σ      : {args.wind_force_std}")
    print(f"  m, l, g     : {args.m}, {args.l}, {args.gravity}")

    gt = GTModel(m=args.m, l=args.l, g=args.gravity,
                 friction_coeff=args.friction_coeff,
                 varying_friction=args.varying_friction,
                 u_dim=3, device=device, dtype=dtype)

    # Scalar loss values (the same numbers train.py would log as
    # train_power_loss / train_V_cons_loss / train_B_cons_loss / train_D_cons_loss).
    L_power = power_balance_loss(gt, train_x_cat, dt).item()
    L_V_t, L_B_t, L_D_t = consistency_subnet_losses(gt, train_x_cat, dt)
    L_V, L_B, L_D = L_V_t.item(), L_B_t.item(), L_D_t.item()

    print(f"\nScalar losses (mean over interior frames):")
    print(f"  L_power = {L_power:.4e}")
    print(f"  L_V     = {L_V:.4e}")
    print(f"  L_B     = {L_B:.4e}")
    print(f"  L_D     = {L_D:.4e}")

    # Per-frame residual distribution stats
    res = _residuals(gt, train_x_cat, dt)
    print(f"\nResidual distribution (per interior frame):")
    print(_fmt('power', res['power'], vector=False))
    print(_fmt('V',     res['V'],     vector=True))
    print(_fmt('B',     res['B'],     vector=True))
    print(_fmt('D',     res['D'],     vector=True))

    # V split into the dynamics-relevant tangent piece and the SO(3)-kernel
    # piece. The tangent piece should match the obs-noise floor; the kernel
    # piece is structural (penalises non-tangent components of ∇_q V_θ).
    print(f"\nL_V breakdown:")
    print(_fmt('V_tan',  res['V_tangent'], vector=True))
    print(_fmt('V_kern', res['V_kernel'],  vector=True))
    pinv_L_V = (res['V_tangent'].pow(2).sum(-1).mean()
                * 0.5  # ½ from the tangent-projection isometry on SO(3)
                + res['V_kernel'].pow(2).sum(-1).mean()).item()
    print(f"  old (pinv 9-dim) L_V would have been : {pinv_L_V:.4e}")
    print(f"  new (tangent 3-dim) L_V              : {L_V:.4e}   ← this is what train.py uses")

    # ṗ_data (central diff) vs ṗ_model (analytic EoM with GT subnets)
    dp_data, dp_model, dp_diff = res['dp_data'], res['dp_model'], res['dp_diff']
    print(f"\nṗ comparison (numerical central diff vs model EoM with GT subnets):")
    print(f"  ṗ_data    : ‖·‖² mean={dp_data.pow(2).sum(-1).mean().item():.4e}  "
          f"max|·|={dp_data.abs().max().item():.4e}  "
          f"std|·|={dp_data.abs().std().item():.4e}")
    print(f"  ṗ_model   : ‖·‖² mean={dp_model.pow(2).sum(-1).mean().item():.4e}  "
          f"max|·|={dp_model.abs().max().item():.4e}  "
          f"std|·|={dp_model.abs().std().item():.4e}")
    print(f"  ṗ_diff    : ‖·‖² mean={dp_diff.pow(2).sum(-1).mean().item():.4e}  "
          f"max|·|={dp_diff.abs().max().item():.4e}  "
          f"std|·|={dp_diff.abs().std().item():.4e}")
    # Component-wise breakdown so you can see which axis the residual lives in
    per_axis_mse = dp_diff.pow(2).mean(dim=(0, 1))                      # (3,)
    print(f"  ṗ_diff per-axis MSE : "
          f"x={per_axis_mse[0].item():.4e}  "
          f"y={per_axis_mse[1].item():.4e}  "
          f"z={per_axis_mse[2].item():.4e}")
    rel = (dp_diff.pow(2).sum(-1).mean()
           / dp_data.pow(2).sum(-1).mean().clamp_min(1e-30)).item()
    print(f"  relative error ‖ṗ_diff‖² / ‖ṗ_data‖² = {rel:.4e}")

    print(f"\nExpected magnitudes with clean GT data:")
    print(f"  central-diff truncation : ‖·‖² ~ O(Δt⁴) ≈ {dt**4:.2e}  per frame")
    print(f"  obs-noise propagation   : ‖dp_data‖ noise ~ σ_obs / Δt = "
          f"{args.obs_noise_std / max(dt, 1e-12):.2e}  per component\n"
          f"                            ⇒ ‖·‖² ~ ({(args.obs_noise_std / max(dt, 1e-12))**2:.2e}) "
          f"× #components")


if __name__ == '__main__':
    main()
