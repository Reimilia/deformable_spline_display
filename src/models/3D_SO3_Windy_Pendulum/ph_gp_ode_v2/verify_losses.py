"""Sanity-check the would-be physics-informed losses for ph_gp_sde_v2 by
plugging in analytic GT subnets — and additionally noise-correct the central-
difference targets using the dataset's obs-noise scale.

For the spherical pendulum (m = l = 1, g = 9.81):
    M⁻¹(q) = (1 / m·l²) I₃
    V(q)   = m·g·l · q[8]
    D(q)   = friction_coeff · I₃
    B(q)   = I₃

Central-difference targets:
    ṗ_data(t) = (p_{t+1} − p_{t−1}) / (2 Δt)

Under iid Gaussian observation noise σ_obs on ω (and hence on p, since
M⁻¹ = I → p = ω), the central diff has per-component noise variance
    σ_eff² = σ_obs² / (2 Δt²),
so the ‖·‖² floor for L_V/B/D is 3 σ_eff² (3 components). The script
reports both the raw losses and noise-corrected versions:

    L_raw_corrected  = max(L_raw − 3 σ_eff², 0)
    L_NLL            = ‖res‖² / (2 σ_eff²)         (per-frame; sum over comp)

The GP_SDE_v2 model carries σ_obs_ω as a frozen field set from the
dataset's obs_noise_std, so it is available to a future training-loop use
of these losses. This script does not load the model — it just verifies
the auxiliary-loss algebra and shows how the noise correction collapses
the obs-noise-driven floor across σ levels.

Usage example:
    python verify_losses.py --obs_noise_std 0.05 --friction_coeff 0.5
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, '../../../..'))
for p in (PROJECT_ROOT,
          os.path.join(PROJECT_ROOT, 'src/utils'),
          os.path.join(PROJECT_ROOT, 'datasets')):
    if p not in sys.path:
        sys.path.insert(0, p)

from windy_pendulum_3d_datagen import get_dataset, arrange_data


# Pendulum constants
M_PEND = 1.0
L_PEND = 1.0
G = 9.81


def gt_residuals(x_seq, dt, friction_coeff):
    """Compute per-frame residuals for L_power, L_V, L_B, L_D using analytic
    GT subnets. x_seq shape (T, B, 15) layout [q (9), ω (3), u (3)].

    Returns a dict of per-interior-frame arrays:
      power : (T-2, B)
      V, B, D : (T-2, B, 3)         tangent-form 3-vec residuals
      dp_data, dp_model : (T-2, B, 3)
    """
    q = x_seq[..., :9]              # (T, B, 9)
    omega = x_seq[..., 9:12]        # (T, B, 3)
    u = x_seq[..., 12:15]           # (T, B, 3)
    T, B, _ = x_seq.shape

    # M⁻¹ = I → p = ω, ∇_p H = ω
    p = omega
    dHdp = omega

    # ∇_q V = m·g·l · e_8 (only q[8] enters V); ∇_q (KE) = 0 since M is constant.
    # So ∇_q H = ∇_q V everywhere; grav_full = grav_V.
    # grav = Σ_i R_i × (∇_q V)_i. With ∇_q V = (0,0, m·g·l) at row 2 only:
    # grav = R_2 × (0, 0, m·g·l) = m·g·l · (R[2,1], -R[2,0], 0)
    R2_x = q[..., 6]
    R2_y = q[..., 7]
    grav_V = np.stack([
        M_PEND * G * L_PEND * R2_y,
        -M_PEND * G * L_PEND * R2_x,
        np.zeros_like(R2_x),
    ], axis=-1)                     # (T, B, 3)
    grav_full = grav_V              # KE doesn't depend on q with constant M

    # gyro = p × ω = ω × ω = 0  (when p = ω with M=I, gyroscopic identically 0)
    gyro = np.cross(p, dHdp, axis=-1)

    # F = B u = I·u = u
    F = u

    # diss = D · ω = friction · ω
    diss = friction_coeff * dHdp

    # H = ½ ω·ω + m·g·l · q[8]
    H = 0.5 * (omega * omega).sum(-1) + M_PEND * G * L_PEND * q[..., 8]   # (T, B)

    # Central-difference targets, interior frames only
    Hdot_lhs = (H[2:] - H[:-2]) / (2.0 * dt)                              # (T-2, B)
    dp_data = (p[2:] - p[:-2]) / (2.0 * dt)                               # (T-2, B, 3)

    # Power-balance RHS
    # power_in = u^T B^T M⁻¹ p = u · ω,     power_diss = ω · D ω = friction ‖ω‖²
    power_in = (u * dHdp).sum(-1)
    power_diss = (dHdp * diss).sum(-1)
    Hdot_rhs = (power_in - power_diss)[1:-1]                              # (T-2, B)

    # ṗ_model = grav_full + gyro − diss + F
    dp_model_full = grav_full - diss + gyro + F                           # (T, B, 3)
    dp_model = dp_model_full[1:-1]

    grav_full_i = grav_full[1:-1]
    grav_V_i = grav_V[1:-1]
    gyro_i = gyro[1:-1]
    F_i = F[1:-1]
    diss_i = diss[1:-1]

    # L_V (tangent form): grav_V − α, with α = ṗ_data − (grav_KE + gyro − diss + F)
    # grav_KE = grav_full − grav_V = 0 here.
    alpha = dp_data - ((grav_full_i - grav_V_i) + gyro_i - diss_i + F_i)
    res_V = grav_V_i - alpha

    # L_B: F − (ṗ_data − grav_full − gyro + diss)
    res_B = F_i - (dp_data - grav_full_i - gyro_i + diss_i)

    # L_D: diss − (−ṗ_data + grav_full + gyro + F)
    res_D = diss_i - (-dp_data + grav_full_i + gyro_i + F_i)

    res_power = Hdot_lhs - Hdot_rhs

    return {
        'power': res_power,
        'V': res_V, 'B': res_B, 'D': res_D,
        'dp_data': dp_data, 'dp_model': dp_model,
    }


def _stats_vec(name, t):
    sq = (t * t).sum(-1)
    return (f"  {name:<10s} mean‖·‖²={sq.mean():.4e}  "
            f"max|·|={np.abs(t).max():.4e}  "
            f"std|·|={np.abs(t).std():.4e}")


def _stats_scalar(name, t):
    sq = t * t
    return (f"  {name:<10s} mean‖·‖²={sq.mean():.4e}  "
            f"max|·|={np.abs(t).max():.4e}  "
            f"std|·|={np.abs(t).std():.4e}")


def main():
    ap = argparse.ArgumentParser(description="Verify aux losses with GT subnets "
                                             "for ph_gp_sde_v2.")
    ap.add_argument('--obs_noise_std', type=float, default=0.0)
    ap.add_argument('--friction_coeff', type=float, default=0.5)
    ap.add_argument('--external_force_std', type=float, default=0.0)
    ap.add_argument('--wind_force_std', type=float, default=0.0)
    ap.add_argument('--external_force_type', type=str, default='sine',
                    choices=['sine', 'square', 'random', 'constant'])
    ap.add_argument('--samples', type=int, default=64)
    ap.add_argument('--timesteps', type=int, default=20)
    ap.add_argument('--num_points', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--data_dir', type=str,
                    default=os.path.join(PROJECT_ROOT,
                                         'datasets/data/windy_pendulum_3d'))
    ap.add_argument('--varying_friction', action='store_true')
    ap.add_argument('--random_u', action='store_true')
    ap.add_argument('--sigma_obs', type=float, default=None,
                    help='σ_obs to assume for the noise correction; defaults '
                         'to --obs_noise_std (i.e. matches the dataset).')
    args = ap.parse_args()

    sigma_obs = args.obs_noise_std if args.sigma_obs is None else args.sigma_obs

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

    train_x, t_eval = arrange_data(data['x'], data['t'],
                                   num_points=args.num_points)
    x_seq = np.concatenate(train_x, axis=1).astype(np.float64)         # (T, B, 15)
    t_eval = np.asarray(t_eval, dtype=np.float64)
    dt = float(t_eval[1] - t_eval[0])

    print("\n=== Verifying losses with GT subnets (ph_gp_sde_v2) ===")
    print(f"  data shape  : {tuple(x_seq.shape)}  (T, B, 15)")
    print(f"  dt          : {dt:.4e}    (Δt² = {dt * dt:.4e})")
    print(f"  obs noise σ : {args.obs_noise_std}    (sigma_obs used = {sigma_obs})")
    print(f"  friction c  : {args.friction_coeff}    varying={args.varying_friction}")

    res = gt_residuals(x_seq, dt, args.friction_coeff)

    # Scalar losses (raw)
    L_power = (res['power'] ** 2).mean()
    L_V = (res['V'] ** 2).sum(-1).mean()
    L_B = (res['B'] ** 2).sum(-1).mean()
    L_D = (res['D'] ** 2).sum(-1).mean()

    # ── Predicted central-diff noise floor on ‖dp_data‖² per frame ─────
    #   data-side per-component variance = σ_obs² / (2 Δt²)
    #   3-component sum                  = 3 σ_obs² / (2 Δt²)
    sigma_eff_sq = (sigma_obs ** 2) / (2.0 * dt * dt) if sigma_obs > 0 else 0.0
    # Data-side noise alone:
    floor_VBD_data = 3.0 * sigma_eff_sq
    # Model-side q-noise excess: ∇_q V_θ = mgl·e_8 → grav_V = mgl·(R[2,1], -R[2,0], 0)
    # carries q-noise on R[2,0] and R[2,1] (2 axes). Per-axis variance after
    # whitening: 2(mgl)²Δt². Summed over the 2 affected axes: 4(mgl)²Δt².
    # In raw (unwhitened) units the contribution is 2·(mgl)²σ_obs² = 4(mgl)²Δt²·σ_eff².
    floor_VBD_model = 2.0 * (M_PEND * G * L_PEND) ** 2 * (sigma_obs ** 2)
    pred_floor_VBD = floor_VBD_data + floor_VBD_model
    # ── Predicted Ḣ_eff² floor for L_power ─────────────────────────────
    # Linearised Ḣ_lhs noise from H = ½‖ω‖² + mgl·q[8]:
    #     σ_H_lin² = (⟨‖ω‖⟩² + (mgl)²) · σ_obs² / (2 Δt²)
    # Ḣ_model = uᵀg(M⁻¹p) − pᵀM⁻¹DM⁻¹p shares the same ω-noise; the negative
    # covariance with Ḣ_lhs subtracts ~half the variance. Empirical 0.5×.
    mean_om_norm_sq = float((x_seq[..., 9:12] ** 2).sum(-1).mean())
    sigma_H_eff_sq_lin = (mean_om_norm_sq + (M_PEND * G * L_PEND) ** 2) \
                          * (sigma_obs ** 2) / (2.0 * dt * dt) if sigma_obs > 0 else 0.0
    sigma_H_eff_sq = 0.5 * sigma_H_eff_sq_lin     # apply the 2× over-prediction correction
    pred_floor_power = sigma_H_eff_sq

    # Noise-corrected (subtract expected variance, clamp at 0)
    L_V_corr = max(L_V - pred_floor_VBD, 0.0)
    L_B_corr = max(L_B - pred_floor_VBD, 0.0)
    L_D_corr = max(L_D - pred_floor_VBD, 0.0)
    L_power_corr = max(L_power - pred_floor_power, 0.0)

    # NLL form: Gaussian per-component on ṗ_data with variance σ_eff².
    # Ignoring constants, NLL = ‖res‖² / (2 σ_eff²).
    if sigma_eff_sq > 0:
        nll_V = L_V / (2.0 * sigma_eff_sq)
        nll_B = L_B / (2.0 * sigma_eff_sq)
        nll_D = L_D / (2.0 * sigma_eff_sq)
    else:
        nll_V = nll_B = nll_D = float('nan')

    print(f"\nRaw scalar losses (mean over interior frames):")
    print(f"  L_power = {L_power:.4e}")
    print(f"  L_V     = {L_V:.4e}")
    print(f"  L_B     = {L_B:.4e}")
    print(f"  L_D     = {L_D:.4e}")

    print(f"\nPredicted noise floors (iid σ_obs={sigma_obs}):")
    print(f"  data-side σ_eff² = σ_obs²/(2Δt²)        = {sigma_eff_sq:.4e}  per component")
    print(f"  L_V/B/D floor  = 3 σ_eff²  (data-side)  = {floor_VBD_data:.4e}")
    print(f"     + model-side q-noise (2 axes via g_V) = {floor_VBD_model:.4e}")
    print(f"     = total expected L_V/B/D floor       = {pred_floor_VBD:.4e}")
    print(f"  σ_H_eff²  (with 0.5× covariance correction) = {pred_floor_power:.4e}")
    print(f"     analytic linearised σ_H_lin² (no correction) = {sigma_H_eff_sq_lin:.4e}")

    print(f"\nNoise-corrected losses (raw − predicted floor, clamped ≥ 0):")
    print(f"  L_power − floor = {L_power_corr:.4e}")
    print(f"  L_V     − floor = {L_V_corr:.4e}")
    print(f"  L_B     − floor = {L_B_corr:.4e}")
    print(f"  L_D     − floor = {L_D_corr:.4e}")

    if sigma_eff_sq > 0:
        print(f"\nGaussian-NLL form  (‖res‖² / (2 σ_eff²)):")
        print(f"  NLL_V = {nll_V:.4e}    (≈ #components/2 = 1.5 if at noise floor)")
        print(f"  NLL_B = {nll_B:.4e}")
        print(f"  NLL_D = {nll_D:.4e}")

    # Per-frame distribution
    print(f"\nResidual distribution (per interior frame, raw):")
    print(_stats_scalar('power', res['power']))
    print(_stats_vec('V',     res['V']))
    print(_stats_vec('B',     res['B']))
    print(_stats_vec('D',     res['D']))

    # ṗ comparison (raw)
    dp_data, dp_model = res['dp_data'], res['dp_model']
    dp_diff = dp_data - dp_model
    print(f"\nṗ comparison (numerical central diff vs analytic GT EoM):")
    print(_stats_vec('ṗ_data',  dp_data))
    print(_stats_vec('ṗ_model', dp_model))
    print(_stats_vec('ṗ_diff',  dp_diff))
    per_axis = (dp_diff ** 2).mean(axis=(0, 1))
    print(f"  ṗ_diff per-axis MSE : x={per_axis[0]:.4e}  "
          f"y={per_axis[1]:.4e}  z={per_axis[2]:.4e}")

    # ── Per-frame whitening ────────────────────────────────────────────
    # Divide each per-frame residual by σ_eff = σ_obs / (Δt √2). Under iid
    # Gaussian obs noise on ω (and hence on p, since M⁻¹ = I), the central-
    # diff noise on ṗ_data has per-component std σ_eff.
    #
    # Reference threshold (correct physics + iid noise):
    #   data-side noise alone gives mean‖res_w‖² ≈ 3.
    #   The MODEL side ALSO carries q-noise via grav_V(q) = mgl·(R[2,1], -R[2,0], 0):
    #     per-axis whitened variance excess on x and y = 2(mgl)²Δt² each
    #     no contribution on z (∇_q V is e_8-only, so g_V·z = 0 identically).
    #   Total expected mean‖res_V/B/D_w‖² = 3 + 4·(mgl)²·Δt²
    #     ≈ 3 + 4·9.81²·0.0025 = 3 + 0.962 ≈ 3.96   (Δt = 0.05, m=l=1)
    #
    # Truncation does NOT account for the excess: truncation is O(Δt²·∂³p/∂t³)
    # which after whitening scales as 1/σ_obs² and would change 50²×=2500× across
    # σ_obs ∈ [0.01, 0.5]. Observed change is ~5%, ruling truncation out.
    if sigma_obs > 0:
        sigma_eff = sigma_obs / (dt * np.sqrt(2.0))
        dp_diff_w = dp_diff / sigma_eff
        res_V_w = res['V'] / sigma_eff
        res_B_w = res['B'] / sigma_eff
        res_D_w = res['D'] / sigma_eff
        print(f"\nPer-frame whitening: divide each residual by σ_eff = "
              f"σ_obs/(Δt√2) = {sigma_eff:.4e}")
        print(f"  Under correct physics + iid noise, components ~ N(0,1) ⇒ "
              f"mean‖·‖² ≈ 3, per-axis MSE ≈ 1.")
        print(_stats_vec('ṗ_diff_w',  dp_diff_w))
        per_axis_w = (dp_diff_w ** 2).mean(axis=(0, 1))
        print(f"  ṗ_diff_w per-axis MSE : x={per_axis_w[0]:.4e}  "
              f"y={per_axis_w[1]:.4e}  z={per_axis_w[2]:.4e}")
        print(_stats_vec('V_w', res_V_w))
        print(_stats_vec('B_w', res_B_w))
        print(_stats_vec('D_w', res_D_w))
        threshold_VBD_w = 3.0 + 4.0 * (M_PEND * G * L_PEND) ** 2 * (dt ** 2)
        print(f"  expected mean‖V/B/D_w‖² = 3 + 4(mgl)²Δt² ≈ {threshold_VBD_w:.4f}  "
              f"(includes model-side q-noise excess)")
        # Whitened L_power: use the SAME σ_H_eff² already computed above so
        # the "subtracted" and "whitened" diagnostics agree. Includes the 0.5×
        # covariance correction (Ḣ_model shares ω-noise with Ḣ_lhs; negative
        # covariance subtracts ~half the variance).
        if sigma_H_eff_sq > 0:
            sigma_H_eff = np.sqrt(sigma_H_eff_sq)
            res_power_w = res['power'] / sigma_H_eff
            print(_stats_scalar('power_w', res_power_w))
            print(f"    (σ_H_eff² = ½·(⟨‖ω‖²⟩ + (mgl)²)·σ_obs²/(2Δt²) = "
                  f"{sigma_H_eff_sq:.4e};  expected mean‖power_w‖² ≈ 1)")
    else:
        print(f"\nPer-frame whitening: skipped (σ_obs = 0, no noise floor "
              f"to whiten against).")

    print(f"\nNote: GP_SDE_v2's `model.sigma_obs_omega` field is frozen at the "
          f"dataset's obs_noise_std (= {args.obs_noise_std}). A future training "
          f"objective can use it to compute exactly the noise-corrected (or "
          f"NLL-weighted) versions printed above.")


if __name__ == "__main__":
    main()
