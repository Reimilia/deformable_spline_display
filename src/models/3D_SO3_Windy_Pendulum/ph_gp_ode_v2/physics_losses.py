"""Noise-aware physics-informed auxiliary losses for ph_gp_ode_v2.

Implements the four whitened auxiliary losses derived in
`new_losses_explaination.md`:

    L_power_w = (1/N_int) Σ_t  (Ḣ_lhs_t − Ḣ_rhs_t)² / σ_H_eff²
    L_V_w     = (1/N_int) Σ_t  ‖res_V(t)‖² / σ_eff²
    L_B_w     = (1/N_int) Σ_t  ‖res_B(t)‖² / σ_eff²
    L_D_w     = (1/N_int) Σ_t  ‖res_D(t)‖² / σ_eff²

with whitening factors

    σ_eff²    = σ_obs_ω² / (2 Δt²)                        (per dp component)
    σ_H_eff²  = ½ · (⟨‖ω‖²⟩ + (mgl)²) · σ_obs_ω² / (2 Δt²) (½ from Ḣ_model/Ḣ_lhs cov.)

At correct physics + iid noise, expectation is

    E[L_X_w]      ≈ 3 + 4(mgl)²Δt²    ≈ 3.96   (X ∈ {V, B, D})
    E[L_power_w]  ≈ 1

so a single λ ∈ [0.1, 1.0] is meaningful across all noise levels.

When σ_obs_ω = 0 the whitening collapses; the loss reduces to the raw
mean-square residual (still useful, but no noise correction).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def _physics_residuals(model, x_seq, dt):
    """Compute per-frame residuals for power / V / B / D and ṗ_data.

    Args:
        model: a KeyedODEModel-style adapter exposing `.M_inv(q)`,
               `._V_call(q)`, `._g_call(q)`, `._Dw_call(q)`,
               `.u_dim`, `.friction`. Inference-mode is up to the caller.
        x_seq: (T, B, 9 + 3 + u_dim).
        dt   : scalar (jnp).

    Returns:
        dict of arrays with shape (T-2, B, ...). Endpoints dropped.
    """
    T, B, _ = x_seq.shape
    u_dim = model.u_dim

    q  = x_seq[..., :9]                                   # (T, B, 9)
    om = x_seq[..., 9:12]                                 # (T, B, 3)
    u  = x_seq[..., 12:12 + u_dim]                        # (T, B, u_dim)

    # vmap subnet calls over (T, B); single-sample subnets expect a 9-vector.
    flat_q = q.reshape(T * B, 9)

    # M⁻¹(q): (T*B, 3, 3)
    M_inv_flat = jax.vmap(model.M_inv)(flat_q)
    M_inv = M_inv_flat.reshape(T, B, 3, 3)

    # p = M·ω = solve(M⁻¹, ω)
    p = jnp.linalg.solve(M_inv, om[..., None])[..., 0]    # (T, B, 3)

    # dHdp = M⁻¹·p ≡ ω (recover from p)
    dHdp = (M_inv @ p[..., None])[..., 0]                 # (T, B, 3)

    # ── ∇_q V_θ via autograd, vmapped per (q, p) frame ─────────────
    # H(q, p) = ½ pᵀ M⁻¹(q) p + V(q); the q-gradient (with p detached
    # at this single q,p-frame slice) is what feeds grav_full.
    def _V_scalar(q_):
        return model._V_call(q_)[0]
    dV_dq = jax.vmap(jax.grad(_V_scalar))(flat_q).reshape(T, B, 9)

    def _H_of_q(q_, p_):
        return 0.5 * jnp.dot(p_, model._M_call(q_) @ p_) + model._V_call(q_)[0]
    dHdq = jax.vmap(lambda q_, p_: jax.grad(_H_of_q)(q_, p_))(
        flat_q, p.reshape(T * B, 3)).reshape(T, B, 9)

    # gV  = -(q^×)ᵀ ∇_q V        = Σ_i R_i × ∂V/∂q_i  (sum over rows)
    # gFu = -(q^×)ᵀ ∇_q H        = Σ_i R_i × ∂H/∂q_i
    R3x3 = q.reshape(T, B, 3, 3)
    grav_V = jnp.sum(jnp.cross(R3x3, dV_dq.reshape(T, B, 3, 3), axis=-1),
                     axis=-2)                              # (T, B, 3)
    grav_full = jnp.sum(jnp.cross(R3x3, dHdq.reshape(T, B, 3, 3), axis=-1),
                        axis=-2)                           # (T, B, 3)

    # gyro = p × ω
    gyro = jnp.cross(p, dHdp, axis=-1)                    # (T, B, 3)

    # F = B_θ(q)·u
    g_q_flat = jax.vmap(model._g_call)(flat_q)            # (T*B, 3, u_dim) or (T*B, 3) when u_dim==1
    if u_dim == 1:
        F = g_q_flat.reshape(T, B, 3) * u
    else:
        F = jnp.einsum('nij,tbj->tbi',
                        g_q_flat, u) if False else \
            (g_q_flat.reshape(T, B, 3, u_dim) @ u[..., None])[..., 0]

    # diss = D_θ(q, p)·dHdp
    if model.friction:
        flat_p = p.reshape(T * B, 3)
        Dw_flat = jax.vmap(model._Dw_call)(flat_q, flat_p)     # (T*B, 3, 3)
        Dw = Dw_flat.reshape(T, B, 3, 3)
        diss = (Dw @ dHdp[..., None])[..., 0]
    else:
        diss = jnp.zeros_like(dHdp)

    # H = ½ pᵀ M⁻¹ p + V  (scalar per frame, used for Ḣ_lhs central diff)
    KE = 0.5 * (p * dHdp).sum(-1)
    V_q = jax.vmap(lambda q_: model._V_call(q_)[0])(flat_q).reshape(T, B)
    H = KE + V_q                                          # (T, B)

    # ── interior central differences ──────────────────────────────────
    Hdot_lhs  = (H[2:] - H[:-2]) / (2.0 * dt)             # (T-2, B)
    dp_data   = (p[2:] - p[:-2]) / (2.0 * dt)             # (T-2, B, 3)
    # detach: targets, no gradient
    Hdot_lhs  = jax.lax.stop_gradient(Hdot_lhs)
    dp_data   = jax.lax.stop_gradient(dp_data)

    # ── interior model quantities ─────────────────────────────────────
    grav_full_i = grav_full[1:-1]
    grav_V_i    = grav_V[1:-1]
    gyro_i      = gyro[1:-1]
    F_i         = F[1:-1]
    diss_i      = diss[1:-1]
    dHdp_i      = dHdp[1:-1]

    # power balance (RHS is analytic; LHS is detached central diff)
    power_in   = (u[1:-1] * (jnp.einsum('tbij,tbj->tbi',
                                         g_q_flat.reshape(T, B, 3, u_dim)
                                          if u_dim > 1 else
                                         g_q_flat.reshape(T, B, 3, 1),
                                         u[1:-1, ..., None]
                                          if u_dim == 1 else u[1:-1])
                              if u_dim > 1 else
                              g_q_flat.reshape(T, B, 3)[1:-1] * u[1:-1])
                   ).sum(-1) if False else None
    # Simpler: power_in = uᵀ Bᵀ M⁻¹ p = u · (g_qᵀ · dHdp) = u · F-via-transpose.
    # For u_dim = 3 with g_q square: u · (g_qᵀ dHdp)  (= F when g_q symmetric).
    # We just compute u · dHdp · (component sum already in F if g_q = I).
    # Clean form: use the model-side B u = F directly, dot with dHdp.
    # Ḣ_model = uᵀ Bᵀ dHdp − dHdpᵀ D dHdp.
    # We have F = B u, so uᵀBᵀ dHdp = (B u)ᵀ dHdp = F · dHdp.
    Hdot_rhs_int = (F_i * dHdp_i).sum(-1) - (dHdp_i * diss_i).sum(-1)

    # ── per-loss residuals ────────────────────────────────────────────
    # Tangent-form L_V: res_V = grav_V − α with α = ṗ_data − [grav_KE + gyro − diss + F]
    grav_KE_i = grav_full_i - grav_V_i
    alpha     = dp_data - (grav_KE_i + gyro_i - diss_i + F_i)
    res_V     = grav_V_i - alpha                          # (T-2, B, 3)
    res_B     = F_i - (dp_data - grav_full_i - gyro_i + diss_i)
    res_D     = diss_i - (-dp_data + grav_full_i + gyro_i + F_i)
    res_power = Hdot_lhs - Hdot_rhs_int                   # (T-2, B)

    # ⟨‖ω‖²⟩ for σ_H_eff (use interior frames; static under jit)
    mean_om_sq = (om[1:-1] ** 2).sum(-1).mean()

    return {
        'res_power': res_power,
        'res_V': res_V, 'res_B': res_B, 'res_D': res_D,
        'mean_om_sq': mean_om_sq,
        'dp_data': dp_data,
    }


def physics_aux_losses(model, x_seq, dt, sigma_omega,
                       m: float = 1.0, l: float = 1.0, g: float = 9.81,
                       whiten: bool = True):
    """Compute (L_power, L_V, L_B, L_D) — whitened by default.

    Args:
        model       : KeyedODEModel-style adapter.
        x_seq       : (T, B, 9 + 3 + u_dim).
        dt          : scalar.
        sigma_omega : scalar JAX array — the ω noise std used as the
                      whitening source. Pass `jnp.exp(model.log_sigma_omega)`
                      to use the trainable rollout-NLL noise scale (this is
                      the "learn-it-from-data" path). The whitening is
                      differentiable through this value, so the aux losses
                      and the rollout NLL co-adapt: as `log_sigma_omega`
                      shrinks toward the true noise level, the aux losses
                      land on their interpretable expectations (~3.96 / ~1).
        m, l, g     : pendulum constants for σ_H_eff (only used if whiten).
        whiten      : if True, divide by σ_eff²/σ_H_eff²; else raw mean-‖·‖².

    Returns:
        dict {'L_power', 'L_V', 'L_B', 'L_D', 'mean_om_sq', 'sigma_omega',
              ...raw versions...} — JAX scalars.
    """
    res = _physics_residuals(model, x_seq, dt)

    L_V_raw = (res['res_V'] ** 2).sum(-1).mean()
    L_B_raw = (res['res_B'] ** 2).sum(-1).mean()
    L_D_raw = (res['res_D'] ** 2).sum(-1).mean()
    L_power_raw = (res['res_power'] ** 2).mean()

    sigma_omega_jnp = jnp.asarray(sigma_omega, dtype=jnp.float32)
    nonzero = sigma_omega_jnp > 0

    # σ_eff² for ṗ_data per-component noise; σ_H_eff² for Ḣ_lhs noise.
    # Both scale by σ_omega² (the trainable / passed-in ω-noise std).
    # The 0.5× factor on σ_H_eff² is the empirical covariance correction:
    # Ḣ_model and Ḣ_lhs share ω-noise → negative covariance subtracts ½.
    sigma_eff_sq    = (sigma_omega_jnp ** 2) / (2.0 * dt * dt)
    sigma_H_eff_sq  = 0.5 * (res['mean_om_sq'] + (m * g * l) ** 2) \
                          * (sigma_omega_jnp ** 2) / (2.0 * dt * dt)

    # Use jnp.where to keep this jit-friendly when σ might be 0 in some
    # warmup configurations. We clip the denominator from below so the
    # division never produces inf; the resulting value is still meaningful
    # because σ_omega is being trained upward from its init (typically 0.1).
    eps = jnp.asarray(1e-12, dtype=sigma_eff_sq.dtype)
    if whiten:
        L_V     = L_V_raw     / jnp.maximum(sigma_eff_sq,    eps)
        L_B     = L_B_raw     / jnp.maximum(sigma_eff_sq,    eps)
        L_D     = L_D_raw     / jnp.maximum(sigma_eff_sq,    eps)
        L_power = L_power_raw / jnp.maximum(sigma_H_eff_sq,  eps)
        L_V     = jnp.where(nonzero, L_V,     L_V_raw)
        L_B     = jnp.where(nonzero, L_B,     L_B_raw)
        L_D     = jnp.where(nonzero, L_D,     L_D_raw)
        L_power = jnp.where(nonzero, L_power, L_power_raw)
    else:
        L_V, L_B, L_D, L_power = L_V_raw, L_B_raw, L_D_raw, L_power_raw

    return {
        'L_power': L_power, 'L_V': L_V, 'L_B': L_B, 'L_D': L_D,
        'L_power_raw': L_power_raw,
        'L_V_raw': L_V_raw, 'L_B_raw': L_B_raw, 'L_D_raw': L_D_raw,
        'mean_om_sq': res['mean_om_sq'],
        'sigma_omega': sigma_omega_jnp,
        'sigma_eff_sq': sigma_eff_sq,
        'sigma_H_eff_sq': sigma_H_eff_sq,
    }
