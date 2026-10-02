"""ELBO components for the variational-GP port-Hamiltonian SDE.

Companion to `loss_utils_jax.py` (which stays the geodesic / L² module).
This file provides the **negative log-likelihood** and **per-subnet KL**
pieces needed to optimise the ELBO

    −ELBO  =  L_NLL(φ, ψ)  +  (β / N) · L_KL(ψ)

where the data-fit term is the **concentrated Gaussian on SO(3)**

    p(R | R̂, σ_R) ∝ exp(−θ²/(2σ_R²)) / Z(σ_R)

with partition function Z(σ_R) ≈ (2π σ_R²)^(3/2) for small σ (Watson /
matrix-vMF small-noise limit on the 3-dimensional SO(3) manifold). The
3 in the normaliser is **not** the dimension of the geodesic-angle scalar
— it's the dimension of the rotation manifold, which is what we're
actually putting a likelihood on. So

    −log p(R_obs | R_hat, σ_R)  =  θ²/(2σ_R²) + 3 log σ_R + 3⁄2 log 2π .

The angular-velocity residual is a plain 3-D iid Gaussian

    −log p(ω_obs | ω_hat, σ_ω) = ‖Δω‖²/(2σ_ω²) + 3 log σ_ω + 3⁄2 log 2π .

The per-subnet KL is the closed-form mean-field Gaussian
KL(N(μ, σ²) ‖ N(0, 1)) already provided by every GP module in
`gp_model.py` via `weight_kl_loss()`.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from .loss_utils_jax import compute_geodesic_distance_from_two_matrices_safe


_LOG_2PI = math.log(2.0 * math.pi)


def gaussian_nll_rotation(R_obs, R_hat, log_sigma_R):
    """Concentrated-Gaussian NLL on SO(3).

    The likelihood is over R, not over θ — so the partition function picks
    up the **manifold dimension (3)**, not the dimension of the scalar θ.
    See module docstring for the small-σ Watson limit.

    Args:
        R_obs, R_hat : (N, 3, 3)
        log_sigma_R  : scalar  (jnp array)

    Returns:
        mean_nll       : scalar — mean over N of −log p(R_obs | R_hat, σ_R)
        mean_theta_sq  : scalar — mean θ² (also reported as the "geodesic MSE"
                         so loss curves stay comparable with the MSE trainer)
    """
    theta = compute_geodesic_distance_from_two_matrices_safe(R_obs, R_hat)  # (N,)
    theta_sq = theta ** 2
    mean_theta_sq = jnp.mean(theta_sq)

    sigma_sq = jnp.exp(2.0 * log_sigma_R)
    mean_nll = 0.5 * mean_theta_sq / sigma_sq + 3.0 * log_sigma_R + 1.5 * _LOG_2PI
    return mean_nll, mean_theta_sq


def gaussian_nll_omega(omega_obs, omega_hat, log_sigma_omega):
    """3-D iid Gaussian NLL on angular-velocity residuals.

    Args:
        omega_obs, omega_hat : (N, 3)
        log_sigma_omega      : scalar (jnp array)

    Returns:
        mean_nll      : scalar — mean over N of −log p(ω_obs | ω_hat, σ_ω)
        mean_sq_norm  : scalar — mean over N of ‖Δω‖² (sum over the 3
                        components of ω, then mean over the N samples).
                        This is the convention used as the "MSE_ω"
                        diagnostic in the trainer prints and matched on
                        the PyTorch side by ph_nn_ode_fp32.
    """
    diff = omega_obs - omega_hat                                # (N, 3)
    sq_norm = jnp.sum(diff * diff, axis=-1)                     # (N,)
    mean_sq_norm = jnp.mean(sq_norm)

    sigma_sq = jnp.exp(2.0 * log_sigma_omega)
    mean_nll = 0.5 * mean_sq_norm / sigma_sq + 3.0 * log_sigma_omega + 1.5 * _LOG_2PI
    return mean_nll, mean_sq_norm


def elbo_nll(traj_obs, traj_hat, log_sigma_R, log_sigma_omega, split=(9, 3, 3)):
    """Per-step Monte-Carlo NLL across a (T, B, 15) or (T, B, S, 15) tensor.

    The control axis (last 3 channels) is sliced off — control inputs are not
    predictions, so they don't enter the likelihood.

    Returns dict with:
        nll_total      : scalar  =  nll_R + nll_omega
        nll_R          : scalar  geodesic-Gaussian NLL on R
        nll_omega      : scalar  3-D Gaussian NLL on ω
        mean_theta_sq  : scalar  geodesic² (legacy "geo MSE")
        mean_omega_sq  : scalar  ‖Δω‖²    (legacy "L2 MSE on ω")
        sigma_R        : scalar  σ_R = exp(log_sigma_R)
        sigma_omega    : scalar  σ_ω = exp(log_sigma_omega)
    """
    rd, ad, _ = split

    q_obs     = traj_obs[..., :rd]
    qdot_obs  = traj_obs[..., rd:rd + ad]
    q_hat     = traj_hat[..., :rd]
    qdot_hat  = traj_hat[..., rd:rd + ad]

    R_obs = q_obs.reshape(-1, 3, 3)
    R_hat = q_hat.reshape(-1, 3, 3)
    omega_obs = qdot_obs.reshape(-1, ad)
    omega_hat = qdot_hat.reshape(-1, ad)

    nll_R, mean_theta_sq = gaussian_nll_rotation(R_obs, R_hat, log_sigma_R)
    nll_omega, mean_omega_sq = gaussian_nll_omega(
        omega_obs, omega_hat, log_sigma_omega
    )

    return {
        'nll_total':     nll_R + nll_omega,
        'nll_R':         nll_R,
        'nll_omega':     nll_omega,
        'mean_theta_sq': mean_theta_sq,
        'mean_omega_sq': mean_omega_sq,
        'sigma_R':       jnp.exp(log_sigma_R),
        'sigma_omega':   jnp.exp(log_sigma_omega),
    }


def pl_loss(model, batch_x_cat, dt, sigma_obs_omega, gp_keys_batch,
            inference_mode=False):
    """Per-increment pseudo-likelihood for σ_φ (and a strong drift signal).

    The rollout NLL gives σ_φ no useful gradient — model and env have
    independent Brownian paths, so increasing σ_φ only adds variance to
    the residual and the optimum is σ_φ → 0. This term replaces the
    rollout with the Euler-Maruyama transition density evaluated at the
    **observed** consecutive snapshots (q_t, ω_t) → (q_{t+Δt}, ω_{t+Δt}):

        Δω_obs ≈ N( μ(q_t,ω_t,u_t)·Δt ,  σ_φ²(q_t)·Δt + 2·σ_obs_ω² )

    The factor of 2 in 2·σ_obs_ω² is because Δω_obs is the difference of
    two independent obs-noisy ω samples (env applies σ_obs to ω in
    `add_proper_noise_3d`, see datasets/windy_pendulum_3d_datagen.py).

    Per-increment NLL (3-D Gaussian normaliser):

        L_t  =  ‖Δω_obs − μ·Δt‖² / (2·Σ_eff)  +  3⁄2 · log Σ_eff

    aggregated as a mean over (T−1, B, S). Gives every subnet a clean
    single-step training signal (V/M/Dw/g via the drift, σ_net via the
    variance) and is the principled loss for the SDE under fixed-Δt
    snapshots — see Tzen & Raginsky / Latent SDE.

    Args:
        model           : DissipativeSO3HamSDE
        batch_x_cat     : (T_obs, B, 15)  observed (q, ω, u)
        dt              : scalar — outer-step size between snapshots
        sigma_obs_omega : scalar (Python float) — ω observation noise.
                          Frozen at the env's `obs_noise_std`; pulled from
                          `model.sigma_obs_omega` (an `eqx.field(static=True)`).
                          Not learnable — taking it as a free parameter let
                          the optimiser absorb model bias into σ_obs and
                          masked V_net / Dw_net errors.
        gp_keys_batch   : dict {M, V, Dw, g, sigma} → (B, S, 2) PRNGKey —
                          same per-(b, s) GP weight sample as used by the
                          rollout, so the variational gradient is coherent
                          across both loss heads.

    Returns:
        dict with:
            pl_loss           : scalar mean
            mean_residual_sq  : scalar — empirical Var(Δω − μΔt) summed over 3
            mean_sigma_phi    : scalar — average σ_φ across (t, b, s)
            sigma_obs_omega   : scalar — the (frozen) σ_obs_ω value
    """
    T_obs, B, _ = batch_x_cat.shape
    sample = next(iter(gp_keys_batch.values()))
    S = sample.shape[1]

    q_t      = batch_x_cat[:-1, :, :9]                           # (T-1, B, 9)
    omega_t  = batch_x_cat[:-1, :, 9:12]                         # (T-1, B, 3)
    u_t      = batch_x_cat[:-1, :, 12:15]                        # (T-1, B, 3)
    delta_om = batch_x_cat[1:, :, 9:12] - omega_t                # (T-1, B, 3)
    Tm1 = T_obs - 1

    # Broadcast the (T-1, B, …) data over a new MC-sample axis S.
    q_tbs      = jnp.broadcast_to(q_t[:, :, None, :],      (Tm1, B, S, 9))
    omega_tbs  = jnp.broadcast_to(omega_t[:, :, None, :],  (Tm1, B, S, 3))
    u_tbs      = jnp.broadcast_to(u_t[:, :, None, :],      (Tm1, B, S, 3))
    dom_tbs    = jnp.broadcast_to(delta_om[:, :, None, :], (Tm1, B, S, 3))

    # gp_keys_batch[name] has shape (B, S, 2). Broadcast along the time
    # axis so the same w-sample drives all transitions within (b, s) —
    # required for variational coherence with the rollout NLL.
    keys_tbs = {k: jnp.broadcast_to(v[None], (Tm1, B, S, 2))
                for k, v in gp_keys_batch.items()}

    sigma_obs_omega_sq = float(sigma_obs_omega) ** 2
    dt_f = jnp.asarray(dt, dtype=q_tbs.dtype)

    def per_step(q, om, u, dom, keys):
        # In inference mode, all subnets use the posterior mean — pass
        # keys=None to drift, and ignore the sigma key (model.sigma also
        # returns the frozen sigma_const regardless of key).
        eff_keys = None if inference_mode else keys
        ksig     = None if inference_mode else keys['sigma']
        mu        = model.drift(q, om, u, keys=eff_keys)         # (3,)
        sigma_phi = model.sigma(q, key=ksig)                     # scalar
        sigma_eff = sigma_phi * sigma_phi * dt_f + 2.0 * sigma_obs_omega_sq
        residual  = dom - mu * dt_f                              # (3,)
        rsq       = jnp.sum(residual * residual)                 # scalar
        nll       = 0.5 * rsq / sigma_eff + 1.5 * jnp.log(sigma_eff)
        return nll, rsq, sigma_phi

    # Triple vmap over (t, b, s).
    vmapped = jax.vmap(jax.vmap(jax.vmap(per_step)))
    nll_arr, rsq_arr, sig_arr = vmapped(
        q_tbs, omega_tbs, u_tbs, dom_tbs, keys_tbs,
    )

    return {
        'pl_loss':          jnp.mean(nll_arr),
        'mean_residual_sq': jnp.mean(rsq_arr),
        'mean_sigma_phi':   jnp.mean(sig_arr),
        'sigma_obs_omega':  jnp.asarray(float(sigma_obs_omega), dtype=jnp.float32),
    }


def kl_per_subnet(model):
    """Per-subnet variational KL for a DissipativeSO3HamSDE model.

    Each GP-flavoured subnet (`M_net`, `V_net`, `Dw_net`, `g_net`,
    `sigma_net`) exposes `weight_kl_loss()` returning the closed-form
    KL(N(μ, σ²) ‖ N(0, 1)) summed over its weights.

    Returns a dict with five subnet keys plus 'total_kl'. All values are
    jnp scalars (so the dict can be used inside a jit'd loss).
    """
    M_kl     = model.M_net.weight_kl_loss()
    V_kl     = model.V_net.weight_kl_loss()
    Dw_kl    = model.Dw_net.weight_kl_loss()
    g_kl     = model.g_net.weight_kl_loss()
    # sigma_net may have been replaced by a frozen `sigma_const` field;
    # in that case it contributes 0 to the variational KL.
    sigma_kl = (model.sigma_net.weight_kl_loss()
                if hasattr(model, 'sigma_net')
                else jnp.zeros((), dtype=jnp.float32))
    return {
        'M_kl':     M_kl,
        'V_kl':     V_kl,
        'Dw_kl':    Dw_kl,
        'g_kl':     g_kl,
        'sigma_kl': sigma_kl,
        'total_kl': M_kl + V_kl + Dw_kl + g_kl + sigma_kl,
    }
