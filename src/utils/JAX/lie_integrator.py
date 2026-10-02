"""Stratonovich Heun integrator on SO(3) × ℝ³ in (q, p) form.

Mirrors the env's Lie-group Heun integrator
(envs/windy_pendulum_3d.py::windy_pendulum_3d._lie_heun_step) geometrically,
but the substep state carried inside the scan is angular **momentum** p
rather than angular velocity ω. ω is reconstructed only when needed for
the SO(3) Lie-step (φ = ω·h) and at outer-step output (so downstream
losses that consume (q, ω) keep working).

Substep state convention (internal):
    x_qp : ℝ¹² = concat(R.flatten(), p)            row-major like the env's obs

Rollout external convention (unchanged):
    x0   : ℝ¹² = concat(R.flatten(), ω₀)           — observed snapshot at t=0
    traj : (n_outer + 1, 12) = (R.flatten, ω)       — rolled trajectory in ω

Conversion at the boundary is done with the model's M_inv:
    ω → p :  p = solve(M⁻¹(q), ω)        (since M_net returns M⁻¹)
    p → ω :  ω = M⁻¹(q) · p

Each substep expects a pre-sampled Wiener increment dW ∈ ℝ³ already scaled
by sqrt(h) so that var(dW) = h (matches the Stratonovich convention used
by the env: dW = rng.normal(0, sqrt(dt_sub), 3)). Setting dW = 0 collapses
the integrator to the deterministic Lie-Heun scheme.

The model is expected to expose:

    drift_p(q, p, u)                 -> ṗ_det        ∈ ℝ³
    stochastic_increment_p(q, dW)    -> dp_stoch     ∈ ℝ³
    M_inv(q)                         -> M⁻¹(q)       ∈ ℝ³ˣ³
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .ode_utils_jax import exp_so3


# ── Single substep (q, p) → (q, p) ────────────────────────────────────

def lie_heun_sde_step(model, x_qp, u, h, dW):
    """One Stratonovich Heun substep on SO(3) × ℝ³ in (q, p) form.

      Stage 1 (current state):
        ṗ₁          = drift_p(q, p, u)
        ω₁          = M⁻¹(q) · p
        φ₁          = ω₁ · h
        dp_stoch₁   = stochastic_increment_p(q, dW)         (= σ(q)·τ(R, dW))

      Stage 2 (predictor — Euler on the manifold, same dW reused):
        R_pred      = R · exp([φ₁]_×)
        p_pred      = p + ṗ₁·h + dp_stoch₁

      Stage 3 (re-evaluate at predicted state):
        ṗ₂          = drift_p(q_pred, p_pred, u)
        ω₂          = M⁻¹(q_pred) · p_pred
        φ₂          = ω₂ · h
        dp_stoch₂   = stochastic_increment_p(q_pred, dW)

      Stage 4 (corrector — average in Lie algebra, exponentiate once):
        φ_avg = (φ₁ + φ₂) / 2
        R_new = R · exp([φ_avg]_×)
        p_new = p + (ṗ₁ + ṗ₂)/2 · h + (dp_stoch₁ + dp_stoch₂) / 2
    """
    q = x_qp[:9]
    p = x_qp[9:12]
    R = q.reshape(3, 3)

    p_dot_1     = model.drift_p(q, p, u)
    omega_1     = model.M_inv(q) @ p
    dp_stoch_1  = model.stochastic_increment_p(q, dW)
    phi_1       = omega_1 * h

    R_pred = R @ exp_so3(phi_1)
    q_pred = R_pred.reshape(9)
    p_pred = p + p_dot_1 * h + dp_stoch_1

    p_dot_2     = model.drift_p(q_pred, p_pred, u)
    omega_2     = model.M_inv(q_pred) @ p_pred
    dp_stoch_2  = model.stochastic_increment_p(q_pred, dW)
    phi_2       = omega_2 * h

    phi_avg = 0.5 * (phi_1 + phi_2)
    R_new = R @ exp_so3(phi_avg)
    q_new = R_new.reshape(9)

    p_new = (
        p
        + 0.5 * (p_dot_1 + p_dot_2) * h
        + 0.5 * (dp_stoch_1 + dp_stoch_2)
    )

    return jnp.concatenate([q_new, p_new])


# ── Full rollout — (q, ω) external, (q, p) internal ────────────────────

def lie_heun_sde_rollout(model, x0, u, h, dW_per_outer):
    """Roll out the SDE, returning the state at every outer-step boundary.

    External I/O is unchanged: the rollout takes and emits ω (matches the
    env's observation and the dataset/loss code). Internally the scan state
    is angular momentum p, with ω↔p conversions only at the entry and at
    each outer-step output.

    Args:
        model        : provides .drift_p, .stochastic_increment_p, .M_inv.
        x0           : (12,) initial (R.flatten + ω) — must lie on SO(3).
        u            : (u_dim,) constant control torque (broadcast to every
                       outer step) OR (n_outer, u_dim) time-varying torque,
                       one row per outer step. Within an outer step u is
                       held constant across the n_substeps Lie-Heun substeps,
                       matching the env's `step()` semantics.
        h            : substep size.  outer_step = h * n_substeps.
        dW_per_outer : (n_outer, n_substeps, 3) Wiener increments,
                       *already scaled* by sqrt(h).

    Returns:
        traj : (n_outer + 1, 12) — x0 prepended to the n_outer post-step
               states, all in (R.flatten, ω) form.
    """
    q0 = x0[:9]
    omega_0 = x0[9:12]
    # ω → p :  p = M(q)·ω = solve(M⁻¹(q), ω)
    p0 = jnp.linalg.solve(model.M_inv(q0), omega_0)
    x0_qp = jnp.concatenate([q0, p0])

    n_outer = dW_per_outer.shape[0]
    u = jnp.asarray(u)
    if u.ndim == 1:
        u_per_outer = jnp.broadcast_to(u[None, :], (n_outer,) + u.shape)
    else:
        u_per_outer = u                                # (n_outer, u_dim)

    def outer_step(x_qp, scan_in):
        u_t, dW_outer = scan_in
        def inner_step(carry, dW):
            return lie_heun_sde_step(model, carry, u_t, h, dW), None
        x_qp_new, _ = jax.lax.scan(inner_step, x_qp, dW_outer)
        # p → ω at the boundary so the emitted trajectory is in (q, ω) form.
        q_new = x_qp_new[:9]
        p_new = x_qp_new[9:12]
        omega_new = model.M_inv(q_new) @ p_new
        x_qomega_new = jnp.concatenate([q_new, omega_new])
        return x_qp_new, x_qomega_new

    _, x_outer_qomega = jax.lax.scan(outer_step, x0_qp,
                                     (u_per_outer, dW_per_outer))
    return jnp.concatenate([x0[None], x_outer_qomega], axis=0)


# ── Deterministic Lie-Heun on SO(3) × ℝ³ (no diffusion) ───────────────
#
# These variants do NOT call `model.stochastic_increment_p`. Use them when
# the model has no diffusion term (e.g. ph_gp_ode_v2's deterministic ODE).
# Geometrically identical to the SDE variants with dW ≡ 0, but they don't
# require the model to expose a stochastic_increment interface.

def lie_heun_ode_step(model, x_qp, u, h):
    """One deterministic Heun substep on SO(3) × ℝ³ in (q, p) form.

    Same as `lie_heun_sde_step` but the stochastic increment is absent
    (the model does not need to expose `stochastic_increment_p`).
    """
    q = x_qp[:9]
    p = x_qp[9:12]
    R = q.reshape(3, 3)

    p_dot_1 = model.drift_p(q, p, u)
    omega_1 = model.M_inv(q) @ p
    phi_1   = omega_1 * h

    R_pred = R @ exp_so3(phi_1)
    q_pred = R_pred.reshape(9)
    p_pred = p + p_dot_1 * h

    p_dot_2 = model.drift_p(q_pred, p_pred, u)
    omega_2 = model.M_inv(q_pred) @ p_pred
    phi_2   = omega_2 * h

    phi_avg = 0.5 * (phi_1 + phi_2)
    R_new = R @ exp_so3(phi_avg)
    q_new = R_new.reshape(9)

    p_new = p + 0.5 * (p_dot_1 + p_dot_2) * h
    return jnp.concatenate([q_new, p_new])


def lie_heun_ode_rollout(model, x0, u, h, n_substeps, n_outer):
    """Roll out the deterministic ODE, returning (n_outer + 1, 12) in (q, ω)
    form. External I/O matches `lie_heun_sde_rollout` with dW absent.

    Args:
        model      : provides .drift_p, .M_inv (no stochastic_increment_p needed).
        x0         : (12,) initial (R.flatten + ω) — must lie on SO(3).
        u          : (u_dim,) constant OR (n_outer, u_dim) time-varying
                     control torque (body frame).
        h          : substep size.  outer_step = h * n_substeps.
        n_substeps : substeps per outer step.
        n_outer    : number of outer steps.
    """
    q0 = x0[:9]
    omega_0 = x0[9:12]
    p0 = jnp.linalg.solve(model.M_inv(q0), omega_0)
    x0_qp = jnp.concatenate([q0, p0])

    u = jnp.asarray(u)
    if u.ndim == 1:
        u_per_outer = jnp.broadcast_to(u[None, :], (n_outer,) + u.shape)
    else:
        u_per_outer = u

    def outer_step(x_qp, u_t):
        def inner_step(carry, _):
            return lie_heun_ode_step(model, carry, u_t, h), None
        x_qp_new, _ = jax.lax.scan(inner_step, x_qp,
                                   jnp.arange(n_substeps))
        q_new = x_qp_new[:9]
        p_new = x_qp_new[9:12]
        omega_new = model.M_inv(q_new) @ p_new
        x_qomega_new = jnp.concatenate([q_new, omega_new])
        return x_qp_new, x_qomega_new

    _, x_outer_qomega = jax.lax.scan(outer_step, x0_qp, u_per_outer)
    return jnp.concatenate([x0[None], x_outer_qomega], axis=0)
