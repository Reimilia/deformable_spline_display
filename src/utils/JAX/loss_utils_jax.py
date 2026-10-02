"""fp32-stable geodesic loss on SO(3) — JAX port.

ph_nn_ode_fp32/loss_utils.py orthogonalised q via Gram-Schmidt before the
geodesic distance, because the ODE output drifted off SO(3).  With the
Lie-group Heun integrator (lie_integrator.py), R is on SO(3) by construction,
so the orthogonalisation step is removed: q is reshaped to (3, 3) directly.

Geodesic distance on SO(3) is computed via the **two-argument atan2 form**

    θ = arctan2( ‖vee((M − Mᵀ)/2)‖ , (tr M − 1)/2 )       M = R₁ R₂ᵀ

which uses both sin θ (from the skew part of M) and cos θ (from the trace).
This is strictly better than the previous arccos((tr M − 1)/2) form:

  • No clipping needed — atan2 is well-defined for arguments slightly past
    ±1 caused by fp32 round-off.
  • Full ulp-accurate precision for small θ; arccos near 1 lost ~6 decimal
    digits and floored our geodesic at ~1e-3 rad in fp32.

The residual gradient singularity at θ = π (antipodal point — the SO(3)
manifold's diameter, where the geodesic axis is ambiguous) is harmless for
training: the squared loss θ² does not visit that point in continuous
optimisation, and a `jnp.where`-style safe-sqrt keeps things finite even
if ‖vee‖ collapses to zero.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .ode_utils_jax import L2_loss


def compute_geodesic_distance_from_two_matrices_safe(m1, m2):
    """m1, m2: (B, 3, 3). Returns θ ∈ [0, π] per batch, fp32-safe.

    Uses θ = arctan2(‖vee((M − Mᵀ)/2)‖, (tr M − 1)/2)  with M = R₁ R₂ᵀ.
    No clipping needed and full small-angle precision (vs the older
    arccos((tr M − 1)/2) form which floored at ~1e-3 rad in fp32).
    """
    M = jnp.matmul(m1, jnp.transpose(m2, (0, 2, 1)))                    # (B, 3, 3)

    # cos θ = (tr M − 1) / 2
    cos_theta = (M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2] - 1.0) / 2.0      # (B,)

    # vee((M − Mᵀ)/2) = sin(θ) · axis_unit   ⇒   ‖vee‖ = |sin θ|.
    sin_axis = jnp.stack([
        (M[:, 2, 1] - M[:, 1, 2]) * 0.5,
        (M[:, 0, 2] - M[:, 2, 0]) * 0.5,
        (M[:, 1, 0] - M[:, 0, 1]) * 0.5,
    ], axis=-1)                                                          # (B, 3)

    # Safe ‖·‖₂ via the **double-where idiom**: a single inside-out
    # `jnp.where(... 1.0)` is not enough because reverse-mode AD still
    # propagates ∂√x/∂x = 1/(2√x) through the *unselected* branch, so a
    # zero input still emits a NaN gradient. By substituting a safe
    # placeholder *before* sqrt and zeroing the gradient *after*, both
    # forward and backward are finite at θ = 0 (correct subgradient is 0,
    # since θ² has a flat minimum there).
    sin_norm_sq = jnp.sum(sin_axis * sin_axis, axis=-1)                  # (B,)
    nonzero = sin_norm_sq > 0.0
    sin_norm_sq_safe = jnp.where(nonzero, sin_norm_sq, 1.0)
    sin_theta = jnp.where(nonzero, jnp.sqrt(sin_norm_sq_safe), 0.0)

    return jnp.arctan2(sin_theta, cos_theta)                             # (B,) ∈ [0, π]


def compute_geodesic_loss_safe(gt_R, pred_R):
    theta = compute_geodesic_distance_from_two_matrices_safe(gt_R, pred_R)
    theta_sq = theta ** 2
    return jnp.mean(theta_sq), theta_sq


def rotmat_L2_geodesic_loss_safe(u, u_hat, split=(9, 3, 3)):
    """Per-window loss over u, u_hat ∈ ℝ^(T, B, 15).

    Returns (total, l2, geo). split = (rotmat_dim, angvel_dim, u_dim).
    Assumes q (= u[..., :9]) is *already* on SO(3) — no orthogonalisation.
    """
    rd, ad, ud = split
    q_hat     = u_hat[..., :rd]
    qdot_hat  = u_hat[..., rd:rd + ad]
    uctrl_hat = u_hat[..., rd + ad:rd + ad + ud]
    q     = u[..., :rd]
    qdot  = u[..., rd:rd + ad]
    uctrl = u[..., rd + ad:rd + ad + ud]

    qdot_u_hat = jnp.concatenate([qdot_hat, uctrl_hat], axis=-1)
    qdot_u     = jnp.concatenate([qdot,     uctrl    ], axis=-1)
    qdot_u_hat = qdot_u_hat.reshape(-1, qdot_u_hat.shape[-1])
    qdot_u     = qdot_u.reshape(-1,     qdot_u.shape[-1])
    l2_loss = L2_loss(qdot_u, qdot_u_hat)

    R_hat = q_hat.reshape(-1, 3, 3)
    R     = q.reshape(-1, 3, 3)
    geo_loss, _ = compute_geodesic_loss_safe(R, R_hat)

    return l2_loss + geo_loss, l2_loss, geo_loss


def _rotmat_L2_geodesic_diff_safe(u, u_hat, split=(9, 3, 3)):
    """Per-sample (no T-axis) per-trajectory diffs. u, u_hat ∈ ℝ^(B, 15)."""
    rd, ad, ud = split
    q_hat     = u_hat[..., :rd]
    qdot_hat  = u_hat[..., rd:rd + ad]
    uctrl_hat = u_hat[..., rd + ad:rd + ad + ud]
    q     = u[..., :rd]
    qdot  = u[..., rd:rd + ad]
    uctrl = u[..., rd + ad:rd + ad + ud]

    qdot_u_hat = jnp.concatenate([qdot_hat, uctrl_hat], axis=-1)
    qdot_u     = jnp.concatenate([qdot,     uctrl    ], axis=-1)
    l2_diff = jnp.sum((qdot_u - qdot_u_hat) ** 2, axis=-1)

    R_hat = q_hat.reshape(-1, 3, 3)
    R     = q.reshape(-1, 3, 3)
    _, geo_diff = compute_geodesic_loss_safe(R, R_hat)

    return l2_diff + geo_diff, l2_diff, geo_diff


def traj_rotmat_L2_geodesic_loss_safe(traj, traj_hat, split=(9, 3, 3)):
    """Same shape as the PyTorch traj_rotmat_L2_geodesic_loss_safe.

    traj, traj_hat: ℝ^(T, B, 15). Returns (total, l2, geo) each ∈ ℝ^(T, B).
    """
    return jax.vmap(_rotmat_L2_geodesic_diff_safe, in_axes=(0, 0, None))(
        traj, traj_hat, split
    )
