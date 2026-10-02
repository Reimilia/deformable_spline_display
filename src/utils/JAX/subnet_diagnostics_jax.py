"""JAX port of src/utils/subnet_diagnostics.py.

Per-subnetwork physics-target MSE diagnostics. Evaluates the trained
M_net, V_net, Dw_net, g_net on the model's own predicted q values and
compares to the true physics for the spherical 3D pendulum.
"""
from __future__ import annotations

from typing import Dict, Union

import jax
import jax.numpy as jnp
import numpy as np


def subnet_physics_mse(
    model,
    x_hat,
    *,
    m: float = 1.0,
    l: float = 1.0,
    g: float = 9.81,
    friction_coeff: Union[float, tuple, list, np.ndarray] = 0.5,
    varying_friction: bool = False,
) -> Dict[str, float]:
    """Compute mean MSE between each subnetwork's outputs and the true physics.

    Args:
        model: a DissipativeSO3HamNODE-like Equinox module exposing M_net,
               V_net, Dw_net, g_net (each a single-sample callable).
        x_hat: diffeqsolve output, shape (T, B, 15) — the initial frame is
               dropped to compare only on predicted timesteps.
        m, l, g, friction_coeff, varying_friction: env constants used to build
               the analytical targets (M = m·l²·I₃, V = m·g·l·R[2,2], etc).

    Returns:
        dict with python floats: {'M_loss', 'V_loss', 'Dw_loss', 'g_loss'}.
    """
    pred = x_hat[1:]                                    # (T-1, B, 15)
    T1, B, _ = pred.shape
    flat = pred.reshape(T1 * B, 15)
    q     = flat[:, :9]                                 # (N, 9)
    q_dot = flat[:, 9:12]                               # (N, 3)
    N = q.shape[0]

    # vmap each single-sample subnet over the N rows.
    M_pred  = jax.vmap(model.M_net)(q)                  # (N, 3, 3)
    V_pred  = jax.vmap(model.V_net)(q).squeeze(-1)      # (N,)
    # Dw_net is (q, p) → (3, 3) when the architecture exposes the
    # |p|-dependent dissipation; older models exposed (q,) → (3, 3).
    # p = M·ω = solve(M⁻¹(q), q_dot) using the model's own M_net.
    p_diag = jnp.linalg.solve(M_pred, q_dot[..., None])[..., 0]   # (N, 3)
    try:
        Dw_pred = jax.vmap(model.Dw_net)(q, p_diag)     # (N, 3, 3)
    except TypeError:
        Dw_pred = jax.vmap(model.Dw_net)(q)             # legacy single-arg
    g_pred  = jax.vmap(model.g_net)(q)                  # (N, 3, 3)

    I3 = jnp.eye(3, dtype=q.dtype)
    # M_net returns M⁻¹ (paper convention), so the analytical target for
    # a unit-mass spherical pendulum is M⁻¹ = (1/(m·l²))·I₃ — NOT M = m·l²·I₃.
    # For m=l=1 these are numerically identical so the bug was invisible,
    # but for non-unit physics constants the comparison was inverted.
    M_tgt = (1.0 / (m * l * l)) * jnp.broadcast_to(I3, (N, 3, 3))

    # V is gauge-free up to a constant — center both before MSE.
    V_tgt_raw = (m * g * l) * q[:, 8]
    V_pred_c  = V_pred - V_pred.mean()
    V_tgt_c   = V_tgt_raw - V_tgt_raw.mean()

    fc = jnp.asarray(friction_coeff, dtype=q.dtype)
    if fc.ndim == 0:
        fc = jnp.broadcast_to(fc, (3,))
    Dw_diag = jnp.broadcast_to(jnp.diag(fc), (N, 3, 3))
    if varying_friction:
        height_term = 0.5 * (1.0 - q[:, 8])
        speed_term  = jnp.tanh(jnp.linalg.norm(q_dot, axis=-1))
        mult = (1.0 + 0.5 * height_term + 0.5 * speed_term).reshape(N, 1, 1)
        Dw_tgt = Dw_diag * mult
    else:
        Dw_tgt = Dw_diag

    g_tgt = jnp.broadcast_to(I3, (N, 3, 3))

    M_loss  = jnp.mean((M_pred  - M_tgt ) ** 2)
    V_loss  = jnp.mean((V_pred_c - V_tgt_c) ** 2)
    Dw_loss = jnp.mean((Dw_pred - Dw_tgt) ** 2)
    g_loss  = jnp.mean((g_pred  - g_tgt ) ** 2)

    return {
        'M_loss':  float(M_loss),
        'V_loss':  float(V_loss),
        'Dw_loss': float(Dw_loss),
        'g_loss':  float(g_loss),
    }
