"""JAX helpers used by the SO(3) windy-pendulum SDE port-Hamiltonian model.

  - L2_loss
  - hat / exp_so3 (Rodrigues) for the Lie-group integrator
  - to_pickle / from_pickle (same API as the PyTorch util)

The Gram-Schmidt orthogonalisation utilities that lived here previously have
been removed: with the Lie-group Heun integrator (lie_integrator.py) the
predicted rotation matrix R stays on SO(3) by construction, so projection
back to SO(3) before computing the geodesic loss is no longer needed.
"""
from __future__ import annotations

import pickle

import jax.numpy as jnp


def L2_loss(u, v):
    return jnp.mean((u - v) ** 2)


# ── SO(3) Lie-group helpers ────────────────────────────────────────────

def hat(w):
    """Skew-symmetric matrix (hat map) for w ∈ ℝ³.  ℝ³ → so(3)."""
    return jnp.array([
        [    0., -w[2],  w[1]],
        [ w[2],     0., -w[0]],
        [-w[1],  w[0],    0.],
    ], dtype=w.dtype)


def exp_so3(phi):
    """Matrix exponential on so(3) via Rodrigues' formula.

        exp([φ]_×) = I + (sin θ / θ) [φ]_×
                       + ((1 − cos θ) / θ²) [φ]_×²,    θ = ‖φ‖

    Uses jnp.where + a safe-divide so the gradient at φ = 0 is finite
    (the standard JAX pattern for the small-angle Taylor branch).
    """
    theta_sq = jnp.dot(phi, phi)
    small = theta_sq < 1e-20
    # Replace theta with 1.0 in the branch we won't use, so we never sqrt(0)
    # or divide-by-0 inside the gradient graph.
    theta_safe = jnp.sqrt(jnp.where(small, 1.0, theta_sq))

    A_full = jnp.sin(theta_safe) / theta_safe
    B_full = (1.0 - jnp.cos(theta_safe)) / (theta_safe * theta_safe)
    A_taylor = 1.0 - theta_sq / 6.0
    B_taylor = 0.5 - theta_sq / 24.0

    A = jnp.where(small, A_taylor, A_full)
    B = jnp.where(small, B_taylor, B_full)

    Phi = hat(phi)
    return jnp.eye(3, dtype=phi.dtype) + A * Phi + B * (Phi @ Phi)


# ── Pickle helpers (same API as src/utils/ode_utils.py) ────────────────

def to_pickle(thing, path, protocol=None):
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle,
                    protocol=pickle.HIGHEST_PROTOCOL if protocol is None else protocol)


def from_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)
