"""Unstructured neural SDE on (R, ω) for the 3D windy pendulum (JAX).

This is the SDE analogue of the reference unstructured SO(3) Neural ODE

    f_net(q, ω) -> (q̇, ω̇) ∈ ℝ¹²       (one MLP, no port-Hamiltonian structure)

with an added diffusion head

    sigma_net(q, ω) -> ℝ³               (softplus → strictly positive scale)

The full SDE (Stratonovich form) is

    dq = f_net(q, ω)[:9]  · dt
    dω = f_net(q, ω)[9:]  · dt + diag(σ(q, ω)) ∘ dW       dW ∈ ℝ³, var = h

Control u is held constant across an integration window, like the PH variants.
It is concatenated with (q, ω) so the network sees it as part of the input.

The integrator is the classical **Stratonovich Heun** predictor–corrector on
ℝ¹² (no SO(3) projection — the state R drifts off the manifold, matching the
unstructured baseline's design). Same dW is reused in the predictor and
corrector stages so the scheme converges to the Stratonovich solution.

The model exposes:

    drift(x, u)              -> (q̇, ω̇)        ∈ ℝ¹²
    diffusion(x, u)          -> diag entries σ ∈ ℝ³  (acts on ω only)
    step(x, u, h, dW)        -> next state          ∈ ℝ¹²

`x` is the 12-dim observation (R.flatten() ‖ ω) used everywhere else in
this codebase.
"""
from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import equinox as eqx

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, '../../../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.JAX.neural_networks import MLP


class NeuralSO3SDE(eqx.Module):
    """Unstructured neural SDE on ℝ¹² (state = R.flatten ‖ ω).

    Single-sample Equinox module — vmap externally for batches.

    Drift and diffusion both consume the augmented input (q, ω, u) ∈ ℝ¹⁵
    so the network can condition on the control.  The drift outputs the
    full ℝ¹² derivative (q̇ ‖ ω̇).  The diffusion outputs three positive
    scales applied diagonally to a 3-D Wiener increment on ω only.
    """
    f_net:     MLP
    sigma_net: MLP

    rotmatdim: int = eqx.field(static=True)
    angveldim: int = eqx.field(static=True)
    u_dim:     int = eqx.field(static=True)

    def __init__(self, *, key, u_dim: int = 3, hidden_dim: int = 500,
                 init_gain: float = 1.0):
        self.rotmatdim = 9
        self.angveldim = 3
        self.u_dim = int(u_dim)

        kf, ks = jax.random.split(key, 2)
        in_dim  = self.rotmatdim + self.angveldim + self.u_dim     # (R, ω, u)
        out_dim = self.rotmatdim + self.angveldim                  # (q̇, ω̇)
        self.f_net = MLP(in_dim, hidden_dim, out_dim,
                         init_gain=init_gain, key=kf)
        # Diffusion outputs 3 scalars (one per ω component).  softplus is
        # applied in `.diffusion(...)` to enforce σ ≥ 0.
        self.sigma_net = MLP(in_dim, hidden_dim, self.angveldim,
                             init_gain=init_gain, key=ks)

    # ── Network heads ────────────────────────────────────────────────────

    def drift(self, x, u):
        """f(x, u) ∈ ℝ¹² — deterministic time derivative (q̇ ‖ ω̇)."""
        z = jnp.concatenate([x, u])
        return self.f_net(z)

    def diffusion(self, x, u):
        """σ(x, u) ∈ ℝ³ (≥ 0) — diagonal diffusion scale on ω."""
        z = jnp.concatenate([x, u])
        return jax.nn.softplus(self.sigma_net(z))

    # ── Stratonovich Heun substep ───────────────────────────────────────

    def _stoch_pad(self, sigma, dW):
        """Pad the diffusion-on-ω increment up to ℝ¹² with zeros on q."""
        stoch_omega = sigma * dW                            # (3,)
        return jnp.concatenate(
            [jnp.zeros(self.rotmatdim, dtype=stoch_omega.dtype), stoch_omega]
        )

    def step(self, x, u, h, dW):
        """One Stratonovich Heun (predictor–corrector) substep on ℝ¹².

        Stage 1 — evaluate drift / diffusion at the current state:
            f₁ = f_θ(x, u),     g₁ = σ_θ(x, u) ⊙ dW    (padded to ℝ¹²)

        Stage 2 — Euler predictor with the SAME Wiener increment:
            x_pred = x + f₁·h + g₁

        Stage 3 — re-evaluate at the predicted state, dW reused:
            f₂ = f_θ(x_pred, u),    g₂ = σ_θ(x_pred, u) ⊙ dW

        Stage 4 — corrector (average drift + average diffusion):
            x_new = x + ½(f₁ + f₂)·h + ½(g₁ + g₂)

        Reusing dW between stages 1 and 3 is what makes this converge
        to the Stratonovich (rather than Itô) solution.

        x  : (12,) state (R.flatten ‖ ω)
        u  : (u_dim,)
        h  : substep size
        dW : (3,) Wiener increment, *already* scaled so var(dW) = h.
        """
        f1     = self.drift(x, u)                           # (12,)
        sigma1 = self.diffusion(x, u)                       # (3,)
        g1     = self._stoch_pad(sigma1, dW)                # (12,)

        x_pred = x + f1 * h + g1

        f2     = self.drift(x_pred, u)                      # (12,)
        sigma2 = self.diffusion(x_pred, u)                  # (3,)
        g2     = self._stoch_pad(sigma2, dW)                # (12,)

        return x + 0.5 * (f1 + f2) * h + 0.5 * (g1 + g2)

    # ── Trajectory rollout ──────────────────────────────────────────────

    def rollout(self, x0, u, h, dW_per_outer):
        """Roll the SDE forward.  External I/O matches `lie_heun_sde_rollout`.

        x0           : (12,) initial state (R.flatten ‖ ω)
        u            : (u_dim,) control held constant across the window
        h            : substep size
        dW_per_outer : (n_outer, n_substeps, 3) Wiener increments scaled
                       so var(dW) = h.

        Returns:
            traj : (n_outer + 1, 12) — x0 prepended to the n_outer post-step
                   states.
        """
        def inner_step(x, dW):
            return self.step(x, u, h, dW), None

        def outer_step(x, dW_outer):
            x_new, _ = jax.lax.scan(inner_step, x, dW_outer)
            return x_new, x_new

        _, x_outer = jax.lax.scan(outer_step, x0, dW_per_outer)
        return jnp.concatenate([x0[None], x_outer], axis=0)
