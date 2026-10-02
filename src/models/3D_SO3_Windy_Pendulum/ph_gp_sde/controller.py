"""SO(3) Energy-Casimir controller for the windy 3D pendulum plant.

Implements the closed-form law derived in §7 of the design doc, with the
sign settled by `verify_control.py`:

    u_p   =  +k_c · log(R^T R_d)  -  d_inj · ω
    u     =  G^+ · u_p

where G is either the identity (env's true input map; default) or the
learned g_theta(R) (toggled by `use_trained_g`).

`verify_control.py` showed the trained g_theta has collapsed to a rank-1
map (the training u-grid only excited the all-ones diagonal of u-space),
so `use_trained_g=True` is currently for diagnostic comparison only — the
ground-truth g = I path is the one to use until g is retrained with
random u.

The controller exposes a numpy-only `act(R, omega) -> u` so it can be
dropped straight into the env's step loop without any JAX runtime cost
on the default (g = I) path.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, '../../../..'))
for _p in (PROJECT_ROOT, THIS_FILE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from envs.windy_pendulum_3d import _log_so3   # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Energy-Casimir controller
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ControllerConfig:
    """Static configuration for `EnergyCasimirController`.

    The four-variant ablation is parametrised by `alpha_D` and `alpha_V`:

        u(α_D, α_V) = k_c · log(R^T R_d) − d_inj · ω
                      + α_D · D_θ(R) · ω
                      + α_V · Σᵢ Rᵢ × ∇V_θ(q)ᵢ

    matching the unified form in the design doc:
      (0, 0) → Variant 1: pure Casimir baseline
      (1, 0) → Variant 2: Casimir + friction cancellation (uses D_θ)
      (0, 1) → Variant 3: Casimir + gravity cancellation (uses V_θ)
      (1, 1) → Variant 4: full IDA-PBC-like (uses both)

    Any α > 0 requires `model` to be passed at construction; the model's
    V_θ / D_θ subnets supply the cancellation terms. The controller stays
    pure-numpy when α_D = α_V = 0.
    """
    R_d: np.ndarray                        # 3×3 target rotation
    k_c: float = 30.0                      # spring stiffness (≈ 3·m·g·l for v1; can be << for v3/v4)
    d_inj: float = 1.0                     # damping injection
    alpha_D: float = 0.0                   # friction-cancel weight (uses D_θ)
    alpha_V: float = 0.0                   # gravity-cancel weight (uses V_θ)
    use_trained_g: bool = False            # False → G = I (env-true)
    antipode_eps: float = 1e-3             # cut-locus deadband
    g_pinv_rcond: float = 1e-2             # rcond for np.linalg.pinv


class EnergyCasimirController:
    """Stateless controller. `act(R, omega)` returns the body-frame torque.

    When `use_trained_g=False` (default), G = I and no JAX code runs at
    inference time — pure numpy, ~µs per call.

    When `use_trained_g=True`, the constructor jit-compiles g_theta(q) and
    each call evaluates it at the current R. The result is then pinv'd
    via numpy (rcond clipped — the trained g is rank-deficient, so pinv
    without an rcond floor would amplify noise wildly).
    """

    def __init__(self, cfg: ControllerConfig,
                 model: Optional[object] = None):
        self.cfg = cfg
        self.R_d = np.asarray(cfg.R_d, dtype=np.float64).reshape(3, 3)

        self._g_call = None
        self._cancel_call = None
        self._jnp = None

        needs_jax = cfg.use_trained_g or cfg.alpha_D > 0.0 or cfg.alpha_V > 0.0
        if needs_jax and model is None:
            raise ValueError(
                "Controller needs a model when use_trained_g=True or "
                "alpha_D>0 or alpha_V>0; got model=None.")

        if needs_jax:
            # Defer JAX import so that the default (pure-Casimir, g=I) path
            # doesn't pay the JAX startup cost.
            import jax
            import jax.numpy as jnp
            self._jnp = jnp

            if cfg.use_trained_g:
                self._g_call = jax.jit(
                    lambda q: model.g_net(q, inference_mode=True))

            if cfg.alpha_D > 0.0 or cfg.alpha_V > 0.0:
                # Single jitted helper that returns BOTH cancellation
                # building blocks; the act() method scales by α at runtime
                # so the same compiled artefact serves all variants.
                #
                #   D_θ_omega   = D_θ(q) · ω           ∈ ℝ³  (friction term)
                #   grav_V      = Σᵢ Rᵢ × ∇V_θ(q)ᵢ    ∈ ℝ³  (gravity term)
                #
                # Note: grav_V uses V_θ ALONE (not full H), the principled
                # gravity-cancel definition. For our isotropic M this
                # coincides with the model's internal `grav` variable
                # (network.py:247) but the V-only form is correct under
                # a future anisotropic M too.
                @jax.jit
                def _cancel(q, omega):
                    Dw = model.Dw_net(q, inference_mode=True)
                    Dw_omega = Dw @ omega
                    def V_scalar(q_):
                        return model.V_net(q_, inference_mode=True)[0]
                    grad_V = jax.grad(V_scalar)(q)               # (9,)
                    R_3x3   = q.reshape(3, 3)
                    gV_3x3  = grad_V.reshape(3, 3)
                    grav_V  = jnp.sum(jnp.cross(R_3x3, gV_3x3,
                                                 axis=-1), axis=0)
                    return Dw_omega, grav_V
                self._cancel_call = _cancel

    # ─────────────────────────────────────────────────────────────────
    # Per-step action
    # ─────────────────────────────────────────────────────────────────
    def act(self, R: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Return body-frame torque u ∈ ℝ³."""
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        omega = np.asarray(omega, dtype=np.float64).reshape(3)

        # Tangent-space error to target.
        # log(R^T R_d) is the body-frame Lie algebra vector that, when
        # exponentiated, takes R to R_d. The sign convention was settled
        # empirically in verify_control.py — `+k_c · log(...)` drives R → R_d.
        err = self._safe_log(R.T @ self.R_d)         # (3,)

        # Energy-shaping torque + damping injection in body frame.
        u_p = self.cfg.k_c * err - self.cfg.d_inj * omega

        # ── Cancellation terms (Variants 2/3/4) ─────────────────────────
        # The network's `dp` uses `+grav` (canonical body-frame Euler:
        # dp/dt = τ - ω×p, where τ here is the body-frame gravity torque
        # = +Σᵢ Rᵢ × ∇V). To cancel the plant's gravity contribution we
        # therefore add `-grav_V_θ` to u. Friction is `-D·ω` in dp, so we
        # add `+D_θ·ω`. Empirically verified by `verify_variants.py`.
        if self._cancel_call is not None:
            jnp = self._jnp
            q = jnp.asarray(R.reshape(-1), dtype=jnp.float32)
            om = jnp.asarray(omega, dtype=jnp.float32)
            Dw_om, grav_V = self._cancel_call(q, om)
            u_p = u_p + (self.cfg.alpha_D * np.asarray(Dw_om, np.float64)
                         - self.cfg.alpha_V * np.asarray(grav_V, np.float64))

        # Map through input matrix.
        if self.cfg.use_trained_g:
            q = self._jnp.asarray(R.reshape(-1), dtype=self._jnp.float32)
            G = np.asarray(self._g_call(q), dtype=np.float64)
            G_pinv = np.linalg.pinv(G, rcond=self.cfg.g_pinv_rcond)
            u = G_pinv @ u_p
        else:
            u = u_p

        return u.astype(np.float64)

    # ─────────────────────────────────────────────────────────────────
    # SO(3) log with antipode deadband
    # ─────────────────────────────────────────────────────────────────
    def _safe_log(self, M: np.ndarray) -> np.ndarray:
        """log_SO(3)(M) with a deadband near the cut locus.

        At trace(M) = -1 the rotation angle is exactly π and the log is
        defined only up to sign; tiny perturbations flip the chosen axis.
        We detect that regime and return a small *fixed* perturbation
        torque to break the symmetry; once the trajectory leaves the
        antipodal sphere the regular log takes over.
        """
        cos_theta = 0.5 * (np.trace(M) - 1.0)
        if cos_theta < -1.0 + self.cfg.antipode_eps:
            # On the cut locus. _log_so3 picks *some* axis; we use it but
            # add a small e_z bias so we never get stuck on the saddle.
            phi = _log_so3(M)
            phi = phi + 1e-3 * np.array([0.0, 0.0, 1.0])
            return phi
        return _log_so3(M)


# ─────────────────────────────────────────────────────────────────────
# Closed-loop energy (for diagnostics / Lyapunov plots)
# ─────────────────────────────────────────────────────────────────────

def closed_loop_energy(R: np.ndarray, omega: np.ndarray,
                       R_d: np.ndarray, k_c: float,
                       m: float = 1.0, l: float = 1.0, g: float = 9.81
                       ) -> float:
    """H_cl(R, ω) = ½ ωᵀ M ω + V_p(R) + (k_c/2) · ‖log(R^T R_d)‖².

    Uses the analytical M = m·l²·I and V_p(R) = m·g·l·(R e_z)·e_z = m·g·l·R[2,2]
    of the spherical pendulum (matches the env's gravity model exactly).
    Useful as a Lyapunov-function proxy when validating the controller in
    the env (we don't have the trained H here — it would require a JAX
    forward pass and shouldn't enter the rollout loop).
    """
    R = np.asarray(R).reshape(3, 3)
    omega = np.asarray(omega).reshape(3)

    KE = 0.5 * (m * l * l) * float(np.dot(omega, omega))
    V_p = m * g * l * float(R[2, 2])
    phi = _log_so3(R.T @ R_d)
    spring = 0.5 * k_c * float(np.dot(phi, phi))
    return KE + V_p + spring


def closed_loop_energy_at_target(R_d: np.ndarray,
                                 m: float = 1.0, l: float = 1.0, g: float = 9.81
                                 ) -> float:
    """H_cl(R_d, 0) — the value all trajectories should converge to (in mean)."""
    return closed_loop_energy(R_d, np.zeros(3), R_d,
                              k_c=0.0, m=m, l=l, g=g)
