"""Dissipative port-Hamiltonian SO(3) **ODE** for the 3D windy pendulum (JAX),
with **GP subnetworks** (random-Fourier-feature variational GPs from
src/utils/JAX/gp_model.py) replacing the earlier MLP/PSD/MatrixNet subnets.

Deterministic ODE structure (state = (q, p), angular momentum):

    dq         = R · ω · dt                (Lie-group integrator; ω = M⁻¹·p)
    dp         = (p × M⁻¹p + Σᵢ Rᵢ × ∂qᵢH − D·M⁻¹p + g·u) · dt

The original SDE diffusion term `R^T · (l · R · e_z × σ(q) · dW)` has been
removed: there is no `sigma_net`, no `stochastic_increment_p`, no
`stochastic_increment`, and the model is fully deterministic.

ω is reconstructed lazily as ω = M⁻¹(q)·p inside the integrator stages (for
the SO(3) exponential) and at the rollout output boundary.

Subnetworks (single-sample Equinox GP modules — vmap externally for batches):
    M_net(q)     PSD_GP_Model    — M⁻¹(q) (paper convention; ε=1.0 fp32 floor)
    V_net(q)     GP_Model→1      — potential energy
    Dw_net(q)    PSD_GP_Model    — dissipation
    g_net(q)     GP_MatrixNet    — control-input coupling

The GP modules accept `(x, key=None, inference_mode=False)`. At dynamics
time we always call them with `inference_mode=True` so the integrator
remains key-free; the GP randomness only enters through the variational
ELBO loss at training time (KL term, plus weight sampling if you want a
stochastic ELBO — exposed via .kl_loss()).

Public model API:
    drift_p(q, p, u)                 -> ṗ_det        ∈ ℝ³   (used by integrator)
    M_inv(q)                         -> M⁻¹(q)       ∈ ℝ³ˣ³ (used by integrator)
    drift(q, q_dot, u)               -> ω̇_det        ∈ ℝ³   (compat wrapper)
    kl_loss()                        -> Σ KL(GP_i)   ∈ ℝ           (for ELBO)
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

from src.utils.JAX.gp_model import GP_Model, PSD_GP_Model, GP_MatrixNet


class FixedInverseMass(eqx.Module):
    """Drop-in replacement for the M_net subnetwork that returns the
    constant ground-truth inverse mass (1/(m·l²)) · I₃. No learnable
    parameters; KL contribution is identically zero.

    Mirrors `ph_nn_ode_v2.network.FixedInverseMass`. The signature
    `(q, key=None, inference_mode=False)` matches `PSD_GP_Model` so it
    can be swapped in at construction without changing call sites.
    """
    m: float = eqx.field(static=True)
    l: float = eqx.field(static=True)

    def __init__(self, m: float = 1.0, l: float = 1.0):
        self.m = float(m)
        self.l = float(l)

    def __call__(self, q, key=None, inference_mode=False):
        scale = 1.0 / (self.m * self.l * self.l)
        return scale * jnp.eye(3, dtype=q.dtype)

    def weight_kl_loss(self):
        return jnp.zeros((), dtype=jnp.float32)


class DissipativeSO3HamODE(eqx.Module):
    """Port-Hamiltonian ODE on SO(3) × ℝ³ with GP subnetworks.

    State (12-dim, used by the integrator):
        x = concat(R.flatten(), ω)    R row-major like envs/windy_pendulum_3d
    Control u ∈ ℝ³ is passed separately (held constant across an integrator step).

    Trainable observation-noise scales `log_sigma_R`, `log_sigma_omega` are
    kept as fields on the model so they're picked up automatically by
    `eqx.filter(model, eqx.is_array)` in the optimiser. They feed the
    Gaussian-NLL likelihood; see `src/utils/JAX/elbo_loss_jax.py`.

    `sigma_obs_omega` is a frozen scalar set from the dataset's `obs_noise_std`,
    used inside per-increment pseudo-likelihoods and noise-aware auxiliary
    losses (whitening of central-difference targets).
    """
    M_net:     eqx.Module        # PSD_GP_Model, or FixedInverseMass when fix_M=True
    V_net:     GP_Model
    Dw_net:    PSD_GP_Model
    g_net:     eqx.Module        # GP_Model if u_dim==1 else GP_MatrixNet

    log_sigma_R:     jnp.ndarray   # scalar — rollout-NLL noise on rotation (geodesic)
    log_sigma_omega: jnp.ndarray   # scalar — rollout-NLL noise on angular velocity

    # Per-snapshot ω observation noise. Frozen `eqx.field(static=True)`,
    # set from the dataset's `obs_noise_std`.
    sigma_obs_omega: float = eqx.field(static=True)

    rotmatdim: int   = eqx.field(static=True)
    angveldim: int   = eqx.field(static=True)
    u_dim:     int   = eqx.field(static=True)
    friction:  bool  = eqx.field(static=True)
    l:         float = eqx.field(static=True)

    def __init__(self, *, key, u_dim: int = 3, init_gain: float = 0.5,
                 friction: bool = True, hidden_dim: int = 20, l: float = 1.0,
                 init_sigma_R: float = 0.1, init_sigma_omega: float = 0.1,
                 init_sigma_obs_omega: float = 0.5,
                 fix_M: bool = False, fix_M_m: float = 1.0,
                 fix_M_l: float = 1.0):
        # init_gain is kept in the signature for API parity with the MLP
        # version, but the GP modules don't use it (their initialisation is
        # spectral / Bayesian and parameterised by ell/nu/m_max).
        del init_gain
        self.rotmatdim = 9
        self.angveldim = 3
        self.u_dim = int(u_dim)
        self.friction = bool(friction)
        self.l = float(l)

        self.log_sigma_R     = jnp.log(jnp.asarray(init_sigma_R,     dtype=jnp.float32))
        self.log_sigma_omega = jnp.log(jnp.asarray(init_sigma_omega, dtype=jnp.float32))
        self.sigma_obs_omega = float(init_sigma_obs_omega)

        kM, kV, kD, kg = jax.random.split(key, 4)

        # M_net : PSD with epsilon=1.0 (fp32 stability floor on diag), OR
        # a constant FixedInverseMass when fix_M=True. The fixed module has
        # no parameters → optax's eqx.filter(..., eqx.is_array) won't
        # register anything for it, and weight_kl_loss() returns 0.
        if fix_M:
            self.M_net = FixedInverseMass(m=fix_M_m, l=fix_M_l)
        else:
            self.M_net = PSD_GP_Model(
                key=kM, input_dim=self.rotmatdim, hidden_dim=hidden_dim,
                diag_dim=self.angveldim, epsilon=1.0,
            )
        # V_net : scalar potential energy
        self.V_net = GP_Model(
            key=kV, input_dim=self.rotmatdim, output_dim=1,
            n_matern_features=hidden_dim,
        )
        # Dw_net : PSD with epsilon=0.0. Input is (q, p) ∈ ℝ¹² so the
        # network can capture state-dependent friction whose magnitude
        # depends on |p| / |ω| (e.g. the env's tanh(|ω|) term in
        # `varying_friction`). Using p (the canonical pH momentum) as
        # the second input rather than ω = M⁻¹·p keeps the input pair
        # (q, p) consistent with the rest of the state.
        self.Dw_net = PSD_GP_Model(
            key=kD, input_dim=self.rotmatdim + self.angveldim,
            hidden_dim=hidden_dim,
            diag_dim=self.angveldim, epsilon=0.0,
        )
        # g_net : (3, u_dim) control coupling
        if self.u_dim == 1:
            self.g_net = GP_Model(
                key=kg, input_dim=self.rotmatdim, output_dim=self.angveldim,
                n_matern_features=hidden_dim,
            )
        else:
            self.g_net = GP_MatrixNet(
                key=kg, input_dim=self.rotmatdim, hidden_dim=hidden_dim,
                output_dim=self.angveldim * self.u_dim,
                shape=(self.angveldim, self.u_dim),
            )

    # ── Key-aware subnet wrappers ────────────────────────────────────────
    # `key=None` → deterministic posterior-mean call (inference_mode=True).
    # `key=<PRNGKey>` → reparameterised weight sample (inference_mode=False).

    def _M_call(self, q, key=None):
        return self.M_net(q, key=key, inference_mode=(key is None))

    def _V_call(self, q, key=None):
        return self.V_net(q, key=key, inference_mode=(key is None))

    def _Dw_call(self, q, p, key=None):
        """Dissipation tensor D(q, p). Concatenated input lets the net
        capture the |p|/|ω|-dependent term of the env's varying friction."""
        x = jnp.concatenate([q, p])
        return self.Dw_net(x, key=key, inference_mode=(key is None))

    def _g_call(self, q, key=None):
        return self.g_net(q, key=key, inference_mode=(key is None))

    def M_inv(self, q, keys=None):
        """M⁻¹(q) ∈ ℝ³ˣ³ — exposed for the (q, p) integrator's ω = M⁻¹·p step."""
        kM = None if keys is None else keys.get('M')
        return self._M_call(q, key=kM)

    # ── Deterministic drift (ṗ) — port-Hamiltonian momentum derivative ─

    def drift_p(self, q, p, u, keys=None):
        """Deterministic momentum derivative ṗ ∈ ℝ³.

            H      = ½ pᵀ M⁻¹(q) p + V(q)
            dHdq   = ∂H/∂q   (autograd through q only; p is an independent state)
            dHdp   = M⁻¹(q) p
            ṗ      = p × dHdp + Σᵢ(Rᵢ × ∂H/∂qᵢ) − Dw(q,p) · dHdp + g(q)·u

        With (q, p) as the integrated state, p is independent of q in the
        autograd graph, so no `stop_gradient` is needed.

        `keys` is an optional dict {M, V, Dw, g} of PRNGKeys (one per GP
        subnet); the same keys are reused for the dHdq autograd through q
        and the outer M⁻¹/g/Dw evaluation so a single coherent w sample
        defines the whole drift.
        """
        kM = None if keys is None else keys.get('M')
        kV = None if keys is None else keys.get('V')
        kD = None if keys is None else keys.get('Dw')
        kg = None if keys is None else keys.get('g')

        def H_of_q(q_):
            return (0.5 * jnp.dot(p, self._M_call(q_, key=kM) @ p)
                    + self._V_call(q_, key=kV)[0])
        dHdq = jax.grad(H_of_q)(q)                                # (9,)

        M_q_inv = self._M_call(q, key=kM)
        g_q     = self._g_call(q, key=kg)

        dHdp = M_q_inv @ p                                        # (3,) = ω
        Dw_q = self._Dw_call(q, p, key=kD)                        # D(q, p)
        F = (g_q * u).reshape(3) if self.u_dim == 1 else g_q @ u  # (3,)

        R_3x3    = q.reshape(3, 3)
        dHdq_3x3 = dHdq.reshape(3, 3)
        grav = jnp.sum(jnp.cross(R_3x3, dHdq_3x3, axis=-1), axis=0)

        if self.friction:
            dp = jnp.cross(p, dHdp) + grav - (Dw_q @ dHdp) + F
        else:
            dp = jnp.cross(p, dHdp) + grav + F
        return dp

    def drift(self, q, q_dot, u, keys=None):
        """ω̇ = M⁻¹·ṗ + (dM⁻¹/dt)·p — compatibility wrapper for ω-space
        per-increment pseudo-likelihoods.

        Reconstructs p = M(q)·ω, calls `drift_p`, then converts ṗ → ω̇
        with the JVP of M⁻¹ along q̇.
        """
        rd = self.rotmatdim
        kM = None if keys is None else keys.get('M')

        # p = M·q_dot   (M_net returns M⁻¹, so M·q_dot = solve(M⁻¹, q_dot))
        M_q_inv = self._M_call(q, key=kM)
        p_val   = jnp.linalg.solve(M_q_inv, q_dot)
        p       = jax.lax.stop_gradient(p_val)

        dp = self.drift_p(q, p, u, keys=keys)

        # Convert ṗ → ω̇ via M⁻¹·ṗ + (dM⁻¹/dt)·p.
        # q̇ in 9-dim form = R × dHdp = R × M⁻¹·p (used only for the JVP).
        dHdp   = M_q_inv @ p
        R_3x3  = q.reshape(3, 3)
        dHdp_b = jnp.broadcast_to(dHdp[None, :], (3, 3))
        dq     = jnp.cross(R_3x3, dHdp_b, axis=-1).reshape(rd)

        _M_kM = lambda q_: self._M_call(q_, key=kM)
        _, dM_inv_dt = jax.jvp(_M_kM, (q,), (dq,))                # (3, 3)
        return M_q_inv @ dp + dM_inv_dt @ p

    # ── ELBO helper ──────────────────────────────────────────────────────

    def kl_loss(self):
        """Sum of variational KL terms across the 4 GP subnets.

        Add this (scaled by some β) to the data-fit term to recover the ELBO:
            ELBO = E[log p(target | x_hat)] − β · kl_loss()
        """
        return (
            self.M_net.weight_kl_loss()
            + self.V_net.weight_kl_loss()
            + self.Dw_net.weight_kl_loss()
            + self.g_net.weight_kl_loss()
        )


# Backward-compat alias so existing imports `from network import
# DissipativeSO3HamSDE` keep working during transition. New code should
# use the ODE name directly.
DissipativeSO3HamSDE = DissipativeSO3HamODE
