"""Dissipative port-Hamiltonian SO(3) **SDE** for the 3D windy pendulum (JAX),
with **GP subnetworks** (random-Fourier-feature variational GPs from
src/utils/JAX/gp_model.py) replacing the earlier MLP/PSD/MatrixNet subnets.

SDE structure (integrated state is now (q, p) — angular momentum, not ω):

    dq         = R · ω · dt                (Lie-group integrator; ω = M⁻¹·p)
    dp_det     = (p × M⁻¹p + Σᵢ Rᵢ × ∂qᵢH − D·M⁻¹p + g·u) · dt
    dp_stoch   = Rᵀ · (l · R · e_z × σ(q) · dW)

ω is reconstructed lazily as ω = M⁻¹(q)·p inside the Heun stages (for the
SO(3) exponential) and at the rollout output boundary (so downstream losses
that consume (q, ω) keep working).

Subnetworks (single-sample Equinox GP modules — vmap externally for batches):
    M_net(q)     PSD_GP_Model    — M⁻¹(q) (paper convention; ε=1.0 fp32 floor)
    V_net(q)     GP_Model→1      — potential energy
    Dw_net(q)    PSD_GP_Model    — dissipation
    g_net(q)     GP_MatrixNet    — control-input coupling
    sigma_net(q) GP_Model→1      — diffusion scale (softplus → σ ≥ 0)

The GP modules accept `(x, key=None, inference_mode=False)`. At dynamics time
we always call them with `inference_mode=True` so the Lie-Heun integrator
remains key-free; the GP randomness only enters through the variational ELBO
loss at training time (KL term, plus weight sampling if you want a stochastic
ELBO — exposed via .kl_loss()).

Public model API:
    drift_p(q, p, u)                 -> ṗ_det        ∈ ℝ³   (used by integrator)
    stochastic_increment_p(q, dW)    -> dp_stoch     ∈ ℝ³   (used by integrator)
    M_inv(q)                         -> M⁻¹(q)       ∈ ℝ³ˣ³ (used by integrator)
    drift(q, q_dot, u)               -> ω̇_det        ∈ ℝ³   (compat wrapper for pl_loss)
    stochastic_increment(q, dW)      -> dω_stoch     ∈ ℝ³   (compat wrapper)
    sigma(q)                         -> σ            ∈ ℝ
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


class DissipativeSO3HamSDE(eqx.Module):
    """Port-Hamiltonian SDE on SO(3) × ℝ³ with GP subnetworks.

    State (12-dim, used by the integrator):
        x = concat(R.flatten(), ω)    R row-major like envs/windy_pendulum_3d
    Control u ∈ ℝ³ is passed separately (held constant across an SDE substep).

    Trainable observation-noise scales `log_sigma_R`, `log_sigma_omega` are
    kept as fields on the model so they're picked up automatically by
    `eqx.filter(model, eqx.is_array)` in the optimiser. They feed the
    Gaussian-NLL likelihood of the ELBO; see `src/utils/JAX/elbo_loss_jax.py`.
    """
    M_net:     PSD_GP_Model
    V_net:     GP_Model
    Dw_net:    PSD_GP_Model
    g_net:     eqx.Module        # GP_Model if u_dim==1 else GP_MatrixNet
    sigma_net: GP_Model          # scalar diffusion-scale GP: σ(q) = softplus(GP(q)+bias)

    log_sigma_R:     jnp.ndarray   # scalar — rollout-NLL noise on rotation (geodesic)
    log_sigma_omega: jnp.ndarray   # scalar — rollout-NLL noise on angular velocity

    # Per-snapshot ω observation noise used inside the per-increment pseudo-
    # likelihood: Σ_eff = σ_φ²·Δt + 2·σ_obs_ω². Frozen `eqx.field(static=True)`,
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
                 init_sigma_const: float = 0.5):
        # init_sigma_const is accepted for backward CLI compatibility but
        # ignored: the static softplus bias on σ_net was removed, so σ(q)
        # initialises at softplus(GP_raw(q)≈0) = ln 2 ≈ 0.693 regardless.
        del init_sigma_const
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

        kM, kV, kD, kg, ks = jax.random.split(key, 5)

        # M_net : PSD with epsilon=1.0  (fp32 stability floor on diag)
        self.M_net = PSD_GP_Model(
            key=kM, input_dim=self.rotmatdim, hidden_dim=hidden_dim,
            diag_dim=self.angveldim, epsilon=1.0,
        )
        # V_net : scalar potential energy
        self.V_net = GP_Model(
            key=kV, input_dim=self.rotmatdim, output_dim=1,
            n_matern_features=hidden_dim,
        )
        # Dw_net : PSD with epsilon=0.0
        self.Dw_net = PSD_GP_Model(
            key=kD, input_dim=self.rotmatdim, hidden_dim=hidden_dim,
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
        # sigma_net : scalar diffusion-scale GP. Output passed through
        # softplus(.) only (no bias) to enforce σ ≥ 0. At initialisation
        # GP_raw ≈ 0 ⇒ σ(q) ≈ softplus(0) = ln 2 ≈ 0.693.
        self.sigma_net = GP_Model(
            key=ks, input_dim=self.rotmatdim, output_dim=1,
            n_matern_features=hidden_dim,
        )

    # ── Key-aware subnet wrappers ────────────────────────────────────────
    # `key=None` → deterministic posterior-mean call (inference_mode=True).
    # `key=<PRNGKey>` → reparameterised weight sample (inference_mode=False).
    # Holding the key fixed across the full rollout ensures the same w sample
    # is used for predictor and corrector inside `lie_heun_sde_step`, and
    # across all substeps within one trajectory — required for the ELBO MC
    # estimate over q_ψ(w).

    def _M_call(self, q, key=None):
        return self.M_net(q, key=key, inference_mode=(key is None))

    def _V_call(self, q, key=None):
        return self.V_net(q, key=key, inference_mode=(key is None))

    def _Dw_call(self, q, key=None):
        return self.Dw_net(q, key=key, inference_mode=(key is None))

    def _g_call(self, q, key=None):
        return self.g_net(q, key=key, inference_mode=(key is None))

    def _sigma_call(self, q, key=None):
        return self.sigma_net(q, key=key, inference_mode=(key is None))

    # ── Diffusion (learnable scalar GP) ──────────────────────────────────

    def sigma(self, q, key=None):
        """Diffusion scale σ(q) = softplus(sigma_net(q)) ≥ 0.

        `key=None` → posterior-mean (deterministic) call.
        `key=<PRNGKey>` → reparameterised weight sample for the ELBO MC term.

        No additive bias: σ(q) starts at softplus(0) = ln 2 ≈ 0.693 at init
        (when GP_raw ≈ 0) and is free to learn whatever level the data
        supports, with no static-field gauge to mismatch at load time.
        """
        raw = self._sigma_call(q, key=key)[0]
        return jax.nn.softplus(raw)

    def M_inv(self, q, keys=None):
        """M⁻¹(q) ∈ ℝ³ˣ³ — exposed for the (q, p) integrator's ω = M⁻¹·p step."""
        kM = None if keys is None else keys.get('M')
        return self._M_call(q, key=kM)

    def stochastic_increment_p(self, q, dW, keys=None):
        """dp_stoch = Rᵀ · (l · R · e_z × σ(q) · dW)  — body-frame torque
        increment in p-units. The (q, p) integrator carries angular momentum
        directly, so the leading M⁻¹ that used to convert torque → dω is
        absorbed back into the model's ω = M⁻¹·p step.
        """
        ksigma = None if keys is None else keys.get('sigma')
        R = q.reshape(3, 3)
        ez = jnp.array([0.0, 0.0, 1.0], dtype=q.dtype)
        r_world = self.l * (R @ ez)                              # (3,)
        dF_stoch = self.sigma(q, key=ksigma) * dW                # (3,)
        torque_world = jnp.cross(r_world, dF_stoch)              # (3,)
        torque_body  = R.T @ torque_world                        # (3,)
        return torque_body                                       # (3,)

    def stochastic_increment(self, q, dW, keys=None):
        """dω_stoch = M⁻¹(q) · dp_stoch  — kept for `pl_loss` (which still
        evaluates the transition density in ω-space)."""
        return self.M_inv(q, keys=keys) @ self.stochastic_increment_p(q, dW, keys=keys)

    # ── Deterministic drift (ṗ_det) — port-Hamiltonian momentum derivative ─

    def drift_p(self, q, p, u, keys=None):
        """Deterministic momentum derivative ṗ ∈ ℝ³.

            H      = ½ pᵀ M⁻¹(q) p + V(q)
            dHdq   = ∂H/∂q   (autograd through q only; p is an independent state)
            dHdp   = M⁻¹(q) p
            ṗ      = p × dHdp + Σᵢ(Rᵢ × ∂H/∂qᵢ) − Dw(q) · dHdp + g(q)·u

        With (q, p) as the integrated state, p is independent of q in the
        autograd graph, so no `stop_gradient` is needed and the JVP of M⁻¹
        used by the old (q, ω) form is gone.

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
        Dw_q    = self._Dw_call(q, key=kD)

        dHdp = M_q_inv @ p                                        # (3,)
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
        """ω̇ = M⁻¹·ṗ + (dM⁻¹/dt)·p  — kept for `pl_loss`, which evaluates
        the per-increment Euler-Maruyama transition density in ω-space.

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
        """Sum of variational KL terms across the 5 GP subnets.

        Add this (scaled by some β) to the data-fit term to recover the ELBO:
            ELBO = E[log p(target | x_hat)] − β · kl_loss()
        """
        return (
            self.M_net.weight_kl_loss()
            + self.V_net.weight_kl_loss()
            + self.Dw_net.weight_kl_loss()
            + self.g_net.weight_kl_loss()
            + self.sigma_net.weight_kl_loss()
        )
