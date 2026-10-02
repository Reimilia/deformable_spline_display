"""Dissipative port-Hamiltonian SO(3) **SDE** for the 3D windy pendulum (JAX).

Replaces the earlier ODE port (ph_nn_ode_fp32 / Diffrax) with a stochastic
formulation that mirrors the env's Stratonovich SDE structure:

    dq           = R · ω · dt          (handled by the Lie-group integrator)
    dω_det       = M⁻¹ · ṗ · dt + (dM⁻¹/dt) · p · dt   (port-Hamiltonian drift)
    dω_stoch     = M⁻¹ · Rᵀ · (l · R · e_z × σ(q) · dW)

Subnetworks (single-sample Equinox modules — vmap externally for batches):
    M_net(q)     PSD        — M⁻¹(q) (paper convention; ε=1.0 fp32 floor)
    V_net(q)     MLP→1      — potential energy
    Dw_net(q)   PSD         — dissipation
    g_net(q)    MatrixNet   — control-input coupling
    sigma_net(q) MLP→1      — diffusion scale  (NEW; softplus → σ ≥ 0)

The model exposes:
    drift(q, q_dot, u)               -> ω̇_det        ∈ ℝ³
    stochastic_increment(q, dW)      -> dω_stoch     ∈ ℝ³

both consumed by src.utils.JAX.lie_integrator.lie_heun_sde_step.
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

from src.utils.JAX.neural_networks import MLP, PSD, MatrixNet


class DissipativeSO3HamSDE(eqx.Module):
    """Port-Hamiltonian SDE on SO(3) × ℝ³.

    State (12-dim, used by the integrator):
        x = concat(R.flatten(), ω)    R row-major like envs/windy_pendulum_3d
    Control u ∈ ℝ³ is passed separately (held constant across an SDE substep).

    Constants l (bob length) is set to 1.0 by default to match the env's
    spherical-pendulum benchmark; exposed so it can be changed if needed.
    """
    M_net:     PSD
    V_net:     MLP
    Dw_net:    PSD
    g_net:     eqx.Module        # MLP if u_dim==1 else MatrixNet
    sigma_net: MLP

    rotmatdim: int  = eqx.field(static=True)
    angveldim: int  = eqx.field(static=True)
    u_dim:     int  = eqx.field(static=True)
    friction:  bool = eqx.field(static=True)
    l:         float = eqx.field(static=True)

    def __init__(self, *, key, u_dim: int = 3, init_gain: float = 0.5,
                 friction: bool = True, hidden_dim: int = 20, l: float = 1.0):
        self.rotmatdim = 9
        self.angveldim = 3
        self.u_dim = int(u_dim)
        self.friction = bool(friction)
        self.l = float(l)

        kM, kV, kD, kg, ksig = jax.random.split(key, 5)

        self.M_net  = PSD(self.rotmatdim, hidden_dim, self.angveldim,
                          init_gain=init_gain, epsilon=1.0, key=kM)
        self.V_net  = MLP(self.rotmatdim, hidden_dim, 1,
                          init_gain=init_gain, key=kV)
        self.Dw_net = PSD(self.rotmatdim, hidden_dim, self.angveldim,
                          init_gain=init_gain, epsilon=0.0, key=kD)
        if self.u_dim == 1:
            self.g_net = MLP(self.rotmatdim, hidden_dim, self.angveldim,
                             init_gain=init_gain, key=kg)
        else:
            self.g_net = MatrixNet(self.rotmatdim, hidden_dim,
                                   self.angveldim * self.u_dim,
                                   shape=(self.angveldim, self.u_dim),
                                   init_gain=init_gain, key=kg)
        # sigma_net outputs a single scalar; softplus is applied in .sigma() to
        # keep the diffusion coefficient strictly positive.
        self.sigma_net = MLP(self.rotmatdim, hidden_dim, 1,
                             init_gain=init_gain, key=ksig)

    # ── Diffusion ────────────────────────────────────────────────────────

    def sigma(self, q):
        """Predicted stochastic-force scale at q (single-sample → scalar)."""
        return jax.nn.softplus(self.sigma_net(q)[0])

    def stochastic_increment(self, q, dW):
        """dω_stoch = M⁻¹(q) · Rᵀ · (l · R · e_z × σ(q) · dW).

        Mirrors envs/windy_pendulum_3d::_compute_omega_rates' stochastic branch
        (with the env's constant `wind_force_std` replaced by σ(q)).
        """
        R = q.reshape(3, 3)
        ez = jnp.array([0.0, 0.0, 1.0], dtype=q.dtype)
        r_world = self.l * (R @ ez)                              # (3,)
        dF_stoch = self.sigma(q) * dW                            # (3,)
        torque_world = jnp.cross(r_world, dF_stoch)              # (3,)
        torque_body  = R.T @ torque_world                        # (3,)
        return self.M_net(q) @ torque_body                       # (3,)

    # ── Deterministic drift (ω̇_det) ────────────────────────────────────

    def drift(self, q, q_dot, u):
        """Deterministic angular acceleration ω̇ ∈ ℝ³.

        Computes the port-Hamiltonian dynamics:

            p          = M(q) · q_dot                      via solve(M⁻¹, q_dot)
            H          = ½ pᵀ M⁻¹(q) p + V(q)
            dHdq       = ∂H/∂q   (autograd through q only — p detached)
            dHdp       = M⁻¹(q) p
            dp/dt      = p × dHdp + Σᵢ(Rᵢ × ∂H/∂qᵢ) − Dw(q) · dHdp + g(q)·u
            dM⁻¹/dt    = (∂M⁻¹/∂q) · q̇   (via JVP, q̇ in matrix form = R × ω)
            ω̇          = M⁻¹ · ṗ + (dM⁻¹/dt) · p

        Only ω̇ is returned; the rotation derivative is handled by the
        Lie-group integrator (R · exp([ω·h]_×)).
        """
        rd = self.rotmatdim

        # p = M·q_dot (M_net returns M⁻¹, so p = solve(M⁻¹, q_dot))
        M_q   = self.M_net(q)
        p_val = jnp.linalg.solve(M_q, q_dot)
        p     = jax.lax.stop_gradient(p_val)

        # dHdq via grad through q only (p frozen)
        def H_of_q(q_):
            return 0.5 * jnp.dot(p, self.M_net(q_) @ p) + self.V_net(q_)[0]
        dHdq = jax.grad(H_of_q)(q)                                # (9,)

        M_q_inv = self.M_net(q)
        g_q     = self.g_net(q)
        Dw_q    = self.Dw_net(q)

        dHdp = M_q_inv @ p                                        # (3,)
        F = (g_q * u).reshape(3) if self.u_dim == 1 else g_q @ u  # (3,)

        # Rotation derivative in 9-dim form (used only for the dM⁻¹/dt JVP).
        R_3x3  = q.reshape(3, 3)
        dHdp_b = jnp.broadcast_to(dHdp[None, :], (3, 3))
        dq     = jnp.cross(R_3x3, dHdp_b, axis=-1).reshape(rd)

        dHdq_3x3 = dHdq.reshape(3, 3)
        grav = jnp.sum(jnp.cross(R_3x3, dHdq_3x3, axis=-1), axis=0)

        if self.friction:
            dp = jnp.cross(p, dHdp) + grav - (Dw_q @ dHdp) + F
        else:
            dp = jnp.cross(p, dHdp) + grav + F

        _, dM_inv_dt = jax.jvp(self.M_net, (q,), (dq,))           # (3, 3)
        omega_dot = M_q_inv @ dp + dM_inv_dt @ p                  # (3,)
        return omega_dot
