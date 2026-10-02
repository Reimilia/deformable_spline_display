"""(R, p)-form integrator wrapper around `windy_pendulum_3d`.

Original env (`envs/windy_pendulum_3d.py`) integrates the rigid-body SDE in
(R, ω) state. This wrapper carries angular **momentum** p = I·ω inside the
Lie-Heun substep instead, recovering ω via ω = I⁻¹·p whenever the SO(3)
exponential needs it.

For the spherical pendulum the inertia is isotropic (M = m·l²·I₃, constant
in q), so the (R, p) and (R, ω) updates are algebraically identical up to
floating-point rounding — this change is *structural*, matching the model's
own (q, p) integrator in `src/utils/JAX/lie_integrator.py`. The benefit is:

  * the diffusion is naturally a torque (added to p) rather than an ω
    increment, so the M⁻¹ that the env applied at the boundary disappears;
  * the integrator topology now exactly mirrors the model side, removing one
    last bookkeeping difference between plant and digital twin.

Public surface stays unchanged: `self.R`, `self.omega`, `self.last_u`,
`self.last_w`, `self.t`, `step`, `reset`, `get_state`, `_get_obs`. Anything
the env's render code reads (`last_w`, `external_force_direction`, etc.)
keeps working.

Limitation: the GroundTruth-subnet model in `eval_ground_truth_match.py`
omits the deterministic wind torque (it was zero in the dataset). This
wrapper, by contrast, *does* still apply the env's deterministic wind, so
it stays a strict superset of the env's behaviour with respect to that
control variable. Only the integration coordinates change.
"""
from __future__ import annotations

import os
import sys

import numpy as np

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, '../../../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.windy_pendulum_3d import (
    windy_pendulum_3d, _exp_so3, _project_to_so3,
)


class QPWindyPendulum3D(windy_pendulum_3d):
    """Lie-Heun on SO(3) × ℝ³ with internal state (R, p) instead of (R, ω).

    Behavioural parity with the parent env is asserted by
    `evaluate_controller.py --mode sweep` (the env-vs-GT-model sweep
    matched to 4 decimal places — that GT model uses the same (q, p)
    integrator topology).
    """

    # ─────────────────────────────────────────────────────────────────
    # (R, p)-form rates — same physics as `_compute_omega_rates`, but
    # we don't pre-multiply by I⁻¹ at the end, and we use the full
    # rigid-body Euler equation ṗ = τ − ω × p (which collapses to ṗ = τ
    # for isotropic M, since p ∥ ω).
    # ─────────────────────────────────────────────────────────────────
    def _compute_p_rates(self, R, p, w, u, dW, sigma):
        ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        omega = self.I_inv @ p                                 # ω = M⁻¹·p
        r_world = self.l * (R @ ez)

        # Gravity torque (world → body)
        Fg = -self.m * self.g * ez
        tau_g_body = R.T @ np.cross(r_world, Fg)

        # Deterministic wind torque (world → body)
        Fw_det = w * self.external_force_direction
        tau_w_body = R.T @ np.cross(r_world, Fw_det)

        # Friction (state-dependent if varying_friction=True)
        fric = self._variable_friction(R, omega)
        tau_fric = fric * omega

        # Control torque is already in body frame.
        tau_det = tau_g_body + tau_w_body + u - tau_fric

        # Rigid-body equation in p-form: ṗ = τ − ω × p.
        # For isotropic I, p = I·ω is parallel to ω so ω×p = 0 and
        # this reduces to ṗ = τ; we keep the cross term explicit so a
        # later anisotropic-inertia variant doesn't need a code change.
        p_dot_det = tau_det - np.cross(omega, p)

        # Stochastic forcing — wind enters as a torque in p-units.
        if sigma > 0.0:
            dF_stoch = sigma * dW
            dp_stoch = R.T @ np.cross(r_world, dF_stoch)
        else:
            dp_stoch = np.zeros(3, dtype=np.float64)

        return p_dot_det, dp_stoch

    # ─────────────────────────────────────────────────────────────────
    # One Lie-Heun substep on (R, p)
    # ─────────────────────────────────────────────────────────────────
    def _lie_heun_step_qp(self, R, p, w, u, h, sigma, dW):
        # Stage 1
        p_dot_1, dp_stoch_1 = self._compute_p_rates(R, p, w, u, dW, sigma)
        omega_1 = self.I_inv @ p
        phi_1 = omega_1 * h

        # Predictor
        R_pred = R @ _exp_so3(phi_1)
        p_pred = p + p_dot_1 * h + dp_stoch_1

        # Stage 2 (re-evaluate, reuse same dW)
        p_dot_2, dp_stoch_2 = self._compute_p_rates(
            R_pred, p_pred, w, u, dW, sigma)
        omega_2 = self.I_inv @ p_pred
        phi_2 = omega_2 * h

        # Corrector — average in Lie algebra, exponentiate once.
        phi_avg = 0.5 * (phi_1 + phi_2)
        R_new = R @ _exp_so3(phi_avg)
        p_new = (p
                 + 0.5 * (p_dot_1 + p_dot_2) * h
                 + 0.5 * (dp_stoch_1 + dp_stoch_2))
        return R_new, p_new

    # ─────────────────────────────────────────────────────────────────
    # Step — same wind/RNG sequence as the parent, just integrated in p
    # ─────────────────────────────────────────────────────────────────
    def step(self, u):
        u = np.asarray(u, dtype=np.float64).reshape(3)
        self.last_u = u.copy()

        self.t += self.dt
        w = self.update_wind(self.t)
        self.last_w = float(w)

        n_substeps = 10
        dt_sub = self.dt / n_substeps
        sigma = self.wind_force_std

        # ω → p at the outer-step boundary (M is q-independent for this
        # plant, so this is a constant rescaling).
        p = self.I @ self.omega

        for _ in range(n_substeps):
            if sigma > 0.0:
                dW = self._np_rng.normal(0.0, np.sqrt(dt_sub), size=3)
            else:
                dW = np.zeros(3, dtype=np.float64)
            self.R, p = self._lie_heun_step_qp(
                self.R, p, w, u, dt_sub, sigma, dW)

        # p → ω at the boundary so downstream code that reads self.omega
        # (env render, controller, observation) keeps working unchanged.
        self.omega = self.I_inv @ p

        # Re-orthogonalise R if floating-point drift accumulated.
        if (abs(np.linalg.det(self.R) - 1.0) > 1e-8
                or np.linalg.norm(self.R.T @ self.R - np.eye(3)) > 1e-8):
            self.R = _project_to_so3(self.R)

        # Reward — same as parent.
        ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        bob_dir = self.R @ ez
        angle_cost = 1.0 - float(bob_dir[2])
        vel_cost = 0.1 * float(np.dot(self.omega, self.omega))
        act_cost = 0.001 * float(np.dot(u, u))
        cost = angle_cost + vel_cost + act_cost

        obs = self._get_obs()
        info = {"wind": self.last_w}
        return obs, -cost, False, False, info
