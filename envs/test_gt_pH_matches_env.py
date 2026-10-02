"""Sanity check: replace every model subnetwork with the ground-truth physics
and verify the Lie-Heun port-Hamiltonian rollout matches the env.

GT subnetworks for the spherical pendulum (m=l=1, isotropic inertia):
    M_inv(q)      = (1 / (m·l²)) · I₃                   # constant
    V(q)          = m·g·l · R_{22}                       # standard pendulum
    D(q, omega)   = friction_coeff · mult(R, omega) · I₃ # depends on omega
                    mult = 1 + 0.5·height_term + 0.5·tanh(|omega|)
                    height_term = 0.5·(1 - R_{22})
    g(q)          = I₃                                    # body-frame torque

Note: the model's architectural Dw_net(q) is a function of q only, so it
*cannot* represent the omega-dependent friction. This test bypasses that
limitation by hand-passing the true D(q, omega) — the result tells us
whether the port-Hamiltonian math is correct, not whether the architecture
is expressive enough.
"""
from __future__ import annotations

import os
import sys
import numpy as np

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, '..'))
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, 'src/utils')):
    if p not in sys.path:
        sys.path.insert(0, p)

from windy_pendulum_3d import (windy_pendulum_3d, _exp_so3,
                               _random_rotation, _hat)


# ──────────────────────────────────────────────────────────────────────
# Ground-truth subnetworks (in pure NumPy, mirroring the model API)
# ──────────────────────────────────────────────────────────────────────

def M_inv_GT(q, m=1.0, l=1.0):
    return (1.0 / (m * l * l)) * np.eye(3)


def V_GT(q, m=1.0, l=1.0, g=9.81):
    """V(R) = m·g·l · R_{22} = m·g·l · q[8]."""
    return m * g * l * q[8]


def dV_dq_GT(q, m=1.0, l=1.0, g=9.81):
    """∂V/∂q = (0,...,0, m·g·l) — only the (2,2) entry of R contributes."""
    out = np.zeros(9, dtype=np.float64)
    out[8] = m * g * l
    return out


def D_GT(q, omega, friction_coeff=0.5, varying_friction=True):
    """env-exact friction operator: D(R, omega) such that D · omega = tau_fric."""
    R = q.reshape(3, 3)
    if not varying_friction:
        mult = 1.0
    else:
        bob_z = R[2, 2]
        height_term = 0.5 * (1.0 - bob_z)
        speed_term  = np.tanh(np.linalg.norm(omega))
        mult = 1.0 + 0.5 * height_term + 0.5 * speed_term
    return friction_coeff * mult * np.eye(3)


def g_GT(q):
    return np.eye(3)


# ──────────────────────────────────────────────────────────────────────
# Port-Hamiltonian drift (same form as ph_gp_ode_v2/network.py:drift_p)
# ──────────────────────────────────────────────────────────────────────

def drift_p_GT(q, p, u, *, m=1.0, l=1.0, g=9.81,
               friction_coeff=0.5, varying_friction=True):
    """ṗ = p × dHdp + Σᵢ Rᵢ × ∂H/∂qᵢ − D(q,ω)·dHdp + g(q)·u

    With M_inv constant, ∂H/∂q comes only from V(q).
    """
    M_q_inv = M_inv_GT(q, m=m, l=l)
    dHdp = M_q_inv @ p           # (3,) — equals ω with constant M⁻¹
    omega = dHdp

    dHdq = dV_dq_GT(q, m=m, l=l, g=g)                   # (9,)
    R_3x3 = q.reshape(3, 3)
    dHdq_3x3 = dHdq.reshape(3, 3)
    grav = np.sum(np.cross(R_3x3, dHdq_3x3, axis=-1), axis=0)

    D_q = D_GT(q, omega, friction_coeff=friction_coeff,
               varying_friction=varying_friction)
    F = g_GT(q) @ u

    dp = np.cross(p, dHdp) + grav - (D_q @ dHdp) + F
    return dp


# ──────────────────────────────────────────────────────────────────────
# Lie-Heun ODE step (deterministic), mirroring lie_heun_ode_step
# ──────────────────────────────────────────────────────────────────────

def lie_heun_ode_step_GT(R, omega, u, h, *, m=1.0, l=1.0, g=9.81,
                          friction_coeff=0.5, varying_friction=True):
    """One deterministic Heun substep on SO(3) × ℝ³ in (R, ω) form,
    using the (q, p) port-Hamiltonian formulation internally."""
    q = R.reshape(9)
    # M_inv constant ⇒ p = M·ω = (m·l²)·ω
    p = (m * l * l) * omega

    p_dot_1 = drift_p_GT(q, p, u, m=m, l=l, g=g,
                          friction_coeff=friction_coeff,
                          varying_friction=varying_friction)
    M_inv_q = M_inv_GT(q, m=m, l=l)
    omega_1 = M_inv_q @ p
    phi_1 = omega_1 * h

    R_pred = R @ _exp_so3(phi_1)
    q_pred = R_pred.reshape(9)
    p_pred = p + p_dot_1 * h

    p_dot_2 = drift_p_GT(q_pred, p_pred, u, m=m, l=l, g=g,
                          friction_coeff=friction_coeff,
                          varying_friction=varying_friction)
    M_inv_qp = M_inv_GT(q_pred, m=m, l=l)
    omega_2 = M_inv_qp @ p_pred
    phi_2 = omega_2 * h

    phi_avg = 0.5 * (phi_1 + phi_2)
    R_new = R @ _exp_so3(phi_avg)
    p_new = p + 0.5 * (p_dot_1 + p_dot_2) * h
    omega_new = (1.0 / (m * l * l)) * p_new

    return R_new, omega_new


# ──────────────────────────────────────────────────────────────────────
# Test driver
# ──────────────────────────────────────────────────────────────────────

def run_test(varying_friction=True, friction_coeff=0.5,
              n_outer_steps=20, n_substeps=10, dt=0.05,
              seed=42, u_scale=2.0, m=1.0, l=1.0, g=9.81):
    rng = np.random.default_rng(seed)

    # ── Env reference rollout ──────────────────────────────────────
    env = windy_pendulum_3d(
        g=g, m=m, l=l, dt=dt,
        friction_coeff=friction_coeff,
        varying_friction=varying_friction,
        external_force_type="constant",
        external_force_std=0.0,
        wind_force_std=0.0,
        seed=seed,
    )
    env.reset(seed=seed)
    R0 = env.R.copy()
    omega0 = env.omega.copy()

    # Pre-sample u per env-step (random in [-u_scale, u_scale]^3)
    u_seq = rng.uniform(-u_scale, u_scale, size=(n_outer_steps, 3))

    env_R_list = [R0.copy()]
    env_omega_list = [omega0.copy()]
    for k in range(n_outer_steps):
        env.step(u_seq[k])
        env_R_list.append(env.R.copy())
        env_omega_list.append(env.omega.copy())
    env.close()
    env_R = np.stack(env_R_list, axis=0)            # (T+1, 3, 3)
    env_omega = np.stack(env_omega_list, axis=0)    # (T+1, 3)

    # ── GT port-Hamiltonian rollout ────────────────────────────────
    h = dt / n_substeps
    R_gt = R0.copy()
    omega_gt = omega0.copy()
    gt_R_list = [R_gt.copy()]
    gt_omega_list = [omega_gt.copy()]
    for k in range(n_outer_steps):
        u = u_seq[k]
        for _ in range(n_substeps):
            R_gt, omega_gt = lie_heun_ode_step_GT(
                R_gt, omega_gt, u, h,
                m=m, l=l, g=g,
                friction_coeff=friction_coeff,
                varying_friction=varying_friction,
            )
        gt_R_list.append(R_gt.copy())
        gt_omega_list.append(omega_gt.copy())
    gt_R = np.stack(gt_R_list, axis=0)
    gt_omega = np.stack(gt_omega_list, axis=0)

    # ── Compare ────────────────────────────────────────────────────
    R_diff = env_R - gt_R                                       # (T+1, 3, 3)
    R_fro = np.linalg.norm(R_diff.reshape(-1, 9), axis=-1)      # (T+1,)
    omega_diff = env_omega - gt_omega                           # (T+1, 3)
    omega_norm = np.linalg.norm(omega_diff, axis=-1)            # (T+1,)

    print(f"varying_friction = {varying_friction}    "
          f"friction_coeff = {friction_coeff}    u_scale = {u_scale}")
    print(f"  steps = {n_outer_steps}  substeps = {n_substeps}  "
          f"dt = {dt}  h = {h}")
    print(f"  ‖R_env − R_GT‖_F per step:")
    for k in range(0, n_outer_steps + 1, max(1, n_outer_steps // 10)):
        print(f"    t={k:>3d}  ‖ΔR‖={R_fro[k]:.3e}  ‖Δω‖={omega_norm[k]:.3e}")
    print(f"  max ‖ΔR‖_F over rollout: {R_fro.max():.3e}")
    print(f"  max ‖Δω‖   over rollout: {omega_norm.max():.3e}")
    return R_fro.max(), omega_norm.max()


if __name__ == "__main__":
    print("=" * 72)
    print("Test 1: constant friction, no varying_friction")
    print("=" * 72)
    run_test(varying_friction=False, friction_coeff=0.5,
              n_outer_steps=20, u_scale=2.0)

    print()
    print("=" * 72)
    print("Test 2: varying friction (height + tanh(|omega|))")
    print("=" * 72)
    run_test(varying_friction=True, friction_coeff=0.5,
              n_outer_steps=20, u_scale=2.0)

    print()
    print("=" * 72)
    print("Test 3: longer rollout to expose any drift")
    print("=" * 72)
    run_test(varying_friction=True, friction_coeff=0.5,
              n_outer_steps=100, u_scale=2.0)
