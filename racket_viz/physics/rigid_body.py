"""Rigid-body dynamics core: quaternion utilities + Euler's equations.

Adapted from `summer-2026/dzhanibekov_display.py` (Sections 2 & 4) and
`summer-2026/rotate.py`. Extends the original torque-free-only integrator with an
optional external-torque hook (`torque_fn`) so controlled trajectories (see
`controllers.py`) can be integrated through real dynamics rather than spliced
between canned states — energy and angular momentum are conserved only when
`torque_fn` is None.

Quaternion convention: q = [q0, q1, q2, q3], scalar-first (q0 is the scalar part).
"""
import numpy as np
from scipy.integrate import solve_ivp


def normalize_q(q):
    """Return q scaled to unit norm."""
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def quat_to_R(q):
    """Rotation matrix (3, 3) for scalar-first quaternion q. Does not mutate q."""
    q0, q1, q2, q3 = normalize_q(q)
    return np.array([
        [1 - 2 * (q2 ** 2 + q3 ** 2),     2 * (q1 * q2 - q0 * q3),     2 * (q1 * q3 + q0 * q2)],
        [    2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2),     2 * (q2 * q3 - q0 * q1)],
        [    2 * (q1 * q3 - q0 * q2),     2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)],
    ])


def quat_deriv(q, w):
    """q_dot = 1/2 q ⊗ [0, w], w = angular velocity in body coordinates."""
    q0, q1, q2, q3 = q
    w1, w2, w3 = w
    return 0.5 * np.array([
        -q1 * w1 - q2 * w2 - q3 * w3,
         q0 * w1 + q2 * w3 - q3 * w2,
         q0 * w2 + q3 * w1 - q1 * w3,
         q0 * w3 + q1 * w2 - q2 * w1,
    ])


def rigid_body_ode(t, state, I, torque_fn=None):
    """RHS for [w1, w2, w3, q0, q1, q2, q3].

    I : array-like (I1, I2, I3), principal moments of inertia.
    torque_fn : optional callable (t, w, q) -> (tau1, tau2, tau3), body-frame
        external torque. None (default) recovers Euler's torque-free equations.

    Euler's equations, vector form: I*wdot = tau - w x (I*w).
    """
    I = np.asarray(I, dtype=float)
    w = state[:3]
    q = state[3:]
    tau = np.zeros(3) if torque_fn is None else np.asarray(torque_fn(t, w, q), dtype=float)
    wdot = (tau - np.cross(w, I * w)) / I
    return np.concatenate([wdot, quat_deriv(q, w)])


def integrate_full(w0, I, T, N, torque_fn=None, q0=None):
    """Integrate angular velocity + orientation quaternion over [0, T].

    Returns (omegas, quats): arrays of shape (N, 3) and (N, 4).
    q0 defaults to the identity quaternion [1, 0, 0, 0].
    """
    I = np.asarray(I, dtype=float)
    if q0 is None:
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
    state0 = np.concatenate([np.asarray(w0, dtype=float), np.asarray(q0, dtype=float)])
    t_eval = np.linspace(0, T, N)
    sol = solve_ivp(
        rigid_body_ode, [0, T], state0, args=(I, torque_fn),
        t_eval=t_eval, rtol=1e-9, atol=1e-9, max_step=T / N * 3,
    )
    omegas = sol.y[:3].T
    quats = np.array([normalize_q(q) for q in sol.y[3:].T])
    return omegas, quats


def integrate_M(M0, I, T, N):
    """Integrate the torque-free angular-momentum ODE in body-frame M-space.

    Used for static phase-portrait background curves (no orientation needed).
    Returns an (N, 3) array, or None if integration fails.
    """
    I = np.asarray(I, dtype=float)

    def ode(t, M):
        m1, m2, m3 = M
        return [
            (1 / I[2] - 1 / I[1]) * m2 * m3,
            (1 / I[0] - 1 / I[2]) * m3 * m1,
            (1 / I[1] - 1 / I[0]) * m1 * m2,
        ]

    t_eval = np.linspace(0, T, N)
    try:
        sol = solve_ivp(
            ode, [0, T], list(M0), t_eval=t_eval,
            rtol=1e-10, atol=1e-10, max_step=T / N * 3,
        )
        if not sol.success:
            return None
        return sol.y.T
    except Exception:
        return None
