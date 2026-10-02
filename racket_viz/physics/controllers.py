"""IDA-PBC controllers for driving the racket to a target spin state.

Adapted from LieSPHGP PR #1 (branch `tennis-racket-effect`, commit 53fe83c):
`src/models/3D_SO3_Tennis_Racket/ph_nn_ode_v2/controller_stageD.py` and
`controller_stageF.py`. Both were validated there with an analytic input coupling
(g = I3, i.e. direct body-frame torque actuation) — this adaptation keeps only that
analytic path (drops the optional learned-g_net / torch branch, which this
precompute-only pipeline has no use for).

PDBodyFrameController = Stage D: proportional control on omega only (stabilize spin
about a target axis). Validated for all three axes (e1, e2, e3).

GeometricAttitudeController = Stage F: full (R*, w*) attitude control ("alignment"),
reduces to Stage D when K_R = 0.
"""
import numpy as np


def hat(v):
    """Skew-symmetric matrix for v in R^3, such that hat(v) @ w == cross(v, w)."""
    v = np.asarray(v, dtype=np.float64).ravel()
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def vee(Omega):
    """Axial vector of a skew-symmetric matrix (inverse of hat)."""
    return np.array([Omega[2, 1], Omega[0, 2], Omega[1, 0]])


def expm_SO3(Omega):
    """Matrix exponential of Omega in so(3) (Rodrigues formula)."""
    v = vee(Omega)
    theta = float(np.linalg.norm(v))
    if theta < 1e-7:
        return np.eye(3) + Omega
    n = v / theta
    return (
        np.cos(theta) * np.eye(3)
        + np.sin(theta) * hat(n)
        + (1 - np.cos(theta)) * np.outer(n, n)
    )


def logm_SO3(R):
    """Matrix logarithm of R in SO(3); vee(logm_SO3(R)) has norm <= pi."""
    cos_theta = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))

    if theta < 1e-7:
        return (R - R.T) / 2.0

    if abs(theta - np.pi) < 1e-4:
        sym = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(sym)))
        n = sym[:, i] / np.sqrt(max(sym[i, i], 1e-15))
        return theta * hat(n)

    return (theta / (2.0 * np.sin(theta))) * (R - R.T)


class PDBodyFrameController:
    """u = Kp * (omega_star - omega). Stage D, analytic g = I3 case only."""

    def __init__(self, omega_star, Kp=0.10, clip=2.0):
        self.omega_star = np.asarray(omega_star, dtype=np.float64).reshape(3)
        self.Kp = float(Kp)
        self.clip = float(clip)

    def __call__(self, R, omega):
        """Return body-frame torque u (3,) given current orientation R (3,3,
        unused in Stage D but kept for a uniform controller call signature) and
        angular velocity omega (3,)."""
        omega = np.asarray(omega, dtype=np.float64).reshape(3)
        tau = self.Kp * (self.omega_star - omega)
        return np.clip(tau, -self.clip, self.clip)


class GeometricAttitudeController:
    """u = -K_R * vee(logm(R_star.T @ R)) - Kp * (omega - omega_star). Stage F."""

    def __init__(self, R_star, omega_star, K_R=0.10, K_p=0.10, clip=2.0):
        self.R_star = np.asarray(R_star, dtype=np.float64).reshape(3, 3)
        self.omega_star = np.asarray(omega_star, dtype=np.float64).reshape(3)
        self.K_R = float(K_R)
        self.K_p = float(K_p)
        self.clip = float(clip)

    def __call__(self, R, omega):
        """Return body-frame torque u (3,) given current orientation R (3,3) and
        angular velocity omega (3,)."""
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        omega = np.asarray(omega, dtype=np.float64).reshape(3)
        e_R = vee(logm_SO3(self.R_star.T @ R))
        e_w = omega - self.omega_star
        tau = -self.K_R * e_R - self.K_p * e_w
        return np.clip(tau, -self.clip, self.clip)
