from __future__ import annotations

from typing import Sequence
import numpy as np

from .pinocchio_backend import PinocchioRigidBodyBackend


def pinocchio_configuration_fim(
    backend: PinocchioRigidBodyBackend,
    q: np.ndarray,
    frame_names: Sequence[str],
    eps: float = 1e-4,
    angle_weight: float = 0.05,
    normalize_trace: bool = True,
) -> np.ndarray:
    """Body/frame-Jacobian pullback metric on internal articulated coordinates."""
    nj = backend.n_actuated
    F = np.zeros((nj, nj), dtype=float)
    for name in frame_names:
        # LOCAL representation removes a common world rotation from the metric.
        J = backend.frame_jacobian(q, name, "LOCAL")[:, backend.base_nv :]
        Jlin = J[:3]
        Jang = J[3:]
        F += Jlin.T @ Jlin + angle_weight * (Jang.T @ Jang)
    F /= max(len(frame_names), 1)
    F += eps * np.eye(nj)
    if normalize_trace:
        F *= nj / max(float(np.trace(F)), eps)
    return F


def pinocchio_empirical_souriau_fisher(
    backend: PinocchioRigidBodyBackend,
    q: np.ndarray,
    v: np.ndarray,
    q_eps: float = 2e-2,
    v_eps: float = 4e-2,
    reg: float = 5e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Local empirical covariance of centroidal [Lz,Px,Py] momentum."""
    samples = []
    for j in range(backend.n_actuated):
        e = np.zeros(backend.nv)
        e[backend.base_nv + j] = q_eps
        dv = np.zeros(backend.nv)
        dv[backend.base_nv + j] = v_eps
        for sign in (-1.0, 1.0):
            qs = backend.integrate(q, sign * e)
            vs = v + sign * dv
            samples.append(backend.planar_centroidal_momentum(qs, vs))
    samples.append(backend.planar_centroidal_momentum(q, v))
    S = np.asarray(samples)
    C = np.cov(S.T, bias=False) if len(S) > 1 else np.zeros((3,3))
    C = C + reg * np.eye(3)
    C *= 3.0 / max(float(np.trace(C)), reg)
    return C, S
