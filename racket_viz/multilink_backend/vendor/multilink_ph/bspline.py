from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from scipy.interpolate import BSpline
from numpy.polynomial.legendre import leggauss


def clamped_uniform_knots(n_ctrl: int, degree: int = 3) -> np.ndarray:
    if n_ctrl <= degree:
        raise ValueError("n_ctrl must be > degree")
    n_internal = n_ctrl - degree - 1
    if n_internal > 0:
        internal = np.linspace(0.0, 1.0, n_internal + 2)[1:-1]
    else:
        internal = np.empty((0,), dtype=float)
    return np.concatenate([
        np.zeros(degree + 1), internal, np.ones(degree + 1)
    ])


def open_bspline_basis(
    times: np.ndarray,
    n_ctrl: int,
    degree: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return basis and first-time-derivative matrices for a clamped spline."""
    times = np.asarray(times, dtype=float)
    knots = clamped_uniform_knots(n_ctrl, degree)
    B = np.zeros((times.size, n_ctrl), dtype=float)
    dB = np.zeros_like(B)
    for j in range(n_ctrl):
        coeff = np.zeros(n_ctrl, dtype=float)
        coeff[j] = 1.0
        spl = BSpline(knots, coeff, degree, extrapolate=False)
        B[:, j] = spl(times)
        dB[:, j] = spl.derivative(1)(times)
    # scipy can return nan infinitesimally outside support; endpoints are well defined.
    B = np.nan_to_num(B)
    dB = np.nan_to_num(dB)
    return B, dB


def periodic_cubic_basis(times: np.ndarray, n_ctrl: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform periodic cardinal cubic B-spline basis on [0,1].

    The returned basis is exactly periodic at t=0 and t=1.  Derivatives are
    with respect to normalized time t, not the local cardinal coordinate.
    """
    if n_ctrl < 4:
        raise ValueError("periodic cubic spline requires at least 4 controls")
    times = np.asarray(times, dtype=float)
    B = np.zeros((times.size, n_ctrl), dtype=float)
    dB = np.zeros_like(B)
    for r, t in enumerate(times):
        tw = float(t % 1.0) if not np.isclose(t, 1.0) else 0.0
        u = n_ctrl * tw
        i = int(np.floor(u))
        s = u - np.floor(u)
        w = np.array([
            (1.0 - s) ** 3 / 6.0,
            (3.0 * s**3 - 6.0 * s**2 + 4.0) / 6.0,
            (-3.0 * s**3 + 3.0 * s**2 + 3.0 * s + 1.0) / 6.0,
            s**3 / 6.0,
        ])
        dw_ds = np.array([
            -0.5 * (1.0 - s) ** 2,
            1.5 * s**2 - 2.0 * s,
            -1.5 * s**2 + s + 0.5,
            0.5 * s**2,
        ])
        inds = [(i - 1) % n_ctrl, i % n_ctrl, (i + 1) % n_ctrl, (i + 2) % n_ctrl]
        for a, j in enumerate(inds):
            B[r, j] += w[a]
            dB[r, j] += n_ctrl * dw_ds[a]
    return B, dB


def spatial_curvature_to_joint_matrix(
    n_links: int,
    n_modes: int,
    degree: int = 3,
    total_length: float = 1.0,
    quad_order: int = 8,
) -> np.ndarray:
    """Integrate a clamped spatial B-spline curvature basis over joint dual cells.

    For joint j=1,...,M-1 located at sigma=j/M, its dual cell is
    [(j-1/2)/M, (j+1/2)/M].  If kappa(sigma)=sum_a B_a(sigma)c_a,
    then alpha = K_sp c with K_sp[j,a] = L int_cell B_a dsigma.
    """
    if n_links < 2:
        raise ValueError("n_links must be >= 2")
    if n_modes <= degree:
        raise ValueError("n_modes must be > degree")
    knots = clamped_uniform_knots(n_modes, degree)
    basis = []
    for a in range(n_modes):
        coeff = np.zeros(n_modes)
        coeff[a] = 1.0
        basis.append(BSpline(knots, coeff, degree, extrapolate=False))

    xg, wg = leggauss(quad_order)
    K = np.zeros((n_links - 1, n_modes), dtype=float)
    for j in range(1, n_links):
        lo = max(0.0, (j - 0.5) / n_links)
        hi = min(1.0, (j + 0.5) / n_links)
        xs = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
        ws = 0.5 * (hi - lo) * wg
        for a, spl in enumerate(basis):
            K[j - 1, a] = total_length * np.sum(ws * spl(xs))
    return K


@dataclass
class TemporalBasis:
    B: torch.Tensor
    dB: torch.Tensor
    periodic: bool

    @property
    def n_steps_plus_one(self) -> int:
        return self.B.shape[0]

    @property
    def n_ctrl(self) -> int:
        return self.B.shape[1]

    def evaluate(self, controls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate trajectory and derivative.

        controls: (..., n_ctrl, n_modes)
        returns: (..., T, n_modes) for c and cdot.
        """
        c = torch.einsum("tn,...nd->...td", self.B, controls)
        cdot = torch.einsum("tn,...nd->...td", self.dB, controls)
        return c, cdot


def make_temporal_basis(
    n_steps: int,
    n_ctrl: int,
    periodic: bool,
    degree: int = 3,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> TemporalBasis:
    times = np.linspace(0.0, 1.0, n_steps + 1)
    if periodic:
        if degree != 3:
            raise NotImplementedError("periodic implementation currently supports cubic splines")
        B, dB = periodic_cubic_basis(times, n_ctrl)
    else:
        B, dB = open_bspline_basis(times, n_ctrl, degree)
    return TemporalBasis(
        B=torch.as_tensor(B, dtype=dtype, device=device),
        dB=torch.as_tensor(dB, dtype=dtype, device=device),
        periodic=periodic,
    )
