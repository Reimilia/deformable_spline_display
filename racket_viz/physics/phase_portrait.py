"""Casimir-sphere phase portrait: fixed points, separatrix, and orbit families.

Adapted from `summer-2026/dzhanibekov_display.py` (Sections 5 & 8). All of this is
torque-free geometry — it describes the |M| = R_cas sphere that a controlled
trajectory will *leave* once an external torque is applied (see `controllers.py`
and `scenarios.py`).
"""
import numpy as np

import rigid_body


def casimir_radius(I, imid, spin_rate):
    """Reference |M| for an IC spinning near the unstable (imid) axis at spin_rate."""
    I = np.asarray(I, dtype=float)
    M_ref = 0.01 * I
    M_ref[imid] = I[imid] * spin_rate
    return float(np.linalg.norm(M_ref))


def energy_at_axes(I, R_cas):
    """H_axis[k] = R_cas^2 / (2*I[k]) for k in (0,1,2); the energy of each pure-axis
    equilibrium. H_axis[imid] is the separatrix energy H_sep."""
    I = np.asarray(I, dtype=float)
    return R_cas ** 2 / (2 * I)


def ic_from_H(H_val, Mmid, I, R_cas, imin, imax, imid):
    """Point on the Casimir sphere (|M|=R_cas) with energy H_val and unstable-axis
    component Mmid. Returns [M0, M1, M2] (M[imin], M[imax] >= 0), or None if no
    real solution exists for the given (H_val, Mmid)."""
    I = np.asarray(I, dtype=float)
    Ia, Ib, Iu = I[imin], I[imax], I[imid]

    S = R_cas ** 2 - Mmid ** 2
    E = 2 * H_val - Mmid ** 2 / Iu
    denom = 1 / Ib - 1 / Ia
    if abs(denom) < 1e-15 or S < 0:
        return None

    Mbsq = (E - S / Ia) / denom
    Masq = S - Mbsq
    if Mbsq < -1e-9 or Masq < -1e-9:
        return None

    M = [0.0, 0.0, 0.0]
    M[imin] = float(np.sqrt(max(0.0, Masq)))
    M[imax] = float(np.sqrt(max(0.0, Mbsq)))
    M[imid] = float(Mmid)
    return M


def build_static_curves(
    I, R_cas, imin, imax, imid,
    n_levels=0, eps_factor=0.001, alpha_small=0.30,
    n_static_pts=900, t_traj_static=80.0, t_sep_static=220.0,
):
    """Background trajectory/separatrix curves for the sphere panel, plus fixed
    points. Returns a dict: {trajectories: [...], separatrices: [...],
    stable_fixed_points: (4,3) array, unstable_fixed_points: (2,3) array}."""
    I = np.asarray(I, dtype=float)
    eps = eps_factor * R_cas
    H_axis = energy_at_axes(I, R_cas)
    H_sep = H_axis[imid]
    stable_axes = (imin, imax)

    def _family_curves(H_k, T, N):
        curves = []
        ic_M = ic_from_H(H_k, eps, I, R_cas, imin, imax, imid)
        if ic_M is None:
            return curves
        for sa in (1, -1):
            for sb in (1, -1):
                M0 = [0.0, 0.0, 0.0]
                M0[imin] = sa * ic_M[imin]
                M0[imax] = sb * ic_M[imax]
                M0[imid] = ic_M[imid]
                curve = rigid_body.integrate_M(M0, I, T, N)
                if curve is not None:
                    curves.append(curve)
        return curves

    trajectories = []
    for ax in stable_axes:
        H_k = H_axis[ax] + alpha_small * (H_sep - H_axis[ax])
        trajectories.extend(_family_curves(H_k, t_traj_static, n_static_pts))

    for ax in stable_axes:
        for k in range(1, n_levels + 1):
            alpha = 0.90 * k / (n_levels + 1)
            H_k = H_axis[ax] + alpha * (H_sep - H_axis[ax])
            trajectories.extend(_family_curves(H_k, t_traj_static, n_static_pts))

    separatrices = []
    for se in (eps, -eps):
        ic_M = ic_from_H(H_sep, se, I, R_cas, imin, imax, imid)
        if ic_M is None:
            continue
        for sa in (1, -1):
            for sb in (1, -1):
                M0 = [0.0, 0.0, 0.0]
                M0[imin] = sa * ic_M[imin]
                M0[imax] = sb * ic_M[imax]
                M0[imid] = se
                curve = rigid_body.integrate_M(M0, I, t_sep_static, n_static_pts)
                if curve is not None:
                    separatrices.append(curve)

    stable_fixed_points = np.array([
        [s * R_cas if i == ax else 0.0 for i in range(3)]
        for ax in stable_axes for s in (1, -1)
    ])
    unstable_fixed_points = np.array([
        [s * R_cas if i == imid else 0.0 for i in range(3)]
        for s in (1, -1)
    ])

    return {
        "trajectories": trajectories,
        "separatrices": separatrices,
        "stable_fixed_points": stable_fixed_points,
        "unstable_fixed_points": unstable_fixed_points,
    }
