import numpy as np
import pytest

import phase_portrait

I_DEFAULT = np.array([0.075, 0.875, 0.950])
IMIN, IMID, IMAX = 0, 1, 2


def test_separatrix_energy_matches_intermediate_axis_energy():
    R_cas = phase_portrait.casimir_radius(I_DEFAULT, IMID, spin_rate=1.5)
    H_axis = phase_portrait.energy_at_axes(I_DEFAULT, R_cas)
    H_sep = H_axis[IMID]
    assert H_sep == pytest.approx(R_cas**2 / (2 * I_DEFAULT[IMID]))


def test_ic_from_H_roundtrip():
    R_cas = phase_portrait.casimir_radius(I_DEFAULT, IMID, spin_rate=1.5)
    H_axis = phase_portrait.energy_at_axes(I_DEFAULT, R_cas)
    H_sep = H_axis[IMID]

    # A small-orbit energy strictly between the stable-axis energy and H_sep.
    alpha = 0.3
    H_val = H_axis[IMIN] + alpha * (H_sep - H_axis[IMIN])
    Mmid = 0.001 * R_cas

    M = phase_portrait.ic_from_H(H_val, Mmid, I_DEFAULT, R_cas, IMIN, IMAX, IMID)
    assert M is not None

    M = np.asarray(M)
    assert np.linalg.norm(M) == pytest.approx(R_cas, rel=1e-6)
    H_recomputed = 0.5 * np.sum(M**2 / I_DEFAULT)
    assert H_recomputed == pytest.approx(H_val, rel=1e-6)


def test_ic_from_H_returns_none_when_infeasible():
    R_cas = phase_portrait.casimir_radius(I_DEFAULT, IMID, spin_rate=1.5)
    # H_val far outside the achievable range for this Mmid must be infeasible.
    M = phase_portrait.ic_from_H(1e6, 0.0, I_DEFAULT, R_cas, IMIN, IMAX, IMID)
    assert M is None


def test_phase_portrait_curves_stay_on_sphere():
    R_cas = phase_portrait.casimir_radius(I_DEFAULT, IMID, spin_rate=1.5)
    curves = phase_portrait.build_static_curves(I_DEFAULT, R_cas, IMIN, IMAX, IMID)

    for curve in curves["trajectories"] + curves["separatrices"]:
        mags = np.linalg.norm(curve, axis=1)
        np.testing.assert_allclose(mags, R_cas, rtol=1e-3)

    for fp in np.vstack([curves["stable_fixed_points"], curves["unstable_fixed_points"]]):
        assert np.linalg.norm(fp) == pytest.approx(R_cas, rel=1e-6)
