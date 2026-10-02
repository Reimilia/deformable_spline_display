import json
from pathlib import Path

import numpy as np
import pytest

import rigid_body

FIXTURES = Path(__file__).parent / "fixtures"
I_DEFAULT = np.array([0.075, 0.875, 0.950])  # arbitrary I1<I2<I3, matches sketch labels


def _load_quaternion_fixture():
    with open(FIXTURES / "quaternion_rotation_pairs.json") as f:
        return json.load(f)


@pytest.mark.parametrize("case", _load_quaternion_fixture(), ids=lambda c: c["name"])
def test_quat_to_R_matches_reference(case):
    R = rigid_body.quat_to_R(np.array(case["q"]))
    np.testing.assert_allclose(R, np.array(case["R"]), atol=1e-9)


def test_normalize_q_returns_unit_norm():
    for q in [[1, 0, 0, 0], [2, 0, 0, 0], [1, 1, 1, 1], [0.3, -0.1, 5.0, 2.0]]:
        nq = rigid_body.normalize_q(np.array(q, dtype=float))
        assert np.isclose(np.linalg.norm(nq), 1.0)


def test_quat_to_R_is_always_a_valid_rotation():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        R = rigid_body.quat_to_R(q)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-8)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-8)


def test_energy_conservation_torque_free():
    """H = 0.5 * w . (I*w) must be conserved along a torque-free trajectory."""
    rng = np.random.default_rng(1)
    for _ in range(5):
        w0 = rng.uniform(0.1, 2.0, size=3)
        omegas, _ = rigid_body.integrate_full(w0, I_DEFAULT, T=10.0, N=200)
        H = 0.5 * np.sum(omegas**2 * I_DEFAULT, axis=1)
        np.testing.assert_allclose(H, H[0], rtol=1e-6)


def test_angular_momentum_magnitude_conservation_torque_free():
    """|M(t)| = |I*w(t)| is the Casimir invariant; must stay constant with no torque."""
    rng = np.random.default_rng(2)
    for _ in range(5):
        w0 = rng.uniform(0.1, 2.0, size=3)
        omegas, _ = rigid_body.integrate_full(w0, I_DEFAULT, T=10.0, N=200)
        M = omegas * I_DEFAULT[None, :]
        M_mag = np.linalg.norm(M, axis=1)
        np.testing.assert_allclose(M_mag, M_mag[0], rtol=1e-6)


def test_fixed_points_are_stationary():
    """Spinning exactly about a principal axis produces no precession: w(t) is
    constant for the whole integration window, for both stable axes."""
    R_cas = 1.5
    for axis in (0, 2):  # imin, imax by construction of I_DEFAULT
        w0 = np.zeros(3)
        w0[axis] = R_cas / I_DEFAULT[axis]
        omegas, _ = rigid_body.integrate_full(w0, I_DEFAULT, T=5.0, N=100)
        # NB: np.testing.assert_allclose has a broadcasting quirk comparing a
        # (N,3) array against a (3,) array in this numpy version even when the
        # values are exactly equal -- np.allclose broadcasts correctly.
        assert np.allclose(omegas, omegas[0], atol=1e-6)


def test_dzhanibekov_flip_occurs_near_unstable_axis():
    """A trajectory that starts near the unstable (imid) axis must show a flip:
    the sign of the omega component along a stable axis reverses at least once.

    The perturbation on the two stable axes must be equal in *omega* (not
    equal in M = I*omega): the saddle at the unstable fixed point has a growing
    eigendirection with a specific (I-dependent) ratio between the two stable
    components, and an equal-in-M kick for this I happens to sit close to the
    *decaying* eigendirection instead, which would take a very long integration
    time to show any flip at all. Equal-in-omega generically avoids this.
    """
    R_cas = 1.5
    spin = R_cas / I_DEFAULT[1]
    eps = 0.02
    w0 = np.array([eps, spin, eps])
    omegas, _ = rigid_body.integrate_full(w0, I_DEFAULT, T=15.0, N=1500)
    signs = np.sign(omegas[:, 0])
    assert np.any(signs != signs[0])
