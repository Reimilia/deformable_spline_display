import numpy as np
import pytest

import controllers
import rigid_body

I_DEFAULT = np.array([0.075, 0.875, 0.950])


def test_hat_vee_are_inverses():
    rng = np.random.default_rng(0)
    for _ in range(10):
        v = rng.normal(size=3)
        np.testing.assert_allclose(controllers.vee(controllers.hat(v)), v, atol=1e-10)


def test_expm_logm_are_inverses_away_from_antipodal():
    rng = np.random.default_rng(1)
    for _ in range(10):
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v) * rng.uniform(0.01, 3.0)  # keep away from theta=pi
        R = controllers.expm_SO3(controllers.hat(v))
        v_back = controllers.vee(controllers.logm_SO3(R))
        np.testing.assert_allclose(v_back, v, atol=1e-6)


# Calibrated numerically against I_DEFAULT (not copied from the PR's cfg0 numbers,
# which used a very different inertia scale, I ~ 0.006-0.013 vs I_DEFAULT's
# 0.075-0.95): Kp=1.0 over T=6.0s converges the axis-2 spin-up to <1% error.
STAGE_D_KP = 1.0
STAGE_D_T = 6.0


def test_controller_stageD_converges_to_target_axis():
    omega_star = np.array([0.0, 0.0, 2 * np.pi])
    ctrl = controllers.PDBodyFrameController(omega_star, Kp=STAGE_D_KP)

    def torque_fn(t, w, q):
        return ctrl(rigid_body.quat_to_R(q), w)

    w0 = np.array([0.0, 2 * np.pi, 0.0])  # start spinning on the unstable axis
    omegas, _ = rigid_body.integrate_full(
        w0, I_DEFAULT, T=STAGE_D_T, N=int(STAGE_D_T * 100), torque_fn=torque_fn
    )

    final_err = np.linalg.norm(omegas[-1] - omega_star)
    assert final_err < 0.05 * np.linalg.norm(omega_star)


def test_controller_stageF_converges_to_target_orientation():
    """K_R=K_p=1.0 over T=8.0s (numerically calibrated against I_DEFAULT, same
    caveat as Stage D above) converges both attitude and rate error below 0.01."""
    R_star = np.eye(3)
    omega_star = np.zeros(3)
    ctrl = controllers.GeometricAttitudeController(R_star, omega_star, K_R=1.0, K_p=1.0)

    def torque_fn(t, w, q):
        return ctrl(rigid_body.quat_to_R(q), w)

    q0 = rigid_body.normalize_q(np.array([0.9, 0.1, 0.2, 0.1]))
    w0 = np.array([0.5, -0.3, 0.2])
    T = 8.0
    omegas, quats = rigid_body.integrate_full(
        w0, I_DEFAULT, T=T, N=int(T * 100), torque_fn=torque_fn, q0=q0
    )

    R_final = rigid_body.quat_to_R(quats[-1])
    attitude_err = np.linalg.norm(controllers.vee(controllers.logm_SO3(R_star.T @ R_final)))
    assert attitude_err < 0.05
    assert np.linalg.norm(omegas[-1]) < 0.05


def test_controlled_energy_is_not_conserved():
    """Sanity check: with an active controller, H(t) must actually move toward
    the target (not sit flat as in the free case) -- guards against a bug where
    the controller is computed but never coupled into the integration."""
    omega_star = np.array([0.0, 0.0, 2 * np.pi])
    H_target = 0.5 * np.sum(omega_star**2 * I_DEFAULT)
    ctrl = controllers.PDBodyFrameController(omega_star, Kp=STAGE_D_KP)

    def torque_fn(t, w, q):
        return ctrl(rigid_body.quat_to_R(q), w)

    w0 = np.array([0.0, 2 * np.pi, 0.0])
    omegas, _ = rigid_body.integrate_full(
        w0, I_DEFAULT, T=STAGE_D_T, N=int(STAGE_D_T * 100), torque_fn=torque_fn
    )
    H = 0.5 * np.sum(omegas**2 * I_DEFAULT, axis=1)

    assert abs(H[0] - H[-1]) > 1e-3, "H did not change -- controller torque may not be applied"
    assert abs(H[-1] - H_target) < abs(H[0] - H_target), "H did not move toward the target"


def test_controlled_angular_momentum_changes():
    """|M(t)| must visibly leave the initial Casimir radius once torque is applied."""
    omega_star = np.array([0.0, 0.0, 2 * np.pi])
    ctrl = controllers.PDBodyFrameController(omega_star, Kp=STAGE_D_KP)

    def torque_fn(t, w, q):
        return ctrl(rigid_body.quat_to_R(q), w)

    w0 = np.array([0.0, 2 * np.pi, 0.0])
    omegas, _ = rigid_body.integrate_full(
        w0, I_DEFAULT, T=STAGE_D_T, N=int(STAGE_D_T * 100), torque_fn=torque_fn
    )
    M = omegas * I_DEFAULT[None, :]
    M_mag = np.linalg.norm(M, axis=1)

    assert abs(M_mag[-1] - M_mag[0]) > 1e-3
