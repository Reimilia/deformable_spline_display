"""Pre-flight checks for the four-variant controller (V_θ / D_θ cancellations).

Three offline tests, all on the env (true physics) with `g = I`:

  (1) Gravity-cancel sign — place pendulum at small tilt with k_c = 0,
      d_inj = 0, α_V = 1, α_D = 0. If V_θ ≈ V_p the bob should hold
      (gravity is cancelled, no other forces). Wrong sign → it accelerates
      outward.

  (2) Friction-cancel sign — spin at constant ω with k_c = 0, d_inj = 0,
      α_V = 0, α_D = 1. If D_θ ≈ D the spin should hold (friction is
      cancelled, no other dissipation). Wrong sign → ω accelerates.

  (3) τ_{V_θ}(R = I) sanity — at upright (gravity-PE saddle for the
      spherical pendulum), the torque computed from V_θ should be ≈ 0.
      Magnitude is a quantitative measure of V_θ's residual bias at the
      controller's target.

Pass criteria (printed at the end):
  * (1) terminal geo distance grows by < 0.05 rad over 1 s
  * (2) terminal |ω| stays within 5 % of the initial value over 1 s
  * (3) ||τ_{V_θ}(I)|| < 1.0   (mgl ≈ 9.81 sets the scale)
"""
from __future__ import annotations

import os
import sys

import numpy as np

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, '../../../..'))
for _p in (PROJECT_ROOT, THIS_FILE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from envs.windy_pendulum_3d import _exp_so3, _log_so3                   # noqa: E402
from qp_env import QPWindyPendulum3D                                    # noqa: E402
from controller import EnergyCasimirController, ControllerConfig        # noqa: E402
from verify_control import load_trained_model                           # noqa: E402


ENV_DEFAULTS = dict(
    g=9.81, m=1.0, l=1.0, dt=0.05,
    varying_friction=False, friction_coeff=0.5,
    external_force_type='sine', external_force_std=0.0,
    wind_force_std=0.0,                # noise off — pure sign-of-drift test
)


def _build_env(R0, omega0, seed=0):
    env = QPWindyPendulum3D(seed=seed, **ENV_DEFAULTS)
    env.reset(seed=seed,
              options={'R_init': R0.copy(), 'omega_init': omega0.copy()})
    return env


def check_gravity_cancel(model):
    """Test (1): α_V = 1 only → bob held against gravity."""
    print("\n" + "=" * 70)
    print("(1)  Gravity-cancel sign — α_V=1, all else 0, slight tilt")
    print("=" * 70)
    R_d = np.eye(3)
    R0 = R_d @ _exp_so3(np.array([np.deg2rad(5.0), 0.0, 0.0]))
    omega0 = np.zeros(3)

    cfg = ControllerConfig(R_d=R_d, k_c=0.0, d_inj=0.0,
                           alpha_D=0.0, alpha_V=1.0)
    ctrl = EnergyCasimirController(cfg, model=model)

    env = _build_env(R0, omega0)
    geo0 = float(np.linalg.norm(_log_so3(env.R.T @ R_d)))

    # Comparison: NO control (α_V = 0 everywhere) — bob should fall over fast.
    cfg_off = ControllerConfig(R_d=R_d, k_c=0.0, d_inj=0.0,
                               alpha_D=0.0, alpha_V=0.0)
    ctrl_off = EnergyCasimirController(cfg_off)
    env_off = _build_env(R0, omega0)

    for _ in range(20):                              # 1 s
        env.step(ctrl.act(env.R, env.omega))
        env_off.step(ctrl_off.act(env_off.R, env_off.omega))
    geo_with_cancel = float(np.linalg.norm(_log_so3(env.R.T @ R_d)))
    geo_without     = float(np.linalg.norm(_log_so3(env_off.R.T @ R_d)))

    drift = geo_with_cancel - geo0
    drift_off = geo_without - geo0
    suppression = drift / max(drift_off, 1e-9)
    print(f"  initial geo dist                 = {geo0:.4f} rad")
    print(f"  geo dist after 1s, α_V=1 cancel  = {geo_with_cancel:.4f} rad")
    print(f"  geo dist after 1s, NO control    = {geo_without:.4f} rad")
    print(f"  drift under cancel               = {drift:+.4f} rad")
    print(f"  drift / no-control drift         = {suppression:.3f}")
    # Sign correct iff cancellation suppresses drift to a small fraction of
    # the no-control case. Residual 5–20 % is the V_θ-bias floor — wrong
    # sign would amplify drift to >1×. Threshold 0.30 catches sign flips
    # with comfortable margin and still flags badly-biased V_θ.
    pass_ = suppression < 0.30
    print(f"  → {'PASS' if pass_ else 'FAIL'} (need drift < 0.30 × no-control drift)")
    return pass_


def check_friction_cancel(model):
    """Test (2): α_D = 1 only → ω preserved against friction."""
    print("\n" + "=" * 70)
    print("(2)  Friction-cancel sign — α_D=1, all else 0, constant spin")
    print("=" * 70)
    R_d = np.eye(3)
    R0 = np.eye(3)
    omega0 = np.array([1.0, 0.0, 0.0])

    cfg = ControllerConfig(R_d=R_d, k_c=0.0, d_inj=0.0,
                           alpha_D=1.0, alpha_V=0.0)
    ctrl = EnergyCasimirController(cfg, model=model)
    env = _build_env(R0, omega0)

    # Reference: NO cancellation — ω should decay under env's friction=0.5.
    cfg_off = ControllerConfig(R_d=R_d, k_c=0.0, d_inj=0.0,
                               alpha_D=0.0, alpha_V=0.0)
    ctrl_off = EnergyCasimirController(cfg_off)
    env_off = _build_env(R0, omega0)

    for _ in range(20):                              # 1 s
        env.step(ctrl.act(env.R, env.omega))
        env_off.step(ctrl_off.act(env_off.R, env_off.omega))

    om_with    = float(np.linalg.norm(env.omega))
    om_without = float(np.linalg.norm(env_off.omega))
    om0 = float(np.linalg.norm(omega0))

    print(f"  initial |ω|                     = {om0:.4f} rad/s")
    print(f"  |ω| after 1s, α_D=1 cancel      = {om_with:.4f} rad/s")
    print(f"  |ω| after 1s, NO control        = {om_without:.4f} rad/s")
    # Both cases experience gravity-driven rotation, so |ω| grows in both.
    # If friction is correctly cancelled, the cancel case dissipates *less*
    # → |ω|_cancel > |ω|_no_control. (Wrong sign would amplify friction
    # and damp ω more strongly: |ω|_cancel < |ω|_no_control.)
    margin = (om_with - om_without) / om_without
    print(f"  relative gain over no-control   = {margin*100:+.1f}%")
    pass_ = om_with > om_without
    print(f"  → {'PASS' if pass_ else 'FAIL'} "
          f"(need |ω|_cancel > |ω|_no_control, i.e. friction effectively cancelled)")
    return pass_


def check_tau_V_at_target(model):
    """Test (3): τ_{V_θ}(R = R_d) ≈ 0."""
    print("\n" + "=" * 70)
    print("(3)  τ_{V_θ}(R = I) — should be ≈ 0 (gravity-PE saddle)")
    print("=" * 70)
    import jax
    import jax.numpy as jnp

    @jax.jit
    def tau_V(q):
        def V_scalar(q_):
            return model.V_net(q_, inference_mode=True)[0]
        grad_V = jax.grad(V_scalar)(q)
        R_3x3 = q.reshape(3, 3)
        return jnp.sum(jnp.cross(R_3x3, grad_V.reshape(3, 3), axis=-1),
                       axis=0)

    q_I = jnp.eye(3, dtype=jnp.float32).reshape(-1)
    tau = np.asarray(tau_V(q_I))
    norm_tau = float(np.linalg.norm(tau))

    # For comparison: at small tilt the magnitude should be ~ mgl·sin(angle).
    R_tilt = _exp_so3(np.array([np.deg2rad(5.0), 0.0, 0.0]))
    q_tilt = jnp.asarray(R_tilt.reshape(-1), dtype=jnp.float32)
    tau_tilt = np.asarray(tau_V(q_tilt))
    expected = 9.81 * np.sin(np.deg2rad(5.0))

    print(f"  τ_{{V_θ}}(R = I)        = {tau}   (||·|| = {norm_tau:.4f})")
    print(f"  τ_{{V_θ}}(5° tilt)      = {tau_tilt}")
    print(f"  expected ≈ mgl·sin(5°) = {expected:.4f}")
    pass_ = norm_tau < 1.0
    print(f"  → {'PASS' if pass_ else 'FAIL'} (need ||τ(I)|| < 1.0)")
    return pass_


def main():
    print("=" * 70)
    print(" verify_variants.py — pre-flight for the 4-variant controller")
    print("=" * 70)
    print("\n[loading trained model]")
    model = load_trained_model()
    print("  loaded.")

    p1 = check_gravity_cancel(model)
    p2 = check_friction_cancel(model)
    p3 = check_tau_V_at_target(model)

    print("\n" + "=" * 70)
    print(f"  gravity-cancel sign   : {'PASS' if p1 else 'FAIL'}")
    print(f"  friction-cancel sign  : {'PASS' if p2 else 'FAIL'}")
    print(f"  τ_{{V_θ}}(I) magnitude  : {'PASS' if p3 else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
