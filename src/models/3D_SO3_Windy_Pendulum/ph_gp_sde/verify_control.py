"""Pre-control sanity checks for the SO(3) Energy-Casimir controller.

Resolves the four open questions before any controller code gets written:

  (1) Sign of   u_p = ±k_c · log(R^T R_d)   on the env.
      Apply a small candidate torque from a slightly-tilted R toward R_d and
      see whether the geodesic distance to R_d shrinks. Fixes the sign
      ambiguity flagged in §7 of the derivation.

  (2) Whether the trained g_theta(R) is close enough to identity that we can
      hardcode g = I in the controller (the env's actuation is u_body added
      directly to the body torque — see envs/windy_pendulum_3d.py
      `_compute_omega_rates`, line "tau_det = ... + u - tau_fric").

  (3) The Itô correction to dH_cl in the *consistent* state-space (the
      integrator carries (q, p), so Hessian = M^-1 and diffusion is on p,
      not on omega). Reports the eigenvalues of Sigma_p Sigma_p^T to expose
      the rank-2 structure of the wind-noise channel.

  (4) A recommended k_c (energy-shaping spring stiffness) and d_inj
      (damping injection) consistent with (3).

Usage (no args needed):
    python src/models/3D_SO3_Windy_Pendulum/ph_gp_sde/verify_control.py

The script never modifies the trained model or env code.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, '../../../..'))
for _p in (PROJECT_ROOT, THIS_FILE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from envs.windy_pendulum_3d import windy_pendulum_3d, _exp_so3, _log_so3, _hat
from network import DissipativeSO3HamSDE  # local import (THIS_FILE_DIR on sys.path)


# ─────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────

DEFAULT_CKPT = os.path.join(
    THIS_FILE_DIR, 'data', 'run_wp3d_jax', 'wp3d-so3hamGPSDE-5p.eqx',
)


def load_trained_model(ckpt_path: str = DEFAULT_CKPT) -> DissipativeSO3HamSDE:
    """Build a fresh template model with the same constructor args used in
    train.py, then deserialise the trained weights into it.

    Static fields (u_dim=3, hidden_dim=20, init_sigma_obs_omega=0.5) match
    the training defaults — see ph_gp_sde/train.py get_args().
    """
    template = DissipativeSO3HamSDE(
        key=jax.random.PRNGKey(0),
        u_dim=3,
        init_gain=0.5,
        friction=True,
        hidden_dim=20,
        l=1.0,
        init_sigma_R=0.1,
        init_sigma_omega=0.1,
        init_sigma_obs_omega=0.5,   # matches training default
        init_sigma_const=0.5,
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    model = eqx.tree_deserialise_leaves(ckpt_path, template)
    return model


# ─────────────────────────────────────────────────────────────────────
# (1) Sign check: does u_p = -k_c log(R^T R_d) drive R toward R_d?
# ─────────────────────────────────────────────────────────────────────

def geodesic_distance(R, R_d):
    return float(np.linalg.norm(_log_so3(R.T @ R_d)))


def check_sign_convention(k_c: float = 30.0, tilt_deg: float = 10.0,
                          n_steps: int = 20):
    """Place a friction-free, noise-free env at R = R_d · exp(tilt · e_x),
    omega = 0. Apply each of the two candidate torques for `n_steps` and
    report which one makes ||log(R^T R_d)|| decrease.
    """
    print("\n" + "=" * 70)
    print("(1)  Sign convention check on env")
    print("=" * 70)

    R_d = np.eye(3)
    tilt_rad = np.deg2rad(tilt_deg)
    R0 = R_d @ _exp_so3(np.array([tilt_rad, 0.0, 0.0]))

    candidates = {
        '+k_c · log(R^T R_d)': +1.0,
        '-k_c · log(R^T R_d)': -1.0,
    }
    results = {}

    for label, sign in candidates.items():
        # Friction off, wind off, noise off — isolate just (gravity + control).
        env = windy_pendulum_3d(
            g=9.81, m=1.0, l=1.0, dt=0.05,
            varying_friction=False, friction_coeff=0.0,
            external_force_type='constant', external_force_std=0.0,
            wind_force_std=0.0, seed=0,
        )
        env.reset(seed=0, options={'R_init': R0.copy(),
                                   'omega_init': np.zeros(3)})

        d0 = geodesic_distance(env.R, R_d)
        d_traj = [d0]
        for _ in range(n_steps):
            phi = _log_so3(env.R.T @ R_d)             # 3-vector body-frame
            u = sign * k_c * phi                      # candidate body torque
            env.step(u)
            d_traj.append(geodesic_distance(env.R, R_d))

        d_end = d_traj[-1]
        d_min = min(d_traj)
        results[label] = (d0, d_end, d_min)
        print(f"  {label:>26s} : d0={d0:.4f}  d_end={d_end:.4f}  "
              f"d_min={d_min:.4f}")

    # Verdict
    plus = results['+k_c · log(R^T R_d)']
    minus = results['-k_c · log(R^T R_d)']
    if plus[1] < plus[0] and minus[1] >= minus[0]:
        verdict = "USE  u_p = +k_c · log(R^T R_d)"
    elif minus[1] < minus[0] and plus[1] >= plus[0]:
        verdict = "USE  u_p = -k_c · log(R^T R_d)"
    elif plus[1] < minus[1]:
        verdict = "USE  u_p = +k_c · log(R^T R_d)  (smaller terminal d)"
    else:
        verdict = "USE  u_p = -k_c · log(R^T R_d)  (smaller terminal d)"
    print("  →", verdict)


# ─────────────────────────────────────────────────────────────────────
# (2) Evaluate trained g_theta(R) at sampled rotations
# ─────────────────────────────────────────────────────────────────────

def random_rotation(rng: np.random.Generator) -> np.ndarray:
    u1, u2, u3 = rng.random(3)
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    x, y, z, w = q1, q2, q3, q4
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def check_g_vs_identity(model: DissipativeSO3HamSDE, n_samples: int = 8):
    """Evaluate g_theta(R) at upright, downright, and 6 random rotations.
    Report Frobenius distance to I, condition number, and singular values.

    The env adds u directly into the body torque (no g matrix), so the
    physical g_true = I. If the trained g_theta is close to I (any
    constant scaling factor would fold into k_c), we can hardcode g = I
    in the controller and skip the pinv; otherwise we need pinv(g_theta).
    """
    print("\n" + "=" * 70)
    print("(2)  Trained g_theta(R) vs identity")
    print("=" * 70)

    rng = np.random.default_rng(0)
    Rs = [np.eye(3),
          np.diag([-1.0, -1.0, 1.0]),                    # downright
          _exp_so3(np.array([0.5, 0.0, 0.0])),
          _exp_so3(np.array([0.0, 0.5, 0.0]))]
    Rs += [random_rotation(rng) for _ in range(n_samples - len(Rs))]

    g_call = jax.jit(lambda q: model.g_net(q, inference_mode=True))

    print(f"  {'R label':<22s} {'||g-I||_F':>10s} {'cond(g)':>10s} "
          f"{'sv_min':>10s} {'sv_max':>10s}")
    g_mats = []
    for i, R in enumerate(Rs):
        q = jnp.asarray(R.reshape(-1), dtype=jnp.float32)
        g = np.asarray(g_call(q))
        g_mats.append(g)
        sv = np.linalg.svd(g, compute_uv=False)
        cond = sv[0] / max(sv[-1], 1e-12)
        diff = np.linalg.norm(g - np.eye(3))
        label = ('upright' if i == 0
                 else 'downright' if i == 1
                 else f'sample {i}')
        print(f"  {label:<22s} {diff:>10.4f} {cond:>10.3f} "
              f"{sv[-1]:>10.4f} {sv[0]:>10.4f}")

    g_stack = np.stack(g_mats)
    mean_g = g_stack.mean(axis=0)
    std_g = g_stack.std(axis=0)
    print(f"\n  mean g across samples =\n{mean_g}")
    print(f"  std  g across samples =\n{std_g}")

    max_diff = max(np.linalg.norm(g - np.eye(3)) for g in g_mats)
    if max_diff < 0.15:
        print("  → g_theta ≈ I across SO(3); hardcode g = I in controller "
              "(skip pinv).")
    elif max_diff < 0.5:
        print("  → g_theta drifts modestly from I; pinv(g_theta) is safer "
              "but g = I will work approximately.")
    else:
        print("  → g_theta is meaningfully non-identity; use pinv(g_theta) "
              "in the controller and watch the condition number.")


# ─────────────────────────────────────────────────────────────────────
# (3) Itô correction in (q, p)-space with the rank-2 wind structure
# ─────────────────────────────────────────────────────────────────────

def check_ito_correction(model: DissipativeSO3HamSDE,
                         R_eval: np.ndarray = None,
                         l_arm: float = 1.0):
    """Compute the Itô correction in p-space at R_eval using the trained
    sigma_net and M_net.

    Plant SDE on p (from network.stochastic_increment_p):
        dp_stoch = R^T ( (l · R e_z) × ( σ(q) · dW ) )
                 = - σ(q) · R^T · [l · R e_z]_x · dW
                 =:   Σ_p(q) · dW

    Hessian of H_p w.r.t. p is M^-1(q) = M_theta^-1(q).

    Itô correction to dH_cl (and thus to L H_cl):
        (1/2) tr( Σ_p^T  M^-1  Σ_p ).

    Eigenvalues of Σ_p Σ_p^T expose the rank-2 wind structure: the radial
    direction along R e_z is annihilated by the cross product, so one
    eigenvalue is exactly zero.
    """
    print("\n" + "=" * 70)
    print("(3)  Itô correction in (q,p)-space at R_eval = I  (rank-2 wind)")
    print("=" * 70)

    if R_eval is None:
        R_eval = np.eye(3)

    q = jnp.asarray(R_eval.reshape(-1), dtype=jnp.float32)
    sigma_q = float(model.sigma(q))                  # scalar > 0
    M_inv_q = np.asarray(model.M_inv(q))             # 3×3 PSD

    ez = np.array([0.0, 0.0, 1.0])
    r_world = l_arm * (R_eval @ ez)                  # bob position
    r_hat = _hat(r_world)                            # 3×3 skew
    # Σ_p = - σ · R^T · [r_world]_x   (no σ-shift; absorbs sign of σ since σ≥0)
    Sigma_p = -sigma_q * (R_eval.T @ r_hat)          # 3×3

    SS = Sigma_p @ Sigma_p.T
    eig_SS = np.linalg.eigvalsh(SS)                  # ascending

    ito_p = 0.5 * float(np.trace(Sigma_p.T @ M_inv_q @ Sigma_p))

    # Cross-check in ω-space: dω_stoch = M^-1 · dp_stoch  ⇒ Σ_ω = M^-1 Σ_p,
    # and Hessian of H w.r.t. ω is M (not M^-1). So
    #   (1/2) tr(Σ_ω^T M Σ_ω) = (1/2) tr(Σ_p^T M^-1 Σ_p)
    # — same scalar, sanity-checked by direct computation.
    M_q = np.linalg.inv(M_inv_q)
    Sigma_w = M_inv_q @ Sigma_p
    ito_w = 0.5 * float(np.trace(Sigma_w.T @ M_q @ Sigma_w))

    print(f"  σ_theta(R_eval)              = {sigma_q:.4f}")
    print(f"  M_theta^-1(R_eval) =\n{M_inv_q}")
    print(f"  l (lever arm)               = {l_arm}")
    print(f"  Σ_p Σ_p^T eigenvalues       = {eig_SS}")
    print(f"    → rank ≈ {int(np.sum(eig_SS > 1e-6))} (expect 2; wind is rank-2)")
    print(f"  Itô correction (p-space)    = {ito_p:.6f}")
    print(f"  Itô correction (ω-space)    = {ito_w:.6f}  "
          f"(should match p-space)")
    return ito_p, sigma_q, M_inv_q


# ─────────────────────────────────────────────────────────────────────
# (4) Recommend k_c and d_inj
# ─────────────────────────────────────────────────────────────────────

def recommend_gains(ito_correction: float,
                    M_inv_at_Rd: np.ndarray,
                    m: float = 1.0, g: float = 9.81, l: float = 1.0):
    """k_c > m·g·l makes R_d a strict local min of V_p + (k_c/2)|log R^T R_d|^2.
    Take 3× margin per the derivation (§9).

    For d_inj, the deterministic-decay budget at omega = 0 must dominate the
    Itô correction in expectation; pick d_inj so that for typical |omega| ≥
    |omega|_thresh, the linear damping torque dissipates more than the noise
    injects. With the trained scalar correction, a conservative choice is

        d_inj ≥ ito_correction / |omega|_thresh^2   (pure-noise floor)

    plus a margin so the system is contracting in mean-square inside the
    target neighborhood. We default to |omega|_thresh = 0.5 rad/s.
    """
    print("\n" + "=" * 70)
    print("(4)  Recommended controller gains")
    print("=" * 70)

    k_c_min_local = m * g * l
    k_c_min_global = 4 * m * g * l / np.pi**2
    k_c_recommended = 3.0 * k_c_min_local

    print(f"  m·g·l                      = {k_c_min_local:.3f}")
    print(f"  k_c > m·g·l (local  min)   ≥ {k_c_min_local:.3f}")
    print(f"  k_c > 4mgl/π² (R_d cheaper than R_down) ≥ "
          f"{k_c_min_global:.3f}")
    print(f"  → k_c (3× margin)          = {k_c_recommended:.3f}")

    omega_thresh = 0.5
    d_floor = ito_correction / max(omega_thresh ** 2, 1e-12)
    # Sanity: d_inj should also be at least O(1) so it visibly damps in
    # finite time at modest |omega|.
    d_inj_recommended = max(2.0 * d_floor, 1.0)
    print(f"  Itô correction (from §3)   = {ito_correction:.6f}")
    print(f"  ω_thresh                   = {omega_thresh:.2f} rad/s")
    print(f"  d_inj floor (= ito/ω²)     = {d_floor:.6f}")
    print(f"  → d_inj (≥ 2× floor, ≥ 1)  = {d_inj_recommended:.3f}")
    return k_c_recommended, d_inj_recommended


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(" verify_control.py  —  pre-controller sanity checks")
    print("=" * 70)

    # (1) — env-only, no model needed
    check_sign_convention(k_c=30.0, tilt_deg=10.0, n_steps=20)

    # Load trained model for (2), (3)
    print("\n[loading trained model]")
    model = load_trained_model()
    print(f"  loaded {DEFAULT_CKPT}")

    # (2)
    check_g_vs_identity(model)

    # (3)
    ito, sigma_q, M_inv_q = check_ito_correction(model)

    # (4)
    recommend_gains(ito, M_inv_q)

    print("\n" + "=" * 70)
    print(" done")
    print("=" * 70)


if __name__ == "__main__":
    main()
