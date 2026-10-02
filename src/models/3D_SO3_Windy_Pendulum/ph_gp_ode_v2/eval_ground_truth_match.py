"""SDE sanity check — does the port-Hamiltonian Stratonovich SDE reproduce
the env's stochastic Lie-group Heun trajectory?

Mirrors ph_nn_ode_fp32/eval_ground_truth_match.py but for the SDE port:
  - Uses lie_heun_sde_rollout (custom Stratonovich Heun on SO(3) × ℝ³)
    instead of Diffrax — same integrator structure as the env.
  - Replaces the four subnets + the new sigma_net with their analytical
    ground-truth values for the spherical pendulum.
  - Mirrors the env's numpy RNG so the SAME Wiener increments dW drive both
    the env and the model (otherwise stochastic trajectories trivially differ).

If the math is correct the env and model trajectories should agree to
*machine precision* in fp32 (no integrator-form mismatch — both use the same
Lie-Heun structure, same substep count, same dW).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

import jax
import jax.numpy as jnp
import equinox as eqx

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, '../../../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if THIS_FILE_DIR not in sys.path:
    sys.path.insert(0, THIS_FILE_DIR)

from envs.windy_pendulum_3d import windy_pendulum_3d
from src.utils.JAX.lie_integrator import lie_heun_sde_rollout


# ─────────────────────────────────────────────────────────────────────
# Ground-truth-subnet SDE model
# ─────────────────────────────────────────────────────────────────────

class GroundTruthHamiltonianSDE(eqx.Module):
    """DissipativeSO3HamSDE shape, subnets replaced with true physics.

    Conventions match the trained network exactly so any bug in the math
    (deterministic *or* stochastic) shows up as nonzero MSE here.

      M_net(q)              = (1/(m·l²)) · I₃                  → I₃ for m=l=1
      V_net(q)              = m·g·l · q[8]                     (R[2,2])
      Dw_net(q)             = friction_coeff · I₃
      g_net(q)              = I₃                                (body-frame torque)
      sigma_net(q)          = wind_force_std                    (env constant)
    """
    m:              float = eqx.field(static=True)
    l:              float = eqx.field(static=True)
    g:              float = eqx.field(static=True)
    friction_coeff: float = eqx.field(static=True)
    wind_force_std: float = eqx.field(static=True)
    rotmatdim:      int   = eqx.field(static=True)
    angveldim:      int   = eqx.field(static=True)
    u_dim:          int   = eqx.field(static=True)

    def __init__(self, m=1.0, l=1.0, g=9.81, friction_coeff=0.5, wind_force_std=0.5):
        self.m = float(m)
        self.l = float(l)
        self.g = float(g)
        self.friction_coeff = float(friction_coeff)
        self.wind_force_std = float(wind_force_std)
        self.rotmatdim = 9
        self.angveldim = 3
        self.u_dim = 3

    # ── Subnet ground truths (single sample, q ∈ ℝ⁹) ─────────────────
    def M_net(self, q):
        return (1.0 / (self.m * self.l ** 2)) * jnp.eye(3, dtype=q.dtype)

    def V_net(self, q):
        return jnp.array([(self.m * self.g * self.l) * q[8]], dtype=q.dtype)

    def Dw_net(self, q):
        return self.friction_coeff * jnp.eye(3, dtype=q.dtype)

    def g_net(self, q):
        return jnp.eye(3, dtype=q.dtype)

    def sigma(self, q):
        # Constant — matches env.wind_force_std
        return jnp.array(self.wind_force_std, dtype=q.dtype)

    # ── M⁻¹ accessor (used by the (q, p) integrator for ω = M⁻¹·p) ─────
    def M_inv(self, q, keys=None):
        del keys
        return self.M_net(q)

    # ── Diffusion in p-units (body-frame torque increment) ───────────────
    def stochastic_increment_p(self, q, dW, keys=None):
        del keys
        R = q.reshape(3, 3)
        ez = jnp.array([0.0, 0.0, 1.0], dtype=q.dtype)
        r_world = self.l * (R @ ez)
        dF_stoch = self.sigma(q) * dW
        torque_world = jnp.cross(r_world, dF_stoch)
        return R.T @ torque_world

    # ── ω-form wrapper kept for any caller that still consumes dω_stoch ─
    def stochastic_increment(self, q, dW):
        return self.M_net(q) @ self.stochastic_increment_p(q, dW)

    # ── Drift in p-form (ṗ = port-Hamiltonian momentum derivative) ──────
    def drift_p(self, q, p, u, keys=None):
        del keys

        def H_of_q(q_):
            return 0.5 * jnp.dot(p, self.M_net(q_) @ p) + self.V_net(q_)[0]
        dHdq = jax.grad(H_of_q)(q)

        M_q_inv = self.M_net(q)
        g_q     = self.g_net(q)
        Dw_q    = self.Dw_net(q)

        dHdp = M_q_inv @ p
        F    = g_q @ u

        R_3x3    = q.reshape(3, 3)
        dHdq_3x3 = dHdq.reshape(3, 3)
        grav = jnp.sum(jnp.cross(R_3x3, dHdq_3x3, axis=-1), axis=0)

        return jnp.cross(p, dHdp) + grav - (Dw_q @ dHdp) + F

    # ── ω-form wrapper kept for any caller that still consumes ω̇ ───────
    def drift(self, q, q_dot, u):
        rd = self.rotmatdim
        M_q_inv = self.M_net(q)
        p_val   = jnp.linalg.solve(M_q_inv, q_dot)
        p       = jax.lax.stop_gradient(p_val)

        dp = self.drift_p(q, p, u)

        dHdp   = M_q_inv @ p
        R_3x3  = q.reshape(3, 3)
        dHdp_b = jnp.broadcast_to(dHdp[None, :], (3, 3))
        dq     = jnp.cross(R_3x3, dHdp_b, axis=-1).reshape(rd)

        _, dM_inv_dt = jax.jvp(self.M_net, (q,), (dq,))
        return M_q_inv @ dp + dM_inv_dt @ p


# ─────────────────────────────────────────────────────────────────────
# Env rollout + matched RNG dW sequence
# ─────────────────────────────────────────────────────────────────────

def run_env_with_recorded_dW(args, dt, n_steps, n_substeps):
    """Roll out the env AND mirror its RNG to produce the same dW sequence.

    The env's _np_rng is consumed in this exact order in reset+step:

        reset(seed=S):
            self.seed(S)                   # self._np_rng = np.random.default_rng(S)
            _random_rotation(rng)          # consumes rng.random(3)
            rng.uniform(-1, 1, size=3)     # omega0  (3 floats)
        step(u):  (per outer step, n_substeps inner)
            if wind_force_std > 0:
                rng.normal(0, sqrt(dt_sub), 3)  # ONE per substep
            else:
                no RNG consumption
    """
    env = windy_pendulum_3d(
        g=9.81, m=1.0, l=1.0, dt=dt,
        friction_coeff=args.friction_coeff, varying_friction=False,
        external_force_type="sine", external_force_std=0.0,
        external_force_direction=(1.0, 0.0, 0.0), wind_force_std=args.wind_force_std,
        ori_rep="rotmat", render_mode=None, seed=args.seed,
    )
    obs, _ = env.reset(seed=args.seed)
    R0, omega0 = env.get_state()

    u_const = np.asarray(args.u, dtype=np.float64)
    traj = [np.concatenate([obs, u_const.astype(np.float32)])]
    for _ in range(n_steps):
        obs, _, _, _, _ = env.step(u_const)
        traj.append(np.concatenate([obs, u_const.astype(np.float32)]))
    env.close()
    x_traj = np.stack(traj, axis=0)
    t = np.arange(x_traj.shape[0]) * dt

    # Mirror RNG to recover the dW path the env saw.
    dt_sub = dt / n_substeps
    rng = np.random.default_rng(args.seed)
    _ = rng.random(3)                  # consumed by _random_rotation(rng)
    _ = rng.uniform(-1.0, 1.0, size=3) # consumed by w0
    dW_path = np.zeros((n_steps, n_substeps, 3), dtype=np.float64)
    if args.wind_force_std > 0.0:
        for i in range(n_steps):
            for j in range(n_substeps):
                dW_path[i, j] = rng.normal(0.0, np.sqrt(dt_sub), size=3)
    return x_traj, t, R0, omega0, dW_path


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--timesteps', type=int, default=20)
    p.add_argument('--dtype', type=str, default='float32',
                   choices=['float32', 'float64'])
    p.add_argument('--friction_coeff', type=float, default=0.5)
    p.add_argument('--wind_force_std', type=float, default=0.5,
                   help='env Stratonovich diffusion scale; reused as ground-truth σ')
    p.add_argument('--n_substeps', type=int, default=10,
                   help='Lie-Heun substeps per dt (env uses 10 — keep matched)')
    p.add_argument('--u', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                   metavar=('UX', 'UY', 'UZ'),
                   help='constant body-frame torque applied during the rollout')
    p.add_argument('--plot_dir', type=str,
                   default=os.path.join(THIS_FILE_DIR, 'data', 'eval_match'),
                   help='where to save the comparison plots')
    p.add_argument('--no_plot', action='store_true', help='skip plot generation')
    p.add_argument('--compile', action='store_true',
                   help='wrap the rollout in eqx.filter_jit')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────

def _rotmat_to_euler(R_flat):
    Rs = R_flat.reshape(*R_flat.shape[:-1], 3, 3)
    R00, R10, R20 = Rs[..., 0, 0], Rs[..., 1, 0], Rs[..., 2, 0]
    R21, R22 = Rs[..., 2, 1], Rs[..., 2, 2]
    R12, R11 = Rs[..., 1, 2], Rs[..., 1, 1]
    sy = np.sqrt(R00 ** 2 + R10 ** 2)
    near = sy < 1e-6
    roll  = np.where(near, np.arctan2(-R12, R11), np.arctan2(R21, R22))
    pitch = np.arctan2(-R20, sy)
    yaw   = np.where(near, np.zeros_like(R00), np.arctan2(R10, R00))
    return np.stack([roll, pitch, yaw], axis=-1)


def _bob_dir(R_flat):
    return R_flat[..., [2, 5, 8]]


def plot_euler_phase_compare(x_env, x_model, t, save_path, title_extra=""):
    eul_e = _rotmat_to_euler(x_env[:, :9])
    eul_m = _rotmat_to_euler(x_model[:, :9])
    om_e  = x_env[:, 9:12]
    om_m  = x_model[:, 9:12]

    axis_labels = ['X (Roll)', 'Y (Pitch)', 'Z (Yaw)']
    fig, axes = plt.subplots(3, 1, figsize=(7, 10), squeeze=True)
    for axis_idx in range(3):
        ax = axes[axis_idx]
        ang_e = np.unwrap(eul_e[:, axis_idx])
        ang_m = np.unwrap(eul_m[:, axis_idx])
        w_e = om_e[:, axis_idx]; w_m = om_m[:, axis_idx]
        ax.plot(ang_e, w_e, color='C0', lw=1.6, label='env (ground truth)', zorder=2)
        ax.plot(ang_m, w_m, color='C3', lw=1.4, ls='--', label='model (predicted)', zorder=3)
        ax.scatter(ang_e[0], w_e[0], color='black', s=40, marker='*',
                   zorder=5, label='start (shared)')
        ax.scatter(ang_e[-1], w_e[-1], color='C0', s=22, marker='o', zorder=4)
        ax.scatter(ang_m[-1], w_m[-1], color='C3', s=22, marker='o', zorder=4)
        ax.set_xlabel("Angle (rad)", fontsize=10)
        ax.set_ylabel(r"$\omega$ (rad/s)", fontsize=10)
        ax.set_title(f"{axis_labels[axis_idx]}", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        if axis_idx == 0:
            ax.legend(loc='best', fontsize=9, framealpha=0.85)
    fig.suptitle(f"Euler-angle phase space — env vs ground-truth-subnet model"
                 + (f"\n{title_extra}" if title_extra else ""),
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_so3_compare(x_env, x_model, t, save_path, m=1.0, l=1.0, g=9.81,
                     title_extra=""):
    bob_e = _bob_dir(x_env[:, :9]); bob_m = _bob_dir(x_model[:, :9])
    tilt_e = np.arccos(np.clip(bob_e[:, 2], -1.0, 1.0))
    tilt_m = np.arccos(np.clip(bob_m[:, 2], -1.0, 1.0))
    om_e = np.linalg.norm(x_env[:, 9:12], axis=-1)
    om_m = np.linalg.norm(x_model[:, 9:12], axis=-1)
    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
    dtilt_e = np.gradient(tilt_e, dt); dtilt_m = np.gradient(tilt_m, dt)
    E_e = 0.5 * (m * l ** 2) * (om_e ** 2) + m * g * l * bob_e[:, 2]
    E_m = 0.5 * (m * l ** 2) * (om_m ** 2) + m * g * l * bob_m[:, 2]

    fig = plt.figure(figsize=(13, 10))
    ax_s2  = fig.add_subplot(2, 2, 1, projection='3d')
    ax_til = fig.add_subplot(2, 2, 2)
    ax_om  = fig.add_subplot(2, 2, 3)
    ax_E   = fig.add_subplot(2, 2, 4)

    uu, vv = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
    sx = np.cos(uu) * np.sin(vv); sy_ = np.sin(uu) * np.sin(vv); sz = np.cos(vv)
    ax_s2.plot_wireframe(sx, sy_, sz, color='lightgray', alpha=0.3, linewidth=0.5)
    ax_s2.scatter([0], [0], [1], color='green', s=40, label='+e_z')
    ax_s2.scatter([0], [0], [-1], color='red', s=40, label='-e_z')
    ax_s2.plot(bob_e[:, 0], bob_e[:, 1], bob_e[:, 2], color='C0', lw=1.6, label='env')
    ax_s2.plot(bob_m[:, 0], bob_m[:, 1], bob_m[:, 2], color='C3', lw=1.4, ls='--', label='model')
    ax_s2.scatter(*bob_e[0], color='black', s=45, marker='*', zorder=10, label='start')
    ax_s2.scatter(*bob_e[-1], color='C0', s=20, marker='o')
    ax_s2.scatter(*bob_m[-1], color='C3', s=20, marker='o')
    ax_s2.set_xlim(-1.1, 1.1); ax_s2.set_ylim(-1.1, 1.1); ax_s2.set_zlim(-1.1, 1.1)
    ax_s2.set_box_aspect([1, 1, 1])
    ax_s2.set_xlabel('x'); ax_s2.set_ylabel('y'); ax_s2.set_zlabel('z')
    ax_s2.set_title("Bob trajectory on S²", fontsize=11, fontweight='bold')
    ax_s2.legend(loc='upper left', fontsize=8)

    ax_til.plot(tilt_e, dtilt_e, color='C0', lw=1.6, label='env')
    ax_til.plot(tilt_m, dtilt_m, color='C3', lw=1.4, ls='--', label='model')
    ax_til.scatter(tilt_e[0], dtilt_e[0], color='black', s=40, marker='*', zorder=5)
    ax_til.set_xlabel(r"Tilt $\alpha$ (rad)"); ax_til.set_ylabel(r"$\dot{\alpha}$ (rad/s)")
    ax_til.set_title("Tilt phase portrait", fontsize=11, fontweight='bold')
    ax_til.grid(True, alpha=0.3, linestyle='--'); ax_til.axhline(0, color='gray', lw=0.5, alpha=0.5)
    ax_til.legend(loc='best', fontsize=9)

    ax_om.plot(t, om_e, color='C0', lw=1.6, label='env')
    ax_om.plot(t, om_m, color='C3', lw=1.4, ls='--', label='model')
    ax_om.set_xlabel("t (s)"); ax_om.set_ylabel(r"$\|\omega\|$ (rad/s)")
    ax_om.set_title("Angular speed over time", fontsize=11, fontweight='bold')
    ax_om.grid(True, alpha=0.3, linestyle='--'); ax_om.legend(loc='best', fontsize=9)

    ax_E.plot(t, E_e, color='C0', lw=1.6, label='env')
    ax_E.plot(t, E_m, color='C3', lw=1.4, ls='--', label='model')
    ax_E.set_xlabel("t (s)"); ax_E.set_ylabel(r"$E$ (J, $m=l=1$)")
    ax_E.set_title("Total energy over time", fontsize=11, fontweight='bold')
    ax_E.grid(True, alpha=0.3, linestyle='--'); ax_E.legend(loc='best', fontsize=9)

    fig.suptitle(f"SO(3)-aware comparison — env vs ground-truth-subnet model"
                 + (f"\n{title_extra}" if title_extra else ""),
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_so3_constraints(x_env, x_model, t, save_path, title_extra=""):
    """How well does each trajectory satisfy SO(3): RᵀR = I  and  det(R) = 1.

    For the env this should be ~1e-15 (Lie-group integrator preserves SO(3)).
    For the model this should also be tiny (Lie-Heun preserves SO(3) too) —
    if it isn't, the rotation update has a bug or fp32 cancellation issue.
    """
    def _constraints(traj_15d):
        R = traj_15d[:, :9].reshape(-1, 3, 3)
        I = np.eye(3)
        ortho_err = np.linalg.norm(np.matmul(np.transpose(R, (0, 2, 1)), R) - I,
                                   axis=(-2, -1))
        det_err = np.abs(np.linalg.det(R) - 1.0)
        return ortho_err, det_err

    ortho_env, det_env = _constraints(x_env)
    ortho_mdl, det_mdl = _constraints(x_model)

    fig, (ax_o, ax_d) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax_o.semilogy(t, np.maximum(ortho_env, 1e-20),
                  color='C0', lw=1.6, label='env')
    ax_o.semilogy(t, np.maximum(ortho_mdl, 1e-20),
                  color='C3', lw=1.4, ls='--', label='model')
    ax_o.set_ylabel(r"$\|R^{\!\top} R - I\|_F$", fontsize=11)
    ax_o.set_title("Orthogonality residual", fontsize=11, fontweight='bold')
    ax_o.grid(True, which='both', alpha=0.3, linestyle='--')
    ax_o.legend(loc='best', fontsize=9)

    ax_d.semilogy(t, np.maximum(det_env, 1e-20),
                  color='C0', lw=1.6, label='env')
    ax_d.semilogy(t, np.maximum(det_mdl, 1e-20),
                  color='C3', lw=1.4, ls='--', label='model')
    ax_d.set_xlabel("t (s)")
    ax_d.set_ylabel(r"$|\det R - 1|$", fontsize=11)
    ax_d.set_title("Determinant residual", fontsize=11, fontweight='bold')
    ax_d.grid(True, which='both', alpha=0.3, linestyle='--')
    ax_d.legend(loc='best', fontsize=9)

    fig.suptitle(f"SO(3) constraint preservation — env vs model"
                 + (f"\n{title_extra}" if title_extra else ""),
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    dtype = jnp.float32 if args.dtype == 'float32' else jnp.float64
    if args.dtype == 'float64':
        jax.config.update("jax_enable_x64", True)

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu))

    print("=" * 72)
    print("Ground-truth-subnet SDE sanity check (JAX / Lie-Heun)")
    print("=" * 72)
    print(f"  jax devices     : {jax.devices()}")
    print(f"  dtype           : {dtype}")
    print(f"  seed            : {args.seed}")
    print(f"  timesteps       : {args.timesteps}")
    print(f"  n_substeps      : {args.n_substeps}  (env uses 10 — keep matched)")
    print(f"  friction_coeff  : {args.friction_coeff}")
    print(f"  wind_force_std  : {args.wind_force_std}"
          + ("  (deterministic)" if args.wind_force_std == 0.0
             else "  (stochastic — RNG mirrored from env so dW path is shared)"))
    print(f"  control u       : {tuple(args.u)}")

    dt = 0.05
    n_outer = args.timesteps - 1
    x_env, t, R0, omega0, dW_path = run_env_with_recorded_dW(
        args, dt, n_outer, args.n_substeps
    )
    print(f"\n  env trajectory  : shape={x_env.shape}, dt={dt}, t in [0, {t[-1]:.3f}]")
    print(f"  R0 (det)        : {np.linalg.det(R0):.6f}  (should be 1.0)")
    print(f"  omega0          : {omega0}")
    print(f"  dW_path         : shape={dW_path.shape}  "
          f"(zeros if wind=0; mirrored from env RNG otherwise)")

    # ── Build ground-truth SDE model and run Lie-Heun rollout ─────────
    model = GroundTruthHamiltonianSDE(
        m=1.0, l=1.0, g=9.81,
        friction_coeff=args.friction_coeff,
        wind_force_std=args.wind_force_std,
    )

    x0 = jnp.asarray(x_env[0, :12], dtype=dtype)            # (12,) — drop u
    u  = jnp.asarray(args.u,        dtype=dtype)
    h  = float(dt / args.n_substeps)
    dW_jax = jnp.asarray(dW_path, dtype=dtype)              # (n_outer, n_sub, 3)

    if args.compile:
        print("\n  jit-compiling rollout...")
        rollout = eqx.filter_jit(lie_heun_sde_rollout)
    else:
        rollout = lie_heun_sde_rollout

    x_model_12 = np.asarray(rollout(model, x0, u, h, dW_jax))   # (n_outer+1, 12)

    # Pad with constant u for plotting / MSE parity with the PyTorch eval format.
    u_col = np.broadcast_to(np.asarray(args.u, dtype=np.float32),
                            (x_model_12.shape[0], 3))
    x_model = np.concatenate([x_model_12, u_col], axis=1).astype(np.float32)

    # ── Compare ──────────────────────────────────────────────────────
    R_env = x_env[:, 0:9]; R_mdl = x_model[:, 0:9]
    w_env = x_env[:, 9:12]; w_mdl = x_model[:, 9:12]
    R_se = ((R_env - R_mdl) ** 2).mean(axis=1)
    w_se = ((w_env - w_mdl) ** 2).mean(axis=1)
    full_se = ((x_env[:, :12] - x_model[:, :12]) ** 2).mean(axis=1)
    geo = []
    for k in range(R_env.shape[0]):
        Re = R_env[k].reshape(3, 3); Rm = R_mdl[k].reshape(3, 3)
        cos = (np.trace(Re @ Rm.T) - 1.0) / 2.0
        geo.append(float(np.arccos(np.clip(cos, -1.0, 1.0))))
    geo = np.array(geo)

    print(f"\n  Per-timestep MSE:")
    print(f"    {'t':>6s}  {'MSE(R)':>12s}  {'MSE(omega)':>12s}  "
          f"{'MSE(all)':>12s}  {'geodesic(rad)':>14s}")
    for k in range(R_env.shape[0]):
        print(f"    {t[k]:>6.3f}  {R_se[k]:>12.3e}  {w_se[k]:>12.3e}  "
              f"{full_se[k]:>12.3e}  {geo[k]:>14.3e}")

    print(f"\n  Aggregate (mean over time):")
    print(f"    MSE(R)        : {R_se.mean():.3e}")
    print(f"    MSE(omega)    : {w_se.mean():.3e}")
    print(f"    MSE(R, omega) : {full_se.mean():.3e}")
    print(f"    mean geodesic : {geo.mean():.3e} rad")
    print(f"    final geodesic: {geo[-1]:.3e} rad")

    print()
    if full_se.mean() < 1e-3:
        print("  port-Hamiltonian SDE MATCHES the env "
              "(MSE near machine-precision level for the shared dW)")
    elif full_se.mean() < 1e-1:
        print("  SDE roughly matches (some drift; check dW mirroring or fp32 noise)")
    else:
        print("  MISMATCH — the SDE does not reproduce the env. "
              "Check the dynamics derivation or RNG matching.")

    # ── Plots ────────────────────────────────────────────────────────
    if not args.no_plot:
        os.makedirs(args.plot_dir, exist_ok=True)
        u_str = '_'.join(f"{v:+.1f}" for v in args.u)
        tag = (f"sde_seed{args.seed}_T{args.timesteps}_n{args.n_substeps}_"
               f"{args.dtype}_fric{args.friction_coeff}_"
               f"wind{args.wind_force_std}_u{u_str}")
        title_extra = (f"SDE Lie-Heun, seed={args.seed}, T={args.timesteps}, "
                       f"dt=0.05, n_sub={args.n_substeps}, dtype={args.dtype}, "
                       f"u={tuple(args.u)}, friction={args.friction_coeff}, "
                       f"wind_std={args.wind_force_std}")

        out1 = os.path.join(args.plot_dir, f"euler_phase_{tag}.png")
        out2 = os.path.join(args.plot_dir, f"so3_compare_{tag}.png")
        out3 = os.path.join(args.plot_dir, f"so3_constraints_{tag}.png")

        print(f"\n  Generating plots in {args.plot_dir}/")
        plot_euler_phase_compare(x_env, x_model, t, out1, title_extra=title_extra)
        plot_so3_compare(x_env, x_model, t, out2,
                         m=1.0, l=1.0, g=9.81, title_extra=title_extra)
        plot_so3_constraints(x_env, x_model, t, out3, title_extra=title_extra)


if __name__ == "__main__":
    main()
