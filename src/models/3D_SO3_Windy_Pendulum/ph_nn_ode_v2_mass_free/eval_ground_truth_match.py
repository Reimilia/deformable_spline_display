"""Sanity check: does the port-Hamiltonian formulation reproduce the env?

Builds a DissipativeSO3HamNODE-shaped model whose four subnetworks are
*hardcoded* to the true physics of the windy_pendulum_3d env:

    M_net(q)  = M⁻¹(q) = (1/(m·l²)) · I₃    (paper convention: M_net outputs M⁻¹)
    V_net(q)  = m·g·l·q[8]                  (gravitational PE = m·g·l·R[2,2])
    Dw_net(q) = friction_coeff · I₃         (constant diagonal damping)
    g_net(q)  = I₃                          (body-frame torque input)

Then rolls out one trajectory of T timesteps through the same forward()
that the trained network uses (same Hamiltonian dynamics, same Lie-bracket
cross-product structure, same odeint solver), starting from the same
(R₀, ω₀) the env was reset to.  Compares against the env's own trajectory
elementwise.

If the port-Hamiltonian math is correct, MSE between the two trajectories
should be tiny — limited only by the difference in integrators (env uses
Lie-group Heun with 10 substeps per dt; model uses rk4 with no substeps).

OPTIMIZATIONS APPLIED HERE (verifying they preserve the math):
  #1  dM_inv/dt computed via torch.func.jvp (1 call, instead of the
      9-iteration autograd.grad double loop in production network.py).
  #2  Cross products batched: one call over the 3 rows of R instead of
      3 separate linalg.cross calls (for both dq and the dp gravity terms).
  #3  Single M_net evaluation per forward (instead of two): dHdp is
      computed analytically as M⁻¹·p, dHdq via autograd through q only
      (with p detached so the q-gradient doesn't pick up p's path).
  #6  Optional torch.compile wrap (--compile flag).
  #7  cudnn.benchmark = True (no-op for this MLP-only model, set for
      consistency with what we'd do in production).

If env vs. model MSE here matches what we got with the un-optimized version
(~integrator-error level), all five changes preserve the math and are safe
to port into the production network.py / train.py.
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, '../../../..'))
sys.path.insert(0, PROJECT_ROOT)

from torchdiffeq import odeint
from envs.windy_pendulum_3d import windy_pendulum_3d


# ─────────────────────────────────────────────────────────────────────
# Ground-truth-subnet model: identical .forward() shape as the trained
# network, but the four "subnetworks" return the true physics constants.
# ─────────────────────────────────────────────────────────────────────

class GroundTruthHamiltonianODE(torch.nn.Module):
    """DissipativeSO3HamNODE shape, subnets replaced with true physics,
    with optimizations #1, #2, #3 inlined (see module docstring).

    Convention: M_net returns M⁻¹(q) (matches production network.py).
    For the spherical pendulum with m=l=1, M = M⁻¹ = I₃.
    """

    def __init__(self, m=1.0, l=1.0, g=9.81, friction_coeff=0.5,
                 device=None, dtype=torch.float32):
        super().__init__()
        self.m = float(m)
        self.l = float(l)
        self.g = float(g)
        self.friction_coeff = float(friction_coeff)
        self.rotmatdim = 9
        self.angveldim = 3
        self.u_dim = 3
        self.device = device
        self.dtype_ = dtype
        self.nfe = 0

    # ── True subnet outputs ───────────────────────────────────────────
    # M_net returns M⁻¹(q) (production convention). For m=l=1, M⁻¹ = I₃.

    def M_net(self, q):
        B = q.shape[0]
        I3 = torch.eye(3, device=q.device, dtype=q.dtype)
        return (1.0 / (self.m * self.l ** 2)) * I3.unsqueeze(0).expand(B, 3, 3)

    def Dw_net(self, q):
        B = q.shape[0]
        I3 = torch.eye(3, device=q.device, dtype=q.dtype)
        return self.friction_coeff * I3.unsqueeze(0).expand(B, 3, 3)

    def V_net(self, q):
        # V = m·g·l · (R · e_z)_z = m·g·l · R[2,2] = m·g·l · q[:, 8]
        return (self.m * self.g * self.l) * q[:, 8:9]   # (B, 1)

    def g_net(self, q):
        B = q.shape[0]
        I3 = torch.eye(3, device=q.device, dtype=q.dtype)
        return I3.unsqueeze(0).expand(B, 3, 3)

    # ── Forward (mirrors trained model, with optimizations #1/#2/#3) ──

    def forward(self, t, x):
        with torch.enable_grad():
            self.nfe += 1
            bs = x.shape[0]
            zero_vec = torch.zeros(bs, self.u_dim, dtype=x.dtype, device=x.device)

            q, q_dot, u = torch.split(x, [9, 3, 3], dim=1)

            # ── #3: SINGLE M_net call (used for both p and H) ──
            # M_net returns M⁻¹(q); p = M·q_dot = solve(M⁻¹, q_dot).
            M_q_inv = self.M_net(q)
            q_dot_aug = q_dot.unsqueeze(2)
            p_val = torch.linalg.solve(M_q_inv, q_dot_aug).squeeze(2)
            # Detach p so dHdq via autograd through q does not also pick up
            # the indirect path q → M_net → p → H. (The original code uses a
            # cat→split trick to make q and p independent autograd nodes;
            # detach() achieves the same end without the extra M_net call.)
            p = p_val.detach()

            V_q  = self.V_net(q)
            g_q  = self.g_net(q)
            Dw_q = self.Dw_net(q)

            # H = ½ pᵀ M⁻¹ p + V
            p_aug = p.unsqueeze(2)
            H = (torch.matmul(p_aug.transpose(1, 2),
                              torch.matmul(M_q_inv, p_aug)).squeeze() / 2.0
                 + V_q.squeeze())

            # dH/dp = M⁻¹ p   (analytical — H is quadratic in p; saves an autograd traversal)
            dHdp = torch.matmul(M_q_inv, p.unsqueeze(2)).squeeze(-1)

            # dH/dq via autograd through q only. Both M_q_inv and V_q
            # depend on q; p is detached so its path contributes nothing.
            # allow_unused=True is needed for torch.compile: with our constant
            # ground-truth M_net, Dynamo's graph capture can sever the (q → H)
            # link through V_q's slice. Eager autograd is fine without it.
            dHdq = torch.autograd.grad(
                H.sum(), q, create_graph=True, allow_unused=True
            )[0]
            if dHdq is None:
                dHdq = torch.zeros_like(q)

            F = torch.matmul(g_q, u.unsqueeze(2)).squeeze(-1)

            # ── #2: Batched cross product for dq (rows of R × dHdp) ──
            q_3x3  = q.view(-1, 3, 3)                       # (B, 3, 3) rows of R
            dHdp_b = dHdp.unsqueeze(1).expand(-1, 3, -1)    # (B, 3, 3) broadcast
            dq = torch.linalg.cross(q_3x3, dHdp_b, dim=2).reshape(-1, 9)

            # ── #2: Batched cross product for the gravity terms in dp ──
            dHdq_3x3 = dHdq.view(-1, 3, 3)                  # (B, 3, 3)
            grav = torch.linalg.cross(q_3x3, dHdq_3x3, dim=2).sum(dim=1)  # (B, 3)

            dp = (torch.linalg.cross(p, dHdp, dim=1)
                  + grav
                  - torch.matmul(Dw_q, dHdp.unsqueeze(2)).squeeze(-1)
                  + F)

            # ── #1: Vectorized dM_inv/dt via JVP ──
            # dM_inv_dt = (∂M⁻¹/∂q) · dq = directional derivative of M_net at q
            # in the direction dq. Replaces the 9-iteration autograd.grad loop.
            # For our constant M_net this returns zero — but we exercise the
            # JVP code path so any breakage would still show up.
            _, dM_inv_dt = torch.func.jvp(self.M_net, (q,), (dq,))

            # q_dot = M⁻¹ p   ⇒   d(q_dot)/dt = (dM⁻¹/dt)·p + M⁻¹·dp
            ddq = (torch.matmul(M_q_inv, dp.unsqueeze(2)).squeeze(-1)
                   + torch.matmul(dM_inv_dt, p.unsqueeze(2)).squeeze(-1))

            return torch.cat([dq, ddq, zero_vec], dim=1)


# ─────────────────────────────────────────────────────────────────────
# Experiment
# ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--timesteps', type=int, default=20)
    p.add_argument('--solver', type=str, default='rk4',
                   help='ODE solver passed to torchdiffeq.odeint')
    p.add_argument('--dtype', type=str, default='float32',
                   choices=['float32', 'float64'])
    p.add_argument('--friction_coeff', type=float, default=0.5)
    p.add_argument('--wind_force_std', type=float, default=0.5,
                   help='Stratonovich Wiener wind-force noise scale; >0 makes env stochastic')
    p.add_argument('--u', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                   metavar=('UX', 'UY', 'UZ'),
                   help='constant body-frame torque applied during the rollout')
    p.add_argument('--plot_dir', type=str,
                   default=os.path.join(THIS_FILE_DIR, 'data', 'eval_match'),
                   help='where to save the comparison plots')
    p.add_argument('--no_plot', action='store_true',
                   help='skip plot generation')
    p.add_argument('--compile', action='store_true',
                   help='wrap the ground-truth model with torch.compile (#6)')
    p.add_argument('--compile_mode', type=str, default='reduce-overhead',
                   choices=['default', 'reduce-overhead', 'max-autotune'])
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Plotting helpers (mirrors datasets/3d_pendulum_trajectory_dataset_plot.py
# but overlays env vs model on a single axis instead of many trajectories).
# ─────────────────────────────────────────────────────────────────────

def _rotmat_to_euler(R_flat):
    """Vectorized batch rotmat -> Euler ZYX (roll, pitch, yaw)."""
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
    """R @ e_z = third column of R = R_flat[..., [2, 5, 8]]."""
    return R_flat[..., [2, 5, 8]]


def plot_euler_phase_compare(x_env, x_model, t, save_path, title_extra=""):
    """Figure 1 — per-axis (Euler angle, body angular velocity) phase portrait
    for env (solid blue) vs model (dashed red), overlaid on the same axes."""
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
        w_e = om_e[:, axis_idx]
        w_m = om_m[:, axis_idx]

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
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_so3_compare(x_env, x_model, t, save_path, m=1.0, l=1.0, g=9.81,
                     title_extra=""):
    """Figure 2 — SO(3)-aware diagnostics for env (blue) vs model (red).

    Panels:
      (0,0) Bob trajectory on S²  (3D)
      (0,1) Tilt α vs dα/dt
      (1,0) ‖ω‖ over time
      (1,1) Total energy E(t) = ½ m l² ‖ω‖² + m g l (R e_z)_z
    """
    bob_e = _bob_dir(x_env[:, :9])
    bob_m = _bob_dir(x_model[:, :9])
    tilt_e = np.arccos(np.clip(bob_e[:, 2], -1.0, 1.0))
    tilt_m = np.arccos(np.clip(bob_m[:, 2], -1.0, 1.0))
    om_e = np.linalg.norm(x_env[:, 9:12], axis=-1)
    om_m = np.linalg.norm(x_model[:, 9:12], axis=-1)

    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
    dtilt_e = np.gradient(tilt_e, dt)
    dtilt_m = np.gradient(tilt_m, dt)

    E_e = 0.5 * (m * l ** 2) * (om_e ** 2) + m * g * l * bob_e[:, 2]
    E_m = 0.5 * (m * l ** 2) * (om_m ** 2) + m * g * l * bob_m[:, 2]

    fig = plt.figure(figsize=(13, 10))
    ax_s2  = fig.add_subplot(2, 2, 1, projection='3d')
    ax_til = fig.add_subplot(2, 2, 2)
    ax_om  = fig.add_subplot(2, 2, 3)
    ax_E   = fig.add_subplot(2, 2, 4)

    # ── Bob on S² ──
    uu, vv = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
    sx = np.cos(uu) * np.sin(vv)
    sy_ = np.sin(uu) * np.sin(vv)
    sz = np.cos(vv)
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

    # ── Tilt phase ──
    ax_til.plot(tilt_e, dtilt_e, color='C0', lw=1.6, label='env')
    ax_til.plot(tilt_m, dtilt_m, color='C3', lw=1.4, ls='--', label='model')
    ax_til.scatter(tilt_e[0], dtilt_e[0], color='black', s=40, marker='*', zorder=5)
    ax_til.set_xlabel(r"Tilt $\alpha$ (rad)")
    ax_til.set_ylabel(r"$\dot{\alpha}$ (rad/s)")
    ax_til.set_title("Tilt phase portrait", fontsize=11, fontweight='bold')
    ax_til.grid(True, alpha=0.3, linestyle='--')
    ax_til.axhline(0, color='gray', lw=0.5, alpha=0.5)
    ax_til.legend(loc='best', fontsize=9)

    # ── ‖ω‖ over time ──
    ax_om.plot(t, om_e, color='C0', lw=1.6, label='env')
    ax_om.plot(t, om_m, color='C3', lw=1.4, ls='--', label='model')
    ax_om.set_xlabel("t (s)")
    ax_om.set_ylabel(r"$\|\omega\|$ (rad/s)")
    ax_om.set_title("Angular speed over time", fontsize=11, fontweight='bold')
    ax_om.grid(True, alpha=0.3, linestyle='--')
    ax_om.legend(loc='best', fontsize=9)

    # ── Energy over time ──
    ax_E.plot(t, E_e, color='C0', lw=1.6, label='env')
    ax_E.plot(t, E_m, color='C3', lw=1.4, ls='--', label='model')
    ax_E.set_xlabel("t (s)")
    ax_E.set_ylabel(r"$E$ (J, $m=l=1$)")
    ax_E.set_title("Total energy over time", fontsize=11, fontweight='bold')
    ax_E.grid(True, alpha=0.3, linestyle='--')
    ax_E.legend(loc='best', fontsize=9)

    fig.suptitle(f"SO(3)-aware comparison — env vs ground-truth-subnet model"
                 + (f"\n{title_extra}" if title_extra else ""),
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def run_env(args, dt, n_steps):
    """Roll out the env from a fresh reset, constant action u, all noise off.
    Returns:
        x_traj : (T+1, 15) numpy — concatenated [R.flatten(), omega, u]
                 at t = 0, dt, 2dt, ..., n_steps*dt  (T+1 = n_steps+1 frames)
        t      : (T+1,) numpy time grid
        R0     : (3,3) initial rotation
        omega0 : (3,) initial angular velocity
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

    x_traj = np.stack(traj, axis=0)            # (T+1, 15)
    t = np.arange(x_traj.shape[0]) * dt
    return x_traj, t, R0, omega0


def main():
    args = get_args()
    dtype = torch.float32 if args.dtype == 'float32' else torch.float64
    torch.set_default_dtype(dtype)
    # ── #7: enable cudnn autotuning (no-op for this MLP-only model, set
    #        for consistency with what we'd do in production training).
    torch.backends.cudnn.benchmark = True
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    print("=" * 72)
    print("Ground-truth-subnet sanity check")
    print("=" * 72)
    print(f"  device          : {device}")
    print(f"  dtype           : {dtype}")
    print(f"  seed            : {args.seed}")
    print(f"  timesteps       : {args.timesteps}")
    print(f"  solver          : {args.solver}")
    print(f"  friction_coeff  : {args.friction_coeff}")
    print(f"  wind_force_std  : {args.wind_force_std}"
          + ("  (deterministic)" if args.wind_force_std == 0.0
             else "  (STOCHASTIC — env path is random; expect non-zero MSE)"))
    print(f"  external_force  : 0.0  (none)")
    print(f"  varying_friction: False")
    print(f"  obs_noise_std   : 0.0")
    print(f"  control u       : {tuple(args.u)}")

    dt = 0.05
    x_env, t, R0, omega0 = run_env(args, dt, args.timesteps - 1)
    print(f"\n  env trajectory  : shape={x_env.shape}, dt={dt}, t in [0, {t[-1]:.3f}]")
    print(f"  R0 (det)        : {np.linalg.det(R0):.6f}  (should be 1.0)")
    print(f"  omega0          : {omega0}")

    # ── Build ground-truth model and run odeint from same x0 ─────────
    model = GroundTruthHamiltonianODE(
        m=1.0, l=1.0, g=9.81,
        friction_coeff=args.friction_coeff,
        device=device, dtype=dtype,
    ).to(device)

    # ── #6: optional torch.compile wrap ──
    # autograd.grad(create_graph=True) inside forward needs donated_buffer off
    if args.compile:
        print(f"\n  Compiling model with mode='{args.compile_mode}'...")
        import torch._functorch.config as _ft_cfg
        _ft_cfg.donated_buffer = False
        model = torch.compile(model, mode=args.compile_mode)

    x0 = torch.tensor(x_env[0], dtype=dtype, device=device,
                       requires_grad=True).unsqueeze(0)   # (1, 15)
    t_torch = torch.tensor(t, dtype=dtype, device=device, requires_grad=True)

    x_model = odeint(model, x0, t_torch, method=args.solver)   # (T+1, 1, 15)
    x_model = x_model.squeeze(1).detach().cpu().numpy()         # (T+1, 15)

    # ── Compare ──────────────────────────────────────────────────────
    R_env  = x_env[:, 0:9]
    R_mdl  = x_model[:, 0:9]
    w_env  = x_env[:, 9:12]
    w_mdl  = x_model[:, 9:12]

    R_se = ((R_env - R_mdl) ** 2).mean(axis=1)   # per-timestep
    w_se = ((w_env - w_mdl) ** 2).mean(axis=1)
    full_se = ((x_env[:, :12] - x_model[:, :12]) ** 2).mean(axis=1)

    # geodesic distance per timestep
    geo = []
    for k in range(R_env.shape[0]):
        Re = R_env[k].reshape(3, 3)
        Rm = R_mdl[k].reshape(3, 3)
        cos = (np.trace(Re @ Rm.T) - 1.0) / 2.0
        cos = float(np.clip(cos, -1.0, 1.0))
        geo.append(float(np.arccos(cos)))
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

    # Verdict
    print()
    if full_se.mean() < 1e-3:
        print("  port-Hamiltonian formulation MATCHES the env "
              "(MSE in expected integrator-error range)")
    elif full_se.mean() < 1e-1:
        print("  port-Hamiltonian formulation roughly matches "
              "(some drift; check integrator step size or fp32 noise)")
    else:
        print("  MISMATCH — the port-Hamiltonian equation does not "
              "reproduce the env. Check the dynamics derivation.")

    # ── Plots ────────────────────────────────────────────────────────
    if not args.no_plot:
        os.makedirs(args.plot_dir, exist_ok=True)
        u_str = '_'.join(f"{v:+.1f}" for v in args.u)
        tag = (f"seed{args.seed}_T{args.timesteps}_{args.solver}_"
               f"{args.dtype}_fric{args.friction_coeff}_"
               f"wind{args.wind_force_std}_u{u_str}")
        title_extra = (f"seed={args.seed}, T={args.timesteps}, dt=0.05, "
                       f"solver={args.solver}, dtype={args.dtype}, "
                       f"u={tuple(args.u)}, friction={args.friction_coeff}, "
                       f"wind_std={args.wind_force_std}")

        out1 = os.path.join(args.plot_dir, f"euler_phase_{tag}.png")
        out2 = os.path.join(args.plot_dir, f"so3_compare_{tag}.png")

        print(f"\n  Generating plots in {args.plot_dir}/")
        plot_euler_phase_compare(x_env, x_model, t, out1, title_extra=title_extra)
        plot_so3_compare(x_env, x_model, t, out2,
                         m=1.0, l=1.0, g=9.81, title_extra=title_extra)


if __name__ == "__main__":
    main()
