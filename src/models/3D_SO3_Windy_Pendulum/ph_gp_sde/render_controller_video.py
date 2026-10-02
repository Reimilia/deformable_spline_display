"""Record an MP4 of the SO(3) Energy-Casimir controller stabilising the
windy 3D pendulum at the upright equilibrium.

Layout mirrors `envs/windy_pendulum_3d.py::windy_pendulum_3d.render`:
  * 3-D scene with the bob, body-frame axes, gravity, and the wind vector.
  * Top-left HUD: t, wind value, |omega|.
  * Top-right HUD (added by this script): u_x / u_y / u_z body-frame torque.

Usage:
    python ph_gp_sde/render_controller_video.py                      # default
    python ph_gp_sde/render_controller_video.py --tilt_deg 60 --horizon 12
    python ph_gp_sde/render_controller_video.py --d_inj 4.0
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, '../../../..'))
for _p in (PROJECT_ROOT, THIS_FILE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from envs.windy_pendulum_3d import windy_pendulum_3d, _exp_so3, _log_so3   # noqa: E402
from qp_env import QPWindyPendulum3D                                        # noqa: E402
from controller import EnergyCasimirController, ControllerConfig            # noqa: E402


# Match `evaluate_controller.py` env defaults so the video reflects the
# same plant the controller was tuned against.
ENV_DEFAULTS = dict(
    g=9.81, m=1.0, l=1.0, dt=0.05,
    varying_friction=False, friction_coeff=0.5,
    external_force_type='sine', external_force_std=0.0,
    wind_force_std=0.5,
)


# ─────────────────────────────────────────────────────────────────────
# Single-frame render (mirrors env.render, plus a top-right u HUD)
# ─────────────────────────────────────────────────────────────────────

def _draw_frame(ax, env, u_now: np.ndarray):
    """Draw the env state plus a top-right u HUD onto `ax`."""
    ax.cla()

    origin = np.zeros(3)
    ez = np.array([0.0, 0.0, 1.0])
    bob = env.l * (env.R @ ez)

    # Pendulum rod
    ax.plot([origin[0], bob[0]], [origin[1], bob[1]], [origin[2], bob[2]],
            color="#555555", lw=3, solid_capstyle='round')
    ax.scatter(*bob, color="#cc4d4d", s=120, depthshade=True, zorder=5)
    ax.scatter(*origin, color="black", s=60, depthshade=True, zorder=5)

    # Body axes at the bob tip
    axis_len = 0.35 * env.l
    axis_colors = ["#e74c3c", "#2ecc71", "#3498db"]
    axis_labels = ["x_b", "y_b", "z_b"]
    for i in range(3):
        e_body = np.zeros(3); e_body[i] = 1.0
        tip = bob + axis_len * (env.R @ e_body)
        ax.quiver(bob[0], bob[1], bob[2],
                  tip[0] - bob[0], tip[1] - bob[1], tip[2] - bob[2],
                  color=axis_colors[i], lw=2, arrow_length_ratio=0.15)
        ax.text(tip[0], tip[1], tip[2], f" {axis_labels[i]}",
                color=axis_colors[i], fontsize=8, fontweight='bold')

    # Wind vector
    if abs(env.last_w) > 1e-6:
        w_vec = env.last_w * env.external_force_direction
        w_scale = 0.5 * env.l
        ax.quiver(0, 0, 0,
                  w_vec[0] * w_scale, w_vec[1] * w_scale, w_vec[2] * w_scale,
                  color="#00bcd4", lw=2.5, arrow_length_ratio=0.18,
                  label=f"wind = {env.last_w:+.2f}")

    # Gravity reference
    g_len = 0.4 * env.l
    ax.quiver(0, 0, 0, 0, 0, -g_len,
              color="#999999", lw=1.5, arrow_length_ratio=0.15,
              linestyle='dashed', label='gravity')

    # Ground disc
    theta = np.linspace(0, 2 * np.pi, 60)
    r_circle = 1.2 * env.l
    xc, yc = r_circle * np.cos(theta), r_circle * np.sin(theta)
    zc = np.zeros_like(theta)
    verts = [list(zip(xc, yc, zc))]
    ground = Poly3DCollection(verts, alpha=0.08, facecolor="#b0bec5",
                              edgecolor="#78909c", linewidth=0.8)
    ax.add_collection3d(ground)

    # Target dashed line — bob target position in world frame
    target_R = getattr(env, '_R_target', np.eye(3))
    target_bob = env.l * (target_R @ ez)
    ax.plot([origin[0], target_bob[0]],
            [origin[1], target_bob[1]],
            [origin[2], target_bob[2]],
            color='#4caf50', lw=1.0, linestyle=':', alpha=0.8,
            label='target $R_d$')

    lim = 1.5 * env.l
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xlabel('X', fontsize=9)
    ax.set_ylabel('Y', fontsize=9)
    ax.set_zlabel('Z', fontsize=9)
    ax.set_title('SO(3) Energy-Casimir control — windy 3D pendulum',
                 fontsize=12, fontweight='bold')
    ax.set_box_aspect([1, 1, 1])

    # ── HUDs ────────────────────────────────────────────────────────
    # Top-left: time / wind / |omega|
    omega_mag = float(np.linalg.norm(env.omega))
    geo = float(np.linalg.norm(_log_so3(env.R.T @ target_R)))
    hud_l = (f"t = {env.t:5.2f} s\n"
             f"wind = {env.last_w:+.2f}\n"
             f"|omega| = {omega_mag:.2f} rad/s\n"
             f"|log R^T R_d| = {geo:.3f} rad")
    ax.text2D(0.02, 0.96, hud_l, transform=ax.transAxes,
              fontsize=9, fontfamily='monospace',
              verticalalignment='top',
              bbox=dict(boxstyle='round,pad=0.3',
                        facecolor='white', alpha=0.85))

    # Top-right: control torque
    hud_r = (f"control u (body)\n"
             f"u_x = {u_now[0]:+7.3f}\n"
             f"u_y = {u_now[1]:+7.3f}\n"
             f"u_z = {u_now[2]:+7.3f}\n"
             f"|u|  = {float(np.linalg.norm(u_now)):7.3f}")
    ax.text2D(0.98, 0.96, hud_r, transform=ax.transAxes,
              fontsize=9, fontfamily='monospace',
              verticalalignment='top', horizontalalignment='right',
              bbox=dict(boxstyle='round,pad=0.3',
                        facecolor='white', alpha=0.85))

    ax.legend(loc='lower right', fontsize=8, framealpha=0.7)


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--k_c', type=float, default=30.0)
    p.add_argument('--d_inj', type=float, default=2.0,
                   help='damping injection (2.0 → ~5° steady-state error)')
    p.add_argument('--tilt_deg', type=float, default=45.0,
                   help='initial tilt about body-x from R_d')
    p.add_argument('--horizon', type=float, default=15.0,
                   help='video length in seconds')
    p.add_argument('--fps', type=int, default=30,
                   help='frames per second (matches env dt: dt=0.05 → 20 fps)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=str,
                   default=os.path.join(THIS_FILE_DIR, 'data',
                                        'controller_eval',
                                        'controller_video.mp4'))
    p.add_argument('--use_trained_g', action='store_true')
    p.add_argument('--no_qp_integrator', action='store_true',
                   help='use the original (R, ω) env integrator instead of '
                        'the (R, p) one (default).')
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # ── Build env ────────────────────────────────────────────────────
    env_cls = windy_pendulum_3d if args.no_qp_integrator else QPWindyPendulum3D
    env = env_cls(seed=args.seed, **ENV_DEFAULTS)
    print(f"plant: {env_cls.__name__}  "
          f"({'(R, p)' if env_cls is QPWindyPendulum3D else '(R, ω)'} integrator)")
    R_d = np.eye(3)
    env._R_target = R_d  # for the dashed target line in the render
    R0 = R_d @ _exp_so3(np.array([np.deg2rad(args.tilt_deg), 0.0, 0.0]))
    env.reset(seed=args.seed, options={'R_init': R0, 'omega_init': np.zeros(3)})

    # ── Build controller ────────────────────────────────────────────
    model = None
    if args.use_trained_g:
        from verify_control import load_trained_model
        model = load_trained_model()
    cfg = ControllerConfig(R_d=R_d, k_c=args.k_c, d_inj=args.d_inj,
                           use_trained_g=args.use_trained_g)
    ctrl = EnergyCasimirController(cfg, model=model)

    # ── Roll out and capture frames ──────────────────────────────────
    n_steps = int(round(args.horizon / env.dt))
    print(f"rendering {n_steps} steps  ({args.horizon}s @ dt={env.dt})  →  {args.out}")
    frames = []
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Initial frame (u = 0 — controller hasn't acted yet)
    _draw_frame(ax, env, np.zeros(3))
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    frames.append(np.asarray(buf)[:, :, :3].copy())

    for k in range(n_steps):
        u = ctrl.act(env.R, env.omega)
        env.step(u)
        _draw_frame(ax, env, u)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frames.append(np.asarray(buf)[:, :, :3].copy())
        if (k + 1) % 30 == 0 or k == n_steps - 1:
            geo = float(np.linalg.norm(_log_so3(env.R.T @ R_d)))
            print(f"  step {k+1:4d}/{n_steps}  geo={geo:.3f} rad  "
                  f"|u|={float(np.linalg.norm(u)):.2f}")

    plt.close(fig)

    # ── Encode mp4 ───────────────────────────────────────────────────
    print(f"encoding {len(frames)} frames at {args.fps} fps …")
    fig_vid, ax_vid = plt.subplots(figsize=(8, 7))
    ax_vid.axis('off')
    im = ax_vid.imshow(frames[0])
    def _update(i):
        im.set_data(frames[i]); return [im]
    anim = FuncAnimation(fig_vid, _update, frames=len(frames),
                         interval=1000.0 / args.fps, blit=True)
    anim.save(args.out, writer='ffmpeg', fps=args.fps, dpi=100)
    plt.close(fig_vid)
    print(f"done → {args.out}")


if __name__ == "__main__":
    main()
