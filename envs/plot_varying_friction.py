"""Visualize the state-dependent friction multiplier used by windy_pendulum_3d.

Friction model (from `_variable_friction`):
    fric = friction_coeff * (1 + 0.5 * height_term + 0.5 * speed_term)
    height_term = 0.5 * (1 - bob_dir_z)         # 0 when bob is up, 1 when down
    speed_term  = tanh(|omega|)                  # 0 .. 1

So the multiplier on the base friction lies in [1, 2).
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def friction_multiplier(tilt_rad: np.ndarray, omega_mag: np.ndarray) -> np.ndarray:
    bob_z = np.cos(tilt_rad)
    height_term = 0.5 * (1.0 - bob_z)
    speed_term = np.tanh(omega_mag)
    return 1.0 + 0.5 * height_term + 0.5 * speed_term


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "varying_friction.png")

    tilt = np.linspace(0.0, np.pi, 200)            # 0 = upright, pi = hanging down
    omega = np.linspace(0.0, 5.0, 200)             # |omega|
    T, W = np.meshgrid(tilt, omega, indexing="xy")
    M = friction_multiplier(T, W)

    fig = plt.figure(figsize=(13, 5))

    # ── Panel 1: 2D heatmap (tilt vs |omega|) ─────────────────────────────
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.pcolormesh(np.degrees(T), W, M, shading="auto", cmap="viridis")
    cs = ax1.contour(np.degrees(T), W, M, levels=8, colors="white",
                     linewidths=0.6, alpha=0.7)
    ax1.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label("friction multiplier  (× friction_coeff)")
    ax1.set_xlabel("tilt angle from upright [deg]")
    ax1.set_ylabel("|omega|  [rad/s]")
    ax1.set_title("Variable friction multiplier")

    # ── Panel 2: slices at fixed |omega| and fixed tilt ───────────────────
    ax2 = fig.add_subplot(1, 2, 2)

    for w in [0.0, 0.5, 1.0, 2.0, 5.0]:
        m = friction_multiplier(tilt, np.full_like(tilt, w))
        ax2.plot(np.degrees(tilt), m, label=f"|omega| = {w:.1f}")
    ax2.set_xlabel("tilt angle from upright [deg]")
    ax2.set_ylabel("friction multiplier")
    ax2.set_title("Slices at fixed |omega|")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "windy_pendulum_3d._variable_friction:  "
        "fric = c · (1 + 0.5·height_term + 0.5·tanh|omega|)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Multiplier range: [{M.min():.3f}, {M.max():.3f}]")


if __name__ == "__main__":
    main()
