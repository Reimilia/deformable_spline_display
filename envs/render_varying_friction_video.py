"""Render a video of the 3D windy pendulum with state-dependent friction,
no wind, and no stochastic forcing.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from windy_pendulum_3d import windy_pendulum_3d


N_STEPS = 500
FPS = 30
SAVE_PATH = "videos/windy_pendulum_3d_varying_friction.mp4"


def main():
    env = windy_pendulum_3d(
        g=9.81,
        m=1.0,
        l=1.0,
        dt=0.05,
        varying_friction=True,
        friction_coeff=0.5,
        external_force_type="constant",
        external_force_std=0.0,
        wind_force_std=0.0,
        render_mode="rgb_array",
        seed=42,
    )
    env.reset(seed=42)

    frames = []
    f0 = env.render()
    if f0 is not None:
        frames.append(f0)

    for step in range(N_STEPS):
        _, reward, _, _, info = env.step([0.0, 0.0, 0.0])
        f = env.render()
        if f is not None:
            frames.append(f)
        if step % 25 == 0:
            R, omega = env.get_state()
            mult = 1.0 + 0.5 * (0.5 * (1.0 - (R @ np.array([0, 0, 1.0]))[2])) \
                       + 0.5 * np.tanh(np.linalg.norm(omega))
            print(f"step {step:3d}/{N_STEPS}  reward={reward:+.3f}  "
                  f"|omega|={np.linalg.norm(omega):.3f}  fric_mult={mult:.3f}")

    env.close()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(os.path.dirname(out_dir),
                             "videos",
                             "windy_pendulum_3d_varying_friction.mp4")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig_vid, ax_vid = plt.subplots(figsize=(7, 7))
    ax_vid.axis("off")
    im = ax_vid.imshow(frames[0])

    def _update(i):
        im.set_data(frames[i])
        return [im]

    anim = FuncAnimation(fig_vid, _update, frames=len(frames),
                         interval=1000 / FPS, blit=True)
    anim.save(save_path, writer="ffmpeg", fps=FPS, dpi=100)
    plt.close(fig_vid)
    print(f"Saved video to: {save_path}")


if __name__ == "__main__":
    main()
