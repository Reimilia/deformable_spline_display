"""
3D Pendulum Phase Space Visualization

Generates TWO output figures per dataset:

  Figure 1 — Original Euler-angle phase space
     6 rows × num_us cols
     Per-axis (Euler angle, body angular velocity) projections.
     Suffers from gimbal lock at pitch=±π/2; useful for quick EDA.

  Figure 2 — SO(3)-aware phase space (new)
     8 rows × num_us cols
     For each of {Train, Test}:
       - Bob trajectory on S² (3D, no parameterization artifacts)
       - Tilt angle α vs tilt rate dα/dt (2D phase portrait)
       - ||ω|| over time
       - Total energy E(t) = T + V (with m=l=1, g from settings)
"""

import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


# ─────────────────── I/O ───────────────────

def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ─────────────────── Geometry helpers ───────────────────

def rotmat_to_euler(R_flat):
    """Vectorized batch rotmat → Euler ZYX (roll, pitch, yaw).
    R_flat: (..., 9) → (..., 3) in radians.
    NOTE: ZYX has gimbal lock at pitch = ±π/2; visualization artifact only.
    """
    Rs = R_flat.reshape(*R_flat.shape[:-1], 3, 3)
    R00 = Rs[..., 0, 0]
    R10 = Rs[..., 1, 0]
    R20 = Rs[..., 2, 0]
    R21 = Rs[..., 2, 1]
    R22 = Rs[..., 2, 2]
    R12 = Rs[..., 1, 2]
    R11 = Rs[..., 1, 1]

    sy = np.sqrt(R00**2 + R10**2)
    near_singular = sy < 1e-6

    roll = np.where(near_singular,
                    np.arctan2(-R12, R11),
                    np.arctan2(R21, R22))
    pitch = np.arctan2(-R20, sy)
    yaw = np.where(near_singular,
                   np.zeros_like(R00),
                   np.arctan2(R10, R00))
    return np.stack([roll, pitch, yaw], axis=-1)


def _rotmat_bob_dir(R_flat):
    """Extract bob direction R @ e_z from row-major flat rotation matrices.
    R_flat: (..., 9) → (..., 3).  R @ e_z is the 3rd column of R = R_flat[..., [2,5,8]].
    """
    return R_flat[..., [2, 5, 8]]


def _resolve_dataset_path(save_dir, filename):
    """Return (filename, file_path), or (None, None) if not found."""
    if filename is None:
        files = sorted([f for f in os.listdir(save_dir) if f.endswith('.pkl')])
        if not files:
            print(f"No .pkl files in {save_dir}")
            return None, None
        filename = files[0]
        print(f"Auto-selected: {filename}")
    file_path = os.path.join(save_dir, filename)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None, None
    return filename, file_path


# ─────────────────── Original: Euler-angle phase space ───────────────────

def plot_3d_phase_space(save_dir, num_trajs_to_plot=15, filename=None):
    """Per-axis (Euler angle, body ω_i) phase portrait.
    Mirrors the original plot but with vectorized Euler conversion and a
    sequential colormap (viridis) since ||u|| is non-negative.
    """
    filename, file_path = _resolve_dataset_path(save_dir, filename)
    if file_path is None:
        return

    print(f"Loading: {file_path}")
    data = load_data(file_path)

    train_data = data['x']                  # (num_us, T, N_train, 15)
    test_data  = data.get('test_x', None)   # (num_us, T, N_test, 15)

    num_us = train_data.shape[0]
    axis_labels = ['X (Roll)', 'Y (Pitch)', 'Z (Yaw)']

    # Global control range for shared colormap normalization
    all_u = [np.linalg.norm(train_data[..., 12:15], axis=-1).flatten()]
    if test_data is not None:
        all_u.append(np.linalg.norm(test_data[..., 12:15], axis=-1).flatten())
    u_max = np.concatenate(all_u).max()
    norm = Normalize(vmin=0, vmax=max(u_max, 0.01))
    cmap = plt.get_cmap('viridis')

    nrows = 6
    ncols = num_us
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(4.5 * ncols, 3 * nrows), squeeze=False)

    datasets = [
        ("Train", train_data, 0),
        ("Test",  test_data,  3),
    ]

    for ds_label, ds_data, row_offset in datasets:
        if ds_data is None:
            continue
        N_avail = ds_data.shape[2]
        n_plot = min(N_avail, num_trajs_to_plot)

        for u_idx in range(num_us):
            batch = ds_data[u_idx]              # (T, N, 15)
            R_all = batch[..., :9]              # (T, N, 9)
            eulers_all = rotmat_to_euler(R_all)  # (T, N, 3)

            for axis_idx in range(3):
                ax = axes[row_offset + axis_idx, u_idx]

                for trial in range(n_plot):
                    angle = np.unwrap(eulers_all[:, trial, axis_idx])
                    omega = batch[:, trial, 9 + axis_idx]
                    u_norms = np.linalg.norm(batch[:, trial, 12:15], axis=1)

                    points = np.array([angle, omega]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    lc = LineCollection(segments, cmap=cmap, norm=norm,
                                        alpha=0.5, linewidth=1.0)
                    lc.set_array(u_norms[:-1])
                    ax.add_collection(lc)
                    ax.scatter(angle[0], omega[0], color='black', s=8, zorder=3)

                ax.autoscale()
                ax.grid(True, alpha=0.3, linestyle='--')

                if u_idx == 0:
                    ax.set_ylabel(f"{ds_label} — {axis_labels[axis_idx]}\n" +
                                  r"$\omega$ (rad/s)", fontsize=9)

                if row_offset + axis_idx == 0:
                    ax.set_title(f"Batch {u_idx}", fontsize=11, fontweight='bold')

                if row_offset + axis_idx == nrows - 1:
                    ax.set_xlabel("Angle (rad)", fontsize=9)
                else:
                    ax.set_xticklabels([])

    # Colorbar
    cbar_ax = fig.add_axes([0.93, 0.12, 0.015, 0.76])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cb.set_label(r'$\|u\|$ (Nm)', fontsize=10)

    fig.suptitle("3D Pendulum Phase Space — Rotation about X, Y, Z (Euler ZYX)",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.subplots_adjust(right=0.91, hspace=0.25, wspace=0.3)

    ds_name = os.path.splitext(filename)[0]
    out_path = os.path.join(save_dir, f'{ds_name}_phase_space_XYZ.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


# ─────────────────── New: SO(3)-aware phase space ───────────────────

def plot_so3_phase_space(save_dir, num_trajs_to_plot=15, filename=None,
                         m=1.0, l=1.0):
    """Geometrically-natural visualizations for SO(3) pendulum data.

    Per dataset (train, test) and per control batch:
      Row 0/4: Bob trajectory on S²  (config space of point-mass bob)
      Row 1/5: Tilt angle α = arccos((R e_z)_z) vs dα/dt   — clean phase portrait
      Row 2/6: ||ω|| (rad/s) over time
      Row 3/7: Total energy E(t) = ½ m l² ||ω||² + m g l (R e_z)_z
    """
    filename, file_path = _resolve_dataset_path(save_dir, filename)
    if file_path is None:
        return

    print(f"Loading: {file_path}")
    data = load_data(file_path)

    t = np.asarray(data['t'])             # (T,)
    settings = data.get('settings', {})
    g = float(settings.get('g', 9.81))

    train_data = data['x']
    test_data = data.get('test_x', None)
    num_us = train_data.shape[0]

    # Global control-magnitude range
    all_u = [np.linalg.norm(train_data[..., 12:15], axis=-1).flatten()]
    if test_data is not None:
        all_u.append(np.linalg.norm(test_data[..., 12:15], axis=-1).flatten())
    u_max = np.concatenate(all_u).max()
    norm = Normalize(vmin=0, vmax=max(u_max, 0.01))
    cmap = plt.get_cmap('viridis')

    nrows = 8
    ncols = num_us
    fig = plt.figure(figsize=(4.5 * ncols, 3.5 * nrows))

    # Build the axes grid manually because rows 0 and 4 need 3D projection
    axes = [[None] * ncols for _ in range(nrows)]
    for r in range(nrows):
        for c in range(ncols):
            idx = r * ncols + c + 1
            if r in (0, 4):
                axes[r][c] = fig.add_subplot(nrows, ncols, idx, projection='3d')
            else:
                axes[r][c] = fig.add_subplot(nrows, ncols, idx)

    datasets = [
        ("Train", train_data, 0),
        ("Test",  test_data,  4),
    ]

    # Wireframe sphere coordinates (reused per panel)
    uu, vv = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
    sx = np.cos(uu) * np.sin(vv)
    sy_ = np.sin(uu) * np.sin(vv)
    sz = np.cos(vv)

    for ds_label, ds_data, row_offset in datasets:
        if ds_data is None:
            continue
        N_avail = ds_data.shape[2]
        n_plot = min(N_avail, num_trajs_to_plot)

        for u_idx in range(num_us):
            batch = ds_data[u_idx]              # (T, N, 15)
            R_all = batch[..., :9]              # (T, N, 9)
            omega_all = batch[..., 9:12]        # (T, N, 3)
            u_all = batch[..., 12:15]           # (T, N, 3)

            bob_all = _rotmat_bob_dir(R_all)                                # (T, N, 3)
            tilt_all = np.arccos(np.clip(bob_all[..., 2], -1.0, 1.0))       # (T, N)
            omega_norm_all = np.linalg.norm(omega_all, axis=-1)             # (T, N)
            u_norm_all = np.linalg.norm(u_all, axis=-1)                     # (T, N)

            T_kin = 0.5 * (m * l ** 2) * (omega_norm_all ** 2)
            V_pot = m * g * l * bob_all[..., 2]
            E_all = T_kin + V_pot

            dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
            tilt_rate_all = np.gradient(tilt_all, dt, axis=0)               # (T, N)

            # ── Row 0/4: Bob on S² ──
            ax_s2 = axes[row_offset + 0][u_idx]
            ax_s2.plot_wireframe(sx, sy_, sz, color='lightgray',
                                 alpha=0.3, linewidth=0.5)
            ax_s2.scatter([0], [0], [1], color='green', s=30, zorder=10)   # Up
            ax_s2.scatter([0], [0], [-1], color='red', s=30, zorder=10)    # Down

            for trial in range(n_plot):
                bob = bob_all[:, trial, :]
                u_n = u_norm_all[:, trial]
                pts = bob.reshape(-1, 1, 3)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                lc3d = Line3DCollection(segs, cmap=cmap, norm=norm,
                                        alpha=0.6, linewidth=1.2)
                lc3d.set_array(u_n[:-1])
                ax_s2.add_collection3d(lc3d)
                ax_s2.scatter(*bob[0], color='black', s=8, zorder=11)

            ax_s2.set_xlim(-1.1, 1.1)
            ax_s2.set_ylim(-1.1, 1.1)
            ax_s2.set_zlim(-1.1, 1.1)
            ax_s2.set_box_aspect([1, 1, 1])
            ax_s2.set_xlabel('x', fontsize=8); ax_s2.set_ylabel('y', fontsize=8); ax_s2.set_zlabel('z', fontsize=8)
            ax_s2.tick_params(labelsize=7)
            if row_offset == 0:  # only top-most row carries the column title
                ax_s2.set_title(f"Batch {u_idx}", fontsize=11, fontweight='bold')
            if u_idx == 0:
                ax_s2.text2D(-0.18, 0.5, f"{ds_label}\nBob on S²",
                             transform=ax_s2.transAxes, fontsize=10,
                             rotation=90, va='center', ha='center', fontweight='bold')

            # ── Row 1/5: Tilt phase portrait ──
            ax_tilt = axes[row_offset + 1][u_idx]
            for trial in range(n_plot):
                a = tilt_all[:, trial]
                ad = tilt_rate_all[:, trial]
                u_n = u_norm_all[:, trial]
                pts = np.array([a, ad]).T.reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                lc = LineCollection(segs, cmap=cmap, norm=norm,
                                    alpha=0.6, linewidth=1.0)
                lc.set_array(u_n[:-1])
                ax_tilt.add_collection(lc)
                ax_tilt.scatter(a[0], ad[0], color='black', s=8, zorder=3)
            ax_tilt.autoscale()
            ax_tilt.grid(True, alpha=0.3, linestyle='--')
            ax_tilt.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
            ax_tilt.set_xlabel(r"Tilt $\alpha$ (rad)", fontsize=9)
            if u_idx == 0:
                ax_tilt.set_ylabel(f"{ds_label}\n" + r"$\dot{\alpha}$ (rad/s)",
                                   fontsize=9)

            # ── Row 2/6: ||ω|| over time ──
            ax_om = axes[row_offset + 2][u_idx]
            for trial in range(n_plot):
                om_n = omega_norm_all[:, trial]
                u_n = u_norm_all[:, trial]
                pts = np.array([t, om_n]).T.reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                lc = LineCollection(segs, cmap=cmap, norm=norm,
                                    alpha=0.6, linewidth=1.0)
                lc.set_array(u_n[:-1])
                ax_om.add_collection(lc)
            ax_om.autoscale()
            ax_om.grid(True, alpha=0.3, linestyle='--')
            ax_om.set_xlabel("t (s)", fontsize=9)
            if u_idx == 0:
                ax_om.set_ylabel(f"{ds_label}\n" + r"$\|\omega\|$ (rad/s)",
                                 fontsize=9)

            # ── Row 3/7: Energy over time ──
            ax_E = axes[row_offset + 3][u_idx]
            for trial in range(n_plot):
                E = E_all[:, trial]
                u_n = u_norm_all[:, trial]
                pts = np.array([t, E]).T.reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                lc = LineCollection(segs, cmap=cmap, norm=norm,
                                    alpha=0.6, linewidth=1.0)
                lc.set_array(u_n[:-1])
                ax_E.add_collection(lc)
            ax_E.autoscale()
            ax_E.grid(True, alpha=0.3, linestyle='--')
            ax_E.set_xlabel("t (s)", fontsize=9)
            if u_idx == 0:
                ax_E.set_ylabel(f"{ds_label}\n" + r"$E$ (J, $m=l=1$)",
                                fontsize=9)

    # Colorbar
    cbar_ax = fig.add_axes([0.93, 0.12, 0.015, 0.76])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cb.set_label(r'$\|u\|$ (Nm)', fontsize=10)

    fig.suptitle("3D Pendulum SO(3)-Aware Phase Space  "
                 r"(green=$\hat{e}_z$ up, red=down)",
                 fontsize=14, fontweight='bold', y=0.99)
    plt.subplots_adjust(right=0.91, hspace=0.4, wspace=0.3)

    ds_name = os.path.splitext(filename)[0]
    out_path = os.path.join(save_dir, f'{ds_name}_phase_space_SO3.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


# ─────────────────── Entry point ───────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot 3D pendulum phase space from a dataset .pkl file.")
    parser.add_argument("pkl_path", type=str, help="Path to the dataset .pkl file.")
    parser.add_argument("--num_trajs", type=int, default=15, help="Number of trajectories to plot per panel.")
    args = parser.parse_args()

    dataset_directory = os.path.dirname(os.path.abspath(args.pkl_path))
    filename = os.path.basename(args.pkl_path)

    plot_3d_phase_space(dataset_directory, num_trajs_to_plot=args.num_trajs, filename=filename)
    plot_so3_phase_space(dataset_directory, num_trajs_to_plot=args.num_trajs, filename=filename)