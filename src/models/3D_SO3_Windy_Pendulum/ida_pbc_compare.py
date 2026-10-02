"""IDA-PBC controller comparison: ph_nn_ode_fp32 (PyTorch) vs ph_gp_sde (JAX).

Builds an IDA-PBC controller from each trained model's V_θ / g_θ / D_w,θ
subnets and rolls out the **same swing-up scenario** (downright → upright)
on the true env (windy_pendulum_3d), using shared dW seeds for fair
comparison.

IDA-PBC law (3-axis SO(3) generalisation of the 1D-pendulum reference code
in `se3hamneuralode.examples.so3-pendulum.controller`, kept verbatim):

      dH_a = Σᵢ Rᵢ × ( -∂V_θ/∂Rᵢ  -  0.5·K_p · R_d,i )
      u_p  = dH_a  -  k_d · ω  +  D_w,θ(R) · ω
      u    = (gᵀg)⁻¹ gᵀ · u_p

The first term cancels the model's gravity (-∂V_θ/∂R), the (0.5·K_p·R_d)
term is the Frobenius spring that energy-shapes R toward R_d, the −k_d·ω
is the damping injection, and +D_w,θ(R)·ω cancels the model's friction so
the closed-loop ω-channel sees only the injected damping.

Trained-model paths (defaults; override with --nn_ckpt / --gp_ckpt):
  ph_nn_ode_fp32 :  data/run_wp3d_fp32/wp3d-so3ham-rk4-5p.tar
  ph_gp_sde      :  data/run_wp3d_jax/wp3d-so3hamGPSDE-5p.eqx

Both were trained with obs_noise=0.01, wind_force_std=0.5, friction=0.5,
no external forces — the env rollout matches that.

Usage (defaults reproduce the report figure):
    python ida_pbc_compare.py
    python ida_pbc_compare.py --K_p 25 --k_d 3 --horizon 8 --n_seeds 8

Output: PNG/PDF in `data/ida_pbc_compare/` next to this script.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

import importlib.util

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, '../../..'))
NN_DIR = os.path.join(THIS_FILE_DIR, 'ph_nn_ode_fp32')
GP_DIR = os.path.join(THIS_FILE_DIR, 'ph_gp_sde')
# Add project root + utils so `from src.utils...` imports inside the GP
# `network.py` resolve. Do NOT add the model dirs themselves — both contain
# a `network.py`, and adding both would collide on `import network`.
for _p in (PROJECT_ROOT,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from envs.windy_pendulum_3d import windy_pendulum_3d, _exp_so3, _log_so3   # noqa: E402


def _load_module_from_path(modname: str, path: str):
    """Import a .py file as a uniquely-named module so two `network.py` files
    don't collide on Python's module cache.
    """
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────
# Defaults — must match the dataset both models were trained on
# ─────────────────────────────────────────────────────────────────────
ENV_DEFAULTS = dict(
    g=9.81, m=1.0, l=1.0, dt=0.05,
    varying_friction=False, friction_coeff=0.5,
    external_force_type='sine', external_force_std=0.0,
    wind_force_std=0.5,
    ori_rep="rotmat", render_mode=None,
)

NN_DEFAULT_CKPT = os.path.join(
    NN_DIR, 'data', 'run_wp3d_fp32', 'wp3d-so3ham-rk4-5p.tar')
GP_DEFAULT_CKPT = os.path.join(
    GP_DIR, 'data', 'run_wp3d_jax', 'wp3d-so3hamGPSDE-5p.eqx')


# ─────────────────────────────────────────────────────────────────────
# IDA-PBC controller — PyTorch (ph_nn_ode_fp32)
# ─────────────────────────────────────────────────────────────────────

def build_ida_pbc_torch(ckpt_path: str, R_d: np.ndarray,
                        K_p: float, k_d: float,
                        gtg_rcond: float = 1e-3,
                        device_str: str = 'auto') -> Callable:
    """Load ph_nn_ode_fp32 and return `act(R, omega) -> u`.

    Uses the model's trained g_θ(R) — the body torque is mapped through the
    regularised pseudo-inverse of g_θ(R) per the IDA-PBC law:
        u = (gᵀg)⁻¹ gᵀ · u_p
    """
    import torch
    nn_mod = _load_module_from_path(
        'ph_nn_ode_fp32_network', os.path.join(NN_DIR, 'network.py'))
    DissipativeSO3HamNODE = nn_mod.DissipativeSO3HamNODE

    if device_str == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    torch.set_default_dtype(torch.float32)
    model = DissipativeSO3HamNODE(device=device, u_dim=3).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"  [NN-IDA-PBC] loaded {ckpt_path}  (device={device})  "
          f"[trained g_θ, lstsq rcond={gtg_rcond}]")

    R_d_t = torch.tensor(R_d.reshape(1, 9), device=device, dtype=torch.float32)

    def act(R: np.ndarray, omega: np.ndarray) -> np.ndarray:
        q = torch.tensor(R.reshape(1, 9), device=device,
                         dtype=torch.float32, requires_grad=True)
        omega_t = torch.tensor(omega.reshape(1, 3), device=device,
                               dtype=torch.float32)

        V_q  = model.V_net(q)                        # (1, 1)
        dV_q = torch.autograd.grad(V_q.sum(), q)[0]  # (1, 9)
        g_q  = model.g_net(q)                        # (1, 3, 3)
        Dw_q = model.Dw_net(q)                       # (1, 3, 3)

        # dH_a = Σᵢ Rᵢ × ( -∂V/∂Rᵢ - 0.5·K_p · R_d,i )
        q_rows    = q.view(1, 3, 3)
        dV_rows   = dV_q.view(1, 3, 3)
        Rd_rows   = R_d_t.view(1, 3, 3)
        target_rows = -dV_rows - 0.5 * K_p * Rd_rows
        dH_a = torch.linalg.cross(q_rows, target_rows, dim=2).sum(dim=1)  # (1,3)

        Dw_omega = torch.matmul(Dw_q, omega_t.unsqueeze(2)).squeeze(2)
        u_p = dH_a - k_d * omega_t + Dw_omega                            # (1,3)

        # u = (gᵀg)⁻¹ gᵀ · u_p   via regularised lstsq
        g_np = g_q.detach().cpu().numpy()[0]                             # (3,3)
        u_p_np = u_p.detach().cpu().numpy()[0]                           # (3,)
        u_np, *_ = np.linalg.lstsq(g_np, u_p_np, rcond=gtg_rcond)
        return u_np.astype(np.float64)

    return act


# ─────────────────────────────────────────────────────────────────────
# IDA-PBC controller — JAX/Equinox (ph_gp_sde)
# ─────────────────────────────────────────────────────────────────────

def build_ida_pbc_jax(ckpt_path: str, R_d: np.ndarray,
                      K_p: float, k_d: float,
                      gtg_rcond: float = 1e-3) -> Callable:
    """Load ph_gp_sde and return `act(R, omega) -> u`.

    Uses the model's trained g_θ(R); body torque mapped through (gᵀg)⁻¹gᵀ
    via regularised lstsq.
    """
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    gp_mod = _load_module_from_path(
        'ph_gp_sde_network', os.path.join(GP_DIR, 'network.py'))
    DissipativeSO3HamSDE = gp_mod.DissipativeSO3HamSDE

    template = DissipativeSO3HamSDE(
        key=jax.random.PRNGKey(0),
        u_dim=3, init_gain=0.5, friction=True, hidden_dim=20, l=1.0,
        init_sigma_R=0.1, init_sigma_omega=0.1,
        init_sigma_obs_omega=0.5, init_sigma_const=0.5,
    )
    model = eqx.tree_deserialise_leaves(ckpt_path, template)
    print(f"  [GP-IDA-PBC] loaded {ckpt_path}  "
          f"[trained g_θ, lstsq rcond={gtg_rcond}]")

    R_d_j = jnp.asarray(R_d.reshape(9), dtype=jnp.float32)

    @jax.jit
    def _kernel(q, omega):
        def V_scalar(q_):
            return model._V_call(q_)[0]
        dV = jax.grad(V_scalar)(q)                                       # (9,)

        g_q  = model._g_call(q)                                          # (3,3)
        Dw_q = model._Dw_call(q)                                         # (3,3)

        q_rows    = q.reshape(3, 3)
        dV_rows   = dV.reshape(3, 3)
        Rd_rows   = R_d_j.reshape(3, 3)
        target_rows = -dV_rows - 0.5 * K_p * Rd_rows
        dH_a = jnp.sum(jnp.cross(q_rows, target_rows, axis=-1), axis=0)  # (3,)

        u_p = dH_a - k_d * omega + Dw_q @ omega                          # (3,)
        return u_p, g_q

    def act(R: np.ndarray, omega: np.ndarray) -> np.ndarray:
        q  = jnp.asarray(R.reshape(9), dtype=jnp.float32)
        om = jnp.asarray(omega, dtype=jnp.float32)
        u_p, g_q = _kernel(q, om)
        u_p_np = np.asarray(u_p, dtype=np.float64)
        g_np   = np.asarray(g_q, dtype=np.float64)
        u_np, *_ = np.linalg.lstsq(g_np, u_p_np, rcond=gtg_rcond)
        return u_np.astype(np.float64)

    return act


# ─────────────────────────────────────────────────────────────────────
# IDA-PBC controller — analytic ground-truth subnetworks
# ─────────────────────────────────────────────────────────────────────

def build_ida_pbc_gt(R_d: np.ndarray, K_p: float, k_d: float,
                     m: float = 1.0, l: float = 1.0, g: float = 9.81,
                     friction_coeff: float = 0.5) -> Callable:
    """IDA-PBC controller using analytical ground-truth subnetworks:

        M⁻¹(q)  = (1 / m·l²)·I₃
        V(q)    = m·g·l · R[2,2]   ⇒   ∂V/∂q has one non-zero entry at q[8]
        D_w(q)  = friction_coeff · I₃
        G(q)    = I₃                   (matches the env's true input map)

    Pure numpy; no autograd, no model load.
    """
    print(f"  [GT-IDA-PBC] analytic subnets  "
          f"m={m} l={l} g={g} friction={friction_coeff}  [G(q) = I]")

    R_d = np.asarray(R_d, dtype=np.float64).reshape(3, 3)
    Dw  = friction_coeff * np.eye(3)

    # ∂V/∂q has only one non-zero entry, at q[8]. As (3, 3): only [2, 2] = m·g·l.
    dV_const = np.zeros((3, 3))
    dV_const[2, 2] = m * g * l

    def act(R: np.ndarray, omega: np.ndarray) -> np.ndarray:
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        omega = np.asarray(omega, dtype=np.float64).reshape(3)

        target_rows = -dV_const - 0.5 * K_p * R_d                 # (3, 3)
        dH_a = np.sum(np.cross(R, target_rows, axis=-1), axis=0)  # (3,)

        u_p = dH_a - k_d * omega + Dw @ omega                     # (3,)
        # G = I ⇒ u = u_p
        return u_p.astype(np.float64)

    return act


# ─────────────────────────────────────────────────────────────────────
# Rollout on the env
# ─────────────────────────────────────────────────────────────────────

def rollout_env(controller: Callable, R0: np.ndarray, omega0: np.ndarray,
                horizon: float = 8.0, env_seed: int = 0,
                env_kwargs: dict | None = None,
                u_clip: float | None = None) -> dict:
    """Roll the env under `controller` for `horizon` seconds.

    Returns dict with t, R, omega, u, geo_dist (||log(R^T R_d)||), H_total.
    """
    env_kwargs = {**ENV_DEFAULTS, **(env_kwargs or {})}
    env = windy_pendulum_3d(seed=env_seed, **env_kwargs)
    env.reset(seed=env_seed,
              options={'R_init': R0.copy(), 'omega_init': omega0.copy()})

    T = int(round(horizon / env.dt))
    Rs = np.zeros((T + 1, 3, 3))
    omegas = np.zeros((T + 1, 3))
    us = np.zeros((T, 3))

    Rs[0] = env.R.copy()
    omegas[0] = env.omega.copy()
    for k in range(T):
        u = controller(env.R.copy(), env.omega.copy())
        if u_clip is not None:
            u = np.clip(u, -u_clip, u_clip)
        us[k] = u
        env.step(u)
        Rs[k + 1] = env.R.copy()
        omegas[k + 1] = env.omega.copy()
    env.close()

    t = np.arange(T + 1) * env.dt
    return dict(t=t, R=Rs, omega=omegas, u=us, T=T, dt=env.dt)


# ─────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────

def geodesic(R: np.ndarray, R_d: np.ndarray) -> np.ndarray:
    """||log(R^T R_d)|| at every time step. R: (T+1, 3, 3)."""
    out = np.zeros(R.shape[0])
    for k in range(R.shape[0]):
        out[k] = np.linalg.norm(_log_so3(R[k].T @ R_d))
    return out


def total_energy(R: np.ndarray, omega: np.ndarray,
                 m=1.0, l=1.0, g=9.81) -> np.ndarray:
    """Open-loop pendulum energy: ½ m l² ‖ω‖² + m g l · R[2,2]."""
    KE = 0.5 * (m * l * l) * np.sum(omega ** 2, axis=1)
    PE = m * g * l * R[:, 2, 2]
    return KE + PE


def aggregate(rollouts: list[dict], R_d: np.ndarray) -> dict:
    """Stack a list of rollouts and compute mean/std curves."""
    t = rollouts[0]['t']
    geos = np.stack([geodesic(r['R'], R_d) for r in rollouts])           # (S, T+1)
    om_mags = np.stack([np.linalg.norm(r['omega'], axis=1) for r in rollouts])
    u_mags = np.stack([np.linalg.norm(r['u'], axis=1) for r in rollouts])
    Es = np.stack([total_energy(r['R'], r['omega']) for r in rollouts])
    return dict(
        t=t,
        geo_mean=geos.mean(0), geo_std=geos.std(0), geos=geos,
        om_mean=om_mags.mean(0), om_std=om_mags.std(0),
        u_mean=u_mags.mean(0), u_std=u_mags.std(0),
        E_mean=Es.mean(0), E_std=Es.std(0),
    )


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────

_STYLE = {
    'NN-IDA-PBC (ph_nn_ode_fp32)': dict(color='C0', lw=1.8, ls='-'),
    'GP-IDA-PBC (ph_gp_sde)':       dict(color='C3', lw=1.6, ls='--'),
    'GT_SDE (analytic GT subnets)': dict(color='C2', lw=1.6, ls='-.'),
}


def plot_compare(aggs: dict[str, dict], save_base: str, header: str = ""):
    """One figure with 4 panels (geo, |ω|, |u|, energy). Mean ± std bands."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    for label, agg in aggs.items():
        st = _STYLE.get(label, dict(lw=1.4))
        col = st.get('color', None)
        t = agg['t']
        t_u = t[:-1]                         # u indexed 0..T-1
        axes[0, 0].plot(t, agg['geo_mean'], label=label, **st)
        axes[0, 0].fill_between(t, agg['geo_mean'] - agg['geo_std'],
                                agg['geo_mean'] + agg['geo_std'],
                                color=col, alpha=0.15)
        axes[0, 1].plot(t, agg['om_mean'], label=label, **st)
        axes[0, 1].fill_between(t, agg['om_mean'] - agg['om_std'],
                                agg['om_mean'] + agg['om_std'],
                                color=col, alpha=0.15)
        axes[1, 0].plot(t_u, agg['u_mean'], label=label, **st)
        axes[1, 0].fill_between(t_u, agg['u_mean'] - agg['u_std'],
                                agg['u_mean'] + agg['u_std'],
                                color=col, alpha=0.15)
        axes[1, 1].plot(t, agg['E_mean'], label=label, **st)
        axes[1, 1].fill_between(t, agg['E_mean'] - agg['E_std'],
                                agg['E_mean'] + agg['E_std'],
                                color=col, alpha=0.15)

    axes[0, 0].axhline(0.0, color='k', lw=0.5)
    axes[0, 0].set_xlabel('t [s]')
    axes[0, 0].set_ylabel(r'$\|\log(R^\top R_d)\|$ [rad]')
    axes[0, 0].set_title('Geodesic distance to target $R_d = I$')
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].set_xlabel('t [s]')
    axes[0, 1].set_ylabel(r'$\|\omega\|$ [rad/s]')
    axes[0, 1].set_title('Angular speed magnitude')
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].set_xlabel('t [s]')
    axes[1, 0].set_ylabel(r'$\|u\|$ [N$\cdot$m]')
    axes[1, 0].set_title('Control torque magnitude')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axhline(ENV_DEFAULTS['m'] * ENV_DEFAULTS['g'] * ENV_DEFAULTS['l'],
                       color='k', lw=0.5, ls=':',
                       label=r'$E$ at $R_d, \omega=0$')
    axes[1, 1].set_xlabel('t [s]')
    axes[1, 1].set_ylabel(r'$E = \frac{1}{2} m l^2 \|\omega\|^2 + m g l\,R_{zz}$ [J]')
    axes[1, 1].set_title('Total open-loop energy')
    axes[1, 1].grid(alpha=0.3)

    axes[0, 0].legend(loc='best', fontsize=10, framealpha=0.9)
    if header:
        fig.suptitle(header, fontsize=12, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()

    fig.savefig(save_base + '.png', dpi=140, bbox_inches='tight')
    fig.savefig(save_base + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {save_base}.png/.pdf")


def plot_per_seed_geo(rollouts_by_label: dict[str, list[dict]],
                      R_d: np.ndarray, save_base: str, header: str = ""):
    """One panel per controller, every individual seed plotted thinly."""
    fig, axes = plt.subplots(1, len(rollouts_by_label),
                              figsize=(6 * len(rollouts_by_label), 4.5),
                              squeeze=False)
    axes = axes[0]
    for ax, (label, rolls) in zip(axes, rollouts_by_label.items()):
        st = _STYLE.get(label, dict(color='C0'))
        col = st.get('color', 'C0')
        for r in rolls:
            ax.plot(r['t'], geodesic(r['R'], R_d), color=col,
                    alpha=0.35, lw=1.0)
        # mean
        geos = np.stack([geodesic(r['R'], R_d) for r in rolls])
        ax.plot(rolls[0]['t'], geos.mean(0), color=col, lw=2.0,
                label=f'{label} mean')
        ax.axhline(0.0, color='k', lw=0.5)
        ax.set_xlabel('t [s]')
        ax.set_ylabel(r'$\|\log(R^\top R_d)\|$ [rad]')
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    if header:
        fig.suptitle(header, fontsize=12, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()
    fig.savefig(save_base + '.png', dpi=140, bbox_inches='tight')
    fig.savefig(save_base + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {save_base}.png/.pdf")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--K_p', type=float, default=20.0,
                   help='IDA-PBC spring stiffness on Frobenius distance.')
    p.add_argument('--k_d', type=float, default=2.0,
                   help='Damping injection.')
    p.add_argument('--init_deg', type=float, default=180.0,
                   help='Initial tilt angle (deg) about --init_axis. Default 180° = downright.')
    p.add_argument('--init_axis', type=float, nargs=3, default=[1.0, 0.05, 0.0],
                   metavar=('AX', 'AY', 'AZ'),
                   help='Rotation axis for the initial tilt (auto-normalized). Slight off-x '
                        'tilt by default to break the cut-locus singularity at exact downright.')
    p.add_argument('--target_deg', type=float, default=30.0,
                   help='Target tilt angle (deg) about --target_axis.')
    p.add_argument('--target_axis', type=float, nargs=3, default=[1.0, 0.0, 0.0],
                   metavar=('AX', 'AY', 'AZ'),
                   help='Rotation axis for R_d (auto-normalized).')
    p.add_argument('--gtg_rcond', type=float, default=1e-3,
                   help='rcond for the lstsq-based pinv of trained g_θ.')
    p.add_argument('--horizon', type=float, default=8.0,
                   help='Seconds.')
    p.add_argument('--n_seeds', type=int, default=8,
                   help='Independent stochastic env rollouts per controller.')
    p.add_argument('--u_clip', type=float, default=50.0,
                   help='Saturate the control torque per axis.')
    p.add_argument('--nn_ckpt', type=str, default=NN_DEFAULT_CKPT)
    p.add_argument('--gp_ckpt', type=str, default=GP_DEFAULT_CKPT)
    p.add_argument('--out_dir', type=str,
                   default=os.path.join(THIS_FILE_DIR, 'data', 'ida_pbc_compare'))
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Target ──────────────────────────────────────────────────────────
    t_axis = np.asarray(args.target_axis, dtype=np.float64)
    t_axis = t_axis / np.linalg.norm(t_axis)
    R_d = _exp_so3(np.deg2rad(args.target_deg) * t_axis)

    # ── IC: tilted by `init_deg` about `init_axis`, zero angular velocity ─
    i_axis = np.asarray(args.init_axis, dtype=np.float64)
    i_axis = i_axis / np.linalg.norm(i_axis)
    R0 = _exp_so3(np.deg2rad(args.init_deg) * i_axis)
    omega0 = np.zeros(3)

    print("=" * 70)
    print(f" IDA-PBC: downright→{args.target_deg:.0f}° tilt  "
          f"(K_p={args.K_p}, k_d={args.k_d}, "
          f"horizon={args.horizon}s, n_seeds={args.n_seeds})")
    print("=" * 70)
    print(f"  R0  = exp_so3({args.init_deg:.1f}° · {tuple(i_axis.round(3))})")
    print(f"  R_d = exp_so3({args.target_deg:.1f}° · {tuple(t_axis.round(3))})")
    print(f"  geo(R0, R_d) = {np.linalg.norm(_log_so3(R0.T @ R_d)):.4f} rad")
    print(f"  g_θ usage:  NN/GP → trained g_θ (pinv);  GT → g = I (analytic)")
    print(f"  env: friction={ENV_DEFAULTS['friction_coeff']}  "
          f"wind_std={ENV_DEFAULTS['wind_force_std']}  "
          f"ext_force={ENV_DEFAULTS['external_force_std']}")

    # ── Build controllers ───────────────────────────────────────────────
    print("\n[building controllers]")
    nn_ctrl = build_ida_pbc_torch(args.nn_ckpt, R_d, args.K_p, args.k_d,
                                  gtg_rcond=args.gtg_rcond)
    gp_ctrl = build_ida_pbc_jax(args.gp_ckpt, R_d, args.K_p, args.k_d,
                                gtg_rcond=args.gtg_rcond)
    gt_ctrl = build_ida_pbc_gt(R_d, args.K_p, args.k_d,
                               m=ENV_DEFAULTS['m'], l=ENV_DEFAULTS['l'],
                               g=ENV_DEFAULTS['g'],
                               friction_coeff=ENV_DEFAULTS['friction_coeff'])

    controllers: dict[str, Callable] = {
        'NN-IDA-PBC (ph_nn_ode_fp32)': nn_ctrl,
        'GP-IDA-PBC (ph_gp_sde)':       gp_ctrl,
        'GT_SDE (analytic GT subnets)': gt_ctrl,
    }

    # ── Run rollouts ────────────────────────────────────────────────────
    rollouts_by_label: dict[str, list[dict]] = {}
    aggs: dict[str, dict] = {}

    for label, ctrl in controllers.items():
        print(f"\n[rollouts] {label}")
        rolls = []
        for seed in range(args.n_seeds):
            r = rollout_env(ctrl, R0, omega0,
                            horizon=args.horizon,
                            env_seed=seed,
                            u_clip=args.u_clip)
            geo_t = geodesic(r['R'], R_d)
            print(f"  seed={seed:>2d}  geo: {geo_t[0]:.3f} → {geo_t[-1]:.3f}   "
                  f"|u|_max={float(np.max(np.linalg.norm(r['u'], axis=1))):.2f}")
            rolls.append(r)
        rollouts_by_label[label] = rolls
        aggs[label] = aggregate(rolls, R_d)

    # ── Plots ───────────────────────────────────────────────────────────
    print("\n[plotting]")
    header = (f"IDA-PBC: {args.init_deg:.0f}° → {args.target_deg:.0f}° tilt  |  "
              f"$K_p$={args.K_p}, $k_d$={args.k_d}, "
              f"horizon={args.horizon}s, $n$={args.n_seeds} seeds  |  "
              "NN/GP use trained $g_\\theta$, GT uses $g=I$")
    plot_compare(aggs,
                 os.path.join(args.out_dir, 'compare_overlay'),
                 header=header)
    plot_per_seed_geo(rollouts_by_label, R_d,
                      os.path.join(args.out_dir, 'compare_per_seed_geo'),
                      header=header)

    # ── Summary table ───────────────────────────────────────────────────
    tail_frac = 0.25
    print("\n[summary — last 25% of horizon]")
    print(f"  {'controller':<35s}  {'mean geo':>10s}  {'std geo':>10s}  "
          f"{'mean |u|':>10s}")
    for label, agg in aggs.items():
        n = len(agg['t'])
        tail = slice(int((1.0 - tail_frac) * n), None)
        tail_u = slice(int((1.0 - tail_frac) * (n - 1)), None)
        print(f"  {label:<35s}  {agg['geo_mean'][tail].mean():>10.4f}  "
              f"{agg['geo_std'][tail].mean():>10.4f}  "
              f"{agg['u_mean'][tail_u].mean():>10.4f}")

    print(f"\nDone. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
