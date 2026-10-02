"""Validate the SO(3) Energy-Casimir controller in the env.

Two modes:

  * `single` (default) — one rollout from a tilted initial condition.
    Plots geodesic distance to R_d, |omega|, control torque norm, and
    H_cl(t) − H_cl(R_d, 0) over time. Quick visual sanity check that the
    controller stabilizes (gains: k_c=30, d_inj=1.0, both from
    verify_control.py).

  * `sweep` — empirical d_inj sweep. For each candidate d_inj, runs
    `--n_seeds` independent stochastic rollouts of length `--horizon`
    seconds. Reports the steady-state mean and std of the energy excess
    H_cl − H_cl(R_d, 0), averaged over the last 25 % of each rollout.

    This is the empirical generator argument from the discussion: instead
    of computing (1/2)tr(Σ^T M^-1 Σ) analytically and dividing by an
    arbitrary ω-threshold, we let the actual closed-loop SDE tell us
    where the steady-state energy plateau sits as a function of d_inj.

CLI:
    python ph_gp_sde/evaluate_controller.py                # single rollout
    python ph_gp_sde/evaluate_controller.py --mode sweep   # d_inj sweep
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, '../../../..'))
for _p in (PROJECT_ROOT, THIS_FILE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from envs.windy_pendulum_3d import windy_pendulum_3d, _exp_so3, _log_so3   # noqa: E402
from qp_env import QPWindyPendulum3D                                        # noqa: E402
from controller import (                                                    # noqa: E402
    EnergyCasimirController, ControllerConfig,
    closed_loop_energy, closed_loop_energy_at_target,
)

# JAX-side imports deferred (only loaded when --compare_plants is set so the
# default env-only path doesn't pay the JAX startup cost).
_jax = None
_jnp = None
_eqx = None
_lie_heun_sde_step = None
_GroundTruthHamiltonianSDE = None
_load_trained_model = None


def _import_jax_stack():
    """Lazy import of JAX + the model classes used for model-side rollouts."""
    global _jax, _jnp, _eqx, _lie_heun_sde_step
    global _GroundTruthHamiltonianSDE, _load_trained_model
    if _jax is not None:
        return
    import jax as jax_mod
    import jax.numpy as jnp_mod
    import equinox as eqx_mod
    from src.utils.JAX.lie_integrator import lie_heun_sde_step as _step
    from eval_ground_truth_match import GroundTruthHamiltonianSDE as _GT
    from verify_control import load_trained_model as _load
    _jax, _jnp, _eqx = jax_mod, jnp_mod, eqx_mod
    _lie_heun_sde_step = _step
    _GroundTruthHamiltonianSDE = _GT
    _load_trained_model = _load


def _override_gp_g_with_identity(gp_model):
    """Replace the trained `g_net` inside `gp_model` with a stub that returns
    the 3×3 identity (the env's true input map). Keeps every other GP subnet
    intact (M, V, D, σ) so the rollout reflects only the *non-g* learned
    physics. Returns a new model — Equinox modules are immutable.
    """
    eqx = _eqx
    jnp = _jnp

    class _IdentityG(eqx.Module):
        out_dim: int = eqx.field(static=True)
        def __init__(self, out_dim: int = 3):
            self.out_dim = int(out_dim)
        def __call__(self, q, key=None, inference_mode=False):
            del key, inference_mode
            return jnp.eye(self.out_dim, dtype=q.dtype)

    return eqx.tree_at(lambda m: m.g_net, gp_model, _IdentityG())


# ─────────────────────────────────────────────────────────────────────
# Defaults — match the dataset the model was trained on
# ─────────────────────────────────────────────────────────────────────

ENV_DEFAULTS = dict(
    g=9.81, m=1.0, l=1.0, dt=0.05,
    varying_friction=False, friction_coeff=0.5,
    external_force_type='sine', external_force_std=0.0,
    wind_force_std=0.5,                      # matches trained dataset
)


# ─────────────────────────────────────────────────────────────────────
# Single rollout (stochastic env, fixed seed)
# ─────────────────────────────────────────────────────────────────────

def rollout_env(controller: EnergyCasimirController,
                R0: np.ndarray, omega0: np.ndarray,
                horizon: float = 6.0, env_seed: int = 0,
                env_kwargs: dict = None,
                use_qp_integrator: bool = True):
    """Roll the env under `controller` for `horizon` seconds.

    `use_qp_integrator=True` (default) uses `QPWindyPendulum3D`, which
    integrates internally in (R, p) form (ω = M⁻¹·p reconstructed at
    each substep). For the spherical pendulum this is algebraically
    equivalent to the (R, ω) integrator in the parent env, but matches
    the model-side integrator topologically.

    Returns dict of (T+1,)/(T+1, 3) arrays:
        t, R (T+1,3,3), omega (T+1,3), u (T,3), H_cl (T+1,), geo_dist (T+1,)
    """
    env_kwargs = {**ENV_DEFAULTS, **(env_kwargs or {})}
    env_cls = QPWindyPendulum3D if use_qp_integrator else windy_pendulum_3d
    env = env_cls(seed=env_seed, **env_kwargs)
    env.reset(seed=env_seed,
              options={'R_init': R0.copy(), 'omega_init': omega0.copy()})

    T = int(round(horizon / env.dt))
    Rs = np.zeros((T + 1, 3, 3))
    omegas = np.zeros((T + 1, 3))
    us = np.zeros((T, 3))

    Rs[0] = env.R.copy()
    omegas[0] = env.omega.copy()

    for k in range(T):
        u = controller.act(env.R, env.omega)
        us[k] = u
        env.step(u)
        Rs[k + 1] = env.R.copy()
        omegas[k + 1] = env.omega.copy()

    H = np.array([
        closed_loop_energy(Rs[k], omegas[k], controller.R_d,
                           controller.cfg.k_c,
                           m=env.m, l=env.l, g=env.g)
        for k in range(T + 1)
    ])
    H_target = closed_loop_energy_at_target(controller.R_d,
                                            m=env.m, l=env.l, g=env.g)

    geo = np.array([np.linalg.norm(_log_so3(Rs[k].T @ controller.R_d))
                    for k in range(T + 1)])

    t = np.arange(T + 1) * env.dt
    return dict(t=t, R=Rs, omega=omegas, u=us,
                H=H, H_target=H_target, geo=geo,
                T=T, dt=env.dt)


# ─────────────────────────────────────────────────────────────────────
# Model-side closed-loop rollout (uses lie_heun_sde_step)
# ─────────────────────────────────────────────────────────────────────

def rollout_model_closed_loop(model, controller,
                              R0: np.ndarray, omega0: np.ndarray,
                              horizon: float = 6.0,
                              dt: float = 0.05, n_substeps: int = 10,
                              dW_seed: int = 0):
    """Closed-loop rollout in `model` (Equinox SDE module) under `controller`.

    The substep loop is jit-scanned per outer step (fixed u within an outer
    step, mirroring how the env applies a constant u across its 10 substeps).
    Controller queries happen between outer steps in numpy.
    """
    _import_jax_stack()
    jnp = _jnp
    eqx = _eqx
    step = _lie_heun_sde_step

    @eqx.filter_jit
    def _outer_step(model_, x_qp_, u_, dWs_, h_):
        def inner(carry, dW):
            return step(model_, carry, u_, h_, dW), None
        final, _ = _jax.lax.scan(inner, x_qp_, dWs_)
        return final

    rng = np.random.default_rng(dW_seed)
    n_outer = int(round(horizon / dt))
    h = dt / n_substeps

    R = R0.copy().astype(np.float64)
    omega = omega0.copy().astype(np.float64)

    # ω → p via M(q) = inv(M⁻¹(q))
    q0_j = jnp.asarray(R.reshape(-1), dtype=jnp.float32)
    M_inv0 = np.asarray(model.M_inv(q0_j))
    p0 = np.linalg.solve(M_inv0, omega)
    x_qp = jnp.concatenate([q0_j, jnp.asarray(p0, dtype=jnp.float32)])

    Rs = [R.copy()]
    omegas = [omega.copy()]
    us = []

    h_j = jnp.asarray(h, dtype=jnp.float32)

    for k in range(n_outer):
        # Read current state for controller
        q_cur = np.asarray(x_qp[:9])
        R_cur = q_cur.reshape(3, 3)
        p_cur = np.asarray(x_qp[9:])
        M_inv_cur = np.asarray(model.M_inv(jnp.asarray(q_cur,
                                                       dtype=jnp.float32)))
        omega_cur = M_inv_cur @ p_cur

        # Closed-loop control
        u = controller.act(R_cur, omega_cur)

        # Wiener increments for this outer step (var = h)
        dWs = rng.normal(0.0, np.sqrt(h), size=(n_substeps, 3))
        dWs_j = jnp.asarray(dWs, dtype=jnp.float32)
        u_j = jnp.asarray(u, dtype=jnp.float32)

        x_qp = _outer_step(model, x_qp, u_j, dWs_j, h_j)

        # Decode for diagnostics
        q_new = np.asarray(x_qp[:9])
        p_new = np.asarray(x_qp[9:])
        M_inv_new = np.asarray(model.M_inv(jnp.asarray(q_new,
                                                       dtype=jnp.float32)))
        omega_new = M_inv_new @ p_new
        Rs.append(q_new.reshape(3, 3))
        omegas.append(omega_new)
        us.append(u)

    Rs = np.stack(Rs)
    omegas = np.stack(omegas)
    us = np.stack(us)

    H = np.array([
        closed_loop_energy(Rs[k], omegas[k], controller.R_d,
                           controller.cfg.k_c)
        for k in range(n_outer + 1)
    ])
    H_target = closed_loop_energy_at_target(controller.R_d)
    geo = np.array([np.linalg.norm(_log_so3(Rs[k].T @ controller.R_d))
                    for k in range(n_outer + 1)])

    t = np.arange(n_outer + 1) * dt
    return dict(t=t, R=Rs, omega=omegas, u=us,
                H=H, H_target=H_target, geo=geo,
                T=n_outer, dt=dt)


# ─────────────────────────────────────────────────────────────────────
# Plotting — overlays multiple plant rollouts
# ─────────────────────────────────────────────────────────────────────

# Style for each plant. Order = legend order.
_PLANT_STYLE = {
    'env (true physics)':  dict(color='C0', lw=1.6, ls='-'),
    'GT-subnet model':     dict(color='C2', lw=1.4, ls='--'),
    'GP-subnet model':     dict(color='C3', lw=1.4, ls=':'),
}


def plot_single(outs: dict, save_path: str):
    """outs : {label -> rollout dict}. Overlays one curve per plant on each axis."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for label, out in outs.items():
        st = _PLANT_STYLE.get(label, dict(lw=1.3))
        axes[0, 0].plot(out['t'], out['geo'], label=label, **st)
        axes[0, 1].plot(out['t'], np.linalg.norm(out['omega'], axis=1),
                        label=label, **st)
        axes[1, 0].plot(out['t'][:-1], np.linalg.norm(out['u'], axis=1),
                        label=label, **st)
        axes[1, 1].plot(out['t'], out['H'] - out['H_target'],
                        label=label, **st)

    axes[0, 0].axhline(0.0, color='k', lw=0.5)
    axes[1, 1].axhline(0.0, color='k', lw=0.5)
    axes[0, 0].set_xlabel('t [s]'); axes[0, 0].set_ylabel(r'$\|\log(R^\top R_d)\|$')
    axes[0, 0].set_title('Geodesic distance to target')
    axes[0, 1].set_xlabel('t [s]'); axes[0, 1].set_ylabel(r'$\|\omega\|$ [rad/s]')
    axes[0, 1].set_title('Angular speed')
    axes[1, 0].set_xlabel('t [s]'); axes[1, 0].set_ylabel(r'$\|u\|$ [N·m]')
    axes[1, 0].set_title('Control torque magnitude')
    axes[1, 1].set_xlabel('t [s]'); axes[1, 1].set_ylabel(r'$H_{cl}(t) - H_{cl}(R_d, 0)$')
    axes[1, 1].set_title('Energy excess')

    axes[0, 0].legend(loc='best', fontsize=9, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {save_path}")


# ─────────────────────────────────────────────────────────────────────
# d_inj sweep — empirical generator argument
# ─────────────────────────────────────────────────────────────────────

def _summarise_rollout(out, tail_frac=0.25):
    """Return (ss_mean_excess, ss_std_excess, geo_tail_mean) for one rollout."""
    n = out['T'] + 1
    tail = slice(int((1.0 - tail_frac) * n), None)
    excess = out['H'][tail] - out['H_target']
    return (float(np.mean(excess)),
            float(np.std(excess)),
            float(np.mean(out['geo'][tail])))


def sweep_d_inj(d_inj_grid, k_c, R_d, *,
                rollout_fn,                # (controller, R0, ω0, horizon, seed) → out
                n_seeds=16, horizon=8.0, tilt_deg=20.0,
                alpha_D=0.0, alpha_V=0.0, model=None):
    """For each d_inj in the grid, run `n_seeds` rollouts of `rollout_fn` and
    aggregate into mean ± std over the last 25 % of each rollout.

    `alpha_D`, `alpha_V`, `model` parametrise the cancellation terms; pass a
    GP-trained model when α > 0."""
    R0 = R_d @ _exp_so3(np.array([np.deg2rad(tilt_deg), 0.0, 0.0]))
    omega0 = np.zeros(3)

    rows = []
    for d_inj in d_inj_grid:
        cfg = ControllerConfig(R_d=R_d, k_c=k_c, d_inj=float(d_inj),
                               alpha_D=alpha_D, alpha_V=alpha_V,
                               use_trained_g=False)
        ctrl = EnergyCasimirController(cfg, model=model)

        ss_means, ss_stds, term_geos = [], [], []
        for seed in range(n_seeds):
            out = rollout_fn(ctrl, R0, omega0, horizon, seed)
            sm, ss, gt = _summarise_rollout(out)
            ss_means.append(sm); ss_stds.append(ss); term_geos.append(gt)

        rows.append(dict(
            d_inj=float(d_inj),
            ss_mean_mean=float(np.mean(ss_means)),
            ss_mean_std=float(np.std(ss_means)),
            ss_std_mean=float(np.mean(ss_stds)),
            geo_tail_mean=float(np.mean(term_geos)),
        ))
        print(f"    d_inj={d_inj:6.3f}  E[H_excess]={rows[-1]['ss_mean_mean']:+.4f}"
              f"  ±{rows[-1]['ss_mean_std']:.4f}   "
              f"σ_path(H)={rows[-1]['ss_std_mean']:.4f}   "
              f"E[geo]={rows[-1]['geo_tail_mean']:.4f}")
    return rows


def plot_sweep(plant_rows: dict, save_path: str):
    """plant_rows : {label -> list[row dict]}. One marker series per plant."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for label, rows in plant_rows.items():
        st = _PLANT_STYLE.get(label, dict())
        d = np.array([r['d_inj'] for r in rows])
        em = np.array([r['ss_mean_mean'] for r in rows])
        es = np.array([r['ss_mean_std'] for r in rows])
        sm = np.array([r['ss_std_mean'] for r in rows])
        gm = np.array([r['geo_tail_mean'] for r in rows])

        col = st.get('color', None)
        ls = st.get('ls', '-')
        axes[0].errorbar(d, em, yerr=es, fmt='o', capsize=3,
                         color=col, linestyle=ls, label=label)
        axes[0].fill_between(d, em - sm, em + sm, alpha=0.10, color=col)
        axes[1].plot(d, gm, 'o', color=col, linestyle=ls, label=label)

    for ax in axes:
        ax.set_xscale('log'); ax.set_xlabel(r'$d_{\rm inj}$')
    axes[0].axhline(0.0, color='k', lw=0.5)
    axes[0].set_ylabel(r'steady-state $E[H_{cl} - H_{cl}(R_d,0)]$')
    axes[0].set_title('Energy plateau vs damping injection')
    axes[0].legend(loc='best', fontsize=9)
    axes[1].set_ylabel(r'steady-state $E[\|\log(R^\top R_d)\|]$')
    axes[1].set_title('Tracking error vs damping injection')
    axes[1].legend(loc='best', fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {save_path}")


# ─────────────────────────────────────────────────────────────────────
# Variant table (α_D, α_V) per the design doc
# ─────────────────────────────────────────────────────────────────────

VARIANTS = {
    'casimir': (0.0, 0.0),    # Variant 1: pure Casimir
    'fric':    (1.0, 0.0),    # Variant 2: + friction cancellation
    'grav':    (0.0, 1.0),    # Variant 3: + gravity cancellation
    'both':    (1.0, 1.0),    # Variant 4: + both
}

VARIANT_LABELS = {
    'casimir': 'V1 Casimir (α=0,0)',
    'fric':    'V2 +Fric  (α_D=1)',
    'grav':    'V3 +Grav  (α_V=1)',
    'both':    'V4 +Both  (α=1,1)',
}

VARIANT_STYLE = {
    'casimir': dict(color='C0', lw=1.6, ls='-'),
    'fric':    dict(color='C1', lw=1.5, ls='--'),
    'grav':    dict(color='C2', lw=1.5, ls='-.'),
    'both':    dict(color='C3', lw=1.5, ls=':'),
}


def plot_variants_single(outs: dict, save_path: str):
    """outs : {variant_label -> rollout dict}, env-plant only."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for variant_key, out in outs.items():
        st = VARIANT_STYLE.get(variant_key, dict(lw=1.3))
        label = VARIANT_LABELS.get(variant_key, variant_key)
        axes[0, 0].plot(out['t'], out['geo'], label=label, **st)
        axes[0, 1].plot(out['t'], np.linalg.norm(out['omega'], axis=1),
                        label=label, **st)
        axes[1, 0].plot(out['t'][:-1], np.linalg.norm(out['u'], axis=1),
                        label=label, **st)
        axes[1, 1].plot(out['t'], out['H'] - out['H_target'],
                        label=label, **st)
    axes[0, 0].axhline(0.0, color='k', lw=0.5)
    axes[1, 1].axhline(0.0, color='k', lw=0.5)
    axes[0, 0].set_xlabel('t [s]'); axes[0, 0].set_ylabel(r'$\|\log(R^\top R_d)\|$')
    axes[0, 0].set_title('Geodesic distance to target')
    axes[0, 1].set_xlabel('t [s]'); axes[0, 1].set_ylabel(r'$\|\omega\|$ [rad/s]')
    axes[0, 1].set_title('Angular speed')
    axes[1, 0].set_xlabel('t [s]'); axes[1, 0].set_ylabel(r'$\|u\|$ [N·m]')
    axes[1, 0].set_title('Control torque magnitude')
    axes[1, 1].set_xlabel('t [s]'); axes[1, 1].set_ylabel(r'$H_{cl}(t) - H_{cl}(R_d, 0)$')
    axes[1, 1].set_title('Energy excess')
    axes[0, 0].legend(loc='best', fontsize=9, framealpha=0.85)
    fig.suptitle('Controller variants: env (true physics) plant',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {save_path}")


def plot_variants_sweep(variant_rows: dict, save_path: str):
    """variant_rows : {variant_key -> list[row dict]} — sweep results per variant."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for variant_key, rows in variant_rows.items():
        st = VARIANT_STYLE.get(variant_key, dict())
        col = st.get('color', None); ls = st.get('ls', '-')
        label = VARIANT_LABELS.get(variant_key, variant_key)
        d  = np.array([r['d_inj']        for r in rows])
        em = np.array([r['ss_mean_mean'] for r in rows])
        es = np.array([r['ss_mean_std']  for r in rows])
        sm = np.array([r['ss_std_mean']  for r in rows])
        gm = np.array([r['geo_tail_mean']for r in rows])
        axes[0].errorbar(d, em, yerr=es, fmt='o', capsize=3,
                         color=col, linestyle=ls, label=label)
        axes[0].fill_between(d, em - sm, em + sm, alpha=0.10, color=col)
        axes[1].plot(d, gm, 'o', color=col, linestyle=ls, label=label)
    for ax in axes:
        ax.set_xscale('log'); ax.set_xlabel(r'$d_{\rm inj}$')
    axes[0].axhline(0.0, color='k', lw=0.5)
    axes[0].set_ylabel(r'steady-state $E[H_{cl} - H_{cl}(R_d,0)]$')
    axes[0].set_title('Energy plateau vs damping injection')
    axes[0].legend(loc='best', fontsize=9)
    axes[1].set_ylabel(r'steady-state $E[\|\log(R^\top R_d)\|]$')
    axes[1].set_title('Tracking error vs damping injection')
    axes[1].legend(loc='best', fontsize=9)
    fig.suptitle('Controller variants on env plant — d_inj ablation',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {save_path}")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['single', 'sweep'], default='single')
    p.add_argument('--k_c', type=float, default=30.0)
    p.add_argument('--d_inj', type=float, default=1.0)
    p.add_argument('--tilt_deg', type=float, default=20.0)
    p.add_argument('--horizon', type=float, default=6.0)
    p.add_argument('--n_seeds', type=int, default=16,
                   help='[sweep] rollouts per d_inj value')
    p.add_argument('--use_trained_g', action='store_true',
                   help='[single] use trained g_theta (and pinv it) instead '
                        'of g = I. The trained g is rank-deficient; expect '
                        'this to perform worse.')
    p.add_argument('--compare_plants', action='store_true',
                   help='also run the controller against the GT-subnet model '
                        'and the GP-subnet trained model (overlaid on env). '
                        'Default ON when --mode single.')
    p.add_argument('--no_compare_plants', action='store_true',
                   help='disable plant comparison; env-only.')
    p.add_argument('--gp_g_keep_trained', action='store_true',
                   help='by default the GP-subnet plant has its trained g_net '
                        'replaced with identity (env-true) so the comparison '
                        'isolates the other learned subnets (M/V/D/σ). Pass '
                        'this flag to keep the rank-1 trained g_θ instead.')
    p.add_argument('--variant',
                   choices=['casimir', 'fric', 'grav', 'both'],
                   default='casimir',
                   help='Which controller variant to use (single-variant runs). '
                        'casimir=baseline, fric=+D_θ cancel, grav=+V_θ cancel, '
                        'both=+D_θ+V_θ. Mapped to (α_D, α_V) ∈ '
                        '{(0,0),(1,0),(0,1),(1,1)}.')
    p.add_argument('--alpha_D', type=float, default=None,
                   help='Override α_D (friction-cancel weight). '
                        'When set, supersedes --variant.')
    p.add_argument('--alpha_V', type=float, default=None,
                   help='Override α_V (gravity-cancel weight). '
                        'When set, supersedes --variant.')
    p.add_argument('--compare_variants', action='store_true',
                   help='Run all 4 variants on the env plant and overlay.')
    p.add_argument('--out_dir', type=str,
                   default=os.path.join(THIS_FILE_DIR, 'data',
                                        'controller_eval'))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    R_d = np.eye(3)
    R0 = R_d @ _exp_so3(np.array([np.deg2rad(args.tilt_deg), 0.0, 0.0]))
    omega0 = np.zeros(3)

    # Resolve variant → (α_D, α_V), with explicit --alpha_D/--alpha_V overrides.
    alpha_D, alpha_V = VARIANTS[args.variant]
    if args.alpha_D is not None:
        alpha_D = float(args.alpha_D)
    if args.alpha_V is not None:
        alpha_V = float(args.alpha_V)
    print(f"variant={args.variant!r}  α_D={alpha_D}  α_V={alpha_V}")

    # ── Variant-comparison short-circuit (env plant only) ────────────────
    if args.compare_variants:
        # Need GP model only as a *V_θ/D_θ provider* for the cancellation
        # terms — it does NOT enter the plant (env is truth here).
        _import_jax_stack()
        print("[loading trained model for V_θ/D_θ cancellation terms]")
        gp_model = _load_trained_model()

        def make_ctrl(alpha_D_v, alpha_V_v):
            cfg = ControllerConfig(
                R_d=R_d, k_c=args.k_c, d_inj=args.d_inj,
                alpha_D=alpha_D_v, alpha_V=alpha_V_v, use_trained_g=False)
            need_model = (alpha_D_v > 0.0) or (alpha_V_v > 0.0)
            return EnergyCasimirController(
                cfg, model=gp_model if need_model else None)

        if args.mode == 'single':
            print(f"variant comparison (single rollout):  k_c={args.k_c} "
                  f"d_inj={args.d_inj}  tilt={args.tilt_deg}°  "
                  f"horizon={args.horizon}s")
            outs = {}
            for vk, (aD, aV) in VARIANTS.items():
                ctrl = make_ctrl(aD, aV)
                out = rollout_env(ctrl, R0, omega0,
                                  horizon=args.horizon, env_seed=0)
                outs[vk] = out
                print(f"  [{VARIANT_LABELS[vk]:>22s}] "
                      f"geo: {out['geo'][0]:.4f} → {out['geo'][-1]:.4f}   "
                      f"H_excess(end)={out['H'][-1] - out['H_target']:+.4f}   "
                      f"|u|_max={float(np.max(np.linalg.norm(out['u'],axis=1))):.2f}")
            plot_variants_single(
                outs,
                os.path.join(args.out_dir, 'single_variants.png'))
        else:
            d_inj_grid = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
            print(f"variant sweep over {d_inj_grid} "
                  f"(n_seeds={args.n_seeds}, horizon={args.horizon}s)")
            variant_rows = {}
            for vk, (aD, aV) in VARIANTS.items():
                print(f"  → {VARIANT_LABELS[vk]}")
                # Build per-d_inj rollout closures that recreate the
                # controller (so each row uses the right d_inj).
                def _rollout_fn(ctrl, R0_, w0_, H, s):
                    return rollout_env(ctrl, R0_, w0_, horizon=H, env_seed=s)

                # Replicate sweep_d_inj-style aggregation but with α-aware ctrl
                rows = []
                for d_inj_v in d_inj_grid:
                    cfg_v = ControllerConfig(
                        R_d=R_d, k_c=args.k_c, d_inj=float(d_inj_v),
                        alpha_D=aD, alpha_V=aV, use_trained_g=False)
                    need_model = (aD > 0.0) or (aV > 0.0)
                    ctrl_v = EnergyCasimirController(
                        cfg_v,
                        model=gp_model if need_model else None)

                    ss_means, ss_stds, term_geos = [], [], []
                    for seed in range(args.n_seeds):
                        out = _rollout_fn(ctrl_v, R0, omega0,
                                          args.horizon, seed)
                        sm, ss, gt = _summarise_rollout(out)
                        ss_means.append(sm); ss_stds.append(ss); term_geos.append(gt)
                    rows.append(dict(
                        d_inj=float(d_inj_v),
                        ss_mean_mean=float(np.mean(ss_means)),
                        ss_mean_std=float(np.std(ss_means)),
                        ss_std_mean=float(np.mean(ss_stds)),
                        geo_tail_mean=float(np.mean(term_geos)),
                    ))
                    print(f"    d_inj={d_inj_v:6.3f}  "
                          f"E[H_excess]={rows[-1]['ss_mean_mean']:+.4f}  "
                          f"±{rows[-1]['ss_mean_std']:.4f}   "
                          f"E[geo]={rows[-1]['geo_tail_mean']:.4f}")
                variant_rows[vk] = rows

            plot_variants_sweep(
                variant_rows,
                os.path.join(args.out_dir, 'd_inj_sweep_variants.png'))
        return

    # Default: compare in single mode unless explicitly disabled. In sweep,
    # only when explicitly requested (3× the runtime).
    do_compare = ((args.mode == 'single' and not args.no_compare_plants)
                  or args.compare_plants)

    # Load auxiliary plants once if needed.
    gp_model = None
    gt_model = None
    if do_compare:
        _import_jax_stack()
        print("[loading plants]")
        gp_model = _load_trained_model()
        if not args.gp_g_keep_trained:
            gp_model = _override_gp_g_with_identity(gp_model)
            print("  GP plant: trained g_net OVERRIDDEN with identity "
                  "(M/V/D/σ still trained)")
        else:
            print("  GP plant: keeping trained g_net (rank-1 collapse)")
        gt_model = _GroundTruthHamiltonianSDE(
            m=ENV_DEFAULTS['m'], l=ENV_DEFAULTS['l'], g=ENV_DEFAULTS['g'],
            friction_coeff=ENV_DEFAULTS['friction_coeff'],
            wind_force_std=ENV_DEFAULTS['wind_force_std'],
        )
        print("  loaded GP-trained model + GT-subnet analytical model")

    if args.mode == 'single':
        # Optionally load trained model for use_trained_g (controller-side).
        model_for_ctrl = None
        if args.use_trained_g:
            model_for_ctrl = (gp_model if gp_model is not None
                              else (lambda: __import__('verify_control',
                                                       fromlist=['load_trained_model']).load_trained_model())())

        cfg = ControllerConfig(R_d=R_d, k_c=args.k_c, d_inj=args.d_inj,
                               alpha_D=alpha_D, alpha_V=alpha_V,
                               use_trained_g=args.use_trained_g)
        # Cancellation terms also need the GP model.
        if (alpha_D > 0.0 or alpha_V > 0.0) and model_for_ctrl is None:
            model_for_ctrl = (gp_model if gp_model is not None
                              else _load_trained_model())
        ctrl = EnergyCasimirController(cfg, model=model_for_ctrl)

        print(f"single rollout: k_c={args.k_c} d_inj={args.d_inj} "
              f"tilt={args.tilt_deg}° horizon={args.horizon}s "
              f"use_trained_g={args.use_trained_g} compare={do_compare}")

        outs = {}
        outs['env (true physics)'] = rollout_env(
            ctrl, R0, omega0, horizon=args.horizon, env_seed=0)
        if do_compare:
            outs['GT-subnet model'] = rollout_model_closed_loop(
                gt_model, ctrl, R0, omega0,
                horizon=args.horizon, dt=ENV_DEFAULTS['dt'],
                n_substeps=10, dW_seed=0)
            outs['GP-subnet model'] = rollout_model_closed_loop(
                gp_model, ctrl, R0, omega0,
                horizon=args.horizon, dt=ENV_DEFAULTS['dt'],
                n_substeps=10, dW_seed=0)

        for label, out in outs.items():
            print(f"  [{label:>22s}] geo: {out['geo'][0]:.4f} → "
                  f"{out['geo'][-1]:.4f}   "
                  f"H_excess(end)={out['H'][-1] - out['H_target']:+.4f}")

        tag = 'trainedG' if args.use_trained_g else 'gI'
        suffix = '_compare' if do_compare else ''
        plot_single(outs, os.path.join(args.out_dir,
                                       f'single_{tag}{suffix}.png'))

    else:
        d_inj_grid = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        print(f"d_inj sweep over {d_inj_grid} "
              f"(n_seeds={args.n_seeds}, horizon={args.horizon}s, "
              f"compare={do_compare})")

        # Build rollout closures bound to each plant.
        env_rollout_fn = lambda c, R0_, w0_, H, s: rollout_env(  # noqa: E731
            c, R0_, w0_, horizon=H, env_seed=s)
        plant_rows = {}

        # If α_D or α_V > 0, the controller needs a model for V_θ/D_θ.
        ctrl_model = None
        if alpha_D > 0.0 or alpha_V > 0.0:
            ctrl_model = gp_model if gp_model is not None else _load_trained_model()

        print("  → env (true physics)")
        plant_rows['env (true physics)'] = sweep_d_inj(
            d_inj_grid, args.k_c, R_d,
            rollout_fn=env_rollout_fn,
            n_seeds=args.n_seeds, horizon=args.horizon,
            tilt_deg=args.tilt_deg,
            alpha_D=alpha_D, alpha_V=alpha_V, model=ctrl_model)

        if do_compare:
            gt_rollout_fn = lambda c, R0_, w0_, H, s: rollout_model_closed_loop(  # noqa: E731
                gt_model, c, R0_, w0_, horizon=H,
                dt=ENV_DEFAULTS['dt'], n_substeps=10, dW_seed=s)
            gp_rollout_fn = lambda c, R0_, w0_, H, s: rollout_model_closed_loop(  # noqa: E731
                gp_model, c, R0_, w0_, horizon=H,
                dt=ENV_DEFAULTS['dt'], n_substeps=10, dW_seed=s)

            print("  → GT-subnet model")
            plant_rows['GT-subnet model'] = sweep_d_inj(
                d_inj_grid, args.k_c, R_d,
                rollout_fn=gt_rollout_fn,
                n_seeds=args.n_seeds, horizon=args.horizon,
                tilt_deg=args.tilt_deg,
                alpha_D=alpha_D, alpha_V=alpha_V, model=ctrl_model)

            print("  → GP-subnet model")
            plant_rows['GP-subnet model'] = sweep_d_inj(
                d_inj_grid, args.k_c, R_d,
                rollout_fn=gp_rollout_fn,
                n_seeds=args.n_seeds, horizon=args.horizon,
                tilt_deg=args.tilt_deg,
                alpha_D=alpha_D, alpha_V=alpha_V, model=ctrl_model)

        suffix = '_compare' if do_compare else ''
        plot_sweep(plant_rows,
                   os.path.join(args.out_dir, f'd_inj_sweep{suffix}.png'))


if __name__ == "__main__":
    main()
