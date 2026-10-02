"""k_c sweep for the gravity-cancel variant (V3) vs Casimir baseline (V1).

The whole point of gravity cancellation is that the equilibrium analysis
no longer requires k_c > m·g·ℓ ≈ 9.81 (the spring is the sole potential
after cancellation). We sweep k_c ∈ {1, 3, 5, 10, 30} at fixed d_inj
(default 2.0 — sweet spot from the d_inj sweep) and check:

  * does V3 still stabilise at small k_c (where V1 cannot)?
  * how does swing-up time and steady-state error trade off?
  * what's the |u| reduction at small k_c?

Usage:
    python ph_gp_sde/kc_sweep_v3.py
    python ph_gp_sde/kc_sweep_v3.py --d_inj 4 --tilt_deg 30 --horizon 15
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

from envs.windy_pendulum_3d import _exp_so3                              # noqa: E402
from controller import EnergyCasimirController, ControllerConfig         # noqa: E402
from evaluate_controller import (                                        # noqa: E402
    rollout_env, ENV_DEFAULTS, VARIANTS, VARIANT_LABELS, VARIANT_STYLE,
)
from verify_control import load_trained_model                            # noqa: E402


def _settling_time(out, geo_tol=0.1):
    """Time to first reach geo < tol and stay there for the rest of the run.

    Returns +inf if it never settles."""
    geo = out['geo']
    t   = out['t']
    # Find latest k where geo > tol; settled iff that k is before the end.
    above = np.where(geo > geo_tol)[0]
    if len(above) == 0:
        return float(t[0])              # was already inside the band
    last_violation = above[-1]
    if last_violation >= len(t) - 1:
        return float('inf')             # still violating at terminal step
    return float(t[last_violation + 1])


def run_sweep(k_c_grid, args, model):
    R_d = np.eye(3)
    R0 = R_d @ _exp_so3(np.array([np.deg2rad(args.tilt_deg), 0.0, 0.0]))
    omega0 = np.zeros(3)

    rows = {'casimir': [], 'grav': []}
    for variant_key in ('casimir', 'grav'):
        aD, aV = VARIANTS[variant_key]
        print(f"\n  → {VARIANT_LABELS[variant_key]}")
        for k_c in k_c_grid:
            cfg = ControllerConfig(
                R_d=R_d, k_c=float(k_c), d_inj=args.d_inj,
                alpha_D=aD, alpha_V=aV, use_trained_g=False)
            ctrl_model = model if (aD > 0.0 or aV > 0.0) else None
            ctrl = EnergyCasimirController(cfg, model=ctrl_model)

            settles, ss_geos, ss_Hs, u_maxes = [], [], [], []
            for seed in range(args.n_seeds):
                out = rollout_env(ctrl, R0, omega0,
                                  horizon=args.horizon, env_seed=seed)
                settles.append(_settling_time(out, geo_tol=args.geo_tol))
                tail = slice(int(0.75 * (out['T'] + 1)), None)
                ss_geos.append(float(np.mean(out['geo'][tail])))
                ss_Hs.append(float(np.mean(out['H'][tail] - out['H_target'])))
                u_maxes.append(float(np.max(np.linalg.norm(out['u'], axis=1))))

            settle_finite = [s for s in settles if np.isfinite(s)]
            row = dict(
                k_c=float(k_c),
                settle_mean=(np.mean(settle_finite)
                             if settle_finite else float('inf')),
                settle_frac_settled=len(settle_finite) / args.n_seeds,
                ss_geo=float(np.mean(ss_geos)),
                ss_geo_std=float(np.std(ss_geos)),
                ss_H=float(np.mean(ss_Hs)),
                u_max_mean=float(np.mean(u_maxes)),
            )
            rows[variant_key].append(row)
            settle_print = (f"{row['settle_mean']:.2f}s"
                            if np.isfinite(row['settle_mean']) else "  ∞")
            print(f"    k_c={k_c:>5.1f}  "
                  f"settle={settle_print:>7s} "
                  f"({int(row['settle_frac_settled']*100)}%)  "
                  f"ss_geo={row['ss_geo']:.4f} ± {row['ss_geo_std']:.4f}  "
                  f"|u|_max={row['u_max_mean']:6.2f}")
    return rows


def plot(rows, save_path, args):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for variant_key, rs in rows.items():
        st = VARIANT_STYLE[variant_key]
        label = VARIANT_LABELS[variant_key]
        kc = np.array([r['k_c'] for r in rs])
        st_mean = np.array([r['settle_mean'] for r in rs])
        ss_geo  = np.array([r['ss_geo'] for r in rs])
        ss_es   = np.array([r['ss_geo_std'] for r in rs])
        u_max   = np.array([r['u_max_mean'] for r in rs])

        # Replace inf settle with horizon for plotting (mark with an x)
        st_plot = np.where(np.isfinite(st_mean), st_mean, args.horizon)
        diverged = ~np.isfinite(st_mean)

        axes[0].plot(kc, st_plot, 'o-', color=st['color'], ls=st['ls'],
                     label=label)
        if diverged.any():
            axes[0].scatter(kc[diverged], st_plot[diverged],
                            marker='x', color=st['color'], s=80, zorder=5)

        axes[1].errorbar(kc, ss_geo, yerr=ss_es, fmt='o-',
                         color=st['color'], ls=st['ls'], capsize=3, label=label)
        axes[2].plot(kc, u_max, 'o-', color=st['color'], ls=st['ls'],
                     label=label)

    # Vertical line at mgl threshold (V1 needs k_c > this)
    mgl = ENV_DEFAULTS['m'] * ENV_DEFAULTS['g'] * ENV_DEFAULTS['l']
    for ax in axes:
        ax.axvline(mgl, color='k', lw=0.7, ls=':',
                   label=r'$mg\ell$' if ax is axes[0] else None)
        ax.set_xscale('log'); ax.set_xlabel(r'$k_c$')

    axes[0].set_ylabel(f'settling time [s]  (geo < {args.geo_tol})')
    axes[0].set_title('Swing-up time vs $k_c$')
    axes[0].legend(loc='best', fontsize=9)
    axes[1].set_ylabel(r'steady-state $\|\log(R^\top R_d)\|$')
    axes[1].set_title('Tracking error vs $k_c$')
    axes[2].set_ylabel(r'$\max\,\|u\|$ along trajectory [N·m]')
    axes[2].set_title('Peak control torque vs $k_c$')

    fig.suptitle(
        f"$k_c$ sweep at $d_{{\\rm inj}}={args.d_inj}$ — "
        f"V1 Casimir vs V3 +Grav (env plant)",
        fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  saved {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--d_inj', type=float, default=2.0,
                   help='damping injection (sweet spot from d_inj sweep)')
    p.add_argument('--tilt_deg', type=float, default=20.0)
    p.add_argument('--horizon', type=float, default=15.0)
    p.add_argument('--n_seeds', type=int, default=8)
    p.add_argument('--geo_tol', type=float, default=0.1,
                   help='settling band: geo < this is "settled"')
    p.add_argument('--out', type=str,
                   default=os.path.join(THIS_FILE_DIR, 'data',
                                        'controller_eval', 'kc_sweep_v3.png'))
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("[loading model for V_θ cancellation in V3]")
    model = load_trained_model()

    k_c_grid = [1.0, 3.0, 5.0, 10.0, 30.0]
    print(f"k_c sweep over {k_c_grid}  d_inj={args.d_inj}  "
          f"tilt={args.tilt_deg}°  horizon={args.horizon}s  "
          f"n_seeds={args.n_seeds}")
    rows = run_sweep(k_c_grid, args, model)
    plot(rows, args.out, args)


if __name__ == "__main__":
    main()
