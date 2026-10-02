"""ELBO trainer for the variational-GP port-Hamiltonian SDE.

Differences from the previous (MSE-based) ph_gp_sde trainer:

  - Rollout switched from diffrax (deterministic ODE) to
    `lie_heun_sde_rollout` (Stratonovich Heun on SO(3) × ℝ³), so the SDE
    structure is actually exercised — σ_net receives gradients via the
    Brownian-noise branch.  Mirrors `ph_nn_sde_debug/train.py`.

  - Loss replaced with the negative ELBO:

        −ELBO  =  L_NLL  +  (β / N) · L_KL

    L_NLL is the geodesic-Gaussian + 3-D Gaussian likelihood from
    `src/utils/JAX/elbo_loss_jax.py`, parameterised by trainable
    σ_R, σ_ω stored as fields on the model.
    L_KL is the closed-form mean-field KL summed over the five GP subnets
    (M_net, V_net, Dw_net, g_net, sigma_net).
    β is annealed linearly from 0 to `--beta_max` over `--kl_anneal_steps`.

  - GP weights are sampled once per (b, s) trajectory via reparameterisation
    (one `key` per subnet, threaded through the integrator's predictor +
    corrector + every substep so a single coherent w sample drives the whole
    rollout).

  - Per-eval printing now includes σ_R, σ_ω, NLL_R, NLL_ω, MSE_R (= θ²),
    MSE_ω (= ‖Δω‖²), per-subnet KLs, β, and the legacy subnet-physics MSE.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, '../../../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if THIS_FILE_DIR not in sys.path:
    sys.path.insert(0, THIS_FILE_DIR)

DATASETS_DIR = os.path.join(PROJECT_ROOT, 'datasets')
if DATASETS_DIR not in sys.path:
    sys.path.insert(0, DATASETS_DIR)

from windy_pendulum_3d_datagen import get_dataset, arrange_data    # noqa: E402

from network import DissipativeSO3HamSDE                           # noqa: E402
from src.utils.JAX.lie_integrator import (                         # noqa: E402
    lie_heun_sde_rollout, lie_heun_sde_step,
)
from src.utils.JAX.loss_utils_jax import (                         # noqa: E402
    traj_rotmat_L2_geodesic_loss_safe,
)
from src.utils.JAX.elbo_loss_jax import (                         # noqa: E402
    elbo_nll, kl_per_subnet, pl_loss,
)
from src.utils.JAX.subnet_diagnostics_jax import subnet_physics_mse  # noqa: E402
from src.utils.JAX.ode_utils_jax import to_pickle                  # noqa: E402


DEFAULT_SAVE_DIR = os.path.join(THIS_FILE_DIR, 'data', 'run_wp3d_jax')
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, 'datasets/data/windy_pendulum_3d')


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description=None)
    p.add_argument('--learn_rate', default=1e-3, type=float)
    p.add_argument('--total_steps', default=10000, type=int)
    p.add_argument('--eval_every', default=50, type=int)
    p.add_argument('--name', default='wp3d', type=str)
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--seed', default=0, type=int)
    p.add_argument('--save_dir', default=DEFAULT_SAVE_DIR, type=str)
    p.add_argument('--data_dir', default=DEFAULT_DATA_DIR, type=str)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--num_points', type=int, default=5)
    p.add_argument('--init_gain', default=0.5, type=float)

    p.add_argument('--samples', type=int, default=64)
    p.add_argument('--timesteps', type=int, default=20)
    p.add_argument('--friction_coeff', type=float, default=0.5)
    p.add_argument('--varying_friction', action='store_true')
    p.add_argument('--external_force_type', type=str, default='sine',
                   choices=['sine', 'square', 'random', 'constant'])
    p.add_argument('--external_force_std', type=float, default=0.0)
    p.add_argument('--wind_force_std', type=float, default=0.0)
    p.add_argument('--obs_noise_std', type=float, default=0.0)
    p.add_argument('--random_u', action='store_true')

    # M_net pretraining
    p.add_argument('--pretrain_M_steps', type=int, default=200)
    p.add_argument('--pretrain_M_lr', type=float, default=1e-3)
    p.add_argument('--pretrain_M_print_every', type=int, default=20)

    # SDE-specific
    p.add_argument('--n_substeps', type=int, default=10,
                   help='Lie-Heun substeps per dt; match env (10) for clean comparison')
    p.add_argument('--grad_clip', type=float, default=1.0,
                   help='global-norm gradient clip')

    # ELBO-specific
    p.add_argument('--init_sigma_R', type=float, default=0.1,
                   help='initial observation-noise scale on rotation (geodesic NLL)')
    p.add_argument('--init_sigma_omega', type=float, default=0.1,
                   help='initial observation-noise scale on angular velocity')
    p.add_argument('--mc_samples', type=int, default=1,
                   help='S in the MC ELBO estimate (GP weights × Brownian path)')
    p.add_argument('--beta_max', type=float, default=1.0,
                   help='final β coefficient on (1/N)·KL')
    p.add_argument('--kl_anneal_steps', type=int, default=1000,
                   help='linear β anneal: β(iter) = beta_max · min(1, iter/this)')

    # Per-increment pseudo-likelihood (gives sigma_net + drift a clean
    # single-step training signal; addresses the σ_net → 0 collapse that
    # the rollout NLL alone produces).
    p.add_argument('--lambda_pl', type=float, default=1.0,
                   help='weight of the per-increment pseudo-likelihood term '
                        '(0 disables; see elbo_loss_jax.pl_loss)')
    p.add_argument('--init_sigma_obs_omega', type=float, default=0.5,
                   help='initial per-snapshot ω observation-noise scale used '
                        'inside Σ_eff = σ_φ²·Δt + 2·σ_obs_ω²; should match the '
                        'dataset\'s --obs_noise_std')
    p.add_argument('--init_sigma_const', type=float, default=0.5,
                   help='DEPRECATED / IGNORED. The static softplus bias on '
                        'sigma_net was removed; σ(q) = softplus(GP_raw(q)) '
                        'now starts at softplus(0) = ln 2 ≈ 0.693 regardless '
                        'of this flag. Kept only for CLI backward compat.')

    # NaN debugging
    p.add_argument('--debug_dump_dir', type=str, default=None,
                   help='where to dump NaN post-mortem pickles (default: save_dir/debug)')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Wrapper that injects per-subnet GP keys into model.drift / .stochastic_increment
# ─────────────────────────────────────────────────────────────────────

class KeyedSDEModel(eqx.Module):
    """Adapter so `lie_heun_sde_rollout` (which only sees `.drift`,
    `.stochastic_increment`) can drive a `DissipativeSO3HamSDE` while
    forwarding a fixed dict of per-subnet PRNGKeys.

    `inference_mode=True` forces every subnet call to the posterior-mean
    path regardless of `keys` — used at eval time so the comparison with
    the deterministic NN_ODE rollout isn't confounded by GP weight-sample
    noise compounding through 4 outer × 10 substeps × 4 subnet calls per
    substep. At train time `inference_mode=False` and the sampled keys
    drive the reparameterised ELBO MC estimate.
    """
    model: DissipativeSO3HamSDE
    keys: dict   # {'M', 'V', 'Dw', 'g', 'sigma'} → PRNGKey
    inference_mode: bool = eqx.field(static=True)

    def _eff_keys(self):
        return (None if self.inference_mode
                else (self.keys if len(self.keys) > 0 else None))

    # ── (q, p) integrator API ────────────────────────────────────────
    def drift_p(self, q, p, u):
        return self.model.drift_p(q, p, u, keys=self._eff_keys())

    def stochastic_increment_p(self, q, dW):
        return self.model.stochastic_increment_p(q, dW, keys=self._eff_keys())

    def M_inv(self, q):
        return self.model.M_inv(q, keys=self._eff_keys())

    # ── ω-form wrappers kept for pl_loss / legacy call sites ─────────
    def drift(self, q, q_dot, u):
        return self.model.drift(q, q_dot, u, keys=self._eff_keys())

    def stochastic_increment(self, q, dW):
        return self.model.stochastic_increment(q, dW, keys=self._eff_keys())


def _split_subnet_keys(key):
    """Split a PRNGKey into a {M, V, Dw, g, sigma} dict of 5 PRNGKeys."""
    kM, kV, kD, kg, ks = jax.random.split(key, 5)
    return {'M': kM, 'V': kV, 'Dw': kD, 'g': kg, 'sigma': ks}


class _DiagShim:
    """Adapter so `subnet_physics_mse` (which expects MLP-style subnets,
    `net(q)` no kwargs) keeps working for the GP-based subnets — bind
    `inference_mode=True` and forward.  Pure Python; not jitted, only used at
    eval cadence.
    """
    def __init__(self, model):
        self.M_net  = lambda q: model.M_net (q, inference_mode=True)
        self.V_net  = lambda q: model.V_net (q, inference_mode=True)
        self.Dw_net = lambda q: model.Dw_net(q, inference_mode=True)
        self.g_net  = lambda q: model.g_net (q, inference_mode=True)


# ─────────────────────────────────────────────────────────────────────
# NaN diagnostics — heavy debug path triggered on first NaN/Inf.
# Ported from ph_nn_sde_debug/train.py and adapted for GP subnets:
#   • Subnet calls thread `inference_mode=True` so the deterministic
#     (posterior-mean) path is exercised first.
#   • The eager substep replay also runs a SECOND pass with the exact
#     per-subnet GP keys from the failing step, so we can tell whether
#     the NaN was driven by a particular weight sample vs. by the
#     posterior mean alone.
#   • Reports σ_R, σ_ω, log_w_covar magnitudes alongside subnet outputs,
#     since those are the parameters whose loosened init plausibly
#     introduces tail samples that drive numerical instability.
# ─────────────────────────────────────────────────────────────────────

def _has_nan_or_inf(tree) -> bool:
    for x in jax.tree.leaves(eqx.filter(tree, eqx.is_array)):
        a = np.asarray(x)
        if not np.all(np.isfinite(a)):
            return True
    return False


def _per_subnet_nan_report(model_or_grads) -> dict:
    """Per-subnet param-array stats. NaN/Inf counts and value range."""
    out = {}
    for name in ('M_net', 'V_net', 'Dw_net', 'g_net', 'sigma_net'):
        sub = getattr(model_or_grads, name, None)
        if sub is None:
            continue
        leaves = jax.tree.leaves(eqx.filter(sub, eqx.is_array))
        n_params = sum(int(np.asarray(l).size) for l in leaves)
        n_nan = sum(int(np.isnan(np.asarray(l)).sum()) for l in leaves)
        n_inf = sum(int(np.isinf(np.asarray(l)).sum()) for l in leaves)
        finite = [
            np.asarray(l).reshape(-1)[np.isfinite(np.asarray(l).reshape(-1))]
            for l in leaves
        ]
        finite_vals = np.concatenate(finite) if finite else np.array([0.0])
        if finite_vals.size == 0:
            v_min, v_max = float('nan'), float('nan')
        else:
            v_min = float(finite_vals.min()); v_max = float(finite_vals.max())
        out[name] = dict(n_params=n_params, n_nan=n_nan, n_inf=n_inf,
                         min=v_min, max=v_max)
    return out


def _gp_log_w_covar_report(model) -> dict:
    """Per-subnet log_w_covar (variational log-std) statistics — useful for
    confirming whether tail weight samples drove the failure."""
    out = {}
    for name in ('M_net', 'V_net', 'Dw_net', 'g_net', 'sigma_net'):
        sub = getattr(model, name, None)
        if sub is None:
            continue   # sigma_net replaced by a constant — no GP weights
        # Both GP_Model and (PSD_GP_Model | GP_MatrixNet) expose log_w_covar
        # through .log_w_covar or .gp_model.log_w_covar
        if hasattr(sub, 'log_w_covar'):
            lwc = np.asarray(sub.log_w_covar)
        elif hasattr(sub, 'gp_model') and hasattr(sub.gp_model, 'log_w_covar'):
            lwc = np.asarray(sub.gp_model.log_w_covar)
        else:
            continue
        sigma_w = np.exp(lwc)
        out[name] = dict(
            log_w_min=float(lwc.min()), log_w_max=float(lwc.max()),
            log_w_mean=float(lwc.mean()),
            sigma_w_min=float(sigma_w.min()), sigma_w_max=float(sigma_w.max()),
        )
    return out


def _subnet_outputs_at_q(model, q, keys=None) -> dict:
    """Call each subnet on a single q sample; report NaN/Inf in outputs.

    With `keys=None`: deterministic posterior-mean path.
    With `keys`: same per-subnet GP keys used in the failing rollout, so we
    can see whether the tail weight sample produced an extreme output.
    """
    out = {}
    def _call(net, q_, k):
        return net(q_, inference_mode=True) if k is None else net(q_, key=k, inference_mode=False)

    pairs = (
        ('M_net',     model.M_net,     None if keys is None else keys.get('M')),
        ('V_net',     model.V_net,     None if keys is None else keys.get('V')),
        ('Dw_net',    model.Dw_net,    None if keys is None else keys.get('Dw')),
        ('g_net',     model.g_net,     None if keys is None else keys.get('g')),
    )
    for name, net, k in pairs:
        try:
            y = np.asarray(_call(net, q, k))
            finite = y[np.isfinite(y)]
            out[name] = dict(
                shape=y.shape,
                has_nan=bool(np.isnan(y).any()),
                has_inf=bool(np.isinf(y).any()),
                min=float(finite.min()) if finite.size else float('nan'),
                max=float(finite.max()) if finite.size else float('nan'),
            )
        except Exception as e:
            out[name] = dict(error=str(e))
    # sigma uses softplus(sigma_net) — call via the model wrapper
    try:
        ksig = None if keys is None else keys.get('sigma')
        y = np.asarray(model.sigma(q, key=ksig))
        out['sigma'] = dict(value=float(y),
                            has_nan=bool(np.isnan(y).any()),
                            has_inf=bool(np.isinf(y).any()))
    except Exception as e:
        out['sigma'] = dict(error=str(e))
    return out


def _substep_replay(model, x0_12, u, h, dW_seq_one, gp_keys_one) -> list:
    """Re-run rollout for ONE trajectory in pure Python (no jit, no scan).

    Uses the exact per-subnet GP keys that were used by the failing step,
    so this reproduces the same w sample. Stops at the first non-finite.

    Args:
        model       : DissipativeSO3HamSDE
        x0_12       : (12,) np or jnp array
        u           : (3,)  np or jnp array
        h           : substep size
        dW_seq_one  : (n_outer, n_substeps, 3) np or jnp array
        gp_keys_one : dict {M, V, Dw, g, sigma} → PRNGKey (2,) — same keys
                      used by the failing step (None → deterministic replay).
    """
    history = []
    x = np.asarray(x0_12)
    n_outer, n_sub, _ = np.asarray(dW_seq_one).shape

    keys = gp_keys_one
    for i in range(n_outer):
        for j in range(n_sub):
            q  = jnp.asarray(x[:9])
            om = jnp.asarray(x[9:12])
            R  = q.reshape(3, 3)
            try:
                drift_v = np.asarray(model.drift(
                    q, om, jnp.asarray(u), keys=keys))
                stoch_v = np.asarray(model.stochastic_increment(
                    q, jnp.asarray(np.asarray(dW_seq_one)[i, j]), keys=keys))
                ksig = None if keys is None else keys.get('sigma')
                sig_v = float(np.asarray(model.sigma(q, key=ksig)))
            except Exception as e:
                history.append(dict(outer=i, substep=j, error=str(e)))
                return history

            stats = dict(
                outer=i, substep=j,
                drift_norm=float(np.linalg.norm(drift_v)),
                stoch_norm=float(np.linalg.norm(stoch_v)),
                sigma=sig_v,
                omega_norm=float(np.linalg.norm(np.asarray(om))),
                det_R=float(np.linalg.det(np.asarray(R))),
                orth_err=float(np.linalg.norm(
                    np.asarray(R).T @ np.asarray(R) - np.eye(3))),
                drift_has_nan=bool(np.isnan(drift_v).any() or np.isinf(drift_v).any()),
                stoch_has_nan=bool(np.isnan(stoch_v).any() or np.isinf(stoch_v).any()),
            )
            history.append(stats)
            if stats['drift_has_nan'] or stats['stoch_has_nan']:
                return history

            # Apply one substep through the same KeyedSDEModel adapter the
            # rollout uses, so we exactly mirror the failing path. The step
            # now operates on (q, p), so convert ω → p on entry and p → ω
            # on exit to keep `x` (the loop variable) in (q, ω) form.
            keyed = KeyedSDEModel(
                model=model, keys=(keys if keys is not None else {}),
                inference_mode=False,
            )
            q_jax = jnp.asarray(x[:9])
            om_jax = jnp.asarray(x[9:12])
            p_jax  = jnp.linalg.solve(keyed.M_inv(q_jax), om_jax)
            x_qp   = jnp.concatenate([q_jax, p_jax])
            x_qp_new = lie_heun_sde_step(
                keyed, x_qp, jnp.asarray(u), h,
                jnp.asarray(np.asarray(dW_seq_one)[i, j]),
            )
            q_new  = x_qp_new[:9]
            p_new  = x_qp_new[9:12]
            om_new = keyed.M_inv(q_new) @ p_new
            x = np.asarray(jnp.concatenate([q_new, om_new]))
            if not np.all(np.isfinite(x)):
                history[-1]['post_step_has_nan'] = True
                return history
    return history


def diagnose_and_dump(*, model_pre, batch_x_cat, dW_batch, gp_keys_batch,
                      grads, opt_state, step_idx, loss_val, aux_val,
                      h, dump_dir, args) -> str:
    """Heavy debug path: collate diagnostics, print, dump pickle.

    Replays the failing step in eager mode using the exact dW + GP keys that
    triggered the NaN, and prints per-substep stats for the first batch
    element under both deterministic (posterior-mean) and stochastic
    (sampled-w) paths.
    """
    os.makedirs(dump_dir, exist_ok=True)
    print("\n" + "!" * 72)
    print(f"!! NaN/Inf DETECTED at step {step_idx}")
    if isinstance(loss_val, float) and np.isfinite(loss_val):
        print(f"!!   loss = {loss_val:.4e}")
    else:
        print(f"!!   loss = {loss_val}")
    print("!" * 72)

    # 1. Per-subnet param report
    print("\n[1] Per-subnet param stats (model JUST BEFORE the failing step):")
    subnet_report = _per_subnet_nan_report(model_pre)
    for name, st in subnet_report.items():
        flag = "  ⚠" if (st['n_nan'] > 0 or st['n_inf'] > 0) else "  ok"
        print(f"   {name:>10s}: n_params={st['n_params']:>5d}  "
              f"n_nan={st['n_nan']:>4d}  n_inf={st['n_inf']:>4d}  "
              f"min={st['min']:.3e}  max={st['max']:.3e}{flag}")

    # 1b. log_sigma_R / log_sigma_omega / σ_obs_ω
    print(f"\n[1b] Obs-noise scales: σ_R = {float(jnp.exp(model_pre.log_sigma_R)):.3e}  "
          f"σ_ω = {float(jnp.exp(model_pre.log_sigma_omega)):.3e}  "
          f"σ_obs_ω = {float(model_pre.sigma_obs_omega):.3e} (frozen)")

    # 1c. Per-subnet log_w_covar magnitudes
    print("\n[1c] Per-subnet variational log_w_covar magnitudes:")
    lw_report = _gp_log_w_covar_report(model_pre)
    for name, st in lw_report.items():
        print(f"   {name:>10s}: log_w in [{st['log_w_min']:.3e}, "
              f"{st['log_w_max']:.3e}]  σ_w in [{st['sigma_w_min']:.3e}, "
              f"{st['sigma_w_max']:.3e}]")

    # 2. Subnet outputs at the first batch element's q at t=0,
    #    both deterministic and with the same GP keys as the failing step.
    q0 = batch_x_cat[0, 0, :9]
    print(f"\n[2] Subnet outputs at q = batch[0, 0, :9] (deterministic, posterior mean):")
    for name, st in _subnet_outputs_at_q(model_pre, q0, keys=None).items():
        print(f"   {name:>10s}: {st}")

    # gp_keys_batch is a dict of (B, S, 2) arrays — pull (b=0, s=0).
    keys_b0s0 = {k: v[0, 0] for k, v in gp_keys_batch.items()}
    print(f"\n[2b] Subnet outputs at q = batch[0, 0, :9] (sampled w, b=0 s=0 keys):")
    for name, st in _subnet_outputs_at_q(model_pre, q0, keys=keys_b0s0).items():
        print(f"   {name:>10s}: {st}")

    # 3. Gradient-NaN per subnet
    print("\n[3] Gradient stats per subnet:")
    grad_report = _per_subnet_nan_report(grads)
    for name, st in grad_report.items():
        flag = "  ⚠" if (st['n_nan'] > 0 or st['n_inf'] > 0) else "  ok"
        print(f"   {name:>10s}: n_grads={st['n_params']:>5d}  "
              f"n_nan={st['n_nan']:>4d}  n_inf={st['n_inf']:>4d}  "
              f"min={st['min']:.3e}  max={st['max']:.3e}{flag}")

    # 4. Eager substep replay — first deterministic, then with sampled w.
    print("\n[4a] Eager substep replay for batch[0] (deterministic, posterior mean):")
    x0_12_np = np.asarray(batch_x_cat[0, 0, :12])
    u_np     = np.asarray(batch_x_cat[0, 0, 12:15])
    dW_b0    = np.asarray(dW_batch[0, 0])             # (n_outer, n_sub, 3) for s=0
    history_det = _substep_replay(model_pre, x0_12_np, u_np, h, dW_b0,
                                  gp_keys_one=None)
    for hh in history_det[:5]:
        print(f"   {hh}")
    if len(history_det) > 5:
        print(f"   ... ({len(history_det) - 5} more substeps; full trace in dump)")
    last = history_det[-1] if history_det else {}
    if last.get('drift_has_nan'):
        print(f"   FIRST FAILURE (det): drift produced NaN/Inf at "
              f"outer={last['outer']} substep={last['substep']}")
    elif last.get('stoch_has_nan'):
        print(f"   FIRST FAILURE (det): stochastic_increment produced NaN/Inf at "
              f"outer={last['outer']} substep={last['substep']}")
    elif last.get('post_step_has_nan'):
        print(f"   FIRST FAILURE (det): integrator output non-finite after "
              f"outer={last['outer']} substep={last['substep']}")

    print("\n[4b] Eager substep replay for batch[0] (with failing-step GP keys):")
    history_stoch = _substep_replay(model_pre, x0_12_np, u_np, h, dW_b0,
                                    gp_keys_one=keys_b0s0)
    for hh in history_stoch[:5]:
        print(f"   {hh}")
    if len(history_stoch) > 5:
        print(f"   ... ({len(history_stoch) - 5} more substeps; full trace in dump)")
    last = history_stoch[-1] if history_stoch else {}
    if last.get('drift_has_nan'):
        print(f"   FIRST FAILURE (stoch): drift produced NaN/Inf at "
              f"outer={last['outer']} substep={last['substep']}")
    elif last.get('stoch_has_nan'):
        print(f"   FIRST FAILURE (stoch): stochastic_increment produced NaN/Inf at "
              f"outer={last['outer']} substep={last['substep']}")
    elif last.get('post_step_has_nan'):
        print(f"   FIRST FAILURE (stoch): integrator output non-finite after "
              f"outer={last['outer']} substep={last['substep']}")

    # 5. Dump.
    base = os.path.join(dump_dir, f"nan_step_{step_idx}")
    eqx.tree_serialise_leaves(base + "_model.eqx", model_pre)
    eqx.tree_serialise_leaves(base + "_grads.eqx", grads)

    aux_finite = {}
    for k, v in (aux_val or {}).items():
        try:
            fv = float(v)
            aux_finite[k] = fv if np.isfinite(fv) else str(v)
        except Exception:
            pass

    payload = dict(
        step_idx=step_idx,
        loss_val=(float(loss_val) if (isinstance(loss_val, float)
                                      and np.isfinite(loss_val)) else str(loss_val)),
        aux=aux_finite,
        h=h,
        args=vars(args),
        subnet_param_report=subnet_report,
        log_w_covar_report=lw_report,
        grad_param_report=grad_report,
        substep_history_det=history_det,
        substep_history_stoch=history_stoch,
        batch_x_cat=np.asarray(batch_x_cat),
        dW_batch=np.asarray(dW_batch),
        gp_keys_batch={k: np.asarray(v) for k, v in gp_keys_batch.items()},
        log_sigma_R=float(model_pre.log_sigma_R),
        log_sigma_omega=float(model_pre.log_sigma_omega),
        sigma_obs_omega=float(model_pre.sigma_obs_omega),
        model_eqx=base + "_model.eqx",
        grads_eqx=base + "_grads.eqx",
    )
    dump_path = base + ".pkl"
    with open(dump_path, 'wb') as f:
        pickle.dump(payload, f)
    print(f"\n   Dump written: {dump_path}")
    print(f"   Model state : {base}_model.eqx")
    print(f"   Grad state  : {base}_grads.eqx")
    return dump_path


# ─────────────────────────────────────────────────────────────────────
# Rollout helpers (ported from ph_nn_sde_debug/train.py)
# ─────────────────────────────────────────────────────────────────────

def _sample_dW(key, batch_size, mc_samples, n_outer, n_substeps, h, dtype):
    """(B, S, n_outer, n_substeps, 3), pre-scaled so var(dW) = h."""
    return jax.random.normal(
        key, (batch_size, mc_samples, n_outer, n_substeps, 3), dtype=dtype
    ) * jnp.sqrt(jnp.asarray(h, dtype=dtype))


def _sample_gp_keys(key, batch_size, mc_samples):
    """(B, S) PRNGKeys per subnet → dict {M, V, Dw, g, sigma}.

    Each (b, s) trajectory gets its own w sample.  For S=1 the trailing
    axis is kept (size 1) for shape uniformity.
    """
    keys_top = _split_subnet_keys(key)                              # 5 base keys
    out = {}
    for name, k in keys_top.items():
        # split into B*S keys, reshape to (B, S, 2)
        sub = jax.random.split(k, batch_size * mc_samples)
        out[name] = sub.reshape(batch_size, mc_samples, 2)
    return out


def _rollout_single(model, x_traj_15, h, dW_one, gp_keys_one, inference_mode=False):
    """One trajectory rollout. Returns (T, 15) — T = n_outer + 1.

    `x_traj_15`       : (T_obs, 15)  full ground-truth (q, ω, u) snapshots.
                        Reads x0 from t=0; reads `u_per_outer = x_traj_15[:-1, 12:15]`
                        (= the u applied during the t→t+1 transition for each
                        outer step). This makes per-step random u in the dataset
                        actually drive the rollout's drift, instead of being
                        silently replaced by u(0) — the bug that prevented
                        `g_θ` from being identifiable on the all-ones-only
                        training grid.
    `dW_one`          : (n_outer, n_substeps, 3)
    `gp_keys_one`     : dict subnet → PRNGKey (2,)  — one w sample for the
                        trajectory (ignored when inference_mode=True)
    `inference_mode`  : if True, all subnets use the posterior mean
                        regardless of gp_keys_one.
    """
    x0_12       = x_traj_15[0, :12]
    u_per_outer = x_traj_15[:-1, 12:15]                            # (n_outer, 3)
    keyed = KeyedSDEModel(model=model, keys=gp_keys_one,
                          inference_mode=inference_mode)
    traj_12 = lie_heun_sde_rollout(keyed, x0_12, u_per_outer, h, dW_one)  # (T, 12)
    T_steps = traj_12.shape[0]
    # Re-attach u so output shape stays (T, 15). Last row's u is unused
    # downstream (no transition after t=T-1) — repeat the last applied u
    # for shape parity with the input trajectory.
    u_full = jnp.concatenate([u_per_outer, u_per_outer[-1:]], axis=0)
    return jnp.concatenate([traj_12, u_full], axis=-1)                    # (T, 15)


def _rollout_batch(model, batch_x_cat, h, dW_batch, gp_keys_batch,
                   inference_mode=False):
    """Vmap over (B, S) → (T, B, S, 15).

    `batch_x_cat`    : (T_obs, B, 15)
    `dW_batch`       : (B, S, n_outer, n_substeps, 3)
    `gp_keys_batch`  : dict name → (B, S, 2) PRNGKeys
    `inference_mode` : if True, posterior-mean path through the integrator.
    """
    # Re-axis to (B, T_obs, 15) so vmap iterates over B.
    x_traj_BT15 = jnp.transpose(batch_x_cat, (1, 0, 2))            # (B, T, 15)

    def per_batch(x_traj_b, dW_b, keys_b):
        def per_sample(dW_s, keys_s):
            return _rollout_single(model, x_traj_b, h, dW_s, keys_s,
                                   inference_mode=inference_mode)
        return jax.vmap(per_sample)(dW_b, keys_b)                  # (S, T, 15)

    traj_BST15 = jax.vmap(per_batch)(x_traj_BT15, dW_batch, gp_keys_batch)  # (B,S,T,15)
    return jnp.transpose(traj_BST15, (2, 0, 1, 3))                 # (T, B, S, 15)


# ─────────────────────────────────────────────────────────────────────
# ELBO loss
# ─────────────────────────────────────────────────────────────────────

def loss_fn(model, batch_x_cat, h, dW_batch, gp_keys_batch, beta, N,
            lambda_pl, dt_outer, inference_mode=False):
    """Negative ELBO + λ_PL · L_PL.

        L_total = L_NLL  +  (β / N) · L_KL  +  λ_PL · L_PL

    where L_PL is the per-increment pseudo-likelihood (Euler-Maruyama
    transition density at observed snapshots; see elbo_loss_jax.pl_loss).
    L_PL gives σ_net a non-collapsing data-fit signal — the rollout NLL
    alone drives σ_φ → 0 because model and env have independent Brownian
    paths, so increasing σ_φ only ever adds variance to the residual.

    Returns (total_loss, aux_dict) for printing/stats.

    `batch_x_cat`    : (T_obs, B, 15)   ground-truth env trajectory
    `dW_batch`       : (B, S, n_outer, n_substeps, 3)
    `gp_keys_batch`  : dict name → (B, S, 2) PRNGKeys (shared with PL term
                       so the variational gradient is coherent across heads)
    `beta`, `N`      : scalars (jnp arrays)
    `lambda_pl`      : scalar (jnp array) — weight on the PL term
    `dt_outer`       : scalar (jnp array) — outer-step Δt between snapshots
                       (= h · n_substeps); used inside PL's Σ_eff.
    """
    traj_15 = _rollout_batch(model, batch_x_cat, h, dW_batch, gp_keys_batch,
                             inference_mode=inference_mode)
    # traj_15 : (T, B, S, 15).  Compare against (T, B, S, 15) target broadcast.
    target = jnp.broadcast_to(
        batch_x_cat[:, :, None, :], traj_15.shape
    )

    # Drop t=0 (initial condition is supplied, not predicted).
    target_hat = traj_15[1:]
    target_obs = target[1:]

    nll = elbo_nll(target_obs, target_hat,
                   model.log_sigma_R, model.log_sigma_omega, split=(9, 3, 3))
    kl  = kl_per_subnet(model)
    pl  = pl_loss(model, batch_x_cat, dt_outer,
                  model.sigma_obs_omega, gp_keys_batch,
                  inference_mode=inference_mode)

    total = (nll['nll_total']
             + (beta / N) * kl['total_kl']
             + lambda_pl * pl['pl_loss'])

    aux = {
        'nll_total':       nll['nll_total'],
        'nll_R':           nll['nll_R'],
        'nll_omega':       nll['nll_omega'],
        'mean_theta_sq':   nll['mean_theta_sq'],
        'mean_omega_sq':   nll['mean_omega_sq'],
        'sigma_R':         nll['sigma_R'],
        'sigma_omega':     nll['sigma_omega'],
        'kl_M':            kl['M_kl'],
        'kl_V':            kl['V_kl'],
        'kl_Dw':           kl['Dw_kl'],
        'kl_g':            kl['g_kl'],
        'kl_sigma':        kl['sigma_kl'],
        'kl_total':        kl['total_kl'],
        'beta':            beta,
        'pl_loss':         pl['pl_loss'],
        'pl_residual_sq':  pl['mean_residual_sq'],
        'mean_sigma_phi':  pl['mean_sigma_phi'],
        'sigma_obs_omega': pl['sigma_obs_omega'],
        'traj_15':         traj_15,        # for downstream subnet-physics MSE
    }
    return total, aux


# ─────────────────────────────────────────────────────────────────────
# M_net pretraining (anchor near M⁻¹ = (1/(m·l²)) · I₃)
# ─────────────────────────────────────────────────────────────────────

def pretrain_M_net(model, q_samples, n_steps, lr, print_every,
                   m: float = 1.0, l: float = 1.0):
    """Anchor M⁻¹ near (1/(m·l²))·I₃ by training only the variational
    parameters (`w_mean`, `log_w_covar`) of M_net's inner GP. The Matérn
    random-Fourier-feature spectrum (`base_rotations_flat`, `omega_angles`,
    etc.) is **frozen** — otherwise pretraining silently corrupts the GP
    kernel approximation by drifting the random frequencies under AdamW.
    """
    if n_steps <= 0:
        return model

    target_scale = 1.0 / (m * l * l)
    print(f"\nPretraining M_net for {n_steps} steps (lr={lr}, "
          f"target = {target_scale:.3f}·I₃)")
    print(f"  using {q_samples.shape[0]} q samples drawn from training data")

    # Filter spec: True only on the variational parameters of the inner GP.
    # PSD_GP_Model owns a `gp_model` field of type GP_Model with `w_mean` and
    # `log_w_covar` as the trainable Bayesian leaves; everything else
    # (Matérn / periodic feature buffers) stays frozen.
    filter_spec = jax.tree.map(lambda _: False, model.M_net)
    filter_spec = eqx.tree_at(
        lambda m_: (m_.gp_model.w_mean, m_.gp_model.log_w_covar),
        filter_spec, replace=(True, True),
    )

    M_params, M_static = eqx.partition(model.M_net, filter_spec)
    optimizer = optax.adamw(learning_rate=lr, weight_decay=1e-4)
    opt_state = optimizer.init(M_params)

    target = target_scale * jnp.eye(3, dtype=q_samples.dtype)

    def pretrain_loss(M_params_inner, q_batch):
        M_net = eqx.combine(M_params_inner, M_static)
        # inference_mode=True — pretraining drives the posterior mean only;
        # the variational variance stays at its (small) init.
        M_preds = jax.vmap(lambda q: M_net(q, inference_mode=True))(q_batch)
        return jnp.mean((M_preds - target) ** 2)

    @eqx.filter_jit
    def step(M_params_inner, opt_state, q_batch):
        loss_val, grads = jax.value_and_grad(pretrain_loss)(M_params_inner, q_batch)
        updates, opt_state = optimizer.update(grads, opt_state, M_params_inner)
        M_params_inner = optax.apply_updates(M_params_inner, updates)
        return M_params_inner, opt_state, loss_val

    initial_loss = None
    for s in range(n_steps):
        M_params, opt_state, loss_val = step(M_params, opt_state, q_samples)
        if initial_loss is None:
            initial_loss = float(loss_val)
        if s % max(1, print_every) == 0 or s == n_steps - 1:
            print(f"  pretrain step {s:>4d}: loss={float(loss_val):.3e}")

    new_M_net = eqx.combine(M_params, M_static)
    M_check = new_M_net(q_samples[0], inference_mode=True)
    deviation = float(jnp.max(jnp.abs(M_check - target)))
    print(f"  pretrain done. initial loss={initial_loss:.3e}  "
          f"final loss={float(loss_val):.3e}  max|M(q₀) − target|={deviation:.3e}")

    return eqx.tree_at(lambda mod: mod.M_net, model, new_M_net)


# ─────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────

def train(args):
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu))
    device = jax.devices()[0]
    if args.verbose:
        print(f"Start training (ELBO/SDE) num_points={args.num_points} "
              f"n_substeps={args.n_substeps} eval_every={args.eval_every} "
              f"S={args.mc_samples} device={device}")

    key = jax.random.PRNGKey(args.seed)
    key_model, key = jax.random.split(key)

    # ── Build model ──
    model = DissipativeSO3HamSDE(
        key=key_model, u_dim=3, init_gain=args.init_gain,
        init_sigma_R=args.init_sigma_R, init_sigma_omega=args.init_sigma_omega,
        init_sigma_obs_omega=args.init_sigma_obs_omega,
        init_sigma_const=args.init_sigma_const,
    )
    n_params = sum(int(np.prod(x.shape)) for x in jax.tree.leaves(
        eqx.filter(model, eqx.is_array)))
    print(f'model contains {n_params} parameters')

    # ── Dataset ──
    us = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (-1.0, -1.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.5, 0.5, 0.5),
        (0.0, -0.5, -0.5),
    )
    data, _ = get_dataset(
        seed=args.seed, samples=args.samples, timesteps=args.timesteps,
        save_dir=args.data_dir, us=us, ori_rep="rotmat",
        friction_coeff=args.friction_coeff,
        varying_friction=args.varying_friction,
        external_force_type=args.external_force_type,
        external_force_std=args.external_force_std,
        wind_force_std=args.wind_force_std,
        obs_noise_std=args.obs_noise_std,
        random_u=args.random_u,
    )

    train_x_np, t_eval_np = arrange_data(data['x'], data['t'],
                                         num_points=args.num_points)
    test_x_np, _ = arrange_data(data['test_x'], data['t'],
                                num_points=args.num_points)
    train_x_cat = np.concatenate(train_x_np, axis=1).astype(np.float32)
    test_x_cat  = np.concatenate(test_x_np,  axis=1).astype(np.float32)
    t_eval_np   = t_eval_np.astype(np.float32)

    train_x_cat = jnp.asarray(train_x_cat)
    test_x_cat  = jnp.asarray(test_x_cat)

    dt   = float(t_eval_np[1] - t_eval_np[0])
    h    = dt / args.n_substeps
    n_outer = args.num_points - 1                   # snapshots between t=0 and t=T_obs
    B_train = int(train_x_cat.shape[1])
    B_test  = int(test_x_cat.shape[1])
    T_obs   = int(train_x_cat.shape[0])

    # N for KL scaling: total observed (b, t) data points (drop t=0).
    N_total = float(B_train * (T_obs - 1))

    print(f"  dt = {dt:.4f}  h = {h:.4f}  n_outer (per window) = {n_outer}  "
          f"n_substeps = {args.n_substeps}")
    print(f"  train batch B = {B_train}, test batch B = {B_test}, "
          f"T_obs = {T_obs}, N (= B·(T−1)) = {int(N_total)}")

    # ── M_net pretraining ──
    if args.pretrain_M_steps > 0:
        q_pretrain = train_x_cat.reshape(-1, 15)[:, :9]
        model = pretrain_M_net(
            model=model, q_samples=q_pretrain,
            n_steps=args.pretrain_M_steps,
            lr=args.pretrain_M_lr,
            print_every=args.pretrain_M_print_every,
            m=1.0, l=1.0,
        )

    # ── Optimiser ──
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(learning_rate=args.learn_rate, weight_decay=1e-4),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    dt_outer_jnp = jnp.asarray(dt, dtype=jnp.float32)
    lambda_pl_jnp = jnp.asarray(args.lambda_pl, dtype=jnp.float32)

    # ── Jitted train / eval steps ──
    # Train uses inference_mode=False (sampled GP weights for the reparameterised
    # ELBO MC estimate). Eval uses inference_mode=True so the diagnostic MSE_R/ω
    # numbers reflect the deterministic posterior-mean rollout — directly
    # comparable to the deterministic NN_ODE trainer's geo/L2 losses, with no
    # GP weight-sample noise compounding through the integrator.
    @eqx.filter_jit
    def train_step(model, opt_state, batch_x_cat, dW_batch, gp_keys_batch,
                   beta, N):
        (loss_val, aux), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(model, batch_x_cat, h, dW_batch, gp_keys_batch, beta, N,
          lambda_pl_jnp, dt_outer_jnp, False)         # inference_mode=False
        # Drop the heavy `traj_15` array from the returned aux to keep host
        # transfers small (we only need the scalar diagnostics).
        aux_scalars = {k: v for k, v in aux.items() if k != 'traj_15'}
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        new_model = eqx.apply_updates(model, updates)
        return new_model, opt_state, loss_val, aux_scalars, grads

    @eqx.filter_jit
    def eval_step(model, batch_x_cat, dW_batch, gp_keys_batch, beta, N):
        loss_val, aux = loss_fn(
            model, batch_x_cat, h, dW_batch, gp_keys_batch, beta, N,
            lambda_pl_jnp, dt_outer_jnp, True,        # inference_mode=True
        )
        return loss_val, aux

    # ── Stats — keep legacy keys (train_loss, train_l2_loss, train_geo_loss)
    # populated for downstream plotting compatibility.  New keys add the ELBO
    # decomposition + per-subnet KL + sigmas + β.
    stats = {
        # legacy
        'train_loss':       [], 'train_l2_loss':  [], 'train_geo_loss':  [],
        'forward_time':     [], 'backward_time':  [],
        'eval_step':        [],
        'test_loss':        [], 'test_l2_loss':   [], 'test_geo_loss':   [],
        'eval_M_loss':      [], 'eval_V_loss':    [],
        'eval_Dw_loss':     [], 'eval_g_loss':    [],
        # ELBO additions
        'train_nll':        [], 'train_nll_R':    [], 'train_nll_omega': [],
        'train_mse_R':      [], 'train_mse_omega': [],
        'train_kl_total':   [],
        'train_kl_M':       [], 'train_kl_V':     [], 'train_kl_Dw':     [],
        'train_kl_g':       [], 'train_kl_sigma': [],
        'sigma_R':          [], 'sigma_omega':    [], 'beta':            [],
        'test_nll':         [], 'test_nll_R':     [], 'test_nll_omega':  [],
        'test_mse_R':       [], 'test_mse_omega': [],
        # PL additions
        'train_pl_loss':         [], 'train_pl_residual_sq': [],
        'train_mean_sigma_phi':  [], 'sigma_obs_omega':      [],
        'test_pl_loss':          [], 'test_pl_residual_sq':  [],
        'test_mean_sigma_phi':   [],
    }

    os.makedirs(args.save_dir, exist_ok=True)
    label = '-so3hamGPSDE'
    stats_path = (f'{args.save_dir}/{args.name}{label}-{args.num_points}p-stats.pkl')

    # PRNG streams: one for dW sampling, one for GP-weight sampling.
    key_dW, key_w = jax.random.split(key)

    debug_dump_dir = (args.debug_dump_dir or
                      os.path.join(args.save_dir, 'debug'))

    for step_idx in range(args.total_steps + 1):
        t_start = time.time()

        key_dW, sub_dW   = jax.random.split(key_dW)
        key_w,  sub_w    = jax.random.split(key_w)
        dW_batch = _sample_dW(sub_dW, B_train, args.mc_samples,
                              n_outer, args.n_substeps, h, train_x_cat.dtype)
        gp_keys  = _sample_gp_keys(sub_w, B_train, args.mc_samples)

        beta = jnp.asarray(
            args.beta_max * min(1.0, step_idx / max(1, args.kl_anneal_steps)),
            dtype=jnp.float32,
        )
        N_jnp = jnp.asarray(N_total, dtype=jnp.float32)

        # Snapshot pre-step state so the NaN diagnostic can replay the
        # exact failing step (model + dW + GP keys all preserved).
        model_pre = model

        new_model, new_opt_state, loss_val, aux, grads = train_step(
            model, opt_state, train_x_cat, dW_batch, gp_keys, beta, N_jnp,
        )
        backward_time = time.time() - t_start

        # Cheap NaN guard: one host sync on the loss scalar.
        loss_f = float(loss_val)
        if not np.isfinite(loss_f) or _has_nan_or_inf(new_model):
            diagnose_and_dump(
                model_pre=model_pre,
                batch_x_cat=train_x_cat,
                dW_batch=dW_batch,
                gp_keys_batch=gp_keys,
                grads=grads,
                opt_state=opt_state,
                step_idx=step_idx,
                loss_val=loss_f,
                aux_val=aux,
                h=h,
                dump_dir=debug_dump_dir,
                args=args,
            )
            print("Aborting training — see dump above for post-mortem.")
            return model_pre, stats

        model = new_model
        opt_state = new_opt_state

        # Append legacy + new train stats; only sync floats we need.
        stats['train_loss'].append(loss_f)
        stats['train_geo_loss'].append(float(aux['mean_theta_sq']))
        stats['train_l2_loss'].append(float(aux['mean_omega_sq']))
        stats['train_nll'].append(float(aux['nll_total']))
        stats['train_nll_R'].append(float(aux['nll_R']))
        stats['train_nll_omega'].append(float(aux['nll_omega']))
        stats['train_mse_R'].append(float(aux['mean_theta_sq']))
        stats['train_mse_omega'].append(float(aux['mean_omega_sq']))
        stats['train_kl_total'].append(float(aux['kl_total']))
        stats['train_kl_M'].append(float(aux['kl_M']))
        stats['train_kl_V'].append(float(aux['kl_V']))
        stats['train_kl_Dw'].append(float(aux['kl_Dw']))
        stats['train_kl_g'].append(float(aux['kl_g']))
        stats['train_kl_sigma'].append(float(aux['kl_sigma']))
        stats['sigma_R'].append(float(aux['sigma_R']))
        stats['sigma_omega'].append(float(aux['sigma_omega']))
        stats['beta'].append(float(aux['beta']))
        stats['train_pl_loss'].append(float(aux['pl_loss']))
        stats['train_pl_residual_sq'].append(float(aux['pl_residual_sq']))
        stats['train_mean_sigma_phi'].append(float(aux['mean_sigma_phi']))
        stats['sigma_obs_omega'].append(float(aux['sigma_obs_omega']))
        stats['forward_time'].append(0.0)
        stats['backward_time'].append(backward_time)

        if step_idx % args.eval_every == 0:
            key_dW, sub_dW_test = jax.random.split(key_dW)
            key_w,  sub_w_test  = jax.random.split(key_w)
            dW_test = _sample_dW(sub_dW_test, B_test, args.mc_samples,
                                 n_outer, args.n_substeps, h, test_x_cat.dtype)
            gp_test = _sample_gp_keys(sub_w_test, B_test, args.mc_samples)

            test_loss, test_aux = eval_step(
                model, test_x_cat, dW_test, gp_test,
                jnp.asarray(args.beta_max, dtype=jnp.float32),  # eval at full β
                jnp.asarray(B_test * (T_obs - 1), dtype=jnp.float32),
            )

            # Subnet-physics MSE uses the (T, B, S, 15) traj — collapse S=0
            # to keep the diagnostics signature unchanged.
            test_traj = test_aux['traj_15'][:, :, 0, :]   # (T, B_test, 15)
            subnet = subnet_physics_mse(
                _DiagShim(model), test_traj,
                m=1.0, l=1.0, g=9.81,
                friction_coeff=args.friction_coeff,
                varying_friction=args.varying_friction,
            )

            stats['eval_step'].append(step_idx)
            stats['test_loss'].append(float(test_loss))
            stats['test_l2_loss'].append(float(test_aux['mean_omega_sq']))
            stats['test_geo_loss'].append(float(test_aux['mean_theta_sq']))
            stats['test_nll'].append(float(test_aux['nll_total']))
            stats['test_nll_R'].append(float(test_aux['nll_R']))
            stats['test_nll_omega'].append(float(test_aux['nll_omega']))
            stats['test_mse_R'].append(float(test_aux['mean_theta_sq']))
            stats['test_mse_omega'].append(float(test_aux['mean_omega_sq']))
            stats['eval_M_loss'].append(subnet['M_loss'])
            stats['eval_V_loss'].append(subnet['V_loss'])
            stats['eval_Dw_loss'].append(subnet['Dw_loss'])
            stats['eval_g_loss'].append(subnet['g_loss'])
            stats['test_pl_loss'].append(float(test_aux['pl_loss']))
            stats['test_pl_residual_sq'].append(float(test_aux['pl_residual_sq']))
            stats['test_mean_sigma_phi'].append(float(test_aux['mean_sigma_phi']))

            print(f"[step {step_idx:>6d}]")
            print(f"  train: −ELBO={loss_f:.4e}  "
                  f"NLL={float(aux['nll_total']):.4e}  "
                  f"NLL_R={float(aux['nll_R']):.4e}  "
                  f"NLL_ω={float(aux['nll_omega']):.4e}  "
                  f"MSE_R={float(aux['mean_theta_sq']):.4e}  "
                  f"MSE_ω={float(aux['mean_omega_sq']):.4e}")
            print(f"  test : −ELBO={float(test_loss):.4e}  "
                  f"NLL={float(test_aux['nll_total']):.4e}  "
                  f"NLL_R={float(test_aux['nll_R']):.4e}  "
                  f"NLL_ω={float(test_aux['nll_omega']):.4e}  "
                  f"MSE_R={float(test_aux['mean_theta_sq']):.4e}  "
                  f"MSE_ω={float(test_aux['mean_omega_sq']):.4e}")
            print(f"  sigma: σ_R={float(aux['sigma_R']):.4e}  "
                  f"σ_ω={float(aux['sigma_omega']):.4e}  "
                  f"β={float(aux['beta']):.4e}")
            print(f"  PL   : train={float(aux['pl_loss']):.4e}  "
                  f"test={float(test_aux['pl_loss']):.4e}  "
                  f"σ_φ̄(train)={float(aux['mean_sigma_phi']):.4e}  "
                  f"σ_φ̄(test)={float(test_aux['mean_sigma_phi']):.4e}  "
                  f"σ_obs_ω={float(aux['sigma_obs_omega']):.4e}  "
                  f"λ_PL={args.lambda_pl:.2f}")
            print(f"  KL   : M={float(aux['kl_M']):.3e}  "
                  f"V={float(aux['kl_V']):.3e}  "
                  f"Dw={float(aux['kl_Dw']):.3e}  "
                  f"g={float(aux['kl_g']):.3e}  "
                  f"sigma={float(aux['kl_sigma']):.3e}  "
                  f"total={float(aux['kl_total']):.3e}")
            print(f"  subnet MSE  M={subnet['M_loss']:.3e}  "
                  f"V={subnet['V_loss']:.3e}  Dw={subnet['Dw_loss']:.3e}  "
                  f"g={subnet['g_loss']:.3e}")

            ckpt = (f'{args.save_dir}/{args.name}{label}-'
                    f'{args.num_points}p-{step_idx}.eqx')
            eqx.tree_serialise_leaves(ckpt, model)
            to_pickle(stats, stats_path)

    # ── Final per-trajectory eval (no subnet diagnostics) ─────────────
    print("\nFinal per-trajectory eval ...")
    train_x_full = jnp.asarray(data['x'].astype(np.float32))      # (num_us, T, B, 15)
    test_x_full  = jnp.asarray(data['test_x'].astype(np.float32))
    t_full_np    = data['t'].astype(np.float32)
    n_outer_full = t_full_np.shape[0] - 1

    def per_us(x_full, base_seed):
        loss_l, l2_l, geo_l, hat_l = [], [], [], []
        for i in range(x_full.shape[0]):
            x_i = x_full[i]                                       # (T, B, 15)
            B = x_i.shape[1]
            key_local = jax.random.PRNGKey(base_seed + i)
            key_dW_i, key_w_i = jax.random.split(key_local)
            dW = _sample_dW(key_dW_i, B, 1, n_outer_full,
                            args.n_substeps, h, x_i.dtype)
            gp_keys = _sample_gp_keys(key_w_i, B, 1)
            traj_15 = _rollout_batch(model, x_i, h, dW, gp_keys,
                                     inference_mode=True)         # (T, B, 1, 15)
            traj_15 = traj_15[:, :, 0, :]                         # collapse S=1
            total, l2, geo = traj_rotmat_L2_geodesic_loss_safe(
                x_i, traj_15, split=(9, 3, 3))
            loss_l.append(total); l2_l.append(l2); geo_l.append(geo)
            hat_l.append(np.asarray(traj_15))
        return loss_l, l2_l, geo_l, hat_l

    train_loss_l, train_l2_l, train_geo_l, train_hat = per_us(
        train_x_full, args.seed + 1000)
    test_loss_l,  test_l2_l,  test_geo_l,  test_hat  = per_us(
        test_x_full,  args.seed + 2000)

    def _per_traj(ll):
        return np.sum(np.asarray(jnp.concatenate(ll, axis=1)), axis=0)

    train_loss_pt = _per_traj(train_loss_l); test_loss_pt = _per_traj(test_loss_l)
    train_l2_pt   = _per_traj(train_l2_l);   test_l2_pt   = _per_traj(test_l2_l)
    train_geo_pt  = _per_traj(train_geo_l);  test_geo_pt  = _per_traj(test_geo_l)

    print('Final trajectory train loss {:.4e} +/- {:.4e}\n'
          'Final trajectory test loss  {:.4e} +/- {:.4e}'.format(
              train_loss_pt.mean(), train_loss_pt.std(),
              test_loss_pt.mean(), test_loss_pt.std()))
    print('Final trajectory train l2 loss {:.4e} +/- {:.4e}\n'
          'Final trajectory test l2 loss  {:.4e} +/- {:.4e}'.format(
              train_l2_pt.mean(), train_l2_pt.std(),
              test_l2_pt.mean(), test_l2_pt.std()))
    print('Final trajectory train geo loss {:.4e} +/- {:.4e}\n'
          'Final trajectory test geo loss  {:.4e} +/- {:.4e}'.format(
              train_geo_pt.mean(), train_geo_pt.std(),
              test_geo_pt.mean(), test_geo_pt.std()))

    stats['traj_train_loss'] = train_loss_pt
    stats['traj_test_loss']  = test_loss_pt
    stats['train_x']         = np.asarray(train_x_full)
    stats['test_x']          = np.asarray(test_x_full)
    stats['train_x_hat']     = np.array(train_hat)
    stats['test_x_hat']      = np.array(test_hat)
    stats['t_eval']          = t_full_np
    return model, stats


if __name__ == "__main__":
    args = get_args()
    model, stats = train(args)

    os.makedirs(args.save_dir, exist_ok=True)
    label = '-so3hamGPSDE'
    final_ckpt = (f'{args.save_dir}/{args.name}{label}-{args.num_points}p.eqx')
    eqx.tree_serialise_leaves(final_ckpt, model)
    final_stats = (f'{args.save_dir}/{args.name}{label}-{args.num_points}p-stats.pkl')
    print("Saved final stats: ", final_stats)
    to_pickle(stats, final_stats)
