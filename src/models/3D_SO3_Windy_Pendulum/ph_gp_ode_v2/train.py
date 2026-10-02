"""Train ph_gp_ode_v2 — deterministic ODE port-Hamiltonian on SO(3) × ℝ³.

This is a fresh, leaner trainer focused on the deterministic case. The
prior SDE-based trainer with PL term and dW sampling is preserved as
`train_legacy_sde.py.bak` for reference.

Loss objective:

    L_total = L_NLL  +  (β/N)·L_KL
            + λ_power · L_power_w
            + λ_V     · L_V_w
            + λ_B     · L_B_w
            + λ_D     · L_D_w

where the four `_w` losses are the noise-aware (whitened) auxiliary
losses from `physics_losses.py`. With `λ_X = 0` the trainer reduces to
plain ELBO on the deterministic rollout.

Run-folder name is auto-built from the relevant args + a YYMMDD-HHMMSS
stamp so each run lives in its own directory.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.abspath(os.path.join(THIS_FILE_DIR, '../../../..'))
DATASETS_DIR  = os.path.join(PROJECT_ROOT, 'datasets')
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, 'src/utils'), DATASETS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from windy_pendulum_3d_datagen import get_dataset, arrange_data    # noqa: E402

from network import DissipativeSO3HamODE                           # noqa: E402
from src.utils.JAX.lie_integrator import lie_heun_ode_rollout      # noqa: E402
from src.utils.JAX.elbo_loss_jax import elbo_nll, kl_per_subnet    # noqa: E402
from src.utils.JAX.subnet_diagnostics_jax import subnet_physics_mse  # noqa: E402
from src.utils.JAX.ode_utils_jax import to_pickle                  # noqa: E402

from physics_losses import physics_aux_losses                      # noqa: E402


DEFAULT_SAVE_DIR = os.path.join(THIS_FILE_DIR, 'data', 'run_wp3d_jax')
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, 'datasets/data/windy_pendulum_3d')


# ──────────────────────────────────────────────────────────────────────
# Run-folder naming
# ──────────────────────────────────────────────────────────────────────

def _fmt_num(x):
    s = f"{x:g}"
    return s.replace('.', 'p').replace('-', 'n').replace('+', '')


def build_run_name(args):
    parts = [
        f"obs{_fmt_num(args.obs_noise_std)}",
        f"fric{_fmt_num(args.friction_coeff)}",
        f"wind{_fmt_num(args.wind_force_std)}",
        f"ext{_fmt_num(args.external_force_std)}-{args.external_force_type}",
        f"lP{_fmt_num(args.lambda_power)}",
        f"lV{_fmt_num(args.lambda_V)}",
        f"lB{_fmt_num(args.lambda_B)}",
        f"lD{_fmt_num(args.lambda_D)}",
        f"lr{_fmt_num(args.learn_rate)}",
        f"s{args.total_steps}",
        f"np{args.num_points}",
        f"smp{args.samples}",
        f"T{args.timesteps}",
        f"seed{args.seed}",
    ]
    if args.varying_friction:
        parts.append('varfric')
    if args.random_u:
        parts.append(f'randu{_fmt_num(args.random_u_scale)}')
    if args.whiten:
        parts.append('w')
    if args.fix_M:
        parts.append('fixM')
    stamp = time.strftime('%y%m%d-%H%M%S')
    return '_'.join(parts) + '_' + stamp


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
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
    p.add_argument('--random_u_scale', type=float, default=1.0,
                   help='if --random_u is set, sample u ~ U(-scale, scale) per step')

    # Integrator
    p.add_argument('--n_substeps', type=int, default=10)
    p.add_argument('--grad_clip', type=float, default=1.0)

    # ELBO observation noise (rollout NLL only)
    p.add_argument('--init_sigma_R', type=float, default=0.1)
    p.add_argument('--init_sigma_omega', type=float, default=0.1)
    p.add_argument('--beta_max', type=float, default=1.0)
    p.add_argument('--kl_anneal_steps', type=int, default=1000)

    # Physics-informed auxiliary losses (whitened by default)
    p.add_argument('--lambda_power', type=float, default=0.0,
                   help='weight for L_power_w (energy bookkeeping)')
    p.add_argument('--lambda_V', type=float, default=0.0,
                   help='weight for L_V_w (V back-solving)')
    p.add_argument('--lambda_B', type=float, default=0.0,
                   help='weight for L_B_w (B back-solving)')
    p.add_argument('--lambda_D', type=float, default=0.0,
                   help='weight for L_D_w (D back-solving)')
    p.add_argument('--whiten', action='store_true', default=True,
                   help='whiten aux losses by σ_obs_ω-derived scales')
    p.add_argument('--no_whiten', dest='whiten', action='store_false',
                   help='use raw mean-‖·‖² aux losses (not noise-invariant)')

    # Mass: fixed I₃·(1/(m·l²)) vs. trainable PSD-GP subnet.
    # --mass_fixed True  → constant FixedInverseMass (no params for M).
    # --mass_fixed False → PSD_GP_Model subnet (default; M⁻¹(q) is learned).
    # --fix_M kept as a back-compatible alias of --mass_fixed True.
    p.add_argument('--mass_fixed', type=lambda s: s.lower() in ('1','true','t','yes','y'),
                   default=False,
                   help='True → use fixed (non-learnable) inverse-mass; '
                        'False → use trainable PSD-GP subnet (default)')
    p.add_argument('--fix_M', action='store_true',
                   help='[deprecated alias] equivalent to --mass_fixed True')
    p.add_argument('--fix_M_m', type=float, default=1.0,
                   help='m used by the fixed M⁻¹ when mass is fixed')
    p.add_argument('--fix_M_l', type=float, default=1.0,
                   help='l used by the fixed M⁻¹ when mass is fixed')
    args = p.parse_args()
    # Reconcile the two flags: either being set means "fix the mass".
    args.fix_M = bool(args.fix_M or args.mass_fixed)
    args.mass_fixed = args.fix_M
    return args


# ──────────────────────────────────────────────────────────────────────
# KeyedODEModel — adapter that injects per-subnet GP keys.
# ──────────────────────────────────────────────────────────────────────

class KeyedODEModel(eqx.Module):
    model: DissipativeSO3HamODE
    keys:  dict
    inference_mode: bool = eqx.field(static=True)

    def _eff_keys(self):
        return None if self.inference_mode else (self.keys or None)

    def drift_p(self, q, p, u):
        return self.model.drift_p(q, p, u, keys=self._eff_keys())

    def M_inv(self, q):
        return self.model.M_inv(q, keys=self._eff_keys())

    # ── Subnet-call passthroughs for physics_losses.py ──
    @property
    def u_dim(self):    return self.model.u_dim
    @property
    def friction(self): return self.model.friction

    def _M_call(self, q):  return self.model._M_call(q, key=None)
    def _V_call(self, q):  return self.model._V_call(q, key=None)
    def _Dw_call(self, q, p): return self.model._Dw_call(q, p, key=None)
    def _g_call(self, q):  return self.model._g_call(q, key=None)


def _split_subnet_keys(key):
    kM, kV, kD, kg = jax.random.split(key, 4)
    return {'M': kM, 'V': kV, 'Dw': kD, 'g': kg}


class _DiagShim:
    """Adapter so subnet_physics_mse (expects MLP-style net(q)) keeps
    working with the GP-based subnets — bind inference_mode=True."""
    def __init__(self, model):
        self.M_net  = lambda q: model.M_net (q, inference_mode=True)
        self.V_net  = lambda q: model.V_net (q, inference_mode=True)
        # Dw_net now takes (q, p) — concatenate before calling.
        self.Dw_net = lambda q, p: model.Dw_net(
            jnp.concatenate([q, p]), inference_mode=True)
        self.g_net  = lambda q: model.g_net (q, inference_mode=True)


# ──────────────────────────────────────────────────────────────────────
# Rollout (deterministic; per-batch / inference_mode true at eval time)
# ──────────────────────────────────────────────────────────────────────

def _rollout_single(model, x0_12, u_seq, u_full, h, gp_keys,
                     n_substeps, n_outer, inference_mode):
    """Roll one (12,) IC + (n_outer, 3) per-step u → (T, 15).

    u_seq  : control applied during each outer step k → k+1 (k = 0..n_outer-1)
    u_full : (T, 3) — u column to concatenate with predicted state for output.
             For consistency with the data convention we just copy u from the
             ground-truth batch (u is a known input, not a learned target).
    """
    keyed = KeyedODEModel(model=model, keys=gp_keys,
                          inference_mode=inference_mode)
    traj_12 = lie_heun_ode_rollout(keyed, x0_12, u_seq, h,
                                    n_substeps, n_outer)
    return jnp.concatenate([traj_12, u_full], axis=-1)           # (T, 15)


def _rollout_batch(model, batch_x_cat, h, gp_keys_batch,
                   n_substeps, n_outer, inference_mode):
    """batch_x_cat: (T, B, 15); gp_keys_batch: dict name → (B, 2) keys.

    Per-step control: u_k = batch_x_cat[k+1, b, 12:15] is the control applied
    over outer step k → k+1 (cf. datagen convention). u for k=0 → 1 equals
    batch_x_cat[1, b, 12:15] which equals batch_x_cat[0, b, 12:15] (= u_0).
    """
    x0_12  = batch_x_cat[0, :, :12]                              # (B, 12)
    # u_seq: outer steps 0..n_outer-1 → use traj[1:n_outer+1, ..., 12:15]
    u_seq  = jnp.transpose(batch_x_cat[1:n_outer+1, :, 12:15],
                           (1, 0, 2))                            # (B, n_outer, 3)
    u_full = jnp.transpose(batch_x_cat[..., 12:15], (1, 0, 2))   # (B, T, 3)

    def per_batch(x0_b, u_seq_b, u_full_b, keys_b):
        return _rollout_single(model, x0_b, u_seq_b, u_full_b, h, keys_b,
                                n_substeps, n_outer, inference_mode)
    traj_BT15 = jax.vmap(per_batch)(x0_12, u_seq, u_full, gp_keys_batch)  # (B, T, 15)
    return jnp.transpose(traj_BT15, (1, 0, 2))                   # (T, B, 15)


# ──────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────

def loss_fn(model, batch_x_cat, h, gp_keys_batch, beta, N,
            dt_outer, n_substeps, n_outer,
            lambda_power, lambda_V, lambda_B, lambda_D,
            whiten, inference_mode=False):
    traj_15 = _rollout_batch(model, batch_x_cat, h, gp_keys_batch,
                              n_substeps, n_outer, inference_mode)

    target_obs = batch_x_cat[1:]
    target_hat = traj_15[1:][..., None, :]   # (T-1, B, 1, 15)
    target_obs_b = target_obs[..., None, :]  # (T-1, B, 1, 15)
    nll = elbo_nll(target_obs_b, target_hat,
                   model.log_sigma_R, model.log_sigma_omega, split=(9, 3, 3))
    kl  = kl_per_subnet(model)

    # Physics-informed aux losses on the GT batch (data-driven, not rollout).
    # Use a posterior-mean shim so the aux losses are deterministic w.r.t. the
    # GP weight sample — they're a calibration term, not a sampled-loss term.
    keyed_inf = KeyedODEModel(model=model, keys={}, inference_mode=True)
    # Whitening source: the *trainable* ω-noise scale `log_sigma_omega`,
    # which is co-trained via the rollout NLL. This is the "we don't know
    # the true obs noise" path — robust to real-world deployment where the
    # dataset's σ_obs is not given. As `log_sigma_omega` adapts toward the
    # true noise level, L_power_w and L_V/B/D_w land on their interpretable
    # expectations (~1 and ~3.96 respectively for the GT-subnet baseline).
    sigma_omega_now = jnp.exp(model.log_sigma_omega)
    aux = physics_aux_losses(
        keyed_inf, batch_x_cat, dt_outer,
        sigma_omega=sigma_omega_now, whiten=whiten,
    )

    total = (nll['nll_total']
             + (beta / N) * kl['total_kl']
             + lambda_power * aux['L_power']
             + lambda_V     * aux['L_V']
             + lambda_B     * aux['L_B']
             + lambda_D     * aux['L_D'])

    aux_out = {
        'nll_total':  nll['nll_total'],
        'mse_R':      nll['mean_theta_sq'],
        'mse_omega':  nll['mean_omega_sq'],
        'kl_total':   kl['total_kl'],
        'sigma_R':    nll['sigma_R'],
        'sigma_omega': nll['sigma_omega'],
        'L_power':    aux['L_power'],
        'L_V':        aux['L_V'],
        'L_B':        aux['L_B'],
        'L_D':        aux['L_D'],
        'L_power_raw': aux['L_power_raw'],
        'L_V_raw':    aux['L_V_raw'],
    }
    return total, aux_out


# ──────────────────────────────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────────────────────────────

def train(args):
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu))
    device = jax.devices()[0]

    # Per-run folder
    run_name = build_run_name(args)
    args.save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Run dir : {args.save_dir}")
    if args.verbose:
        print(f"Start training (deterministic ODE) eval_every={args.eval_every} "
              f"device={device}  whiten={args.whiten}")

    key = jax.random.PRNGKey(args.seed)
    key_model, key = jax.random.split(key)

    model = DissipativeSO3HamODE(
        key=key_model, u_dim=3, init_gain=args.init_gain,
        init_sigma_R=args.init_sigma_R, init_sigma_omega=args.init_sigma_omega,
        init_sigma_obs_omega=args.obs_noise_std,
        fix_M=args.fix_M, fix_M_m=args.fix_M_m, fix_M_l=args.fix_M_l,
    )
    if args.fix_M:
        print(f"M_net pinned to (1/(m·l²))·I₃ with "
              f"m={args.fix_M_m}  l={args.fix_M_l} — M_net does not train.")
    n_params = sum(int(np.prod(x.shape)) for x in jax.tree.leaves(
        eqx.filter(model, eqx.is_array)))
    print(f'model contains {n_params} parameters')

    us = ((0.0, 0.0, 0.0), (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0),
          (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))
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
        random_u_scale=args.random_u_scale,
    )

    train_x_np, t_eval_np = arrange_data(data['x'], data['t'],
                                          num_points=args.num_points)
    test_x_np, _ = arrange_data(data['test_x'], data['t'],
                                 num_points=args.num_points)
    train_x_cat = jnp.asarray(np.concatenate(train_x_np, axis=1).astype(np.float32))
    test_x_cat  = jnp.asarray(np.concatenate(test_x_np,  axis=1).astype(np.float32))

    dt = float(t_eval_np[1] - t_eval_np[0])
    h  = dt / args.n_substeps
    n_outer = args.num_points - 1
    B_train = int(train_x_cat.shape[1])
    B_test  = int(test_x_cat.shape[1])
    T_obs   = int(train_x_cat.shape[0])
    N_total = float(B_train * (T_obs - 1))

    print(f"  dt = {dt:.4f}  h = {h:.4f}  n_outer = {n_outer}  "
          f"B_train = {B_train}  T_obs = {T_obs}")

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(learning_rate=args.learn_rate, weight_decay=1e-4),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    dt_outer_jnp = jnp.asarray(dt, dtype=jnp.float32)

    @eqx.filter_jit
    def train_step(model, opt_state, batch_x_cat, gp_keys, beta, N):
        (loss_val, aux), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(model, batch_x_cat, h, gp_keys, beta, N,
          dt_outer_jnp, args.n_substeps, n_outer,
          args.lambda_power, args.lambda_V, args.lambda_B, args.lambda_D,
          args.whiten, False)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        new_model = eqx.apply_updates(model, updates)
        return new_model, opt_state, loss_val, aux

    @eqx.filter_jit
    def eval_step(model, batch_x_cat, gp_keys, beta, N):
        return loss_fn(
            model, batch_x_cat, h, gp_keys, beta, N,
            dt_outer_jnp, args.n_substeps, n_outer,
            args.lambda_power, args.lambda_V, args.lambda_B, args.lambda_D,
            args.whiten, True,
        )

    stats = {k: [] for k in (
        'train_loss', 'train_l2_loss', 'train_geo_loss',
        'train_nll', 'train_kl_total',
        'train_L_power', 'train_L_V', 'train_L_B', 'train_L_D',
        'train_L_power_raw', 'train_L_V_raw',
        'sigma_R', 'sigma_omega',
        'eval_step',
        'test_loss', 'test_l2_loss', 'test_geo_loss',
        'eval_M_loss', 'eval_V_loss', 'eval_Dw_loss', 'eval_g_loss',
    )}
    stats_path = f'{args.save_dir}/{args.name}-so3hamGPODE-{args.num_points}p-stats.pkl'

    key_w = jax.random.split(key, 1)[0]

    def _sample_keys(rng_key, B):
        sub = jax.random.split(rng_key, B * 4).reshape(4, B, 2)
        return {'M': sub[0], 'V': sub[1], 'Dw': sub[2], 'g': sub[3]}

    for step_idx in range(args.total_steps + 1):
        t0 = time.time()
        key_w, sub_w = jax.random.split(key_w)
        gp_keys = _sample_keys(sub_w, B_train)

        beta  = jnp.asarray(args.beta_max * min(1.0, step_idx /
                                                 max(1, args.kl_anneal_steps)),
                            dtype=jnp.float32)
        N_jnp = jnp.asarray(N_total, dtype=jnp.float32)

        model, opt_state, loss_val, aux = train_step(
            model, opt_state, train_x_cat, gp_keys, beta, N_jnp,
        )

        loss_f = float(loss_val)
        if not np.isfinite(loss_f):
            print(f"NaN/Inf loss at step {step_idx}; aborting.")
            return model, stats

        stats['train_loss'].append(loss_f)
        stats['train_geo_loss'].append(float(aux['mse_R']))
        stats['train_l2_loss'].append(float(aux['mse_omega']))
        stats['train_nll'].append(float(aux['nll_total']))
        stats['train_kl_total'].append(float(aux['kl_total']))
        stats['train_L_power'].append(float(aux['L_power']))
        stats['train_L_V'].append(float(aux['L_V']))
        stats['train_L_B'].append(float(aux['L_B']))
        stats['train_L_D'].append(float(aux['L_D']))
        stats['train_L_power_raw'].append(float(aux['L_power_raw']))
        stats['train_L_V_raw'].append(float(aux['L_V_raw']))
        stats['sigma_R'].append(float(aux['sigma_R']))
        stats['sigma_omega'].append(float(aux['sigma_omega']))

        if step_idx % args.eval_every == 0:
            key_w, sub_w_test = jax.random.split(key_w)
            gp_test = _sample_keys(sub_w_test, B_test)
            test_loss, test_aux = eval_step(
                model, test_x_cat, gp_test,
                jnp.asarray(args.beta_max, dtype=jnp.float32),
                jnp.asarray(B_test * (T_obs - 1), dtype=jnp.float32),
            )
            sub_diag = subnet_physics_mse(
                _DiagShim(model), test_x_cat[1:],
                m=1.0, l=1.0, g=9.81,
                friction_coeff=args.friction_coeff,
                varying_friction=args.varying_friction,
            )

            stats['eval_step'].append(step_idx)
            stats['test_loss'].append(float(test_loss))
            stats['test_l2_loss'].append(float(test_aux['mse_omega']))
            stats['test_geo_loss'].append(float(test_aux['mse_R']))
            stats['eval_M_loss'].append(sub_diag['M_loss'])
            stats['eval_V_loss'].append(sub_diag['V_loss'])
            stats['eval_Dw_loss'].append(sub_diag['Dw_loss'])
            stats['eval_g_loss'].append(sub_diag['g_loss'])

            print(f"[step {step_idx:>6d}]  "
                  f"loss={loss_f:.3e}  nll={float(aux['nll_total']):.3e}  "
                  f"kl={float(aux['kl_total']):.3e}")
            print(f"  rollout L2={float(aux['mse_omega']):.3e}  "
                  f"geo²={float(aux['mse_R']):.3e}    "
                  f"σ_R={float(aux['sigma_R']):.3e}  "
                  f"σ_ω={float(aux['sigma_omega']):.3e}")
            print(f"  aux (whitened)  L_power={float(aux['L_power']):.3e}  "
                  f"L_V={float(aux['L_V']):.3e}  "
                  f"L_B={float(aux['L_B']):.3e}  "
                  f"L_D={float(aux['L_D']):.3e}")
            print(f"  subnet MSE  M={sub_diag['M_loss']:.3e}  "
                  f"V={sub_diag['V_loss']:.3e}  "
                  f"Dw={sub_diag['Dw_loss']:.3e}  "
                  f"g={sub_diag['g_loss']:.3e}")

            ckpt = f'{args.save_dir}/{args.name}-so3hamGPODE-{args.num_points}p-{step_idx}.eqx'
            eqx.tree_serialise_leaves(ckpt, model)
            to_pickle(stats, stats_path)

        if args.verbose and step_idx % 10 == 0:
            print(f"step {step_idx}  step_time={time.time() - t0:.3f}s",
                  flush=True)

    return model, stats


if __name__ == "__main__":
    a = get_args()
    train(a)
