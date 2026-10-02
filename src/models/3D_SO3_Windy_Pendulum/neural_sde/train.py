"""Trainer for the unstructured neural SDE on the 3D windy pendulum.

Mirrors ph_gp_sde/train.py and ph_nn_sde_debug/train.py at a high level —
same dataset (`windy_pendulum_3d_datagen`), same loss bookkeeping
(`rotmat_L2_geodesic_loss_safe` + per-trajectory final eval), same stat
keys for downstream comparison plotting.

Differences:
  - No port-Hamiltonian subnets (M, V, Dw, g) — a single drift MLP.
  - No Lie-group integrator — plain Euler-Maruyama on ℝ¹² (matches the
    unstructured baseline philosophy of the reference SO(3) NODE).
  - No GP / ELBO / KL terms — straight MSE-style loss between the
    rolled-out trajectory and the env target trajectory.
"""
from __future__ import annotations

import argparse
import os
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

from windy_pendulum_3d_datagen import get_dataset, arrange_data        # noqa: E402

from network import NeuralSO3SDE                                       # noqa: E402
from src.utils.JAX.loss_utils_jax import (                             # noqa: E402
    rotmat_L2_geodesic_loss_safe,
    traj_rotmat_L2_geodesic_loss_safe,
)
from src.utils.JAX.ode_utils_jax import to_pickle                      # noqa: E402


DEFAULT_SAVE_DIR = os.path.join(THIS_FILE_DIR, 'data', 'run_wp3d_neural_sde')
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
    p.add_argument('--init_gain', default=1.0, type=float)
    p.add_argument('--hidden_dim', default=500, type=int,
                   help='MLP hidden width for both drift and diffusion heads '
                        '(matches the reference UnstructuredSO3NODE = 500)')

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

    # SDE-specific
    p.add_argument('--n_substeps', type=int, default=10,
                   help='Euler-Maruyama substeps per dt; match env (10) for '
                        'clean comparison with ph_gp_sde')
    p.add_argument('--grad_clip', type=float, default=1.0,
                   help='global-norm gradient clip')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Rollout helpers
# ─────────────────────────────────────────────────────────────────────

def _sample_dW(key, batch_size, n_outer, n_substeps, h, dtype):
    """(B, n_outer, n_substeps, 3) Wiener increments scaled so var(dW) = h."""
    return jax.random.normal(
        key, (batch_size, n_outer, n_substeps, 3), dtype=dtype
    ) * jnp.sqrt(jnp.asarray(h, dtype=dtype))


def _batched_rollout(model, x12_init, u, h, dW_batch):
    """vmap model.rollout over the batch axis.

    x12_init : (B, 12)
    u        : (B, u_dim)
    dW_batch : (B, n_outer, n_substeps, 3)
    Returns  : (n_outer + 1, B, 12) — time-major to match the loss helper.
    """
    rollout_one = lambda x0_b, u_b, dW_b: model.rollout(x0_b, u_b, h, dW_b)
    traj_b = jax.vmap(rollout_one)(x12_init, u, dW_batch)         # (B, T+1, 12)
    return jnp.transpose(traj_b, (1, 0, 2))                        # (T+1, B, 12)


def _pad_with_u(traj_12, u_const):
    """traj_12 : (T, B, 12), u_const : (B, u_dim) → (T, B, 12 + u_dim)."""
    T, B, _ = traj_12.shape
    u_b = jnp.broadcast_to(u_const[None, :, :],
                           (T, B, u_const.shape[-1]))
    return jnp.concatenate([traj_12, u_b], axis=-1)


def loss_fn(model, batch_x_cat, h, dW_batch):
    """batch_x_cat : (T, B, 15)  env trajectories.

    Returns total loss + (l2, geo) auxiliaries.
    """
    x0_15    = batch_x_cat[0]                              # (B, 15)
    x12_init = x0_15[:, :12]                               # (B, 12)
    u_const  = x0_15[:, 12:15]                             # (B, 3)

    traj_12 = _batched_rollout(model, x12_init, u_const, h, dW_batch)
    traj_15 = _pad_with_u(traj_12, u_const)

    target     = batch_x_cat[1:]
    target_hat = traj_15[1:]
    total, l2, geo = rotmat_L2_geodesic_loss_safe(
        target, target_hat, split=(9, 3, 3)
    )
    return total, (l2, geo)


# ─────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────

def train(args):
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu))
    device = jax.devices()[0]
    if args.verbose:
        print(f"Start training (neural SDE) num_points={args.num_points} "
              f"n_substeps={args.n_substeps} eval_every={args.eval_every} "
              f"device={device}")

    key = jax.random.PRNGKey(args.seed)
    key_model, key_dW = jax.random.split(key)

    model = NeuralSO3SDE(
        key=key_model, u_dim=3,
        hidden_dim=args.hidden_dim, init_gain=args.init_gain,
    )
    n_params = sum(int(np.prod(x.shape)) for x in jax.tree.leaves(
        eqx.filter(model, eqx.is_array)))
    print(f'model contains {n_params} parameters')

    # ── Dataset ──
    us = (
        (0.0, 0.0, 0.0),
        (-1.0, -1.0, -1.0),
        (1.0, 1.0, 1.0),
        (-2.0, -2.0, -2.0),
        (2.0, 2.0, 2.0),
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

    dt      = float(t_eval_np[1] - t_eval_np[0])
    h       = dt / args.n_substeps
    n_outer = args.num_points - 1
    B_train = int(train_x_cat.shape[1])
    B_test  = int(test_x_cat.shape[1])

    print(f"  dt = {dt:.4f}  h = {h:.4f}  n_outer (per window) = {n_outer}  "
          f"n_substeps = {args.n_substeps}")
    print(f"  train batch B = {B_train}, test batch B = {B_test}")

    # ── Optimiser ──
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(learning_rate=args.learn_rate, weight_decay=1e-4),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, batch_x_cat, dW_batch):
        (loss_val, (l2, geo)), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(model, batch_x_cat, h, dW_batch)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        new_model = eqx.apply_updates(model, updates)
        return new_model, opt_state, loss_val, l2, geo

    @eqx.filter_jit
    def eval_step(model, batch_x_cat, dW_batch):
        loss_val, (l2, geo) = loss_fn(model, batch_x_cat, h, dW_batch)
        return loss_val, l2, geo

    stats = {
        'train_loss':    [], 'train_l2_loss':  [], 'train_geo_loss': [],
        'forward_time':  [], 'backward_time':  [],
        'eval_step':     [],
        'test_loss':     [], 'test_l2_loss':   [], 'test_geo_loss':  [],
    }

    os.makedirs(args.save_dir, exist_ok=True)
    label = '-neuralSDE'
    stats_path = (f'{args.save_dir}/{args.name}{label}-{args.num_points}p-stats.pkl')

    for step_idx in range(args.total_steps + 1):
        t0 = time.time()
        key_dW, sub = jax.random.split(key_dW)
        dW_batch = _sample_dW(sub, B_train, n_outer, args.n_substeps,
                              h, train_x_cat.dtype)

        new_model, new_opt_state, loss_val, l2_val, geo_val = train_step(
            model, opt_state, train_x_cat, dW_batch
        )
        backward_time = time.time() - t0

        loss_f = float(loss_val)
        if not np.isfinite(loss_f):
            print(f"[step {step_idx}] non-finite loss={loss_f}; aborting.")
            return model, stats

        model = new_model
        opt_state = new_opt_state

        stats['train_loss'].append(loss_f)
        stats['train_l2_loss'].append(float(l2_val))
        stats['train_geo_loss'].append(float(geo_val))
        stats['forward_time'].append(0.0)
        stats['backward_time'].append(backward_time)

        if step_idx % args.eval_every == 0:
            key_dW, sub = jax.random.split(key_dW)
            dW_test = _sample_dW(sub, B_test, n_outer, args.n_substeps,
                                 h, test_x_cat.dtype)
            test_loss, test_l2, test_geo = eval_step(
                model, test_x_cat, dW_test
            )

            stats['eval_step'].append(step_idx)
            stats['test_loss'].append(float(test_loss))
            stats['test_l2_loss'].append(float(test_l2))
            stats['test_geo_loss'].append(float(test_geo))

            print(f"[step {step_idx:>6d}]")
            print(f"  train: total={loss_f:.4e}  L2={float(l2_val):.4e}  "
                  f"geo={float(geo_val):.4e}  "
                  f"step_time={backward_time*1e3:.1f}ms")
            print(f"  test : total={float(test_loss):.4e}  "
                  f"L2={float(test_l2):.4e}  geo={float(test_geo):.4e}")

            ckpt = (f'{args.save_dir}/{args.name}{label}-'
                    f'{args.num_points}p-{step_idx}.eqx')
            eqx.tree_serialise_leaves(ckpt, model)
            to_pickle(stats, stats_path)

    # ── Final per-trajectory eval ─────────────────────────────────────
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
            dW = _sample_dW(key_local, B, n_outer_full,
                            args.n_substeps, h, x_i.dtype)
            x_init_15 = x_i[0]
            u_const   = x_init_15[:, 12:15]
            traj_12 = _batched_rollout(model, x_init_15[:, :12], u_const, h, dW)
            traj_15 = _pad_with_u(traj_12, u_const)               # (T, B, 15)
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
    label = '-neuralSDE'
    final_ckpt = (f'{args.save_dir}/{args.name}{label}-{args.num_points}p.eqx')
    eqx.tree_serialise_leaves(final_ckpt, model)
    final_stats = (f'{args.save_dir}/{args.name}{label}-{args.num_points}p-stats.pkl')
    print("Saved final stats: ", final_stats)
    to_pickle(stats, final_stats)
