"""SDE trainer with NaN-detection instrumentation.

Trains DissipativeSO3HamSDE end-to-end on the windy_pendulum_3d dataset.
Differs from the (deprecated) ODE trainer:
  - Uses lie_heun_sde_rollout (Stratonovich Heun on SO(3) × ℝ³) instead of
    Diffrax. Substep count must match the env (default 10).
  - At each train step, samples fresh Wiener increments dW from a JAX PRNG
    key. The env target trajectory was generated with a DIFFERENT dW path,
    so the loss compares two independent SDE samples — the gradient signal
    is intrinsically noisier than ODE training. Drift learns the env mean,
    sigma_net learns the spread.
  - Optimizer chain prepends optax.clip_by_global_norm before adamw to
    suppress NaN-exploding gradients before they corrupt the params.

NaN debugging:
  - Every step we sync loss + grad_norm and check finiteness (the only
    extra host-roundtrip vs a vanilla trainer).
  - On NaN: a heavy diagnostic path re-runs the failing step in eager
    mode (no jit, no scan), printing per-substep ‖drift‖, ‖σ‖, ‖stoch‖,
    det(R), ‖RᵀR-I‖, ‖ω‖. Reports which subnet first emitted NaN at the
    failing batch's q[0]. Dumps a pickle with model + batch + dW + grads
    for offline post-mortem, then exits.
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
    lie_heun_sde_rollout,
    lie_heun_sde_step,
)
from src.utils.JAX.loss_utils_jax import (                         # noqa: E402
    rotmat_L2_geodesic_loss_safe,
    traj_rotmat_L2_geodesic_loss_safe,
)
from src.utils.JAX.subnet_diagnostics_jax import subnet_physics_mse  # noqa: E402
from src.utils.JAX.ode_utils_jax import to_pickle, exp_so3          # noqa: E402


DEFAULT_SAVE_DIR = os.path.join(THIS_FILE_DIR, 'data', 'run_wp3d_sde_debug')
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
                   help='global-norm gradient clip (passed to optax.clip_by_global_norm)')

    # Debug-specific
    p.add_argument('--debug_dump_dir', type=str, default=None,
                   help='where to dump NaN post-mortem pickles (default: save_dir/debug)')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Rollout helpers
# ─────────────────────────────────────────────────────────────────────

def _sample_dW(key, batch_size, n_outer, n_substeps, h, dtype):
    """Wiener increments scaled so var(dW) = h, shape (B, n_outer, n_substeps, 3)."""
    return jax.random.normal(
        key, (batch_size, n_outer, n_substeps, 3), dtype=dtype
    ) * jnp.sqrt(jnp.asarray(h, dtype=dtype))


def _batched_rollout(model, x12_init, u, h, dW_batch):
    """Vmap lie_heun_sde_rollout over the batch axis.

    x12_init : (B, 12)
    u        : (B, 3)
    dW_batch : (B, n_outer, n_substeps, 3)
    Returns  : (n_outer + 1, B, 12)  — time-major to match the loss helper.
    """
    rollout_one = lambda x0_b, u_b, dW_b: lie_heun_sde_rollout(
        model, x0_b, u_b, h, dW_b
    )
    traj_b = jax.vmap(rollout_one)(x12_init, u, dW_batch)   # (B, T+1, 12)
    return jnp.transpose(traj_b, (1, 0, 2))                  # (T+1, B, 12)


def _pad_with_u(traj_12, u_const):
    """traj_12 : (T, B, 12), u_const : (B, 3) → (T, B, 15)."""
    T, B, _ = traj_12.shape
    u_broad = jnp.broadcast_to(u_const[None, :, :], (T, B, 3))
    return jnp.concatenate([traj_12, u_broad], axis=-1)


def loss_fn(model, batch_x_cat, h, dW_batch):
    """batch_x_cat : (T, B, 15)  env trajectories.

    Returns scalar total loss + (l2, geo) auxiliaries.
    """
    x0_15 = batch_x_cat[0]                                   # (B, 15)
    x12_init = x0_15[:, :12]                                 # (B, 12)
    u_const  = x0_15[:, 12:15]                               # (B, 3)

    traj_12 = _batched_rollout(model, x12_init, u_const, h, dW_batch)  # (T, B, 12)
    traj_15 = _pad_with_u(traj_12, u_const)                  # (T, B, 15)

    target     = batch_x_cat[1:]
    target_hat = traj_15[1:]
    total, l2, geo = rotmat_L2_geodesic_loss_safe(
        target, target_hat, split=(9, 3, 3)
    )
    return total, (l2, geo)


def _global_grad_norm(grads):
    leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
    return jnp.sqrt(sum(jnp.sum(g * g) for g in leaves))


# ─────────────────────────────────────────────────────────────────────
# NaN diagnostics (executed only when a NaN/Inf trips the cheap check)
# ─────────────────────────────────────────────────────────────────────

def _has_nan_or_inf(tree) -> bool:
    for x in jax.tree.leaves(eqx.filter(tree, eqx.is_array)):
        a = np.asarray(x)
        if not np.all(np.isfinite(a)):
            return True
    return False


def _per_subnet_nan_report(model) -> dict:
    """Per-subnet param-array stats. NaN/Inf counts and value range."""
    out = {}
    for name in ('M_net', 'V_net', 'Dw_net', 'g_net', 'sigma_net'):
        sub = getattr(model, name)
        leaves = jax.tree.leaves(eqx.filter(sub, eqx.is_array))
        n_params = sum(int(np.asarray(l).size) for l in leaves)
        n_nan = sum(int(np.isnan(np.asarray(l)).sum()) for l in leaves)
        n_inf = sum(int(np.isinf(np.asarray(l)).sum()) for l in leaves)
        finite_vals = np.concatenate([
            np.asarray(l).reshape(-1)[np.isfinite(np.asarray(l).reshape(-1))]
            for l in leaves
        ]) if leaves else np.array([0.0])
        if finite_vals.size == 0:
            v_min, v_max = float('nan'), float('nan')
        else:
            v_min = float(finite_vals.min())
            v_max = float(finite_vals.max())
        out[name] = dict(n_params=n_params, n_nan=n_nan, n_inf=n_inf,
                         min=v_min, max=v_max)
    return out


def _subnet_outputs_at_q(model, q) -> dict:
    """Call each subnet on a single q sample; report NaN/Inf in outputs."""
    out = {}
    for name in ('M_net', 'V_net', 'Dw_net', 'g_net'):
        try:
            y = np.asarray(getattr(model, name)(q))
            out[name] = dict(
                shape=y.shape,
                has_nan=bool(np.isnan(y).any()),
                has_inf=bool(np.isinf(y).any()),
                min=float(y[np.isfinite(y)].min()) if np.isfinite(y).any() else float('nan'),
                max=float(y[np.isfinite(y)].max()) if np.isfinite(y).any() else float('nan'),
            )
        except Exception as e:
            out[name] = dict(error=str(e))
    try:
        y = np.asarray(model.sigma(q))
        out['sigma'] = dict(
            shape=y.shape, value=float(y) if y.size == 1 else None,
            has_nan=bool(np.isnan(y).any()),
            has_inf=bool(np.isinf(y).any()),
        )
    except Exception as e:
        out['sigma'] = dict(error=str(e))
    return out


def _substep_replay(model, x0_12, u, h, dW_seq_one) -> list:
    """Re-run rollout for ONE trajectory in pure Python (no jit, no scan).

    Returns a list of per-substep stats so we can locate where things explode.
    Stops at the first substep that produces a non-finite value.
    dW_seq_one : (n_outer, n_substeps, 3) — for ONE batch element.
    """
    history = []
    x = np.asarray(x0_12)
    n_outer, n_sub, _ = dW_seq_one.shape
    for i in range(n_outer):
        for j in range(n_sub):
            # Compute pre-step diagnostics
            q  = jnp.asarray(x[:9])
            om = jnp.asarray(x[9:12])
            R  = q.reshape(3, 3)
            try:
                drift_v = np.asarray(model.drift(q, om, jnp.asarray(u)))
                stoch_v = np.asarray(model.stochastic_increment(
                    q, jnp.asarray(dW_seq_one[i, j])))
                sig_v = float(np.asarray(model.sigma(q)))
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

            # Apply one substep
            x_jax = jnp.asarray(x)
            x = np.asarray(lie_heun_sde_step(
                model, x_jax, jnp.asarray(u), h, jnp.asarray(dW_seq_one[i, j])
            ))
            if not np.all(np.isfinite(x)):
                history[-1]['post_step_has_nan'] = True
                return history
    return history


def diagnose_and_dump(model_pre_step, batch_x_cat, dW_batch_np, grads,
                      opt_state, step_idx, loss_val, l2_val, geo_val,
                      grad_norm, h, dump_dir, args) -> str:
    """Heavy debug path: collate diagnostics, print, dump pickle, return path."""
    os.makedirs(dump_dir, exist_ok=True)
    print("\n" + "!" * 72)
    print(f"!! NaN/Inf DETECTED at step {step_idx}")
    print(f"!!   loss = {loss_val}   l2 = {l2_val}   geo = {geo_val}   "
          f"grad_norm = {grad_norm}")
    print("!" * 72)

    # 1. Per-subnet param report
    print("\n[1] Per-subnet param stats (model state JUST BEFORE the failing step):")
    subnet_report = _per_subnet_nan_report(model_pre_step)
    for name, st in subnet_report.items():
        flag = "  ⚠" if (st['n_nan'] > 0 or st['n_inf'] > 0) else "  ok"
        print(f"   {name:>10s}: n_params={st['n_params']:>5d}  "
              f"n_nan={st['n_nan']:>4d}  n_inf={st['n_inf']:>4d}  "
              f"min={st['min']:.3e}  max={st['max']:.3e}{flag}")

    # 2. Subnet outputs at the first batch element's q at t=0
    q0 = batch_x_cat[0, 0, :9]
    print(f"\n[2] Subnet outputs at q = batch[0, 0, :9] = "
          f"{np.asarray(q0).tolist()[:3]}…:")
    sub_out = _subnet_outputs_at_q(model_pre_step, q0)
    for name, st in sub_out.items():
        print(f"   {name:>10s}: {st}")

    # 3. Gradient-NaN per subnet
    print("\n[3] Gradient stats per subnet:")
    grad_report = _per_subnet_nan_report(grads)
    for name, st in grad_report.items():
        flag = "  ⚠" if (st['n_nan'] > 0 or st['n_inf'] > 0) else "  ok"
        print(f"   {name:>10s}: n_grads={st['n_params']:>5d}  "
              f"n_nan={st['n_nan']:>4d}  n_inf={st['n_inf']:>4d}  "
              f"min={st['min']:.3e}  max={st['max']:.3e}{flag}")

    # 4. Substep replay — eager Python loop, b=0 only (representative)
    print("\n[4] Eager substep replay for batch[0]:")
    x0_12_np = np.asarray(batch_x_cat[0, 0, :12])
    u_np     = np.asarray(batch_x_cat[0, 0, 12:15])
    history = _substep_replay(model_pre_step, x0_12_np, u_np, h,
                              np.asarray(dW_batch_np[0]))
    if history:
        for h_ in history[:5]:
            print(f"   {h_}")
        if len(history) > 5:
            print(f"   ... ({len(history) - 5} more substeps; full trace in dump)")
        last = history[-1]
        if last.get('drift_has_nan'):
            print(f"   FIRST FAILURE: drift produced NaN/Inf at "
                  f"outer={last['outer']} substep={last['substep']}")
        elif last.get('stoch_has_nan'):
            print(f"   FIRST FAILURE: stochastic_increment produced NaN/Inf at "
                  f"outer={last['outer']} substep={last['substep']}")
        elif last.get('post_step_has_nan'):
            print(f"   FIRST FAILURE: integrator output non-finite after "
                  f"outer={last['outer']} substep={last['substep']}")

    # 5. Dump.
    # Model + grads are eqx pytrees containing un-pickleable static leaves
    # (e.g. activation function references) — serialise their array leaves
    # via eqx.tree_serialise_leaves to a sibling .eqx file, and pickle only
    # the safely-serialisable diagnostics.
    base = os.path.join(dump_dir, f"nan_step_{step_idx}")
    eqx.tree_serialise_leaves(base + "_model.eqx", model_pre_step)
    eqx.tree_serialise_leaves(base + "_grads.eqx", grads)

    payload = dict(
        step_idx=step_idx,
        loss_val=float(loss_val) if np.isfinite(loss_val) else str(loss_val),
        l2_val=float(l2_val) if np.isfinite(l2_val) else str(l2_val),
        geo_val=float(geo_val) if np.isfinite(geo_val) else str(geo_val),
        grad_norm=float(grad_norm) if np.isfinite(grad_norm) else str(grad_norm),
        h=h,
        args=vars(args),
        subnet_param_report=subnet_report,
        subnet_outputs_at_q0=sub_out,
        grad_param_report=grad_report,
        substep_history=history,
        batch_x_cat=np.asarray(batch_x_cat),
        dW_batch=np.asarray(dW_batch_np),
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
# M_net pretraining (anchor near M⁻¹ = (1/(m·l²)) · I₃)
# ─────────────────────────────────────────────────────────────────────

def pretrain_M_net(model, q_samples, n_steps, lr, print_every,
                   m: float = 1.0, l: float = 1.0):
    if n_steps <= 0:
        return model
    target_scale = 1.0 / (m * l * l)
    print(f"\nPretraining M_net for {n_steps} steps "
          f"(lr={lr}, target = {target_scale:.3f}·I₃)")
    print(f"  using {q_samples.shape[0]} q samples drawn from training data")

    M_params, M_static = eqx.partition(model.M_net, eqx.is_array)
    optimizer = optax.adamw(learning_rate=lr, weight_decay=1e-4)
    opt_state = optimizer.init(M_params)
    target = target_scale * jnp.eye(3, dtype=q_samples.dtype)

    def pre_loss(M_params_inner, q_batch):
        M_net = eqx.combine(M_params_inner, M_static)
        return jnp.mean((jax.vmap(M_net)(q_batch) - target) ** 2)

    @eqx.filter_jit
    def step(M_params_inner, opt_state, q_batch):
        loss_val, grads = jax.value_and_grad(pre_loss)(M_params_inner, q_batch)
        updates, opt_state = optimizer.update(grads, opt_state, M_params_inner)
        return optax.apply_updates(M_params_inner, updates), opt_state, loss_val

    initial_loss = None
    for s in range(n_steps):
        M_params, opt_state, loss_val = step(M_params, opt_state, q_samples)
        if initial_loss is None:
            initial_loss = float(loss_val)
        if s % max(1, print_every) == 0 or s == n_steps - 1:
            print(f"  pretrain step {s:>4d}: loss={float(loss_val):.3e}")

    new_M_net = eqx.combine(M_params, M_static)
    M_check = new_M_net(q_samples[0])
    deviation = float(jnp.max(jnp.abs(M_check - target)))
    print(f"  pretrain done. initial loss={initial_loss:.3e}  "
          f"final loss={float(loss_val):.3e}  max|M(q₀) − target|={deviation:.3e}")

    new_model = eqx.tree_at(lambda mod: mod.M_net, model, new_M_net)
    if _has_nan_or_inf(new_model):
        raise RuntimeError("M_net pretraining produced NaN/Inf params — abort.")
    return new_model


# ─────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────

def train(args):
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu))
    device = jax.devices()[0]
    if args.verbose:
        print(f"Start training (SDE+debug) num_points={args.num_points} "
              f"n_substeps={args.n_substeps} eval_every={args.eval_every} "
              f"device={device}")

    key = jax.random.PRNGKey(args.seed)
    key_model, key_dW = jax.random.split(key)

    model = DissipativeSO3HamSDE(key=key_model, u_dim=3, init_gain=args.init_gain)
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

    dt   = float(t_eval_np[1] - t_eval_np[0])
    h    = dt / args.n_substeps
    n_outer = args.num_points - 1   # number of intervals between snapshots
    B_train = train_x_cat.shape[1]
    B_test  = test_x_cat.shape[1]

    print(f"  dt = {dt:.4f}  h = {h:.4f}  n_outer (per window) = {n_outer}  "
          f"n_substeps = {args.n_substeps}")
    print(f"  train batch B = {B_train}, test batch B = {B_test}")

    # ── M_net pretraining ──
    if args.pretrain_M_steps > 0:
        q_pretrain = train_x_cat.reshape(-1, 15)[:, :9]
        model = pretrain_M_net(
            model=model, q_samples=q_pretrain,
            n_steps=args.pretrain_M_steps, lr=args.pretrain_M_lr,
            print_every=args.pretrain_M_print_every, m=1.0, l=1.0,
        )

    # Sanity: any NaN in init params?
    if _has_nan_or_inf(model):
        raise RuntimeError("Model has NaN/Inf params at start of training — abort.")

    # ── Optimiser ──
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(learning_rate=args.learn_rate, weight_decay=1e-4),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # ── Train step (jitted) ──
    @eqx.filter_jit
    def train_step(model, opt_state, batch_x_cat, dW_batch):
        (loss_val, (l2, geo)), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(model, batch_x_cat, h, dW_batch)
        grad_norm = _global_grad_norm(grads)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        new_model = eqx.apply_updates(model, updates)
        return new_model, opt_state, loss_val, l2, geo, grad_norm, grads

    @eqx.filter_jit
    def eval_step(model, batch_x_cat, dW_batch):
        loss_val, (l2, geo) = loss_fn(model, batch_x_cat, h, dW_batch)
        # diagnostics on test prediction
        x0_15 = batch_x_cat[0]
        x12_init = x0_15[:, :12]; u_const = x0_15[:, 12:15]
        traj_12 = _batched_rollout(model, x12_init, u_const, h, dW_batch)
        traj_15 = _pad_with_u(traj_12, u_const)
        return loss_val, l2, geo, traj_15

    debug_dump_dir = (args.debug_dump_dir or
                      os.path.join(args.save_dir, 'debug'))

    stats = {
        'train_loss': [], 'train_l2_loss': [], 'train_geo_loss': [],
        'train_grad_norm': [], 'step_time': [],
        'eval_step': [],
        'test_loss': [], 'test_l2_loss': [], 'test_geo_loss': [],
        'eval_M_loss': [], 'eval_V_loss': [],
        'eval_Dw_loss': [], 'eval_g_loss': [],
    }

    os.makedirs(args.save_dir, exist_ok=True)
    label = '-so3hamSDE'
    stats_path = (f'{args.save_dir}/{args.name}{label}-{args.num_points}p-stats.pkl')

    for step_idx in range(args.total_steps + 1):
        t0 = time.time()
        key_dW, sub = jax.random.split(key_dW)
        dW_batch = _sample_dW(sub, B_train, n_outer, args.n_substeps,
                              h, train_x_cat.dtype)

        # snapshot model+opt_state BEFORE the step so the dump can be replayed
        model_pre = model

        new_model, new_opt_state, loss_val, l2_val, geo_val, grad_norm, grads = (
            train_step(model, opt_state, train_x_cat, dW_batch)
        )

        # Per-step finite-check (the cheap NaN guard)
        loss_f = float(loss_val)
        gnorm_f = float(grad_norm)
        if (not np.isfinite(loss_f)) or (not np.isfinite(gnorm_f)):
            diagnose_and_dump(
                model_pre_step=model_pre,
                batch_x_cat=train_x_cat,
                dW_batch_np=dW_batch,
                grads=grads,
                opt_state=opt_state,
                step_idx=step_idx,
                loss_val=loss_f, l2_val=float(l2_val), geo_val=float(geo_val),
                grad_norm=gnorm_f,
                h=h, dump_dir=debug_dump_dir, args=args,
            )
            print("Aborting training — see dump above for post-mortem.")
            return model_pre, stats

        model = new_model
        opt_state = new_opt_state
        step_time = time.time() - t0

        stats['train_loss'].append(loss_f)
        stats['train_l2_loss'].append(float(l2_val))
        stats['train_geo_loss'].append(float(geo_val))
        stats['train_grad_norm'].append(gnorm_f)
        stats['step_time'].append(step_time)

        if step_idx % args.eval_every == 0:
            # Even periodic check that params didn't quietly go NaN.
            if _has_nan_or_inf(model):
                diagnose_and_dump(
                    model_pre_step=model_pre,
                    batch_x_cat=train_x_cat, dW_batch_np=dW_batch,
                    grads=grads, opt_state=opt_state, step_idx=step_idx,
                    loss_val=loss_f, l2_val=float(l2_val), geo_val=float(geo_val),
                    grad_norm=gnorm_f, h=h,
                    dump_dir=debug_dump_dir, args=args,
                )
                print("Aborting — model params went NaN/Inf after step.")
                return model_pre, stats

            key_dW, sub = jax.random.split(key_dW)
            dW_test = _sample_dW(sub, B_test, n_outer, args.n_substeps,
                                 h, test_x_cat.dtype)
            test_loss, test_l2, test_geo, test_traj_15 = eval_step(
                model, test_x_cat, dW_test
            )
            test_pack = jax.device_get(jnp.stack([test_loss, test_l2, test_geo]))
            subnet = subnet_physics_mse(
                model, test_traj_15,
                m=1.0, l=1.0, g=9.81,
                friction_coeff=args.friction_coeff,
                varying_friction=args.varying_friction,
            )

            stats['eval_step'].append(step_idx)
            stats['test_loss'].append(float(test_pack[0]))
            stats['test_l2_loss'].append(float(test_pack[1]))
            stats['test_geo_loss'].append(float(test_pack[2]))
            stats['eval_M_loss'].append(subnet['M_loss'])
            stats['eval_V_loss'].append(subnet['V_loss'])
            stats['eval_Dw_loss'].append(subnet['Dw_loss'])
            stats['eval_g_loss'].append(subnet['g_loss'])

            print(f"[step {step_idx:>6d}]")
            print(f"  train: total={loss_f:.4e}  L2={float(l2_val):.4e}  "
                  f"geo={float(geo_val):.4e}  ‖g‖={gnorm_f:.3e}  "
                  f"step_time={step_time*1e3:.1f}ms")
            print(f"  test : total={test_pack[0]:.4e}  L2={test_pack[1]:.4e}  "
                  f"geo={test_pack[2]:.4e}")
            print(f"  subnet MSE  M={subnet['M_loss']:.3e}  "
                  f"V={subnet['V_loss']:.3e}  Dw={subnet['Dw_loss']:.3e}  "
                  f"g={subnet['g_loss']:.3e}")

            ckpt = (f'{args.save_dir}/{args.name}{label}-{args.num_points}p-'
                    f'{step_idx}.eqx')
            eqx.tree_serialise_leaves(ckpt, model)
            to_pickle(stats, stats_path)

    # ── Final per-trajectory eval ──
    print("\nFinal per-trajectory eval ...")
    train_x_full = jnp.asarray(data['x'].astype(np.float32))    # (num_us, T, B, 15)
    test_x_full  = jnp.asarray(data['test_x'].astype(np.float32))
    t_full_np    = data['t'].astype(np.float32)
    n_outer_full = t_full_np.shape[0] - 1

    @eqx.filter_jit
    def full_rollout_one(model, x0_15, u, dW_one):
        traj_12 = lie_heun_sde_rollout(model, x0_15[:12], u, h, dW_one)
        return _pad_with_u(traj_12[:, None, :], u[None, :])[:, 0, :]   # (T, 15)

    def per_us(x_full):
        loss_l, l2_l, geo_l, hat_l = [], [], [], []
        for i in range(x_full.shape[0]):
            x_i = x_full[i]                                  # (T, B, 15)
            B = x_i.shape[1]
            key_local = jax.random.PRNGKey(args.seed + 1000 + i)
            dW = _sample_dW(key_local, B, n_outer_full, args.n_substeps,
                            h, x_i.dtype)
            x_init_15 = x_i[0]; u_const = x_init_15[:, 12:15]
            traj_12 = _batched_rollout(model, x_init_15[:, :12], u_const, h, dW)
            traj_15 = _pad_with_u(traj_12, u_const)          # (T, B, 15)
            total, l2, geo = traj_rotmat_L2_geodesic_loss_safe(
                x_i, traj_15, split=(9, 3, 3))
            loss_l.append(total); l2_l.append(l2); geo_l.append(geo)
            hat_l.append(np.asarray(traj_15))
        return loss_l, l2_l, geo_l, hat_l

    train_loss_l, train_l2_l, train_geo_l, train_hat = per_us(train_x_full)
    test_loss_l,  test_l2_l,  test_geo_l,  test_hat  = per_us(test_x_full)

    def _per_traj(ll):
        return np.sum(np.asarray(jnp.concatenate(ll, axis=1)), axis=0)

    train_loss_pt = _per_traj(train_loss_l); test_loss_pt = _per_traj(test_loss_l)
    train_l2_pt   = _per_traj(train_l2_l);   test_l2_pt   = _per_traj(test_l2_l)
    train_geo_pt  = _per_traj(train_geo_l);  test_geo_pt  = _per_traj(test_geo_l)
    print(f'Final trajectory train loss {train_loss_pt.mean():.4e} +/- {train_loss_pt.std():.4e}')
    print(f'Final trajectory test  loss {test_loss_pt.mean():.4e} +/- {test_loss_pt.std():.4e}')
    print(f'Final trajectory train l2   {train_l2_pt.mean():.4e} +/- {train_l2_pt.std():.4e}')
    print(f'Final trajectory test  l2   {test_l2_pt.mean():.4e} +/- {test_l2_pt.std():.4e}')
    print(f'Final trajectory train geo  {train_geo_pt.mean():.4e} +/- {train_geo_pt.std():.4e}')
    print(f'Final trajectory test  geo  {test_geo_pt.mean():.4e} +/- {test_geo_pt.std():.4e}')

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
    label = '-so3hamSDE'
    final_ckpt = (f'{args.save_dir}/{args.name}{label}-{args.num_points}p.eqx')
    eqx.tree_serialise_leaves(final_ckpt, model)
    final_stats = (f'{args.save_dir}/{args.name}{label}-{args.num_points}p-stats.pkl')
    print("Saved final stats: ", final_stats)
    to_pickle(stats, final_stats)
