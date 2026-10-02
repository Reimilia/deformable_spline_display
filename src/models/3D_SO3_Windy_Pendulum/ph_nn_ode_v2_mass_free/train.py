"""fp32 production training for 3D windy pendulum SO(3) Hamiltonian NODE.

Differences from the fp64 production train script:
  - dtype defaults to float32
  - imports the fp32-stable network (epsilon=1.0, linalg.solve)
  - imports the fp32-stable loss (arccos clamp)
  - windowed eval (no_grad) every --eval_every steps with subnet diagnostics
  - checkpoint + stats saved every --eval_every steps
  - final per-trajectory eval block runs once at the end (no diagnostics there)
"""
import torch, argparse
import numpy as np
import os, sys
import time

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, '../../../..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src/utils'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'datasets'))
sys.path.insert(0, THIS_FILE_DIR)

from torchdiffeq import odeint

from ode_utils import to_pickle
from subnet_diagnostics import subnet_physics_mse
from windy_pendulum_3d_datagen import get_dataset, arrange_data
from network import DissipativeSO3HamNODE
from loss_utils import (
    rotmat_L2_geodesic_loss_safe as rotmat_L2_geodesic_loss,
    traj_rotmat_L2_geodesic_loss_safe as traj_rotmat_L2_geodesic_loss,
    power_balance_loss,
    consistency_subnet_losses,
)


DEFAULT_SAVE_DIR = os.path.join(THIS_FILE_DIR, 'data', 'run_wp3d_fp32')
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, 'datasets/data/windy_pendulum_3d')


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--learn_rate', default=1e-3, type=float)
    parser.add_argument('--total_steps', default=10000, type=int)
    parser.add_argument('--eval_every', default=50, type=int,
                        help='windowed eval + diagnostics + checkpoint cadence')
    parser.add_argument('--name', default='wp3d', type=str)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--save_dir', default=DEFAULT_SAVE_DIR, type=str)
    parser.add_argument('--data_dir', default=DEFAULT_DATA_DIR, type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_points', type=int, default=5)
    parser.add_argument('--solver', default='rk4', type=str)
    parser.add_argument('--init_gain', default=0.5, type=float)

    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--timesteps', type=int, default=20)
    parser.add_argument('--friction_coeff', type=float, default=0.5)
    parser.add_argument('--varying_friction', action='store_true')
    parser.add_argument('--external_force_type', type=str, default='sine',
                        choices=['sine', 'square', 'random', 'constant'])
    parser.add_argument('--external_force_std', type=float, default=0.0)
    parser.add_argument('--wind_force_std', type=float, default=0.0)
    parser.add_argument('--obs_noise_std', type=float, default=0.0)
    parser.add_argument('--random_u', action='store_true')

    # M_net pretraining (anchors M_net near the true target M⁻¹ = I₃ for
    # m = l = 1 spherical pendulum, before main joint training begins).
    parser.add_argument('--pretrain_M_steps', type=int, default=200,
                        help='gradient steps to pretrain M_net to identity (0 disables)')
    parser.add_argument('--pretrain_M_lr', type=float, default=1e-3,
                        help='learning rate for M_net pretraining')
    parser.add_argument('--pretrain_M_print_every', type=int, default=20,
                        help='print interval during M_net pretraining')

    # Physics-informed auxiliary losses (off by default).
    # Suggested starting range per the PH math reference: [0.01, 0.1].
    parser.add_argument('--lambda_power', type=float, default=0.0,
                        help='weight for power-balance loss (Loss 1); 0 disables')
    # Per-subnetwork back-solving (Loss 2): isolates V, B, D individually so
    # each loss localizes which subnet is most to blame.
    parser.add_argument('--lambda_V', type=float, default=0.0,
                        help='weight for V back-solving loss (Loss 2 / V); 0 disables')
    parser.add_argument('--lambda_B', type=float, default=0.0,
                        help='weight for B (input matrix) back-solving loss; 0 disables')
    parser.add_argument('--lambda_D', type=float, default=0.0,
                        help='weight for D (dissipation) back-solving loss; 0 disables')

    return parser.parse_args()


def get_model_parm_nums(model):
    return sum(p.nelement() for p in model.parameters())


def _fmt_num(x):
    """Format a number for filesystem paths: 0.01 → 0p01, -1.0 → n1, 1e-3 → 0p001."""
    s = f"{x:g}"
    return s.replace('.', 'p').replace('-', 'n').replace('+', '')


def build_run_name(args):
    """Build a run-folder name encoding the training specs + a YYMMDD-HHMM stamp.
    Example:
      obs0p01_fric0p5_wind0p0_ext0p0-sine_lP1_lV1_lB1_lD1_lr0p001_s10000_np5_smp64_T20_rk4_seed0_260504-1530
    """
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
        f"{args.solver}",
        f"seed{args.seed}",
    ]
    if args.varying_friction:
        parts.append('varfric')
    if args.random_u:
        parts.append('randu')
    stamp = time.strftime('%y%m%d-%H%M%S')
    return '_'.join(parts) + '_' + stamp


def _state_dict(model):
    return (model._orig_mod if hasattr(model, '_orig_mod') else model).state_dict()


def _inner(model):
    """Unwrap a possibly torch.compile-wrapped model."""
    return model._orig_mod if hasattr(model, '_orig_mod') else model


def pretrain_M_net(model, q_samples, n_steps, lr, print_every,
                   m=1.0, l=1.0):
    """Pretrain M_net to output the true mass-inverse, M⁻¹(q) = (1/(m·l²))·I₃.

    For the spherical pendulum with m = l = 1, the target is exactly I₃ for
    any q. We minimize MSE between M_net(q) and this target on q values
    drawn from the training distribution. Only M_net's parameters are
    updated; V_net / Dw_net / g_net stay at random init.

    Pretraining anchors M_net near a sane starting point so the joint
    training doesn't drift M_net to pathological values (e.g. entries of
    ±50, which we observed caused trajectory blowup at step ~7500).
    """
    if n_steps <= 0:
        return

    inner = _inner(model)
    target_scale = 1.0 / (m * l * l)
    print(f"\nPretraining M_net for {n_steps} steps (lr={lr}, target = {target_scale:.3f}·I₃)")
    print(f"  using {q_samples.shape[0]} q samples drawn from training data")

    optim = torch.optim.Adam(inner.M_net.parameters(), lr=lr, weight_decay=1e-4)
    I3 = torch.eye(3, device=q_samples.device, dtype=q_samples.dtype)
    target = (target_scale * I3).unsqueeze(0).expand(q_samples.shape[0], 3, 3)

    q_no_grad = q_samples.detach()  # don't carry training graph into pretraining

    initial_loss = None
    for step in range(n_steps):
        M_pred = inner.M_net(q_no_grad)
        loss = (M_pred - target).pow(2).mean()
        loss.backward()
        optim.step()
        optim.zero_grad()
        if initial_loss is None:
            initial_loss = loss.item()
        if step % max(1, print_every) == 0 or step == n_steps - 1:
            print(f"  pretrain step {step:>4d}: loss={loss.item():.3e}")

    # Final summary
    with torch.no_grad():
        M_check = inner.M_net(q_no_grad[:1])
        deviation = (M_check - I3 * target_scale).abs().max().item()
    print(f"  pretrain done. initial loss={initial_loss:.3e}  final loss={loss.item():.3e}  "
          f"max|M(q₀) − target|={deviation:.3e}")


def train(args):
    float_type = torch.float32
    torch.set_default_dtype(torch.float32)

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Per-run subfolder so each invocation gets its own checkpoint directory.
    # The default save_dir acts as the parent for all runs.
    run_name = build_run_name(args)
    args.save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Run dir : {args.save_dir}")

    if args.verbose:
        print(f"Start training (fp32) num_points={args.num_points} solver={args.solver} "
              f"eval_every={args.eval_every} device={device}")

    model = DissipativeSO3HamNODE(device=device, u_dim=3, init_gain=args.init_gain).to(device)

    print(f'model contains {get_model_parm_nums(model)} parameters')

    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)

    us = (
        (0.0, 0.0, 0.0),
        (-1.0, -1.0, -1.0),
        (1.0, 1.0, 1.0),
        (-2.0, -2.0, -2.0),
        (2.0, 2.0, 2.0),
    )

    data, _ = get_dataset(
        seed=args.seed,
        samples=args.samples,
        timesteps=args.timesteps,
        save_dir=args.data_dir,
        us=us,
        ori_rep="rotmat",
        friction_coeff=args.friction_coeff,
        varying_friction=args.varying_friction,
        external_force_type=args.external_force_type,
        external_force_std=args.external_force_std,
        wind_force_std=args.wind_force_std,
        obs_noise_std=args.obs_noise_std,
        random_u=args.random_u,
    )

    train_x, t_eval = arrange_data(data['x'], data['t'], num_points=args.num_points)
    test_x, _ = arrange_data(data['test_x'], data['t'], num_points=args.num_points)
    train_x_cat = np.concatenate(train_x, axis=1)
    test_x_cat = np.concatenate(test_x, axis=1)

    train_x_cat = torch.tensor(train_x_cat, requires_grad=True, dtype=float_type).to(device)
    test_x_cat = torch.tensor(test_x_cat, requires_grad=True, dtype=float_type).to(device)
    t_eval = torch.tensor(t_eval, requires_grad=True, dtype=float_type).to(device)

    # ── M_net pretraining ─────────────────────────────────────────────
    # Sample q values from the entire training tensor (all time slices,
    # all batch indices) — gives the widest coverage of the SO(3) region
    # the model will see during joint training.
    if args.pretrain_M_steps > 0:
        # train_x_cat shape: (num_points, batch, 15) → flatten to (num_points·batch, 9)
        q_pretrain = train_x_cat.detach().reshape(-1, 15)[:, :9]
        pretrain_M_net(
            model=model,
            q_samples=q_pretrain,
            n_steps=args.pretrain_M_steps,
            lr=args.pretrain_M_lr,
            print_every=args.pretrain_M_print_every,
            m=1.0, l=1.0,
        )

    split = [9, 3, 3]

    stats = {
        'train_loss': [], 'train_l2_loss': [], 'train_geo_loss': [],
        'train_power_loss': [],
        'train_V_cons_loss': [], 'train_B_cons_loss': [], 'train_D_cons_loss': [],
        'forward_time': [], 'backward_time': [], 'nfe': [],
        # Eval (windowed) — recorded every eval_every steps
        'eval_step': [],
        'test_loss': [], 'test_l2_loss': [], 'test_geo_loss': [],
        'eval_M_loss': [], 'eval_V_loss': [],
        'eval_Dw_loss': [], 'eval_g_loss': [],
    }

    dt_train = (t_eval[1] - t_eval[0]).detach().item()

    os.makedirs(args.save_dir, exist_ok=True)
    label = '-so3ham'
    stats_path = f'{args.save_dir}/{args.name}{label}-{args.solver}-{args.num_points}p-stats.pkl'

    # ── #4: per-step loss buffer kept on GPU; drained once per eval_every
    #        to amortize CUDA syncs (avoids 3-6 syncs per training step).
    loss_buffer = []   # list of (3,) tensors: [train_loss, train_l2, train_geo]
    fwd_buffer = []    # python floats — no GPU sync needed for time.time()
    bwd_buffer = []
    nfe_buffer = []    # python ints from model.nfe — no sync

    for step in range(args.total_steps + 1):
        # ── train step ───────────────────────────────────────────────
        t = time.time()
        train_x_hat = odeint(model, train_x_cat[0, :, :], t_eval, method=args.solver)
        forward_time = time.time() - t

        target = train_x_cat[1:, :, :]
        target_hat = train_x_hat[1:, :, :]
        train_loss, train_l2_loss, train_geo_loss = rotmat_L2_geodesic_loss(
            target, target_hat, split=split
        )

        # Physics-informed aux losses on the GT trajectory. Computed only
        # when at least one weight is non-zero so the default training path
        # is untouched. The three subnet losses share one autograd.grad call
        # over H, so we always compute all three when any λ_{V,B,D} > 0.
        if args.lambda_power > 0.0:
            L_power = power_balance_loss(model, train_x_cat, dt_train)
        else:
            L_power = torch.zeros((), device=device, dtype=float_type)

        if args.lambda_V > 0.0 or args.lambda_B > 0.0 or args.lambda_D > 0.0:
            L_V, L_B, L_D = consistency_subnet_losses(model, train_x_cat, dt_train)
        else:
            L_V = torch.zeros((), device=device, dtype=float_type)
            L_B = torch.zeros((), device=device, dtype=float_type)
            L_D = torch.zeros((), device=device, dtype=float_type)

        total_loss = (train_loss
                      + args.lambda_power * L_power
                      + args.lambda_V     * L_V
                      + args.lambda_B     * L_B
                      + args.lambda_D     * L_D)

        t = time.time()
        total_loss.backward()
        optim.step()
        optim.zero_grad()
        backward_time = time.time() - t

        # #4: stack the losses as a single tensor and keep on GPU (no sync here)
        loss_buffer.append(torch.stack(
            [total_loss.detach(), train_l2_loss.detach(), train_geo_loss.detach(),
             L_power.detach(), L_V.detach(), L_B.detach(), L_D.detach()]
        ))
        fwd_buffer.append(forward_time)
        bwd_buffer.append(backward_time)
        nfe = getattr(model, 'nfe', getattr(getattr(model, '_orig_mod', model), 'nfe', 0))
        nfe_buffer.append(nfe)

        # ── eval + diagnostics + checkpoint every eval_every steps ──
        if step % args.eval_every == 0:
            with torch.no_grad():
                test_x_hat = odeint(model, test_x_cat[0, :, :], t_eval, method=args.solver)
                tgt = test_x_cat[1:, :, :]
                tgt_hat = test_x_hat[1:, :, :]
                test_loss, test_l2_loss, test_geo_loss = rotmat_L2_geodesic_loss(
                    tgt, tgt_hat, split=split
                )
                subnet = subnet_physics_mse(
                    model, test_x_hat,
                    m=1.0, l=1.0, g=9.81,
                    friction_coeff=args.friction_coeff,
                    varying_friction=args.varying_friction,
                )

            # #4: drain GPU loss buffer with a SINGLE .cpu() sync for all
            # buffered steps (typically 50 entries × 7 losses).
            if loss_buffer:
                drained = torch.stack(loss_buffer, dim=0).cpu().numpy()  # (N, 7)
                stats['train_loss'].extend(drained[:, 0].tolist())
                stats['train_l2_loss'].extend(drained[:, 1].tolist())
                stats['train_geo_loss'].extend(drained[:, 2].tolist())
                stats['train_power_loss'].extend(drained[:, 3].tolist())
                stats['train_V_cons_loss'].extend(drained[:, 4].tolist())
                stats['train_B_cons_loss'].extend(drained[:, 5].tolist())
                stats['train_D_cons_loss'].extend(drained[:, 6].tolist())
                stats['forward_time'].extend(fwd_buffer)
                stats['backward_time'].extend(bwd_buffer)
                stats['nfe'].extend(nfe_buffer)
                loss_buffer = []; fwd_buffer = []; bwd_buffer = []; nfe_buffer = []

            # #4: stack the 3 test losses + 4 subnet losses, sync once
            test_pack = torch.stack([
                test_loss.detach(), test_l2_loss.detach(), test_geo_loss.detach()
            ]).cpu().numpy()
            # most-recent train losses (drained from the GPU buffer above)
            train_total = stats['train_loss'][-1]
            train_l2    = stats['train_l2_loss'][-1]
            train_geo   = stats['train_geo_loss'][-1]

            stats['eval_step'].append(step)
            stats['test_loss'].append(float(test_pack[0]))
            stats['test_l2_loss'].append(float(test_pack[1]))
            stats['test_geo_loss'].append(float(test_pack[2]))
            stats['eval_M_loss'].append(subnet['M_loss'])
            stats['eval_V_loss'].append(subnet['V_loss'])
            stats['eval_Dw_loss'].append(subnet['Dw_loss'])
            stats['eval_g_loss'].append(subnet['g_loss'])

            train_pow = stats['train_power_loss'][-1]
            train_LV  = stats['train_V_cons_loss'][-1]
            train_LB  = stats['train_B_cons_loss'][-1]
            train_LD  = stats['train_D_cons_loss'][-1]

            print(f"[step {step:>6d}]")
            print(f"  train: total={train_total:.4e}  "
                  f"L2={train_l2:.4e}  geo={train_geo:.4e}  "
                  f"power={train_pow:.4e}")
            print(f"  cons : L_V={train_LV:.4e}  L_B={train_LB:.4e}  L_D={train_LD:.4e}")
            print(f"  test : total={test_pack[0]:.4e}  "
                  f"L2={test_pack[1]:.4e}  geo={test_pack[2]:.4e}")
            print(f"  subnet MSE  M={subnet['M_loss']:.3e}  "
                  f"V={subnet['V_loss']:.3e}  Dw={subnet['Dw_loss']:.3e}  "
                  f"g={subnet['g_loss']:.3e}  | nfe={nfe}")

            # checkpoint + stats save
            ckpt = f'{args.save_dir}/{args.name}{label}-{args.solver}-{args.num_points}p-{step}.tar'
            torch.save(_state_dict(model), ckpt)
            to_pickle(stats, stats_path)

    # ── Final per-trajectory eval (no subnet diagnostics) ─────────────
    train_x_full = torch.tensor(data['x'], requires_grad=True, dtype=float_type).to(device)
    test_x_full = torch.tensor(data['test_x'], requires_grad=True, dtype=float_type).to(device)
    t_full = torch.tensor(data['t'], requires_grad=True, dtype=float_type).to(device)

    train_loss_l, test_loss_l = [], []
    train_l2_l, test_l2_l = [], []
    train_geo_l, test_geo_l = [], []
    train_data_hat, test_data_hat = [], []

    for i in range(train_x_full.shape[0]):
        train_x_hat = odeint(model, train_x_full[i, 0, :, :], t_full, method=args.solver)
        total_loss, l2_loss, geo_loss = traj_rotmat_L2_geodesic_loss(
            train_x_full[i, :, :, :], train_x_hat, split=split
        )
        train_loss_l.append(total_loss); train_l2_l.append(l2_loss); train_geo_l.append(geo_loss)
        train_data_hat.append(train_x_hat.detach().cpu().numpy())

        test_x_hat = odeint(model, test_x_full[i, 0, :, :], t_full, method=args.solver)
        total_loss, l2_loss, geo_loss = traj_rotmat_L2_geodesic_loss(
            test_x_full[i, :, :, :], test_x_hat, split=split
        )
        test_loss_l.append(total_loss); test_l2_l.append(l2_loss); test_geo_l.append(geo_loss)
        test_data_hat.append(test_x_hat.detach().cpu().numpy())

    def _per_traj(loss_list):
        return torch.sum(torch.cat(loss_list, dim=1), dim=0)

    train_loss_pt = _per_traj(train_loss_l); test_loss_pt = _per_traj(test_loss_l)
    train_l2_pt = _per_traj(train_l2_l); test_l2_pt = _per_traj(test_l2_l)
    train_geo_pt = _per_traj(train_geo_l); test_geo_pt = _per_traj(test_geo_l)

    print('Final trajectory train loss {:.4e} +/- {:.4e}\n'
          'Final trajectory test loss  {:.4e} +/- {:.4e}'.format(
              train_loss_pt.mean().item(), train_loss_pt.std().item(),
              test_loss_pt.mean().item(), test_loss_pt.std().item()))
    print('Final trajectory train l2 loss {:.4e} +/- {:.4e}\n'
          'Final trajectory test l2 loss  {:.4e} +/- {:.4e}'.format(
              train_l2_pt.mean().item(), train_l2_pt.std().item(),
              test_l2_pt.mean().item(), test_l2_pt.std().item()))
    print('Final trajectory train geo loss {:.4e} +/- {:.4e}\n'
          'Final trajectory test geo loss  {:.4e} +/- {:.4e}'.format(
              train_geo_pt.mean().item(), train_geo_pt.std().item(),
              test_geo_pt.mean().item(), test_geo_pt.std().item()))

    stats['traj_train_loss'] = train_loss_pt.detach().cpu().numpy()
    stats['traj_test_loss'] = test_loss_pt.detach().cpu().numpy()
    stats['train_x'] = train_x_full.detach().cpu().numpy()
    stats['test_x'] = test_x_full.detach().cpu().numpy()
    stats['train_x_hat'] = np.array(train_data_hat)
    stats['test_x_hat'] = np.array(test_data_hat)
    stats['t_eval'] = t_full.detach().cpu().numpy()
    return model, stats


if __name__ == "__main__":
    args = get_args()
    model, stats = train(args)

    os.makedirs(args.save_dir, exist_ok=True)
    label = '-so3ham'
    final_ckpt = f'{args.save_dir}/{args.name}{label}-{args.solver}-{args.num_points}p.tar'
    torch.save(_state_dict(model), final_ckpt)
    final_stats = f'{args.save_dir}/{args.name}{label}-{args.solver}-{args.num_points}p-stats.pkl'
    print("Saved final stats: ", final_stats)
    to_pickle(stats, final_stats)
