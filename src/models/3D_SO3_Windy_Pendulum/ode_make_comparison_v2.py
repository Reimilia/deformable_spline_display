"""Cross-framework comparison: ph_nn_ode_v2 (torch) vs ph_gp_ode_v2 (JAX).

Compares four hand-picked runs (defaults below — override via --runs):

    nn_ode_old : ph_nn_ode_v2  λ=(0,0,0,0)
    nn_ode_new : ph_nn_ode_v2  λ=(0.5, 0.5, 0.5, 0.5)
    gp_ode_old : ph_gp_ode_v2  λ=(0,0,0,0)
    gp_ode_new : ph_gp_ode_v2  λ=(0.5, 0.5, 0.5, 0.5)

Output PDF mirrors `ph_nn_ode_v2/make_comparison_pdf.py`:
    1. Summary table
    2-7. Loss-curve overlays (where available)
    8. Trajectory MSE (geo² + L2) vs GT, ensemble
    9. Energy ensemble + single
    10. SO(3) violation (det, orth)
    11. State trajectories (Euler + ω) ensemble + single
    12. Phase portraits
    13. Subnet evolution (M⁻¹, V, D, B) along 5 GT trajectories
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, "../../.."))
NN_ODE_V2_DIR = os.path.join(THIS_FILE_DIR, "ph_nn_ode_v2")
GP_ODE_V2_DIR = os.path.join(THIS_FILE_DIR, "ph_gp_ode_v2")
for p in (PROJECT_ROOT,
          os.path.join(PROJECT_ROOT, "src/utils"),
          os.path.join(PROJECT_ROOT, "datasets"),
          os.path.join(PROJECT_ROOT, "envs"),
          NN_ODE_V2_DIR,
          GP_ODE_V2_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from envs.windy_pendulum_3d import windy_pendulum_3d


# ──────────────────────────────────────────────────────────────────────────
# Default run dirs (the 4 runs the user asked to compare)
# ──────────────────────────────────────────────────────────────────────────

DEFAULT_RUNS = [
    # (kind, run_dir, label_override_or_None)
    ("nn_ode", os.path.join(
        NN_ODE_V2_DIR, "data/run_wp3d_fp32",
        "obs0p05_fric0p5_wind0_ext0-sine_lP0_lV0_lB0_lD0_lr0p001_s10000_np5_smp64_T20_rk4_seed0_fixM_260504-232931"
    ), "nn_ode_old"),
    ("nn_ode", os.path.join(
        NN_ODE_V2_DIR, "data/run_wp3d_fp32",
        "obs0p05_fric0p5_wind0_ext0-sine_lP0p5_lV0p5_lB0p5_lD0p5_lr0p001_s10000_np5_smp64_T20_rk4_seed0_fixM_260504-225515"
    ), "nn_ode_new"),
    ("gp_ode", os.path.join(
        GP_ODE_V2_DIR, "data/run_wp3d_jax",
        "obs0p05_fric0p5_wind0_ext0-sine_lP0_lV0_lB0_lD0_lr0p001_s10000_np5_smp64_T20_seed0_w_260505-144433"
    ), "gp_ode_old"),
    ("gp_ode", os.path.join(
        GP_ODE_V2_DIR, "data/run_wp3d_jax",
        "obs0p05_fric0p5_wind0_ext0-sine_lP0p5_lV0p5_lB0p5_lD0p5_lr0p001_s10000_np5_smp64_T20_seed0_w_260505-144220"
    ), "gp_ode_new"),
]

DEFAULT_OUT_PDF = os.path.join(THIS_FILE_DIR, "comparison_v2_4way.pdf")

# Env / rollout settings
ENV_KW = dict(
    g=9.81, m=1.0, l=1.0, dt=0.05,
    friction_coeff=0.5, varying_friction=False,
    external_force_type="sine", external_force_std=0.0,
    wind_force_std=0.0,
)
N_SUBSTEPS = 10
N_OUTER = 200
N_TRAJ_ENSEMBLE = 10
N_SUB_TRAJ = 1
GT_FRICTION = 0.5
GT_M, GT_L, GT_G = 1.0, 1.0, 9.81
I_PERP = GT_M * GT_L * GT_L
I_PARA = I_PERP


# ──────────────────────────────────────────────────────────────────────────
# Run-folder parsing  (lP/lV/lB/lD → bool is_new, fix_M flag, num_points)
# ──────────────────────────────────────────────────────────────────────────

def _str_to_num(s):
    return float(s.replace('n', '-').replace('p', '.'))


def parse_run_dir(name):
    obs_m = re.search(r"obs([^_]+)_", name)
    if not obs_m:
        return None
    obs = _str_to_num(obs_m.group(1))
    lambdas = []
    for k in ('lP', 'lV', 'lB', 'lD'):
        m = re.search(rf"{k}([^_]+)_", name)
        lambdas.append(_str_to_num(m.group(1)) if m else 0.0)
    fix_M = "_fixM_" in name or name.endswith("_fixM")
    np_m = re.search(r"np(\d+)_", name)
    num_points = int(np_m.group(1)) if np_m else 5
    return {
        "obs": obs, "obs_str": f"{obs:g}",
        "is_new": any(l > 0 for l in lambdas),
        "fix_M": fix_M, "num_points": num_points, "lambdas": lambdas,
    }


# ──────────────────────────────────────────────────────────────────────────
# Torch (ph_nn_ode_v2) loading + rollout + subnet eval
# ──────────────────────────────────────────────────────────────────────────

_torch_loaded = False
def _ensure_torch():
    global _torch_loaded, torch, odeint, DissipativeSO3HamNODE, FixedInverseMass
    if _torch_loaded:
        return
    import importlib.util
    import torch as _torch
    from torchdiffeq import odeint as _odeint
    spec = importlib.util.spec_from_file_location(
        "_nn_ode_v2_network",
        os.path.join(NN_ODE_V2_DIR, "network.py"))
    nn_net = importlib.util.module_from_spec(spec)
    sys.modules["_nn_ode_v2_network"] = nn_net
    spec.loader.exec_module(nn_net)
    torch = _torch
    odeint = _odeint
    DissipativeSO3HamNODE = nn_net.DissipativeSO3HamNODE
    FixedInverseMass = nn_net.FixedInverseMass
    _torch_loaded = True


def find_latest_ckpt_nn(run_dir, num_points):
    pat = re.compile(rf"wp3d-so3ham-rk4-{num_points}p-(\d+)\.tar$")
    cands = []
    for fn in os.listdir(run_dir):
        m = pat.match(fn)
        if m:
            cands.append((int(m.group(1)), fn))
    if not cands:
        return None
    cands.sort()
    return os.path.join(run_dir, cands[-1][1])


def find_stats_nn(run_dir, num_points):
    p = os.path.join(run_dir, f"wp3d-so3ham-rk4-{num_points}p-stats.pkl")
    return p if os.path.exists(p) else None


def load_nn_ode(ckpt_path, device, fix_M):
    _ensure_torch()
    model = DissipativeSO3HamNODE(
        device=device, u_dim=3, init_gain=0.5, friction=True,
    ).to(device)
    if fix_M:
        model.M_net = FixedInverseMass(m=GT_M, l=GT_L).to(device).to(torch.float32)
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def rollout_nn_ode(model, R0, omega0, u_const, t_eval, device):
    _ensure_torch()
    x0 = np.concatenate([R0.reshape(-1), omega0, u_const]).astype(np.float32)
    x0_t = torch.tensor(x0[None, :], dtype=torch.float32,
                        device=device, requires_grad=True)
    t_t = torch.tensor(t_eval, dtype=torch.float32, device=device)
    traj = odeint(model, x0_t, t_t, method="rk4")
    return traj[:, 0, :].detach().cpu().numpy()


def eval_subnets_nn(model, traj_12, device):
    _ensure_torch()
    qs = torch.tensor(traj_12[:, :9], dtype=torch.float32, device=device)
    with torch.no_grad():
        M_inv = model.M_net(qs).cpu().numpy()
        V = model.V_net(qs).cpu().numpy().squeeze(-1)
        D = model.Dw_net(qs).cpu().numpy()
        B = model.g_net(qs).cpu().numpy()
    return {"M": M_inv, "V": V, "D": D, "B": B}


# ──────────────────────────────────────────────────────────────────────────
# JAX (ph_gp_ode_v2) loading + rollout + subnet eval
# ──────────────────────────────────────────────────────────────────────────

_jax_loaded = False
def _ensure_jax():
    global _jax_loaded, jax, jnp, eqx, DissipativeSO3HamODE, lie_heun_ode_rollout
    if _jax_loaded:
        return
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax as _jax
    import jax.numpy as _jnp
    import equinox as _eqx
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_gp_ode_v2_network",
        os.path.join(GP_ODE_V2_DIR, "network.py"))
    gp_net = importlib.util.module_from_spec(spec)
    sys.modules["_gp_ode_v2_network"] = gp_net
    spec.loader.exec_module(gp_net)
    from src.utils.JAX.lie_integrator import lie_heun_ode_rollout as _rollout
    jax = _jax
    jnp = _jnp
    eqx = _eqx
    DissipativeSO3HamODE = gp_net.DissipativeSO3HamODE
    lie_heun_ode_rollout = _rollout
    _jax_loaded = True


def find_latest_ckpt_gp(run_dir, num_points):
    pat = re.compile(rf"wp3d-so3hamGPODE-{num_points}p-(\d+)\.eqx$")
    cands = []
    for fn in os.listdir(run_dir):
        m = pat.match(fn)
        if m:
            cands.append((int(m.group(1)), fn))
    if not cands:
        return None
    cands.sort()
    return os.path.join(run_dir, cands[-1][1])


def find_stats_gp(run_dir, num_points):
    p = os.path.join(run_dir, f"wp3d-so3hamGPODE-{num_points}p-stats.pkl")
    return p if os.path.exists(p) else None


def load_gp_ode(ckpt_path, fix_M=False):
    """Load a GP-ODE checkpoint. `fix_M` must match the training-time flag,
    because the pytree structure of `M_net` differs between the two cases
    (PSD_GP_Model vs FixedInverseMass) and equinox's deserialiser is
    structure-sensitive."""
    _ensure_jax()
    template = DissipativeSO3HamODE(
        key=jax.random.PRNGKey(0), u_dim=3, init_gain=0.5, friction=True,
        fix_M=fix_M,
    )
    return eqx.tree_deserialise_leaves(ckpt_path, template)


def rollout_gp_ode(model, R0, omega0, u_const, n_substeps, n_outer, dt):
    _ensure_jax()
    h = dt / n_substeps
    x0_12 = jnp.concatenate([
        jnp.asarray(R0.reshape(-1), dtype=jnp.float32),
        jnp.asarray(omega0,         dtype=jnp.float32),
    ])
    u = jnp.asarray(u_const, dtype=jnp.float32)
    traj_12 = lie_heun_ode_rollout(model, x0_12, u, jnp.float32(h),
                                    n_substeps, n_outer)
    return np.asarray(traj_12)


def eval_subnets_gp(model, traj_12):
    _ensure_jax()
    qs = jnp.asarray(traj_12[:, :9], dtype=jnp.float32)
    def per_state(q):
        return (
            model.M_net (q, inference_mode=True),
            model.V_net (q, inference_mode=True)[0],
            model.Dw_net(q, inference_mode=True),
            model.g_net (q, inference_mode=True),
        )
    M_inv, V, D, B = jax.vmap(per_state)(qs)
    return {
        "M": np.asarray(M_inv),
        "V": np.asarray(V),
        "D": np.asarray(D),
        "B": np.asarray(B),
    }


# ──────────────────────────────────────────────────────────────────────────
# Run discovery / preparation
# ──────────────────────────────────────────────────────────────────────────

def prepare_runs(run_specs, device):
    out = []
    for kind, run_dir, label_override in run_specs:
        if not os.path.isdir(run_dir):
            print(f"  skip (missing dir): {run_dir}")
            continue
        meta = parse_run_dir(os.path.basename(run_dir))
        if meta is None:
            print(f"  skip (unparsed): {run_dir}")
            continue
        if kind == "nn_ode":
            ckpt = find_latest_ckpt_nn(run_dir, meta["num_points"])
            stats_p = find_stats_nn(run_dir, meta["num_points"])
        elif kind == "gp_ode":
            ckpt = find_latest_ckpt_gp(run_dir, meta["num_points"])
            stats_p = find_stats_gp(run_dir, meta["num_points"])
        else:
            print(f"  skip (unknown kind={kind}): {run_dir}")
            continue
        if ckpt is None:
            print(f"  skip (no ckpt): {run_dir}")
            continue
        with open(stats_p, "rb") as f:
            stats = pickle.load(f)
        meta.update({
            "kind": kind, "dir": run_dir, "ckpt": ckpt,
            "stats": stats,
            "label": label_override or
                     f"{kind}_{'new' if meta['is_new'] else 'old'}",
        })
        out.append(meta)
    return out


def assign_colors(runs):
    """Colour by (framework, variant). 4 distinguishable colours."""
    palette = {
        ("nn_ode", False): ("#1f77b4", "--"),  # blue dashed
        ("nn_ode", True):  ("#1f77b4", "-"),   # blue solid
        ("gp_ode", False): ("#d62728", "--"),  # red dashed
        ("gp_ode", True):  ("#d62728", "-"),   # red solid
    }
    for m in runs:
        key = (m["kind"], m["is_new"])
        m["color"], m["linestyle"] = palette.get(key, ("gray", ":"))
    return runs


# ──────────────────────────────────────────────────────────────────────────
# GT rollout + diagnostics
# ──────────────────────────────────────────────────────────────────────────

def rollout_gt(env, R0, omega0, u_const, dW_per_outer):
    n_outer, n_sub, _ = dW_per_outer.shape
    h_sub = env.dt / n_sub
    sigma = env.wind_force_std
    R, omega = R0.copy(), omega0.copy()
    traj = np.zeros((n_outer + 1, 12), dtype=np.float64)
    traj[0, :9] = R.reshape(-1); traj[0, 9:12] = omega
    t = 0.0
    for k in range(n_outer):
        t += env.dt
        w_force = env.update_wind(t)
        for s in range(n_sub):
            R, omega = env._lie_heun_step(
                R, omega, w_force, u_const, h_sub, sigma, dW_per_outer[k, s])
        traj[k + 1, :9] = R.reshape(-1); traj[k + 1, 9:12] = omega
    return traj


def rotmat_to_euler(R_flat):
    Rs = np.asarray(R_flat).reshape(-1, 3, 3)
    R00, R10, R20 = Rs[:, 0, 0], Rs[:, 1, 0], Rs[:, 2, 0]
    R21, R22, R12, R11 = Rs[:, 2, 1], Rs[:, 2, 2], Rs[:, 1, 2], Rs[:, 1, 1]
    sy = np.sqrt(R00 ** 2 + R10 ** 2)
    near = sy < 1e-6
    roll = np.where(near, np.arctan2(-R12, R11), np.arctan2(R21, R22))
    pitch = np.arctan2(-R20, sy)
    yaw = np.where(near, 0.0, np.arctan2(R10, R00))
    return np.stack([roll, pitch, yaw], axis=-1)


def get_energy(traj):
    pe = GT_G * (1.0 - traj[:, 8])
    omega = traj[:, 9:12]
    I = np.diag([I_PERP, I_PERP, I_PARA])
    return 0.5 * np.einsum("ti,ij,tj->t", omega, I, omega) + pe


def so3_orth_residual(traj_12):
    R = traj_12[:, :9].reshape(-1, 3, 3)
    diff = np.einsum("tji,tjk->tik", R, R) - np.eye(3)[None]
    return np.linalg.norm(diff.reshape(-1, 9), axis=1)


def so3_det_residual_abs(traj_12):
    R = traj_12[:, :9].reshape(-1, 3, 3)
    return np.abs(np.linalg.det(R) - 1.0)


def _geodesic_sq(R_pred_flat, R_gt_flat):
    R_pred = R_pred_flat.reshape(*R_pred_flat.shape[:-1], 3, 3)
    R_gt   = R_gt_flat  .reshape(*R_gt_flat  .shape[:-1], 3, 3)
    M = np.einsum("...ji,...jk->...ik", R_pred, R_gt)
    cos_t = np.clip((np.trace(M, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cos_t) ** 2


def _smooth(y, w=21):
    y = np.asarray(y, dtype=np.float64); n = len(y)
    if n < w + 1:
        return y
    cumsum = np.cumsum(np.insert(y, 0, 0.0))
    half = w // 2
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = (cumsum[hi] - cumsum[lo]) / (hi - lo)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Plotting helpers — same look as ph_nn_ode_v2/make_comparison_pdf.py
# ──────────────────────────────────────────────────────────────────────────

def fig_loss_curves_train(runs, key, title, ylab, logy=True, smooth=True):
    """`key` may be a single str or list (sums them). Skip silently if missing."""
    fig, ax = plt.subplots(figsize=(11, 6))
    keys = [key] if isinstance(key, str) else list(key)
    any_data = False
    for m in runs:
        st = m["stats"]
        if st is None:
            continue
        try:
            arr = np.zeros(len(st[keys[0]]))
            for k in keys:
                arr = arr + np.asarray(st[k])
        except KeyError:
            continue
        if smooth:
            arr = _smooth(arr)
        ax.plot(np.arange(len(arr)), arr, lw=1.2,
                color=m["color"], linestyle=m["linestyle"], label=m["label"])
        any_data = True
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("training step"); ax.set_ylabel(ylab); ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if any_data:
        ax.legend(fontsize="x-small", ncol=2, loc="best")
    fig.tight_layout()
    return fig


def fig_loss_curves_eval(runs, key, title, ylab, logy=True):
    fig, ax = plt.subplots(figsize=(11, 6))
    any_data = False
    for m in runs:
        st = m["stats"]
        if st is None or "eval_step" not in st or key not in st:
            continue
        x = np.asarray(st["eval_step"]); y = np.asarray(st[key])
        ax.plot(x, y, lw=1.4, color=m["color"], linestyle=m["linestyle"],
                label=m["label"])
        any_data = True
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("training step"); ax.set_ylabel(ylab); ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if any_data:
        ax.legend(fontsize="x-small", ncol=2, loc="best")
    fig.tight_layout()
    return fig


def fig_energy_ensemble(t_eval, gt_e_all, model_e):
    fig, ax = plt.subplots(figsize=(11, 6))
    gt_m, gt_s = gt_e_all.mean(0), gt_e_all.std(0)
    ax.plot(t_eval, gt_m, "k-", lw=2, label="GT")
    ax.fill_between(t_eval, gt_m - 2 * gt_s, gt_m + 2 * gt_s,
                    color="black", alpha=0.15)
    for label, (em, es, color, ls) in model_e.items():
        ax.plot(t_eval, em, color=color, linestyle=ls, lw=1.6, label=label)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Energy (J)")
    ax.set_title("Hamiltonian energy — ensemble mean")
    ax.legend(fontsize="x-small", ncol=2); ax.grid(True, alpha=0.3)
    fig.tight_layout(); return fig


def fig_energy_single(t_eval, gt_single_e, model_single_e):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(t_eval, gt_single_e, "k-", lw=2, label="GT")
    for label, (e, color, ls) in model_single_e.items():
        ax.plot(t_eval, e, color=color, linestyle=ls, lw=1.6, label=label)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Energy (J)")
    ax.set_title("Hamiltonian energy — single trajectory")
    ax.legend(fontsize="x-small", ncol=2); ax.grid(True, alpha=0.3)
    fig.tight_layout(); return fig


def fig_traj_mse(t_eval, gt_trajs_all, model_trajs):
    R_gt = gt_trajs_all[..., :9]; om_gt = gt_trajs_all[..., 9:12]
    geo_per, l2_per = {}, {}
    for label, (trajs, color, ls) in model_trajs.items():
        geo_per[label] = (_geodesic_sq(trajs[..., :9], R_gt), color, ls)
        l2_per [label] = (np.sum((trajs[..., 9:12] - om_gt) ** 2, axis=-1),
                          color, ls)
    figs = []
    for source, ylab, title in (
        (geo_per, "geodesic² (rad²)",
         "Trajectory geodesic error vs GT (10-traj ensemble)"),
        (l2_per, "‖Δω‖² (rad²/s²)",
         "Trajectory MSE angular velocity vs GT (10-traj ensemble)"),
    ):
        fig, ax = plt.subplots(figsize=(11, 6))
        for label, (arr, color, ls) in source.items():
            m_, s_ = arr.mean(0), arr.std(0)
            ax.plot(t_eval, m_, color=color, linestyle=ls, lw=1.6, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("Time (s)"); ax.set_ylabel(ylab); ax.set_title(title)
        ax.grid(True, alpha=0.3); ax.legend(fontsize="x-small", ncol=2)
        fig.tight_layout(); figs.append(fig)
    return figs


def fig_so3_violation(t_eval, gt_metric_all, model_metric, ylab, title):
    fig, ax = plt.subplots(figsize=(11, 6))
    if gt_metric_all is not None:
        ax.plot(t_eval, gt_metric_all.mean(0), "k--", lw=2, label="GT (mean)")
    for label, (mm, color, ls) in model_metric.items():
        ax.plot(t_eval, mm, color=color, linestyle=ls, lw=1.4, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)"); ax.set_ylabel(ylab); ax.set_title(title)
    ax.grid(True, alpha=0.3); ax.legend(fontsize="x-small", ncol=2)
    fig.tight_layout(); return fig


def fig_state_ensemble(t_eval, gt_eul_all, gt_om_all, model_states):
    labels_ang = ["Roll (rad)", "Pitch (rad)", "Yaw (rad)"]
    labels_om = ["Omega X", "Omega Y", "Omega Z"]
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    gt_eul_m = gt_eul_all.mean(0); gt_eul_s = gt_eul_all.std(0)
    gt_om_m  = gt_om_all .mean(0); gt_om_s  = gt_om_all .std(0)
    for i in range(3):
        axes[i, 0].plot(t_eval, gt_eul_m[:, i], "k-", lw=2,
                        label="GT" if i == 0 else None)
        axes[i, 0].fill_between(t_eval, gt_eul_m[:, i] - 2 * gt_eul_s[:, i],
                                gt_eul_m[:, i] + 2 * gt_eul_s[:, i],
                                color="black", alpha=0.15)
        axes[i, 1].plot(t_eval, gt_om_m[:, i], "k-", lw=2,
                        label="GT" if i == 0 else None)
        axes[i, 1].fill_between(t_eval, gt_om_m[:, i] - 2 * gt_om_s[:, i],
                                gt_om_m[:, i] + 2 * gt_om_s[:, i],
                                color="black", alpha=0.15)
    for label, (eul_all, om_all, color, ls) in model_states.items():
        em = eul_all.mean(0); om_ = om_all.mean(0)
        for i in range(3):
            axes[i, 0].plot(t_eval, em[:, i], color=color, linestyle=ls, lw=1.4,
                            label=label if i == 0 else None)
            axes[i, 1].plot(t_eval, om_[:, i], color=color, linestyle=ls, lw=1.4,
                            label=label if i == 0 else None)
    for i in range(3):
        axes[i, 0].set_ylabel(labels_ang[i]); axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].set_ylabel(labels_om[i]);  axes[i, 1].grid(True, alpha=0.3)
    axes[2, 0].set_xlabel("Time (s)"); axes[2, 1].set_xlabel("Time (s)")
    axes[0, 0].legend(fontsize="x-small", ncol=2)
    axes[0, 1].legend(fontsize="x-small", ncol=2)
    fig.suptitle("State trajectories (10-traj ensemble means)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def fig_state_single(t_eval, gt_traj, model_trajs_single):
    labels_ang = ["Roll (rad)", "Pitch (rad)", "Yaw (rad)"]
    labels_om = ["Omega X", "Omega Y", "Omega Z"]
    gt_eul = rotmat_to_euler(gt_traj[:, :9])
    gt_om = gt_traj[:, 9:12]
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    for i in range(3):
        axes[i, 0].plot(t_eval, gt_eul[:, i], "k-", lw=2,
                        label="GT" if i == 0 else None)
        axes[i, 1].plot(t_eval, gt_om[:, i], "k-", lw=2,
                        label="GT" if i == 0 else None)
    for label, (tr, color, ls) in model_trajs_single.items():
        eul = rotmat_to_euler(tr[:, :9])
        for i in range(3):
            axes[i, 0].plot(t_eval, eul[:, i], color=color, linestyle=ls, lw=1.4,
                            label=label if i == 0 else None)
            axes[i, 1].plot(t_eval, tr[:, 9 + i], color=color, linestyle=ls, lw=1.4,
                            label=label if i == 0 else None)
    for i in range(3):
        axes[i, 0].set_ylabel(labels_ang[i]); axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].set_ylabel(labels_om[i]);  axes[i, 1].grid(True, alpha=0.3)
    axes[2, 0].set_xlabel("Time (s)"); axes[2, 1].set_xlabel("Time (s)")
    axes[0, 0].legend(fontsize="x-small", ncol=2)
    axes[0, 1].legend(fontsize="x-small", ncol=2)
    fig.suptitle("Single-trajectory state", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def fig_phase_portraits(gt_eul_all, gt_om_all, model_states):
    labels_ang = ["Roll (rad)", "Pitch (rad)", "Yaw (rad)"]
    labels_om = ["Omega X", "Omega Y", "Omega Z"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i in range(3):
        axes[i].plot(gt_eul_all[0][:, i], gt_om_all[0][:, i],
                     "k-", lw=2, label="GT" if i == 0 else None)
        for label, (eul_all, om_all, color, ls) in model_states.items():
            axes[i].plot(eul_all[0][:, i], om_all[0][:, i],
                         color=color, linestyle=ls, lw=1.4,
                         label=label if i == 0 else None)
        axes[i].set_xlabel(labels_ang[i]); axes[i].set_ylabel(labels_om[i])
        axes[i].grid(True, alpha=0.3)
    axes[0].legend(fontsize="x-small", ncol=2)
    fig.suptitle("Phase portraits (Angle vs Omega) — single trajectory",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_matrix_3x3_multi(t, comp_multi, comp_key, gt_flat, model_meta):
    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    for idx in range(9):
        r, c = divmod(idx, 3); ax = axes[r, c]
        if gt_flat is not None:
            ax.axhline(y=gt_flat[idx], color="k", linestyle=":", lw=1.5,
                       label="GT" if idx == 0 else None)
        for label, (color, ls) in model_meta.items():
            first = True
            for comp in comp_multi[label]:
                if comp_key not in comp:
                    break
                arr = comp[comp_key]
                if arr.ndim != 3:
                    break
                ax.plot(t, arr.reshape(arr.shape[0], 9)[:, idx],
                        color=color, linestyle=ls, alpha=0.7, lw=1.0,
                        label=label if (idx == 0 and first) else None)
                first = False
        ax.set_xlabel("Time (s)"); ax.set_title(f"({r},{c})")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize="x-small", ncol=2)
    return fig


def plot_potential_multi(t, comp_multi, gt_v_all, model_meta):
    fig, ax = plt.subplots(figsize=(11, 6))
    for ti in range(gt_v_all.shape[0]):
        ax.plot(t, gt_v_all[ti], "k:", lw=1.0, alpha=0.6,
                label="GT" if ti == 0 else None)
    for label, (color, ls) in model_meta.items():
        first = True
        for comp in comp_multi[label]:
            if "V" not in comp:
                break
            v = comp["V"].flatten()
            ax.plot(t, v, color=color, linestyle=ls, alpha=0.7, lw=1.0,
                    label=label if first else None)
            first = False
    ax.set_xlabel("Time (s)"); ax.set_ylabel("V(q)")
    ax.legend(fontsize="x-small", ncol=2); ax.grid(True, alpha=0.3)
    fig.tight_layout(); return fig


def _compute_table_metrics(gt_all, pred_all):
    N, T, _ = gt_all.shape
    R_gt = gt_all[..., :9].reshape(N, T, 3, 3)
    R_pr = pred_all[..., :9].reshape(N, T, 3, 3)
    om_gt = gt_all[..., 9:12]; om_pr = pred_all[..., 9:12]
    M = np.einsum("ntji,ntjk->ntik", R_pr, R_gt)
    cos_t = np.clip((np.trace(M, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    geo_sq = np.arccos(cos_t) ** 2
    om_sq  = np.sum((om_pr - om_gt) ** 2, axis=-1)
    geo_traj_mean = geo_sq.mean(axis=1)
    om_traj_mean  = om_sq .mean(axis=1)
    geo_final = geo_sq[:, -1]; om_final = om_sq[:, -1]
    det_resid  = np.abs(np.linalg.det(R_pr) - 1.0)
    orth_resid = np.linalg.norm(
        (np.einsum("ntji,ntjk->ntik", R_pr, R_pr)
         - np.eye(3)[None, None]).reshape(N, T, 9), axis=-1)
    det_max  = det_resid .max(axis=1)
    orth_max = orth_resid.max(axis=1)
    e_gt = np.array([get_energy(gt_all  [n]) for n in range(N)])
    e_pr = np.array([get_energy(pred_all[n]) for n in range(N)])
    e_err = np.abs(e_pr - e_gt).mean(axis=1)
    def ms(x):
        return f"{float(np.mean(x)):.2e}±{float(np.std(x)):.1e}"
    return {
        "geo² mean": ms(geo_traj_mean),
        "‖Δω‖² mean": ms(om_traj_mean),
        "geo² final": ms(geo_final),
        "‖Δω‖² final": ms(om_final),
        "|ΔE| mean": ms(e_err),
        "max|det−1|": ms(det_max),
        "max‖RᵀR−I‖": ms(orth_max),
    }


def fig_comparison_table(gt_trajs_all, models, last_step=None,
                         horizon_seconds=None):
    """`last_step`: optional dict {label: {"test geo² (10k train step)": str,
       "test ω-MSE (10k train step)": str}} appended as extra rows.
    `horizon_seconds`: if set, annotates the title with the horizon."""
    per = {label: _compute_table_metrics(gt_trajs_all, trajs)
           for label, trajs in models.items()}
    if last_step is not None:
        for label in per:
            per[label].update(last_step.get(label, {}))
    metric_names = list(next(iter(per.values())).keys())
    model_names  = list(per.keys())
    cell = [[per[m].get(k, "—") for m in model_names] for k in metric_names]
    fig, ax = plt.subplots(figsize=(2 + 2.0 * len(model_names),
                                    1.5 + 0.55 * len(metric_names)))
    ax.axis("off")
    horiz_str = (f" — horizon {horizon_seconds:g}s"
                 if horizon_seconds is not None else "")
    title = (f"Cross-framework comparison{horiz_str} — "
             f"{gt_trajs_all.shape[0]}-rollout ensemble vs GT (mean ± std)")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    tbl = ax.table(cellText=cell, rowLabels=metric_names,
                   colLabels=model_names, cellLoc="center", rowLoc="left",
                   loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.0, 1.6)
    for j in range(len(model_names)):
        tbl[0, j].set_text_props(weight="bold")
    fig.tight_layout(); return fig


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_pdf", default=DEFAULT_OUT_PDF)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu",
                    help="torch device for ph_nn_ode_v2 (cpu / cuda:0)")
    ap.add_argument("--u", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--runs", type=str, default=None,
                    help="optional override: path to a JSON list of "
                         "[kind, run_dir, label] tuples. If omitted, the "
                         "DEFAULT_RUNS in the script are used.")
    args = ap.parse_args()

    if args.runs:
        import json
        with open(args.runs) as f:
            run_specs = [tuple(r) for r in json.load(f)]
    else:
        run_specs = DEFAULT_RUNS

    print("Preparing runs:")
    for kind, d, lbl in run_specs:
        print(f"  {lbl or '<auto>':<14s}  kind={kind}  dir={os.path.basename(d)}")

    runs = prepare_runs(run_specs, args.device)
    if not runs:
        print("No runs available."); return
    runs = assign_colors(runs)
    print(f"\nLoaded {len(runs)} runs:")
    for m in runs:
        print(f"  {m['label']:<14s}  kind={m['kind']}  fix_M={m['fix_M']}  "
              f"λ={m['lambdas']}  ckpt={os.path.basename(m['ckpt'])}")

    # ── Build ensemble of distinct ICs (one per trajectory) ──
    u_const = np.asarray(args.u, dtype=np.float64)
    dt = ENV_KW["dt"]
    dt_sub = dt / N_SUBSTEPS
    sqrt_h = float(np.sqrt(dt_sub))
    rng = np.random.default_rng(args.seed)
    dW_ensemble = [rng.normal(0.0, sqrt_h, size=(N_OUTER, N_SUBSTEPS, 3))
                   for _ in range(N_TRAJ_ENSEMBLE)]
    t_eval = np.arange(N_OUTER + 1) * dt

    ic_list = []
    for ti in range(N_TRAJ_ENSEMBLE):
        env_ic = windy_pendulum_3d(seed=args.seed + ti, **ENV_KW)
        env_ic.reset(seed=args.seed + ti)
        ic_list.append((env_ic.R.copy(), env_ic.omega.copy()))
    R0, omega0 = ic_list[0]   # kept as reference IC for any single-traj plot

    # ── GT rollouts (each from its own IC) ──
    print(f"\nRolling out {N_TRAJ_ENSEMBLE} GT trajectories with distinct ICs ...")
    gt_trajs_all = np.zeros((N_TRAJ_ENSEMBLE, N_OUTER + 1, 12))
    for ti in range(N_TRAJ_ENSEMBLE):
        env = windy_pendulum_3d(seed=args.seed + ti, **ENV_KW)
        env.reset(seed=args.seed + ti)
        R_ti, om_ti = ic_list[ti]
        gt_trajs_all[ti] = rollout_gt(env, R_ti, om_ti, u_const, dW_ensemble[ti])

    # ── Per-model: load, rollout from each IC, eval subnets ──
    print("Loading models and rolling out ...")
    for m in runs:
        trajs = np.zeros((N_TRAJ_ENSEMBLE, N_OUTER + 1, 12))
        if m["kind"] == "nn_ode":
            _ensure_torch()
            device = torch.device(args.device)
            model = load_nn_ode(m["ckpt"], device, m["fix_M"])
            for ti in range(N_TRAJ_ENSEMBLE):
                R_ti, om_ti = ic_list[ti]
                trajs[ti] = rollout_nn_ode(
                    model, R_ti, om_ti, u_const, t_eval, device)[:, :12]
            comp = [eval_subnets_nn(model, gt_trajs_all[ti], device)
                    for ti in range(N_SUB_TRAJ)]
        else:                    # gp_ode
            _ensure_jax()
            model = load_gp_ode(m["ckpt"], fix_M=m["fix_M"])
            for ti in range(N_TRAJ_ENSEMBLE):
                R_ti, om_ti = ic_list[ti]
                trajs[ti] = rollout_gp_ode(
                    model, R_ti, om_ti, u_const, N_SUBSTEPS, N_OUTER, dt)
            comp = [eval_subnets_gp(model, gt_trajs_all[ti])
                    for ti in range(N_SUB_TRAJ)]
        m["traj_single"] = trajs[0]
        m["trajs_all"] = trajs
        m["comp_multi"] = comp

    # ── Pre-compute Euler / omega / energy ──
    def euler_om_energy(trajs):
        eul = np.array([rotmat_to_euler(trajs[ti, :, :9])
                        for ti in range(trajs.shape[0])])
        om  = trajs[:, :, 9:12]
        en  = np.array([get_energy(trajs[ti]) for ti in range(trajs.shape[0])])
        return eul, om, en

    gt_eul_all, gt_om_all, gt_e_all = euler_om_energy(gt_trajs_all)
    for m in runs:
        m["eul_all"], m["om_all"], m["e_all"] = euler_om_energy(m["trajs_all"])

    # GT subnet refs
    gt_m_flat = (np.eye(3) / I_PERP).flatten()
    gt_d_flat = (GT_FRICTION * np.eye(3)).flatten()
    gt_b_flat = np.eye(3).flatten()
    gt_v_all = np.array([
        GT_M * GT_G * GT_L * gt_trajs_all[ti, :, 8]
        for ti in range(N_SUB_TRAJ)
    ])

    # ── Build PDF ──
    print(f"\nWriting PDF: {args.out_pdf}")
    os.makedirs(os.path.dirname(args.out_pdf) or ".", exist_ok=True)
    with PdfPages(args.out_pdf) as pdf:
        # Pages 1-3 — summary tables at horizons 1s, 2s, 10s
        last_step = {}
        for m in runs:
            st = m.get("stats")
            def _last(key, _st=st):
                if _st is None or key not in _st:
                    return "—"
                arr = np.asarray(_st[key]).ravel()
                if arr.size == 0:
                    return "—"
                return f"{float(arr[-1]):.3e}"
            last_step[m["label"]] = {
                "test geo² (10k train step)":  _last("test_geo_loss"),
                "test ω-MSE (10k train step)": _last("test_l2_loss"),
            }

        for horizon_s in (1.0, 2.0, ENV_KW["dt"] * N_OUTER):
            n_keep = int(round(horizon_s / ENV_KW["dt"])) + 1
            n_keep = min(n_keep, gt_trajs_all.shape[1])
            gt_slice = gt_trajs_all[:, :n_keep]
            table_models = {m["label"]: m["trajs_all"][:, :n_keep] for m in runs}
            # Test losses describe the trained model, not the rollout horizon —
            # only attach them on the full-horizon page.
            ls_arg = (last_step
                      if abs(horizon_s - ENV_KW["dt"] * N_OUTER) < 1e-9
                      else None)
            pdf.savefig(fig_comparison_table(gt_slice, table_models,
                                             last_step=ls_arg,
                                             horizon_seconds=horizon_s))
            plt.close()

        # Loss curves — keys differ per framework, plot any that exist
        # Train losses common to both
        for key, title, ylab in (
            ("train_loss",     "Train total loss",    "loss"),
            ("train_l2_loss",  "Train L2 (ω)",         "L2"),
            ("train_geo_loss", "Train geodesic²",      "geo²"),
        ):
            pdf.savefig(fig_loss_curves_train(runs, key, title, ylab))
            plt.close()

        # Aux losses (key names differ between frameworks; both included)
        # nn_ode_v2: train_power_loss, train_V_cons_loss, ...
        # gp_ode_v2: train_L_power, train_L_V, ...
        for nn_key, gp_key, title, ylab in (
            ("train_power_loss",  "train_L_power", "Train L_power",   "L_power"),
            ("train_V_cons_loss", "train_L_V",     "Train L_V",       "L_V"),
        ):
            fig, ax = plt.subplots(figsize=(11, 6))
            any_data = False
            for m in runs:
                st = m["stats"];  k = nn_key if m["kind"] == "nn_ode" else gp_key
                if st is None or k not in st:
                    continue
                arr = _smooth(np.asarray(st[k]))
                ax.plot(np.arange(len(arr)), arr, lw=1.2,
                        color=m["color"], linestyle=m["linestyle"], label=m["label"])
                any_data = True
            ax.set_yscale("log")
            ax.set_xlabel("training step"); ax.set_ylabel(ylab); ax.set_title(title)
            ax.grid(True, alpha=0.3)
            if any_data:
                ax.legend(fontsize="x-small", ncol=2, loc="best")
            fig.tight_layout(); pdf.savefig(fig); plt.close()

        # Eval losses
        for key, title, ylab in (
            ("test_l2_loss",  "Test L2(ω)",       "L2"),
            ("test_geo_loss", "Test geodesic²",    "geo²"),
        ):
            pdf.savefig(fig_loss_curves_eval(runs, key, title, ylab))
            plt.close()

        # Trajectory MSE overlays
        traj_models = {m["label"]: (m["trajs_all"], m["color"], m["linestyle"])
                       for m in runs}
        for f in fig_traj_mse(t_eval, gt_trajs_all, traj_models):
            pdf.savefig(f); plt.close(f)

        # Energy
        model_e = {m["label"]: (m["e_all"].mean(0), m["e_all"].std(0),
                                m["color"], m["linestyle"]) for m in runs}
        pdf.savefig(fig_energy_ensemble(t_eval, gt_e_all, model_e)); plt.close()
        single_e = {m["label"]: (m["e_all"][0], m["color"], m["linestyle"])
                    for m in runs}
        pdf.savefig(fig_energy_single(t_eval, gt_e_all[0], single_e)); plt.close()

        # SO(3) violation
        def _det_all(trajs):
            return np.array([so3_det_residual_abs(trajs[ti])
                             for ti in range(trajs.shape[0])])
        def _orth_all(trajs):
            return np.array([so3_orth_residual(trajs[ti])
                             for ti in range(trajs.shape[0])])
        det_metrics = {m["label"]: (_det_all(m["trajs_all"]).mean(0),
                                    m["color"], m["linestyle"]) for m in runs}
        pdf.savefig(fig_so3_violation(t_eval, _det_all(gt_trajs_all),
                                       det_metrics, "|det(R)−1|",
                                       "SO(3) violation — determinant"))
        plt.close()
        orth_metrics = {m["label"]: (_orth_all(m["trajs_all"]).mean(0),
                                     m["color"], m["linestyle"]) for m in runs}
        pdf.savefig(fig_so3_violation(t_eval, _orth_all(gt_trajs_all),
                                       orth_metrics, "‖RᵀR−I‖_F",
                                       "SO(3) violation — orthogonality"))
        plt.close()

        # State trajectories
        model_states = {m["label"]: (m["eul_all"], m["om_all"],
                                     m["color"], m["linestyle"]) for m in runs}
        pdf.savefig(fig_state_ensemble(t_eval, gt_eul_all, gt_om_all, model_states))
        plt.close()
        single_trajs = {m["label"]: (m["traj_single"], m["color"], m["linestyle"])
                        for m in runs}
        pdf.savefig(fig_state_single(t_eval, gt_trajs_all[0], single_trajs))
        plt.close()
        pdf.savefig(fig_phase_portraits(gt_eul_all, gt_om_all, model_states))
        plt.close()

        # Subnet evolution
        comp_multi = {m["label"]: m["comp_multi"] for m in runs}
        model_meta = {m["label"]: (m["color"], m["linestyle"]) for m in runs}

        def _finalise(fig, title):
            fig.suptitle(title, fontsize=13, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig); plt.close(fig)

        _finalise(plot_matrix_3x3_multi(t_eval, comp_multi, "M",
                                        gt_m_flat, model_meta),
                  "Inverse mass M⁻¹(q) along 5 GT trajectories")
        _finalise(plot_matrix_3x3_multi(t_eval, comp_multi, "D",
                                        gt_d_flat, model_meta),
                  "Dissipation D(q) along 5 GT trajectories")
        _finalise(plot_matrix_3x3_multi(t_eval, comp_multi, "B",
                                        gt_b_flat, model_meta),
                  "Control gain B(q) along 5 GT trajectories")
        _finalise(plot_potential_multi(t_eval, comp_multi, gt_v_all,
                                        model_meta),
                  "Potential energy V(q) along 5 GT trajectories")

    print(f"Done: {args.out_pdf}")


if __name__ == "__main__":
    main()
