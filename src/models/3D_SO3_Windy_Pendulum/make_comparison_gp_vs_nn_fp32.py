"""GP-ODE-v2 (JAX/Equinox) vs NN-ODE-fp32 (PyTorch) comparison PDF.

Generates a comprehensive comparison PDF for one pair of runs:
  - ph_gp_ode_v2  (JAX) checkpoint
  - ph_nn_ode_fp32 (PyTorch) checkpoint

Usage:
  python make_comparison_gp_vs_nn_fp32.py \\
    --gp_dir <path>  --nn_dir <path>  --out_pdf comparison.pdf \\
    [--seed 42] [--device cuda:0] [--obs_label "obs=0.01"]

Handles Dw(q, p) architecture (12-dim input) for both models.
GT rollouts use varying_friction=True (matching the training setup).
Rollout eval uses constant u=0 for a fair, deterministic comparison.
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, "../../.."))
NN_FP32_DIR = os.path.join(THIS_FILE_DIR, "ph_nn_ode_fp32")
GP_ODE_V2_DIR = os.path.join(THIS_FILE_DIR, "ph_gp_ode_v2")

for p in (PROJECT_ROOT,
          os.path.join(PROJECT_ROOT, "src/utils"),
          os.path.join(PROJECT_ROOT, "datasets"),
          os.path.join(PROJECT_ROOT, "envs")):
    if p not in sys.path:
        sys.path.insert(0, p)

from envs.windy_pendulum_3d import windy_pendulum_3d

# ── Env / rollout constants ──────────────────────────────────────────────────
ENV_KW = dict(
    g=9.81, m=1.0, l=1.0, dt=0.05,
    friction_coeff=0.5, varying_friction=True,
    external_force_type="sine", external_force_std=0.0,
    wind_force_std=0.0,
)
N_SUBSTEPS = 10
N_OUTER = 200
N_TRAJ_ENSEMBLE = 10
GT_FRICTION = 0.5
GT_M, GT_L, GT_G = 1.0, 1.0, 9.81
I_PERP = GT_M * GT_L * GT_L

COLORS = {"gp_ode": "#d62728", "nn_ode": "#1f77b4"}   # red / blue
LS     = {"gp_ode": "-",        "nn_ode": "--"}


# ── Lazy imports ─────────────────────────────────────────────────────────────

_torch_loaded = False
def _ensure_torch():
    global _torch_loaded, torch, odeint, DissipativeSO3HamNODE_fp32
    if _torch_loaded:
        return
    import torch as _torch
    from torchdiffeq import odeint as _odeint
    spec = importlib.util.spec_from_file_location(
        "_nn_fp32_network", os.path.join(NN_FP32_DIR, "network.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_nn_fp32_network"] = mod
    spec.loader.exec_module(mod)
    torch = _torch
    odeint = _odeint
    DissipativeSO3HamNODE_fp32 = mod.DissipativeSO3HamNODE
    _torch_loaded = True


_jax_loaded = False
def _ensure_jax():
    global _jax_loaded, jax, jnp, eqx, DissipativeSO3HamODE, lie_heun_ode_rollout
    if _jax_loaded:
        return
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax as _jax
    import jax.numpy as _jnp
    import equinox as _eqx
    spec = importlib.util.spec_from_file_location(
        "_gp_ode_v2_network", os.path.join(GP_ODE_V2_DIR, "network.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_gp_ode_v2_network"] = mod
    spec.loader.exec_module(mod)
    from src.utils.JAX.lie_integrator import lie_heun_ode_rollout as _rollout
    jax = _jax
    jnp = _jnp
    eqx = _eqx
    DissipativeSO3HamODE = mod.DissipativeSO3HamODE
    lie_heun_ode_rollout = _rollout
    _jax_loaded = True


# ── Checkpoint discovery ─────────────────────────────────────────────────────

def _find_latest(run_dir, pattern):
    cands = []
    for fn in os.listdir(run_dir):
        m = re.match(pattern, fn)
        if m:
            cands.append((int(m.group(1)), os.path.join(run_dir, fn)))
    return sorted(cands)[-1][1] if cands else None


def find_nn_ckpt(run_dir):
    return _find_latest(run_dir, r"wp3d-so3ham-rk4-5p-(\d+)\.tar$")


def find_gp_ckpt(run_dir):
    return _find_latest(run_dir, r"wp3d-so3hamGPODE-5p-(\d+)\.eqx$")


def load_stats(run_dir, prefix):
    p = os.path.join(run_dir, f"{prefix}-stats.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


# ── Model loading ────────────────────────────────────────────────────────────

def load_nn_model(ckpt_path, device):
    _ensure_torch()
    model = DissipativeSO3HamNODE_fp32(
        device=device, u_dim=3, init_gain=0.01, friction=True).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict):
        if any(k.startswith("module.") for k in sd):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        # checkpoint may be wrapped in another dict
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def load_gp_model(ckpt_path):
    _ensure_jax()
    template = DissipativeSO3HamODE(
        key=jax.random.PRNGKey(0), u_dim=3, init_gain=0.5, friction=True,
        fix_M=False)
    return eqx.tree_deserialise_leaves(ckpt_path, template)


# ── Rollouts ─────────────────────────────────────────────────────────────────

def rollout_nn(model, R0, omega0, t_eval, device):
    _ensure_torch()
    u0 = np.zeros(3, dtype=np.float32)
    x0 = np.concatenate([R0.reshape(-1), omega0, u0]).astype(np.float32)
    x0_t = torch.tensor(x0[None, :], dtype=torch.float32, device=device)
    t_t = torch.tensor(t_eval, dtype=torch.float32, device=device)
    # NN forward uses torch.enable_grad() internally — do not wrap in no_grad
    traj = odeint(model, x0_t, t_t, method="rk4")
    return traj[:, 0, :12].detach().cpu().numpy()  # (T, 12)


def rollout_gp(model, R0, omega0, n_substeps, n_outer, dt):
    _ensure_jax()
    h = dt / n_substeps
    x0 = jnp.concatenate([
        jnp.asarray(R0.reshape(-1), dtype=jnp.float32),
        jnp.asarray(omega0,          dtype=jnp.float32),
    ])
    u = jnp.zeros(3, dtype=jnp.float32)
    return np.asarray(
        lie_heun_ode_rollout(model, x0, u, jnp.float32(h), n_substeps, n_outer))


def rollout_gt(env, R0, omega0, dW_per_outer):
    n_outer, n_sub, _ = dW_per_outer.shape
    h_sub = env.dt / n_sub
    sigma = env.wind_force_std
    u_const = np.zeros(3, dtype=np.float64)
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


# ── Subnet evaluation ────────────────────────────────────────────────────────

def eval_subnets_nn(model, traj_12, device):
    """Returns dicts with M (N,3,3), V (N,), D (N,3,3), B (N,3,3)."""
    _ensure_torch()
    qs = torch.tensor(traj_12[:, :9], dtype=torch.float32, device=device)
    omegas = torch.tensor(traj_12[:, 9:12], dtype=torch.float32, device=device)
    with torch.no_grad():
        M_inv = model.M_net(qs)                                         # (N,3,3)
        p = torch.linalg.solve(M_inv, omegas.unsqueeze(-1)).squeeze(-1) # (N,3)
        V = model.V_net(qs).cpu().numpy().squeeze(-1)
        D = model.Dw_net(torch.cat([qs, p], dim=1)).cpu().numpy()
        B = model.g_net(qs).cpu().numpy()
        M_np = M_inv.cpu().numpy()
    return {"M": M_np, "V": V, "D": D, "B": B}


def eval_subnets_gp(model, traj_12):
    """Returns dicts with M (N,3,3), V (N,), D (N,3,3), B (N,3,3)."""
    _ensure_jax()
    qs = jnp.asarray(traj_12[:, :9], dtype=jnp.float32)
    omegas = jnp.asarray(traj_12[:, 9:12], dtype=jnp.float32)

    def per_sample(q, omega):
        M_inv = model.M_net(q, inference_mode=True)          # (3,3)
        V = model.V_net(q, inference_mode=True)[0]           # scalar
        p = jnp.linalg.solve(M_inv, omega)                   # (3,)  = M * omega
        D = model._Dw_call(q, p)                             # (3,3)
        B = model.g_net(q, inference_mode=True)              # (3,3)
        return M_inv, V, D, B

    M_inv, V, D, B = jax.vmap(per_sample)(qs, omegas)
    return {
        "M": np.asarray(M_inv),
        "V": np.asarray(V),
        "D": np.asarray(D),
        "B": np.asarray(B),
    }


# ── GT subnet ground truth ───────────────────────────────────────────────────

def gt_D_varying(traj_12, friction_coeff=0.5):
    """Compute the varying-friction dissipation matrix along a trajectory.
    D_gt(q, ω) = friction_coeff * mult * I₃
    mult = 1 + 0.5 * height_term + 0.5 * speed_term
    """
    q = traj_12[:, :9]
    omega = traj_12[:, 9:12]
    height_term = 0.5 * (1.0 - q[:, 8])
    speed_term = np.tanh(np.linalg.norm(omega, axis=-1))
    mult = 1.0 + 0.5 * height_term + 0.5 * speed_term
    N = q.shape[0]
    D = np.zeros((N, 3, 3), dtype=np.float64)
    for i in range(3):
        D[:, i, i] = friction_coeff * mult
    return D


# ── Utility ──────────────────────────────────────────────────────────────────

def rotmat_to_euler(R_flat):
    Rs = np.asarray(R_flat).reshape(-1, 3, 3)
    sy = np.sqrt(Rs[:, 0, 0] ** 2 + Rs[:, 1, 0] ** 2)
    near = sy < 1e-6
    roll  = np.where(near, np.arctan2(-Rs[:, 1, 2], Rs[:, 1, 1]),
                           np.arctan2( Rs[:, 2, 1], Rs[:, 2, 2]))
    pitch = np.arctan2(-Rs[:, 2, 0], sy)
    yaw   = np.where(near, 0.0, np.arctan2(Rs[:, 1, 0], Rs[:, 0, 0]))
    return np.stack([roll, pitch, yaw], axis=-1)


def get_energy(traj):
    pe = GT_G * (1.0 - traj[:, 8])
    omega = traj[:, 9:12]
    I = I_PERP * np.eye(3)
    return 0.5 * np.einsum("ti,ij,tj->t", omega, I, omega) + pe


def geodesic_sq(traj_pred, traj_gt):
    """(N_traj, T) array of geodesic² errors."""
    R_pred = traj_pred[..., :9].reshape(*traj_pred.shape[:-1], 3, 3)
    R_gt   = traj_gt  [..., :9].reshape(*traj_gt  .shape[:-1], 3, 3)
    M = np.einsum("...ji,...jk->...ik", R_pred, R_gt)
    cos_t = np.clip((np.trace(M, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cos_t) ** 2


def omega_sq(traj_pred, traj_gt):
    return np.sum((traj_pred[..., 9:12] - traj_gt[..., 9:12]) ** 2, axis=-1)


def so3_det_err(traj):
    R = traj[:, :9].reshape(-1, 3, 3)
    return np.abs(np.linalg.det(R) - 1.0)


def so3_orth_err(traj):
    R = traj[:, :9].reshape(-1, 3, 3)
    diff = np.einsum("tji,tjk->tik", R, R) - np.eye(3)[None]
    return np.linalg.norm(diff.reshape(-1, 9), axis=1)


def _smooth(y, w=21):
    y = np.asarray(y, dtype=np.float64); n = len(y)
    if n < w + 1:
        return y
    cumsum = np.cumsum(np.insert(y, 0, 0.0))
    half = w // 2
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = (cumsum[hi] - cumsum[lo]) / (hi - lo)
    return out


def _ms(x):
    return f"{float(np.mean(x)):.2e}±{float(np.std(x)):.1e}"


# ── Plotting helpers ─────────────────────────────────────────────────────────

def _two_ax(title):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_title(title); ax.grid(True, alpha=0.3)
    return fig, ax


def fig_train_loss(stats_gp, stats_nn, key, title, ylab, logy=True, smooth=True):
    fig, ax = _two_ax(title)
    for st, kind, label in ((stats_gp, "gp_ode", "GP"), (stats_nn, "nn_ode", "NN")):
        if st is None or key not in st:
            continue
        arr = np.asarray(st[key])
        if smooth:
            arr = _smooth(arr)
        ax.plot(arr, color=COLORS[kind], ls=LS[kind], lw=1.2, label=label)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel(ylab)
    ax.legend(fontsize="small"); fig.tight_layout()
    return fig


def fig_aux_loss(stats_gp, stats_nn, gp_key, nn_key, title, ylab):
    fig, ax = _two_ax(title)
    for st, key, kind, label in (
            (stats_gp, gp_key, "gp_ode", "GP"),
            (stats_nn, nn_key, "nn_ode", "NN")):
        if st is None or key not in st:
            continue
        arr = _smooth(np.asarray(st[key]))
        ax.plot(arr, color=COLORS[kind], ls=LS[kind], lw=1.2, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel(ylab)
    ax.legend(fontsize="small"); fig.tight_layout()
    return fig


def fig_eval_loss(stats_gp, stats_nn, key, title, ylab, logy=True):
    fig, ax = _two_ax(title)
    for st, kind, label in ((stats_gp, "gp_ode", "GP"), (stats_nn, "nn_ode", "NN")):
        if st is None or "eval_step" not in st or key not in st:
            continue
        ax.plot(st["eval_step"], st[key],
                color=COLORS[kind], ls=LS[kind], lw=1.4, label=label)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel(ylab)
    ax.legend(fontsize="small"); fig.tight_layout()
    return fig


def fig_summary_table(gt_trajs, preds_gp, preds_nn, stats_gp, stats_nn,
                       horizon_s, dt, obs_label):
    """Single summary table for a given horizon."""
    n_keep = min(int(round(horizon_s / dt)) + 1, gt_trajs.shape[1])
    gt_sl = gt_trajs[:, :n_keep]

    def _metrics(preds):
        pred_sl = preds[:, :n_keep]
        N, T, _ = gt_sl.shape
        geo  = geodesic_sq(pred_sl, gt_sl)
        om   = omega_sq(pred_sl, gt_sl)
        geo_mean = geo.mean(axis=1)
        om_mean  = om.mean(axis=1)
        e_gt = np.array([get_energy(gt_sl[n]) for n in range(N)])
        e_pr = np.array([get_energy(pred_sl[n]) for n in range(N)])
        e_err = np.abs(e_pr - e_gt).mean(axis=1)
        R_pr = pred_sl[..., :9].reshape(N, T, 3, 3)
        det_max  = np.abs(np.linalg.det(R_pr) - 1.0).max(axis=1)
        orth_max = np.linalg.norm(
            (np.einsum("ntji,ntjk->ntik", R_pr, R_pr) - np.eye(3)[None, None]
             ).reshape(N, T, 9), axis=-1).max(axis=1)
        return {
            "geo² mean":    _ms(geo_mean),
            "‖Δω‖² mean":  _ms(om_mean),
            "geo² final":   _ms(geo[:, -1]),
            "‖Δω‖² final": _ms(om[:, -1]),
            "|ΔE| mean":   _ms(e_err),
            "max|det−1|":  _ms(det_max),
            "max‖RᵀR−I‖": _ms(orth_max),
        }

    gp_m = _metrics(preds_gp)
    nn_m = _metrics(preds_nn)

    # Append final-step test losses on full-horizon table
    if abs(horizon_s - dt * (gt_trajs.shape[1] - 1)) < 1e-9:
        for key, label in (("test_geo_loss", "test geo²"),
                            ("test_l2_loss",  "test ω-MSE")):
            for st, m in ((stats_gp, gp_m), (stats_nn, nn_m)):
                if st and key in st:
                    arr = np.asarray(st[key]).ravel()
                    m[f"{label} (final step)"] = (
                        f"{float(arr[-1]):.3e}" if arr.size else "—")
                else:
                    m.setdefault(f"{label} (final step)", "—")

    row_names = list(gp_m.keys())
    cell = [[gp_m.get(k, "—"), nn_m.get(k, "—")] for k in row_names]
    fig, ax = plt.subplots(figsize=(8, 1.5 + 0.55 * len(row_names)))
    ax.axis("off")
    ax.set_title(
        f"{obs_label} — horizon {horizon_s:g}s — {gt_trajs.shape[0]}-rollout ensemble vs GT",
        fontsize=12, fontweight="bold", pad=12)
    tbl = ax.table(cellText=cell, rowLabels=row_names,
                   colLabels=["GP-ODE-v2", "NN-ODE-fp32"],
                   cellLoc="center", rowLoc="left", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.6)
    for j in range(2):
        tbl[0, j].set_text_props(weight="bold")
    fig.tight_layout()
    return fig


def fig_traj_mse(t_eval, gt_all, preds_gp, preds_nn):
    figs = []
    for metric_fn, ylab, title in (
        (geodesic_sq, "geodesic² (rad²)", "Geodesic error vs GT"),
        (omega_sq,    "‖Δω‖² (rad²/s²)", "Angular velocity MSE vs GT"),
    ):
        fig, ax = _two_ax(title)
        for preds, kind, label in ((preds_gp, "gp_ode", "GP"),
                                   (preds_nn, "nn_ode", "NN")):
            err = metric_fn(preds, gt_all)
            ax.plot(t_eval, err.mean(0), color=COLORS[kind], ls=LS[kind],
                    lw=1.6, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("Time (s)"); ax.set_ylabel(ylab)
        ax.legend(fontsize="small"); fig.tight_layout()
        figs.append(fig)
    return figs


def fig_energy(t_eval, gt_all, preds_gp, preds_nn):
    gt_e = np.array([get_energy(gt_all[n]) for n in range(gt_all.shape[0])])
    gp_e = np.array([get_energy(preds_gp[n]) for n in range(preds_gp.shape[0])])
    nn_e = np.array([get_energy(preds_nn[n]) for n in range(preds_nn.shape[0])])

    fig1, ax = _two_ax("Hamiltonian energy — ensemble mean")
    ax.plot(t_eval, gt_e.mean(0), "k-", lw=2, label="GT")
    ax.fill_between(t_eval, gt_e.mean(0)-2*gt_e.std(0), gt_e.mean(0)+2*gt_e.std(0),
                    color="black", alpha=0.15)
    ax.plot(t_eval, gp_e.mean(0), color=COLORS["gp_ode"], ls=LS["gp_ode"], lw=1.6, label="GP")
    ax.plot(t_eval, nn_e.mean(0), color=COLORS["nn_ode"], ls=LS["nn_ode"], lw=1.6, label="NN")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Energy (J)")
    ax.legend(fontsize="small"); fig1.tight_layout()

    fig2, ax2 = _two_ax("Hamiltonian energy — single trajectory (traj 0)")
    ax2.plot(t_eval, gt_e[0], "k-", lw=2, label="GT")
    ax2.plot(t_eval, gp_e[0], color=COLORS["gp_ode"], ls=LS["gp_ode"], lw=1.6, label="GP")
    ax2.plot(t_eval, nn_e[0], color=COLORS["nn_ode"], ls=LS["nn_ode"], lw=1.6, label="NN")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Energy (J)")
    ax2.legend(fontsize="small"); fig2.tight_layout()
    return fig1, fig2


def fig_so3(t_eval, gt_all, preds_gp, preds_nn):
    figs = []
    for metric_fn, ylab, title in (
        (so3_det_err,  "|det(R)−1|",   "SO(3) violation — determinant"),
        (so3_orth_err, "‖RᵀR−I‖_F",   "SO(3) violation — orthogonality"),
    ):
        fig, ax = _two_ax(title)
        gt_m = np.array([metric_fn(gt_all[n]) for n in range(gt_all.shape[0])]).mean(0)
        ax.plot(t_eval, gt_m, "k--", lw=1.5, label="GT")
        for preds, kind, label in ((preds_gp, "gp_ode", "GP"),
                                   (preds_nn, "nn_ode", "NN")):
            m = np.array([metric_fn(preds[n]) for n in range(preds.shape[0])]).mean(0)
            ax.plot(t_eval, m, color=COLORS[kind], ls=LS[kind], lw=1.4, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("Time (s)"); ax.set_ylabel(ylab)
        ax.legend(fontsize="small"); fig.tight_layout()
        figs.append(fig)
    return figs


def fig_state_ensemble(t_eval, gt_all, preds_gp, preds_nn):
    labels_ang = ["Roll (rad)", "Pitch (rad)", "Yaw (rad)"]
    labels_om  = ["Omega X",    "Omega Y",     "Omega Z"]
    gt_eul = np.array([rotmat_to_euler(gt_all[n, :, :9]) for n in range(gt_all.shape[0])])
    gt_om  = gt_all[:, :, 9:12]
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    gt_em = gt_eul.mean(0); gt_es = gt_eul.std(0)
    gt_om_m = gt_om.mean(0); gt_om_s = gt_om.std(0)
    for i in range(3):
        axes[i, 0].plot(t_eval, gt_em[:, i], "k-", lw=2,
                        label="GT" if i == 0 else None)
        axes[i, 0].fill_between(t_eval, gt_em[:, i]-2*gt_es[:, i],
                                gt_em[:, i]+2*gt_es[:, i], color="black", alpha=0.15)
        axes[i, 1].plot(t_eval, gt_om_m[:, i], "k-", lw=2,
                        label="GT" if i == 0 else None)
        axes[i, 1].fill_between(t_eval, gt_om_m[:, i]-2*gt_om_s[:, i],
                                gt_om_m[:, i]+2*gt_om_s[:, i], color="black", alpha=0.15)
    for preds, kind, label in ((preds_gp, "gp_ode", "GP"),
                               (preds_nn, "nn_ode", "NN")):
        eul = np.array([rotmat_to_euler(preds[n, :, :9]) for n in range(preds.shape[0])])
        em  = eul.mean(0); om_m = preds[:, :, 9:12].mean(0)
        for i in range(3):
            axes[i, 0].plot(t_eval, em[:, i], color=COLORS[kind], ls=LS[kind],
                            lw=1.4, label=label if i == 0 else None)
            axes[i, 1].plot(t_eval, om_m[:, i], color=COLORS[kind], ls=LS[kind],
                            lw=1.4, label=label if i == 0 else None)
    for i in range(3):
        axes[i, 0].set_ylabel(labels_ang[i]); axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].set_ylabel(labels_om[i]);  axes[i, 1].grid(True, alpha=0.3)
    axes[2, 0].set_xlabel("Time (s)"); axes[2, 1].set_xlabel("Time (s)")
    axes[0, 0].legend(fontsize="x-small", ncol=2)
    axes[0, 1].legend(fontsize="x-small", ncol=2)
    fig.suptitle("State trajectories — 10-traj ensemble means", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def fig_state_single(t_eval, gt_single, traj_gp, traj_nn):
    labels_ang = ["Roll (rad)", "Pitch (rad)", "Yaw (rad)"]
    labels_om  = ["Omega X",    "Omega Y",     "Omega Z"]
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    gt_eul = rotmat_to_euler(gt_single[:, :9])
    for i in range(3):
        axes[i, 0].plot(t_eval, gt_eul[:, i], "k-", lw=2,
                        label="GT" if i == 0 else None)
        axes[i, 1].plot(t_eval, gt_single[:, 9+i], "k-", lw=2,
                        label="GT" if i == 0 else None)
    for traj, kind, label in ((traj_gp, "gp_ode", "GP"),
                              (traj_nn, "nn_ode", "NN")):
        eul = rotmat_to_euler(traj[:, :9])
        for i in range(3):
            axes[i, 0].plot(t_eval, eul[:, i], color=COLORS[kind], ls=LS[kind],
                            lw=1.4, label=label if i == 0 else None)
            axes[i, 1].plot(t_eval, traj[:, 9+i], color=COLORS[kind], ls=LS[kind],
                            lw=1.4, label=label if i == 0 else None)
    for i in range(3):
        axes[i, 0].set_ylabel(labels_ang[i]); axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].set_ylabel(labels_om[i]);  axes[i, 1].grid(True, alpha=0.3)
    axes[2, 0].set_xlabel("Time (s)"); axes[2, 1].set_xlabel("Time (s)")
    axes[0, 0].legend(fontsize="x-small")
    axes[0, 1].legend(fontsize="x-small")
    fig.suptitle("Single trajectory (traj 0) state", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def fig_phase_portraits(gt_all, preds_gp, preds_nn):
    labels_ang = ["Roll (rad)", "Pitch (rad)", "Yaw (rad)"]
    labels_om  = ["Omega X",    "Omega Y",     "Omega Z"]
    gt_eul = np.array([rotmat_to_euler(gt_all[n, :, :9]) for n in range(gt_all.shape[0])])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i in range(3):
        axes[i].plot(gt_eul[0, :, i], gt_all[0, :, 9+i], "k-", lw=2,
                     label="GT" if i == 0 else None)
        for preds, kind, label in ((preds_gp, "gp_ode", "GP"),
                                   (preds_nn, "nn_ode", "NN")):
            eul = rotmat_to_euler(preds[0, :, :9])
            axes[i].plot(eul[:, i], preds[0, :, 9+i], color=COLORS[kind],
                         ls=LS[kind], lw=1.4, label=label if i == 0 else None)
        axes[i].set_xlabel(labels_ang[i]); axes[i].set_ylabel(labels_om[i])
        axes[i].grid(True, alpha=0.3)
    axes[0].legend(fontsize="x-small")
    fig.suptitle("Phase portraits — single trajectory (traj 0)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def fig_subnet_matrix(t_eval, sub_gp, sub_nn, gt_traj, comp_key,
                       gt_val_fn, title):
    """3x3 grid of matrix element time-series; gt_val_fn(t_step) returns (9,) or None."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    gt_vals = gt_val_fn(gt_traj) if gt_val_fn else None  # (T, 9) or None

    for idx in range(9):
        r, c = divmod(idx, 3); ax = axes[r, c]
        if gt_vals is not None:
            ax.plot(t_eval, gt_vals[:, idx], "k:", lw=1.5,
                    label="GT" if idx == 0 else None)
        for sub, kind, label in ((sub_gp, "gp_ode", "GP"),
                                  (sub_nn, "nn_ode", "NN")):
            arr = sub.get(comp_key)
            if arr is not None and arr.ndim == 3:
                ax.plot(t_eval, arr.reshape(arr.shape[0], 9)[:, idx],
                        color=COLORS[kind], ls=LS[kind], lw=1.0, alpha=0.85,
                        label=label if idx == 0 else None)
        ax.set_title(f"({r},{c})"); ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize="x-small")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def fig_subnet_V(t_eval, sub_gp, sub_nn, gt_traj):
    gt_v = GT_M * GT_G * GT_L * gt_traj[:, 8]
    fig, ax = _two_ax("Potential energy V(q) along GT trajectory (traj 0)")
    ax.plot(t_eval, gt_v, "k:", lw=1.5, label="GT")
    for sub, kind, label in ((sub_gp, "gp_ode", "GP"),
                              (sub_nn, "nn_ode", "NN")):
        v = sub.get("V")
        if v is not None:
            # centre to remove gauge ambiguity
            ax.plot(t_eval, v - v.mean() + gt_v.mean(),
                    color=COLORS[kind], ls=LS[kind], lw=1.0, alpha=0.85, label=label)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("V(q)")
    ax.legend(fontsize="small"); fig.tight_layout()
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gp_dir",    required=True,  help="ph_gp_ode_v2 run dir")
    ap.add_argument("--nn_dir",    required=True,  help="ph_nn_ode_fp32 run dir")
    ap.add_argument("--out_pdf",   required=True)
    ap.add_argument("--obs_label", default="",     help="title annotation (e.g. obs=0.01)")
    ap.add_argument("--seed",      type=int,  default=42)
    ap.add_argument("--device",    default="cpu")
    args = ap.parse_args()

    # ── Find checkpoints & stats ──
    gp_ckpt = find_gp_ckpt(args.gp_dir)
    nn_ckpt = find_nn_ckpt(args.nn_dir)
    assert gp_ckpt, f"No GP checkpoint in {args.gp_dir}"
    assert nn_ckpt, f"No NN checkpoint in {args.nn_dir}"
    stats_gp = load_stats(args.gp_dir,  "wp3d-so3hamGPODE-5p")
    stats_nn = load_stats(args.nn_dir,  "wp3d-so3ham-rk4-5p")
    print(f"GP ckpt : {os.path.basename(gp_ckpt)}")
    print(f"NN ckpt : {os.path.basename(nn_ckpt)}")

    # ── Build ICs & noise ──
    rng = np.random.default_rng(args.seed)
    dt = ENV_KW["dt"]
    dt_sub = dt / N_SUBSTEPS
    sqrt_h = float(np.sqrt(dt_sub))
    dW_ensemble = [rng.normal(0.0, sqrt_h, (N_OUTER, N_SUBSTEPS, 3))
                   for _ in range(N_TRAJ_ENSEMBLE)]
    t_eval = np.arange(N_OUTER + 1) * dt

    ic_list = []
    for ti in range(N_TRAJ_ENSEMBLE):
        env = windy_pendulum_3d(seed=args.seed + ti, **ENV_KW)
        env.reset(seed=args.seed + ti)
        ic_list.append((env.R.copy(), env.omega.copy()))

    # ── GT rollouts ──
    print(f"Rolling out {N_TRAJ_ENSEMBLE} GT trajectories ...")
    gt_all = np.zeros((N_TRAJ_ENSEMBLE, N_OUTER + 1, 12))
    for ti in range(N_TRAJ_ENSEMBLE):
        env = windy_pendulum_3d(seed=args.seed + ti, **ENV_KW)
        env.reset(seed=args.seed + ti)
        gt_all[ti] = rollout_gt(env, ic_list[ti][0], ic_list[ti][1], dW_ensemble[ti])

    # ── Load models ──
    print("Loading GP model ...")
    gp_model = load_gp_model(gp_ckpt)

    print("Loading NN model ...")
    _ensure_torch()
    device = torch.device(args.device)
    nn_model = load_nn_model(nn_ckpt, device)

    # ── Model rollouts ──
    print("Rolling out GP model ...")
    preds_gp = np.zeros((N_TRAJ_ENSEMBLE, N_OUTER + 1, 12))
    for ti in range(N_TRAJ_ENSEMBLE):
        R_ti, om_ti = ic_list[ti]
        preds_gp[ti] = rollout_gp(gp_model, R_ti, om_ti, N_SUBSTEPS, N_OUTER, dt)

    print("Rolling out NN model ...")
    preds_nn = np.zeros((N_TRAJ_ENSEMBLE, N_OUTER + 1, 12))
    for ti in range(N_TRAJ_ENSEMBLE):
        R_ti, om_ti = ic_list[ti]
        preds_nn[ti] = rollout_nn(nn_model, R_ti, om_ti, t_eval, device)

    # ── Subnet eval on GT trajectory 0 ──
    print("Evaluating subnets on GT traj 0 ...")
    sub_gp = eval_subnets_gp(gp_model, gt_all[0])
    sub_nn = eval_subnets_nn(nn_model, gt_all[0], device)

    # ── GT subnet references ──
    gt_M_inv_flat = (1.0 / I_PERP) * np.eye(3)  # M⁻¹ = I₃ for m=l=1
    gt_B_flat = np.eye(3)

    def gt_M_vals(gt_traj):
        T = gt_traj.shape[0]
        return np.tile(gt_M_inv_flat.reshape(-1), (T, 1))

    def gt_B_vals(gt_traj):
        T = gt_traj.shape[0]
        return np.tile(gt_B_flat.reshape(-1), (T, 1))

    def gt_D_vals(gt_traj):
        D_all = gt_D_varying(gt_traj, GT_FRICTION)  # (T, 3, 3)
        return D_all.reshape(D_all.shape[0], 9)

    # ── Write PDF ──
    print(f"Writing PDF: {args.out_pdf}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_pdf)), exist_ok=True)
    obs = args.obs_label or os.path.basename(args.gp_dir)

    with PdfPages(args.out_pdf) as pdf:
        # ── Summary tables at 1s, 2s, full horizon ──
        for horizon_s in (1.0, 2.0, dt * N_OUTER):
            pdf.savefig(fig_summary_table(
                gt_all, preds_gp, preds_nn, stats_gp, stats_nn,
                horizon_s, dt, obs))
            plt.close()

        # ── Loss curves ──
        for key, title, ylab in (
            ("train_loss",     "Train total loss",     "loss"),
            ("train_l2_loss",  "Train L2 (ω)",          "L2"),
            ("train_geo_loss", "Train geodesic²",       "geo²"),
        ):
            pdf.savefig(fig_train_loss(stats_gp, stats_nn, key, title, ylab))
            plt.close()

        # Aux losses
        for gp_key, nn_key, title, ylab in (
            ("train_L_power", "train_power_loss", "Train L_power", "L_power"),
            ("train_L_V",     "train_V_cons_loss","Train L_V",     "L_V"),
        ):
            pdf.savefig(fig_aux_loss(stats_gp, stats_nn, gp_key, nn_key, title, ylab))
            plt.close()

        # Eval losses
        for key, title, ylab in (
            ("test_l2_loss",  "Test L2(ω)",        "L2"),
            ("test_geo_loss", "Test geodesic²",     "geo²"),
            ("eval_M_loss",   "Eval M⁻¹ MSE",       "MSE"),
            ("eval_V_loss",   "Eval V MSE",          "MSE"),
            ("eval_Dw_loss",  "Eval Dw MSE",         "MSE"),
            ("eval_g_loss",   "Eval g MSE",          "MSE"),
        ):
            pdf.savefig(fig_eval_loss(stats_gp, stats_nn, key, title, ylab))
            plt.close()

        # ── Trajectory metrics ──
        for f in fig_traj_mse(t_eval, gt_all, preds_gp, preds_nn):
            pdf.savefig(f); plt.close(f)

        # Energy
        e1, e2 = fig_energy(t_eval, gt_all, preds_gp, preds_nn)
        pdf.savefig(e1); plt.close(e1)
        pdf.savefig(e2); plt.close(e2)

        # SO(3)
        for f in fig_so3(t_eval, gt_all, preds_gp, preds_nn):
            pdf.savefig(f); plt.close(f)

        # State trajectories
        pdf.savefig(fig_state_ensemble(t_eval, gt_all, preds_gp, preds_nn))
        plt.close()
        pdf.savefig(fig_state_single(t_eval, gt_all[0], preds_gp[0], preds_nn[0]))
        plt.close()
        pdf.savefig(fig_phase_portraits(gt_all, preds_gp, preds_nn))
        plt.close()

        # Subnet evolution along GT traj 0
        pdf.savefig(fig_subnet_matrix(
            t_eval, sub_gp, sub_nn, gt_all[0],
            "M", gt_M_vals, "M⁻¹(q) along GT traj 0"))
        plt.close()
        pdf.savefig(fig_subnet_matrix(
            t_eval, sub_gp, sub_nn, gt_all[0],
            "D", gt_D_vals, "Dissipation D(q,p) along GT traj 0 (GT = varying friction)"))
        plt.close()
        pdf.savefig(fig_subnet_matrix(
            t_eval, sub_gp, sub_nn, gt_all[0],
            "B", gt_B_vals, "Control gain B(q) along GT traj 0"))
        plt.close()
        pdf.savefig(fig_subnet_V(t_eval, sub_gp, sub_nn, gt_all[0]))
        plt.close()

    print("Done.")


if __name__ == "__main__":
    main()
