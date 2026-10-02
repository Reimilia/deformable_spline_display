"""Per-checkpoint subnetwork phase plots: ph_gp_sde vs ph_nn_ode_fp32 vs GT.

One PDF page per training step in {1k, 2k, ..., 10k}. Page layout:

    cols = 3   (component i ∈ {x, y, z})
    rows = 12  (subnetwork × model, in this order:
                M⁻¹ (GP_SDE), M⁻¹ (NN_ODE), M⁻¹ (GT),
                V   (GP_SDE), V   (NN_ODE), V   (GT),
                D   (GP_SDE), D   (NN_ODE), D   (GT),
                B   (GP_SDE), B   (NN_ODE), B   (GT))

For each cell (subnet, model, axis i):
    x = ω_i                            (angular velocity; ω = ∇_pH = M⁻¹·p)
    y depends on subnet (matching ṗ-decomposition in the PHS spec):
        M⁻¹ : (p^× · ∇_pH)_i = (p × ω)_i
        V   : (−(q^×)ᵀ · ∇_qH)_i  = −(Σⱼ rⱼ × ∂_{qⱼ}V)_i   (potential-only)
        D   : (−D(q,p) · ∇_pH)_i = −(D · ω)_i
        B   : (B(q) · u)_i

Trajectories come from the env GT rollout (same Lie-Heun loop as
make_comparison_pdf.py). With --ensemble, N trajectories are overlaid per
cell; otherwise a single trajectory is shown.

A single port-Hamiltonian scale-invariance β is estimated per model per
checkpoint from M⁻¹(q) along the GT trajectory(ies) (β = (1/(m·l²)) /
mean trace(M⁻¹)/3). It rescales the model output to the GT scale so all
three columns are directly comparable:
    M⁻¹ → M⁻¹·β,  V → V/β,  D → D/β,  B → B/β
which propagates as 1/β on every y-quantity above. The x-axis (ω) is
β-invariant — ω is sourced from the GT trajectory directly, and after
β-correction M⁻¹·p = ω still holds, so x carries no β factor.
"""
from __future__ import annotations

import argparse
import os
import sys

# Force CPU for JAX (env may have CUDA but cuSolver fails on this host).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, "../../.."))
for p in (PROJECT_ROOT,
          os.path.join(PROJECT_ROOT, "src/utils"),
          os.path.join(PROJECT_ROOT, "datasets"),
          os.path.join(PROJECT_ROOT, "envs"),
          os.path.join(THIS_FILE_DIR, "ph_gp_sde"),
          os.path.join(THIS_FILE_DIR, "ph_nn_ode_fp32")):
    if p not in sys.path:
        sys.path.insert(0, p)

import jax
import jax.numpy as jnp
import equinox as eqx
import torch

from envs.windy_pendulum_3d import windy_pendulum_3d

import importlib.util


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_gp_sde_net = _load_module(
    "_gp_sde_net", os.path.join(THIS_FILE_DIR, "ph_gp_sde", "network.py"))
_nn_ode_net = _load_module(
    "_nn_ode_net", os.path.join(THIS_FILE_DIR, "ph_nn_ode_fp32", "network.py"))
DissipativeSO3HamSDE = _gp_sde_net.DissipativeSO3HamSDE
DissipativeSO3HamNODE = _nn_ode_net.DissipativeSO3HamNODE


# ─────────────────────────────────────────────────────────────────────
# Defaults — match make_comparison_pdf.py
# ─────────────────────────────────────────────────────────────────────
GP_CKPT_DIR = os.path.join(THIS_FILE_DIR, "ph_gp_sde/data/run_wp3d_jax")
NN_CKPT_DIR = os.path.join(THIS_FILE_DIR, "ph_nn_ode_fp32/data/run_wp3d_fp32")
GP_CKPT_FMT = "wp3d-so3hamGPSDE-5p-{step}.eqx"
NN_CKPT_FMT = "wp3d-so3ham-rk4-5p-{step}.tar"

DEFAULT_OUT_PDF = os.path.join(
    THIS_FILE_DIR, "phaseplot_subnetwork_comparision.pdf")

ENV_KW = dict(
    g=9.81, m=1.0, l=1.0, dt=0.05,
    friction_coeff=0.5, varying_friction=False,
    external_force_type="sine", external_force_std=0.0,
    wind_force_std=0.5,
)
N_SUBSTEPS = 10
N_OUTER = 200                  # 200 outer steps of dt=0.05 → 10 s
N_TRAJ_ENSEMBLE = 10           # ensemble size when --ensemble
GT_FRICTION = 0.5
GT_M = 1.0
GT_L = 1.0
GT_G = 9.81
I_PERP = GT_M * GT_L * GT_L

GP_COLOR = "#d62728"
NN_COLOR = "#1f77b4"
GT_COLOR = "#000000"

CKPT_STEPS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]


# ─────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────

def load_gp_sde(step):
    ckpt = os.path.join(GP_CKPT_DIR, GP_CKPT_FMT.format(step=step))
    template = DissipativeSO3HamSDE(
        key=jax.random.PRNGKey(0), u_dim=3, init_gain=0.5, friction=True,
        init_sigma_R=0.5, init_sigma_omega=0.5, init_sigma_obs_omega=0.5,
    )
    return eqx.tree_deserialise_leaves(ckpt, template)


def load_nn_ode(step, device):
    ckpt = os.path.join(NN_CKPT_DIR, NN_CKPT_FMT.format(step=step))
    model = DissipativeSO3HamNODE(
        device=device, u_dim=3, init_gain=0.5, friction=True,
    ).to(device)
    sd = torch.load(ckpt, map_location=device)
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────
# GT rollout
# ─────────────────────────────────────────────────────────────────────

def rollout_gt(env, R0, omega0, u_const, dW_per_outer):
    n_outer, n_sub, _ = dW_per_outer.shape
    h_sub = env.dt / n_sub
    sigma = env.wind_force_std
    R, omega = R0.copy(), omega0.copy()
    traj = np.zeros((n_outer + 1, 12), dtype=np.float64)
    traj[0, :9] = R.reshape(-1)
    traj[0, 9:12] = omega
    t = 0.0
    for k in range(n_outer):
        t += env.dt
        w_force = env.update_wind(t)
        for s in range(n_sub):
            R, omega = env._lie_heun_step(
                R, omega, w_force, u_const, h_sub, sigma, dW_per_outer[k, s])
        traj[k + 1, :9] = R.reshape(-1)
        traj[k + 1, 9:12] = omega
    return traj


# ─────────────────────────────────────────────────────────────────────
# Per-trajectory subnet evaluation (raw, β not yet applied)
# ─────────────────────────────────────────────────────────────────────

def _gp_subnets(model, qs):
    """qs: jnp.ndarray (T, 9). Returns numpy:
       (M_inv (T,3,3), neg_grav_V (T,3), D (T,3,3), B (T,3,3))
    where neg_grav_V = −(q^×)ᵀ · ∇_qV = −Σⱼ rⱼ × ∂_{qⱼ}V is the
    V-only ṗ-contribution exactly as it appears in the PHS spec."""
    def per(q):
        M_inv = model.M_net(q,  inference_mode=True)
        D     = model.Dw_net(q, inference_mode=True)
        B     = model.g_net(q,  inference_mode=True)
        return M_inv, D, B
    M_inv_a, D_a, B_a = jax.vmap(per)(qs)

    def V_q(q):
        return model.V_net(q, inference_mode=True)[0]

    def neg_gravV_one(q):
        dVdq = jax.grad(V_q)(q)
        R    = q.reshape(3, 3)
        dV33 = dVdq.reshape(3, 3)
        return -jnp.sum(jnp.cross(R, dV33, axis=-1), axis=0)
    neg_grav_V = jax.vmap(neg_gravV_one)(qs)

    return (np.asarray(M_inv_a), np.asarray(neg_grav_V),
            np.asarray(D_a),     np.asarray(B_a))


def _nn_subnets(model, qs_np, device):
    """qs_np: (T, 9). Returns numpy 4-tuple:
       (M_inv (T,3,3), neg_grav_V (T,3), D (T,3,3), B (T,3,3))
    where neg_grav_V = −(q^×)ᵀ · ∇_qV = −Σⱼ rⱼ × ∂_{qⱼ}V is the
    V-only ṗ-contribution exactly as it appears in the PHS spec."""
    qs = torch.tensor(qs_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        M_inv = model.M_net(qs).cpu().numpy()
        D     = model.Dw_net(qs).cpu().numpy()
        B     = model.g_net(qs).cpu().numpy()

    qs_v   = qs.clone().requires_grad_(True)
    V_sum  = model.V_net(qs_v).sum()
    dVdq   = torch.autograd.grad(V_sum, qs_v)[0]                  # (T, 9)
    R_v    = qs.view(-1, 3, 3)
    dV33   = dVdq.view(-1, 3, 3)
    neg_grav_V = (-torch.linalg.cross(R_v, dV33, dim=2)
                  .sum(dim=1)).cpu().numpy()

    return M_inv, neg_grav_V, D, B


# ─────────────────────────────────────────────────────────────────────
# β scale-invariance correction (mirrors make_comparison_pdf.estimate_beta)
# ─────────────────────────────────────────────────────────────────────

def _estimate_beta(M_inv_arr, gt_target=1.0 / I_PERP):
    if M_inv_arr is None or M_inv_arr.size == 0:
        return 1.0
    diag_mean = float(np.mean(np.trace(M_inv_arr, axis1=1, axis2=2) / 3.0))
    if diag_mean < 1e-12:
        return 1.0
    return gt_target / diag_mean


# ─────────────────────────────────────────────────────────────────────
# Phase-plot quantities (x, y_M, y_V, y_D, y_B)
# ─────────────────────────────────────────────────────────────────────

def _quantities_model(M_inv, neg_grav_V, D, B, omegas, u_const):
    """All per-axis quantities for a model along one trajectory.
    Shapes: M_inv (T,3,3), neg_grav_V (T,3), D (T,3,3),
            B (T,3,3), omegas (T,3), u_const (3,).
    Returns x, y_M, y_V, y_D, y_B each of shape (T, 3).
    x = ω (angular velocity, β-invariant).
    y_M = p × ω,  y_V = neg_grav_V,  y_D = −D·ω,  y_B = B·u.
    """
    T = omegas.shape[0]
    om = omegas[..., None]                                # (T, 3, 1)
    p  = np.linalg.solve(M_inv, om)[..., 0]               # (T, 3)
    y_M = np.cross(p, omegas, axis=-1)                    # p^× · ∇_pH
    y_D = -(D @ om)[..., 0]                               # −D · ∇_pH
    u_b = np.broadcast_to(u_const.reshape(1, 3, 1), (T, 3, 1))
    y_B = (B @ u_b)[..., 0]
    return omegas, y_M, neg_grav_V, y_D, y_B


def _quantities_gt(traj_12, u_const):
    qs     = traj_12[:, :9]
    omegas = traj_12[:, 9:12]
    T      = omegas.shape[0]
    R      = qs.reshape(T, 3, 3)
    # M⁻¹_GT = (1/(m l²)) I  ⇒  p_GT = m·l²·ω, so p × ω = (m·l²)(ω × ω) = 0.
    y_M = np.zeros_like(omegas)
    # V = m·g·l·R[2,2]; ∂V/∂R has only entry (2,2) = m·g·l, so
    # Σⱼ Rⱼ × ∂_{qⱼ}V = R[2,:] × (m·g·l·ẑ). Spec form is the negative.
    z_force = np.array([0.0, 0.0, GT_M * GT_G * GT_L])
    y_V = -np.cross(R[:, 2, :], z_force[None, :], axis=-1)
    y_D = -GT_FRICTION * omegas
    y_B = np.broadcast_to(u_const.reshape(1, 3), (T, 3)).copy()
    return omegas, y_M, y_V, y_D, y_B


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────

SUBNETS = ["M⁻¹", "V", "D", "B"]
MODELS  = ["GP_SDE", "NN_ODE", "GT"]
COLORS  = {"GP_SDE": GP_COLOR, "NN_ODE": NN_COLOR, "GT": GT_COLOR}
AXIS_LABELS = ["x", "y", "z"]

# matplotlib's mathtext doesn't accept a literal "ₚ" in $...$; use \nabla_p
SUBNET_Y_LABEL = {
    "M⁻¹": r"$(p^{{\times}} \nabla_p H)_{{{i}}} = (p \times \omega)_{{{i}}}$",
    "V":   r"$(-(q^{{\times}})^\top \nabla_q V)_{{{i}}}$",
    "D":   r"$(-D \nabla_p H)_{{{i}}} = -(D \omega)_{{{i}}}$",
    "B":   r"$(B u)_{{{i}}}$",
}
SUBNET_FULL_NAME = {
    "M⁻¹": "Inverse mass M⁻¹",
    "V":   "Potential V",
    "D":   "Dissipation D",
    "B":   "Control input B",
}


def make_page(step, per_model_quants, beta_info, ensemble):
    """per_model_quants[name]: list of (x, y_M, y_V, y_D, y_B), one per traj."""
    fig, axes = plt.subplots(12, 3, figsize=(13, 36))
    rows = [(s, m) for s in SUBNETS for m in MODELS]
    for r, (sub, model) in enumerate(rows):
        col = COLORS[model]
        for c in range(3):
            ax = axes[r, c]
            label_done = False
            for trajs in per_model_quants[model]:
                x, y_M, y_V, y_D, y_B = trajs
                y = {"M⁻¹": y_M, "V": y_V, "D": y_D, "B": y_B}[sub]
                lab = model if not label_done else None
                ax.plot(x[:, c], y[:, c], color=col, lw=1.0,
                        alpha=0.45 if ensemble else 0.9, label=lab)
                # mark trajectory start to disambiguate the parametric loop
                ax.scatter(x[0, c], y[0, c], color=col, s=14, zorder=3,
                           marker="o", edgecolors="white", linewidths=0.6)
                label_done = True
            ax.set_xlabel(rf"$\omega_{AXIS_LABELS[c]}$")
            ax.set_ylabel(SUBNET_Y_LABEL[sub].format(i=AXIS_LABELS[c]))
            ax.grid(True, alpha=0.3)
            if c == 0:
                title = f"{SUBNET_FULL_NAME[sub]}  —  {model}"
                if model in beta_info:
                    title += f"   (β = {beta_info[model]:.3f})"
                ax.set_title(title, loc="left", fontsize=10, fontweight="bold")

    fig.suptitle(
        f"Subnetwork phase plots — checkpoint step = {step}",
        fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.0, 1, 0.985))
    return fig


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def _apply_beta(M_inv, neg_grav_V, D, B, beta):
    """β rescales the model output to the GT scale:
        M⁻¹ → M⁻¹·β  (so β·M⁻¹_β ≈ M⁻¹_GT)
        V   → V/β    ⇒ ∇_qV / β ⇒ neg_grav_V / β
        D   → D/β
        B   → B/β
    x = ω is sourced from the GT trajectory, β-invariant — no rescaling.
    """
    return M_inv * beta, neg_grav_V / beta, D / beta, B / beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_pdf", default=DEFAULT_OUT_PDF)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--u", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                    help="constant body-frame torque held across rollouts")
    ap.add_argument("--ensemble", action="store_true",
                    help="overlay N=%d trajectories per cell instead of 1"
                         % N_TRAJ_ENSEMBLE)
    ap.add_argument("--steps", type=int, nargs="+", default=CKPT_STEPS,
                    help="training-step checkpoints to render (one page each)")
    args = ap.parse_args()

    device  = torch.device(args.device)
    u_const = np.asarray(args.u, dtype=np.float64)

    # ── GT trajectory(ies) ─────────────────────────────────────────
    env_setup = windy_pendulum_3d(seed=args.seed, **ENV_KW)
    env_setup.reset(seed=args.seed)
    R0     = env_setup.R.copy()
    omega0 = env_setup.omega.copy()

    dt_sub = ENV_KW["dt"] / N_SUBSTEPS
    sqrt_h = float(np.sqrt(dt_sub))
    rng    = np.random.default_rng(args.seed)
    n_traj = N_TRAJ_ENSEMBLE if args.ensemble else 1
    dWs = [rng.normal(0.0, sqrt_h, size=(N_OUTER, N_SUBSTEPS, 3))
           for _ in range(n_traj)]

    print(f"Rolling out {n_traj} GT trajectory{'ies' if n_traj > 1 else ''} ...")
    gt_trajs = []
    for ti in range(n_traj):
        env = windy_pendulum_3d(seed=args.seed + ti, **ENV_KW)
        env.reset(seed=args.seed + ti)            # advances RNG; we override dW
        gt_trajs.append(rollout_gt(env, R0, omega0, u_const, dWs[ti]))
    gt_trajs = np.stack(gt_trajs, axis=0)         # (n_traj, T+1, 12)

    # GT y-quantities (no checkpoint dependence)
    gt_quants = [_quantities_gt(gt_trajs[ti], u_const) for ti in range(n_traj)]

    # ── per-checkpoint pages ───────────────────────────────────────
    print(f"Writing PDF: {args.out_pdf}")
    os.makedirs(os.path.dirname(args.out_pdf) or ".", exist_ok=True)
    with PdfPages(args.out_pdf) as pdf:
        for step in args.steps:
            print(f"  step {step} ...")
            gp = load_gp_sde(step)
            nn = load_nn_ode(step, device)

            # raw subnets along each GT trajectory
            gp_raw, nn_raw = [], []
            for ti in range(n_traj):
                qs_np = gt_trajs[ti, :, :9]
                gp_raw.append(_gp_subnets(
                    gp, jnp.asarray(qs_np, dtype=jnp.float32)))
                nn_raw.append(_nn_subnets(
                    nn, qs_np.astype(np.float32), device))

            # one β per model, estimated across all trajectories' M⁻¹
            gp_M_all = np.concatenate([r[0] for r in gp_raw], axis=0)
            nn_M_all = np.concatenate([r[0] for r in nn_raw], axis=0)
            beta_g   = _estimate_beta(gp_M_all)
            beta_n   = _estimate_beta(nn_M_all)
            print(f"    β: GP_SDE={beta_g:.4f}  NN_ODE={beta_n:.4f}")

            gp_quants, nn_quants = [], []
            for ti in range(n_traj):
                omegas = gt_trajs[ti, :, 9:12]
                Mg, gVg, Dg, Bg = _apply_beta(*gp_raw[ti], beta_g)
                Mn, gVn, Dn, Bn = _apply_beta(*nn_raw[ti], beta_n)
                gp_quants.append(_quantities_model(
                    Mg, gVg, Dg, Bg, omegas, u_const))
                nn_quants.append(_quantities_model(
                    Mn, gVn, Dn, Bn, omegas, u_const))

            per_model = {
                "GP_SDE": gp_quants,
                "NN_ODE": nn_quants,
                "GT":     gt_quants,
            }
            beta_info = {"GP_SDE": beta_g, "NN_ODE": beta_n}
            fig = make_page(step, per_model, beta_info, args.ensemble)
            pdf.savefig(fig); plt.close(fig)

    print(f"Done: {args.out_pdf}")


if __name__ == "__main__":
    main()
