"""Subnet vector-field plots on a regular (φ_i, p_i) grid.

One PDF page per training-step checkpoint in {1k, 2k, ..., 10k}. Page layout:

    cols = body axis i ∈ {x, y, z}  (roll/pitch/yaw on x; p_x/p_y/p_z on y)
    rows = 12  (subnet × model):
        M⁻¹ (GP_SDE), M⁻¹ (NN_ODE), M⁻¹ (GT),
        V   (GP_SDE), V   (NN_ODE), V   (GT),
        D   (GP_SDE), D   (NN_ODE), D   (GT),
        B   (GP_SDE), B   (NN_ODE), B   (GT)

Each cell shows a regular vector field (quiver) on a 2D grid in the
(φ_i, p_i) plane. The OTHER eight Euler / two p coordinates are fixed to
the mean of the model's predicted (q, p) trajectory at that checkpoint
(so the slice is anchored at "the operating regime" the model actually
visits). The grid spans the trajectory's (φ_i, p_i) range with light
padding.

At every grid point (φ_i, p_i):
    1. Reconstruct full state:  full_eulers = mean_eulers but with
       [i] = grid φ_i;  full_p = mean_p but with [i] = grid p_i.
    2. R = ZYX_intrinsic(full_eulers).  Body-frame ω = M⁻¹(R) · full_p.
    3. Arrow x-component:  φ̇_i, computed from ω via the analytic ZYX
       Euler-rate matrix.
    4. Arrow y-component:  ṗ_subnet_i — that row's PHS subnet contribution
       to ṗ component i:
            M⁻¹ : (p × ω)_i                = (p^× · ∇_pH)_i
            V   : (−Σⱼ rⱼ × ∂_{qⱼ}V)_i     = (−(q^×)ᵀ ∇_qV)_i  (potential-only)
            D   : −(D · ω)_i                = (−D · ∇_pH)_i
            B   : (B · u)_i

Predicted trajectory (used only for grid range and slice anchor — NOT
plotted as a line):
    GP_SDE : Lie-Heun SDE rollout from src.utils.JAX.lie_integrator
    NN_ODE : torchdiffeq RK4 rollout
    GT     : env Lie-Heun rollout (envs.windy_pendulum_3d)
all from a shared (R0, ω0) and dW.

β scale-invariance correction (from M⁻¹ along the predicted trajectory):
    M⁻¹ → M⁻¹·β,  V → V/β,  D → D/β,  B → B/β,  p → p_raw/β.
ω = M⁻¹·p is β-invariant; φ̇_i is β-invariant; the subnet ṗ-arrow
y-components carry 1/β.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

# Force CPU for JAX (the env may have CUDA but cuSolver fails on this host).
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
from torchdiffeq import odeint

from envs.windy_pendulum_3d import windy_pendulum_3d
from src.utils.JAX.lie_integrator import lie_heun_sde_rollout


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
# Defaults
# ─────────────────────────────────────────────────────────────────────
GP_CKPT_DIR = os.path.join(THIS_FILE_DIR, "ph_gp_sde/data/run_wp3d_jax")
NN_CKPT_DIR = os.path.join(THIS_FILE_DIR, "ph_nn_ode_fp32/data/run_wp3d_fp32")
GP_CKPT_FMT = "wp3d-so3hamGPSDE-5p-{step}.eqx"
NN_CKPT_FMT = "wp3d-so3ham-rk4-5p-{step}.tar"

DEFAULT_OUT_PDF = os.path.join(
    THIS_FILE_DIR, "vectorfield_subnetwork_comparision.pdf")

ENV_KW = dict(
    g=9.81, m=1.0, l=1.0, dt=0.05,
    friction_coeff=0.5, varying_friction=False,
    external_force_type="sine", external_force_std=0.0,
    wind_force_std=0.5,
)
N_SUBSTEPS = 10
N_OUTER    = 200
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
# Predicted-trajectory rollouts (used for slice mean + grid range only)
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


def rollout_gp_sde(model, R0, omega0, u_const, dW_per_outer, h_sub):
    x0 = jnp.concatenate(
        [jnp.asarray(R0).reshape(-1), jnp.asarray(omega0)]).astype(jnp.float32)
    u_jnp = jnp.asarray(u_const, dtype=jnp.float32)
    dW_jnp = jnp.asarray(dW_per_outer, dtype=jnp.float32)
    traj = lie_heun_sde_rollout(model, x0, u_jnp, jnp.float32(h_sub), dW_jnp)
    return np.asarray(traj)


def rollout_nn_ode(model, R0, omega0, u_const, t_eval, device):
    x0 = np.concatenate([R0.reshape(-1), omega0, u_const]).astype(np.float32)
    x0_t = torch.tensor(x0[None, :], dtype=torch.float32,
                        device=device, requires_grad=True)
    t_t = torch.tensor(t_eval, dtype=torch.float32, device=device)
    traj = odeint(model, x0_t, t_t, method="rk4")
    return traj[:, 0, :].detach().cpu().numpy()       # (T, 15)


# ─────────────────────────────────────────────────────────────────────
# Geometry: ZYX intrinsic Euler ↔ rotation matrix and Euler-rate
# ─────────────────────────────────────────────────────────────────────
# Convention: R = Rz(yaw) · Ry(pitch) · Rx(roll). Matches the
# rotmat_to_euler in make_comparison_pdf.py:
#     roll  = atan2(R[2,1], R[2,2])
#     pitch = atan2(-R[2,0], sqrt(R[0,0]² + R[1,0]²))
#     yaw   = atan2(R[1,0], R[0,0])

def rotmat_to_euler(R_flat):
    Rs = np.asarray(R_flat).reshape(-1, 3, 3)
    R00, R10, R20 = Rs[:, 0, 0], Rs[:, 1, 0], Rs[:, 2, 0]
    R21, R22, R12, R11 = Rs[:, 2, 1], Rs[:, 2, 2], Rs[:, 1, 2], Rs[:, 1, 1]
    sy = np.sqrt(R00 ** 2 + R10 ** 2)
    near = sy < 1e-6
    roll  = np.where(near, np.arctan2(-R12, R11), np.arctan2(R21, R22))
    pitch = np.arctan2(-R20, sy)
    yaw   = np.where(near, 0.0, np.arctan2(R10, R00))
    return np.stack([roll, pitch, yaw], axis=-1)


def euler_zyx_to_R(eulers):
    """eulers: (..., 3) = (roll, pitch, yaw). Returns R: (..., 3, 3)."""
    phi   = eulers[..., 0]
    theta = eulers[..., 1]
    psi   = eulers[..., 2]
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth,  sth  = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    R = np.empty(eulers.shape[:-1] + (3, 3), dtype=eulers.dtype)
    R[..., 0, 0] = cpsi * cth
    R[..., 0, 1] = cpsi * sth * sphi - spsi * cphi
    R[..., 0, 2] = cpsi * sth * cphi + spsi * sphi
    R[..., 1, 0] = spsi * cth
    R[..., 1, 1] = spsi * sth * sphi + cpsi * cphi
    R[..., 1, 2] = spsi * sth * cphi - cpsi * sphi
    R[..., 2, 0] = -sth
    R[..., 2, 1] = cth * sphi
    R[..., 2, 2] = cth * cphi
    return R


def omega_to_euler_rate_zyx(eulers, omegas, eps=1e-6):
    """Body-frame ω → (φ̇, θ̇, ψ̇) for ZYX intrinsic Euler.

        φ̇ = ω_x + (ω_y sin φ + ω_z cos φ) tan θ
        θ̇ = ω_y cos φ − ω_z sin φ
        ψ̇ = (ω_y sin φ + ω_z cos φ) / cos θ

    Floor cos θ to avoid the gimbal singularity at θ = ±π/2.
    """
    phi   = eulers[..., 0]
    theta = eulers[..., 1]
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    cth_safe = np.where(np.abs(cth) > eps, cth, np.sign(cth + eps) * eps)
    tth = sth / cth_safe
    wx, wy, wz = omegas[..., 0], omegas[..., 1], omegas[..., 2]
    phi_dot   = wx + (wy * sphi + wz * cphi) * tth
    theta_dot = wy * cphi - wz * sphi
    psi_dot   = (wy * sphi + wz * cphi) / cth_safe
    return np.stack([phi_dot, theta_dot, psi_dot], axis=-1)


# ─────────────────────────────────────────────────────────────────────
# Subnet evaluation at arbitrary q-states
# ─────────────────────────────────────────────────────────────────────

def _gp_subnet_outputs(model, qs):
    """qs: jnp (T, 9). Returns numpy (M_inv (T,3,3), neg_grav_V (T,3),
    D (T,3,3), B (T,3,3))."""
    def per(q):
        return (model.M_net(q,  inference_mode=True),
                model.Dw_net(q, inference_mode=True),
                model.g_net(q,  inference_mode=True))
    M_inv, D, B = jax.vmap(per)(qs)

    def V_q(q):
        return model.V_net(q, inference_mode=True)[0]
    def neg_gravV_one(q):
        dVdq = jax.grad(V_q)(q)
        R    = q.reshape(3, 3)
        dV33 = dVdq.reshape(3, 3)
        return -jnp.sum(jnp.cross(R, dV33, axis=-1), axis=0)
    neg_grav_V = jax.vmap(neg_gravV_one)(qs)
    return (np.asarray(M_inv), np.asarray(neg_grav_V),
            np.asarray(D),     np.asarray(B))


def _nn_subnet_outputs(model, qs_np, device):
    qs = torch.tensor(qs_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        M_inv = model.M_net(qs).cpu().numpy()
        D     = model.Dw_net(qs).cpu().numpy()
        B     = model.g_net(qs).cpu().numpy()
    qs_v   = qs.clone().requires_grad_(True)
    V_sum  = model.V_net(qs_v).sum()
    dVdq   = torch.autograd.grad(V_sum, qs_v)[0]
    R_v    = qs.view(-1, 3, 3)
    dV33   = dVdq.view(-1, 3, 3)
    neg_grav_V = (-torch.linalg.cross(R_v, dV33, dim=2)
                  .sum(dim=1)).cpu().numpy()
    return M_inv, neg_grav_V, D, B


def _gt_subnet_outputs(qs_np, T):
    R = qs_np.reshape(T, 3, 3)
    M_inv = np.tile(np.eye(3) / I_PERP, (T, 1, 1))
    z_force = np.array([0.0, 0.0, GT_M * GT_G * GT_L])
    neg_grav_V = -np.cross(R[:, 2, :], z_force[None, :], axis=-1)
    D = np.tile(GT_FRICTION * np.eye(3), (T, 1, 1))
    B = np.tile(np.eye(3), (T, 1, 1))
    return M_inv, neg_grav_V, D, B


def _estimate_beta(M_inv_arr, gt_target=1.0 / I_PERP):
    if M_inv_arr is None or M_inv_arr.size == 0:
        return 1.0
    diag_mean = float(np.mean(np.trace(M_inv_arr, axis1=1, axis2=2) / 3.0))
    if diag_mean < 1e-12:
        return 1.0
    return gt_target / diag_mean


# ─────────────────────────────────────────────────────────────────────
# Grid construction in (φ_i, p_i)
# ─────────────────────────────────────────────────────────────────────

def _grid_axis(lo_hi, n, pad=0.1):
    lo, hi = lo_hi
    span = max(hi - lo, 1e-3)
    return np.linspace(lo - pad * span, hi + pad * span, n)


def _build_grid_states(eulers_traj, p_traj, axis_i, n_grid):
    """Regular grid in (φ_axis_i, p_axis_i). Other 8 Euler / 2 p
    coordinates fixed to the trajectory mean.
    Returns: eulers_grid (N², 3), p_grid (N², 3), Phi_mesh (N, N), P_mesh (N, N)."""
    mean_eulers = np.mean(eulers_traj, axis=0)
    mean_p      = np.mean(p_traj,      axis=0)

    phi_axis = _grid_axis(
        (eulers_traj[:, axis_i].min(), eulers_traj[:, axis_i].max()), n_grid)
    p_axis   = _grid_axis(
        (p_traj[:, axis_i].min(),      p_traj[:, axis_i].max()),      n_grid)

    Phi_mesh, P_mesh = np.meshgrid(phi_axis, p_axis, indexing="xy")
    N = Phi_mesh.size

    eulers_grid = np.broadcast_to(mean_eulers[None, :], (N, 3)).copy()
    eulers_grid[:, axis_i] = Phi_mesh.ravel()
    p_grid = np.broadcast_to(mean_p[None, :], (N, 3)).copy()
    p_grid[:, axis_i] = P_mesh.ravel()

    return eulers_grid, p_grid, Phi_mesh, P_mesh


# ─────────────────────────────────────────────────────────────────────
# Per-model processing: subnet evaluation on the (φ_i, p_i) grid
# ─────────────────────────────────────────────────────────────────────

def _eval_subnets(name, qs_np, gp_model=None, nn_model=None, device=None):
    if name == "GP_SDE":
        return _gp_subnet_outputs(gp_model, jnp.asarray(qs_np, dtype=jnp.float32))
    if name == "NN_ODE":
        return _nn_subnet_outputs(nn_model, qs_np.astype(np.float32), device)
    return _gt_subnet_outputs(qs_np, qs_np.shape[0])


def _process_model(name, traj_12, n_grid, u_const,
                   gp_model=None, nn_model=None, device=None):
    qs_traj     = traj_12[:, :9]
    omegas_traj = traj_12[:, 9:12]
    eulers_traj = rotmat_to_euler(qs_traj)

    # β + trajectory p — needed for the grid range and for the slice's mean p.
    M_inv_traj, _, _, _ = _eval_subnets(
        name, qs_traj, gp_model=gp_model, nn_model=nn_model, device=device)
    if name == "GT":
        beta = 1.0
        p_traj = I_PERP * omegas_traj                 # GT: M = m·l²·I, β = 1
    else:
        beta = _estimate_beta(M_inv_traj)
        # PHS gauge: M⁻¹_phys = β·M⁻¹_model, p_phys = p_model/β. So
        # p_phys = solve(β·M⁻¹_model, ω) = solve(M⁻¹_model, ω) / β = p_raw/β.
        # This is consistent with the M⁻¹·β / V/β / D/β / B/β scaling applied
        # to the four subnets at the grid (lines 398-401), so ω = M⁻¹·p
        # is gauge-invariant and every ṗ-contribution carries a clean 1/β.
        p_traj = np.linalg.solve(
            M_inv_traj * beta, omegas_traj[..., None])[..., 0]

    color = {"GP_SDE": GP_COLOR, "NN_ODE": NN_COLOR, "GT": GT_COLOR}[name]

    per_axis = []
    for i in range(3):
        eulers_grid, p_grid, Phi_mesh, P_mesh = _build_grid_states(
            eulers_traj, p_traj, i, n_grid)

        qs_grid = euler_zyx_to_R(eulers_grid).reshape(-1, 9)

        M_inv, neg_grav_V, D, B = _eval_subnets(
            name, qs_grid, gp_model=gp_model, nn_model=nn_model, device=device)
        if name != "GT":
            M_inv      = M_inv      * beta
            neg_grav_V = neg_grav_V / beta
            D          = D          / beta
            B          = B          / beta

        N2 = qs_grid.shape[0]
        omegas_grid = (M_inv @ p_grid[..., None])[..., 0]              # (N², 3)
        phi_dot = omega_to_euler_rate_zyx(eulers_grid, omegas_grid)    # (N², 3)

        # subnet ṗ-contributions
        y_M = np.cross(p_grid, omegas_grid, axis=-1)                   # (N², 3)
        y_V = neg_grav_V                                               # (N², 3)
        y_D = -(D @ omegas_grid[..., None])[..., 0]                    # (N², 3)
        u_b = np.broadcast_to(u_const.reshape(1, 3, 1), (N2, 3, 1))
        y_B = (B @ u_b)[..., 0]                                        # (N², 3)

        shape2d = Phi_mesh.shape
        per_axis.append(dict(
            Phi_mesh = Phi_mesh,
            P_mesh   = P_mesh,
            phi_dot_i = phi_dot[:, i].reshape(shape2d),
            y_M_i = y_M[:, i].reshape(shape2d),
            y_V_i = y_V[:, i].reshape(shape2d),
            y_D_i = y_D[:, i].reshape(shape2d),
            y_B_i = y_B[:, i].reshape(shape2d),
        ))

    return dict(per_axis=per_axis, color=color), beta


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────

SUBNETS = ["M⁻¹", "V", "D", "B"]
MODELS  = ["GP_SDE", "NN_ODE", "GT"]
SUBNET_FULL = {
    "M⁻¹": "Inverse mass M⁻¹  (ṗ-contribution: p × ω)",
    "V":   "Potential V  (ṗ-contribution: −Σⱼ rⱼ × ∂_{qⱼ}V)",
    "D":   "Dissipation D  (ṗ-contribution: −D · ω)",
    "B":   "Control input B  (ṗ-contribution: B · u)",
}
SUBNET_KEY = {"M⁻¹": "y_M_i", "V": "y_V_i", "D": "y_D_i", "B": "y_B_i"}
EULER_LABEL = [r"$\phi_x$  (roll, rad)",
               r"$\phi_y$  (pitch, rad)",
               r"$\phi_z$  (yaw, rad)"]
P_LABEL = [r"$p_x$", r"$p_y$", r"$p_z$"]


def make_page(step, per_model_data, betas):
    fig, axes = plt.subplots(12, 3, figsize=(14, 38))
    rows = [(s, m) for s in SUBNETS for m in MODELS]

    for r, (sub, model) in enumerate(rows):
        d = per_model_data[model]
        col = d["color"]
        for c in range(3):
            ax = axes[r, c]
            a = d["per_axis"][c]
            ax.quiver(a["Phi_mesh"], a["P_mesh"],
                      a["phi_dot_i"], a[SUBNET_KEY[sub]],
                      color=col, alpha=0.8,
                      width=0.005, headwidth=4, headlength=5)
            ax.set_xlabel(EULER_LABEL[c])
            ax.set_ylabel(P_LABEL[c])
            ax.grid(True, alpha=0.3)
            if c == 0:
                title = f"{SUBNET_FULL[sub]}  —  {model}"
                if model in betas:
                    title += f"   (β = {betas[model]:.3f})"
                ax.set_title(title, loc="left", fontsize=9.5,
                             fontweight="bold")

    fig.suptitle(
        f"Subnet vector fields on (φ_i, p_i) grid  —  step = {step}",
        fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.0, 1, 0.985))
    return fig


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_pdf", default=DEFAULT_OUT_PDF)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--u", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                    help="constant body-frame torque held across rollouts")
    ap.add_argument("--steps", type=int, nargs="+", default=CKPT_STEPS,
                    help="training-step checkpoints to render (one page each)")
    ap.add_argument("--n_grid", type=int, default=14,
                    help="grid resolution per axis (n_grid × n_grid arrows)")
    args = ap.parse_args()

    device  = torch.device(args.device)
    u_const = np.asarray(args.u, dtype=np.float64)

    env_setup = windy_pendulum_3d(seed=args.seed, **ENV_KW)
    env_setup.reset(seed=args.seed)
    R0     = env_setup.R.copy()
    omega0 = env_setup.omega.copy()

    dt_sub = ENV_KW["dt"] / N_SUBSTEPS
    sqrt_h = float(np.sqrt(dt_sub))
    rng    = np.random.default_rng(args.seed)
    dW     = rng.normal(0.0, sqrt_h, size=(N_OUTER, N_SUBSTEPS, 3))
    t_eval = np.arange(N_OUTER + 1) * ENV_KW["dt"]

    print("Rolling out GT trajectory ...")
    env_gt = windy_pendulum_3d(seed=args.seed, **ENV_KW)
    env_gt.reset(seed=args.seed)
    gt_traj = rollout_gt(env_gt, R0, omega0, u_const, dW)
    gt_data, _ = _process_model(
        "GT", gt_traj, args.n_grid, u_const)

    print(f"Writing PDF: {args.out_pdf}")
    os.makedirs(os.path.dirname(args.out_pdf) or ".", exist_ok=True)
    with PdfPages(args.out_pdf) as pdf:
        for step in args.steps:
            print(f"  step {step} ...")
            gp = load_gp_sde(step)
            nn = load_nn_ode(step, device)

            gp_traj = rollout_gp_sde(gp, R0, omega0, u_const, dW, dt_sub)
            nn_traj = rollout_nn_ode(nn, R0, omega0, u_const,
                                     t_eval, device)[:, :12]

            gp_data, beta_g = _process_model(
                "GP_SDE", gp_traj, args.n_grid, u_const, gp_model=gp)
            nn_data, beta_n = _process_model(
                "NN_ODE", nn_traj, args.n_grid, u_const,
                nn_model=nn, device=device)
            print(f"    β: GP_SDE={beta_g:.4f}  NN_ODE={beta_n:.4f}")

            data  = {"GP_SDE": gp_data, "NN_ODE": nn_data, "GT": gt_data}
            betas = {"GP_SDE": beta_g, "NN_ODE": beta_n}
            fig = make_page(step, data, betas)
            pdf.savefig(fig); plt.close(fig)

    print(f"Done: {args.out_pdf}")


if __name__ == "__main__":
    main()
