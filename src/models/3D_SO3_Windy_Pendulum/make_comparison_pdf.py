"""Compare ph_gp_sde, ph_nn_ode_fp32, and neural_sde on the 3D windy pendulum.

Mirrors the structure of the legacy `Report_3D_Extended.pdf` generator: a
10-trajectory ensemble for state/energy comparisons (mean ±2σ bands), per-
component 3×3 matrix grids for M⁻¹/Dw/g, a scalar V(q) page, phase portraits,
and a 5-trajectory subnetwork-evolution suite at the bottom.

Output PDF pages (in order):
  A. Loss curves
     1. Train MSE (rotation + ω)        — NN_ODE vs GP_SDE
     2. Train geodesic error               — NN_ODE vs GP_SDE
     3. Train MSE angular vel            — NN_ODE vs GP_SDE
     4. Test MSE (rotation + ω)
     5. Test geodesic error
     6. Test MSE angular velocity
     7. Total NLL  (GP_SDE)
     8. Total KL   (GP_SDE)
     9. Per-subnet KL (M, V, Dw, g, sigma)

  B. Ensemble dynamics (10 GT trajectories, 10 model rollouts each)
     10. Hamiltonian energy E(t)         — GT vs NN_ODE vs GP_SDE  (mean ±2σ)
     11. Hamiltonian energy — single trajectory
     12. SO(3) violation: |det(R) − 1|
     13. SO(3) violation: ‖RᵀR − I‖_F
     14. State trajectories (Euler angles + ω) — mean ±2σ
     15. State trajectories — single trajectory
     16. Phase portraits (Euler angle vs ω, 3 panels)

  C. Subnetwork evolution along 5 GT trajectories
     17. Inverse mass M⁻¹(q) — 3×3 grid
     18. Dissipation  Dw(q)  — 3×3 grid
     19. Control gain g(q)   — 3×3 grid
     20. Potential V(q)
     21. Diffusion σ(q)      — 3 panels (ω-axes)
"""
from __future__ import annotations

import argparse
import os
import pickle
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
          os.path.join(THIS_FILE_DIR, "ph_nn_ode_fp32"),
          os.path.join(THIS_FILE_DIR, "neural_sde")):
    if p not in sys.path:
        sys.path.insert(0, p)

import jax
import jax.numpy as jnp
import equinox as eqx
import torch
from torchdiffeq import odeint

from envs.windy_pendulum_3d import windy_pendulum_3d
from src.utils.JAX.lie_integrator import lie_heun_sde_rollout

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
_neural_sde_net = _load_module(
    "_neural_sde_net", os.path.join(THIS_FILE_DIR, "neural_sde", "network.py"))
DissipativeSO3HamSDE = _gp_sde_net.DissipativeSO3HamSDE
DissipativeSO3HamNODE = _nn_ode_net.DissipativeSO3HamNODE
NeuralSO3SDE = _neural_sde_net.NeuralSO3SDE


# ──────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────

DEFAULT_GP_SDE_CKPT = os.path.join(
    THIS_FILE_DIR,
    "ph_gp_sde/data/run_wp3d_jax/wp3d-so3hamGPSDE-5p-10000.eqx")
DEFAULT_GP_SDE_STATS = os.path.join(
    THIS_FILE_DIR,
    "ph_gp_sde/data/run_wp3d_jax/wp3d-so3hamGPSDE-5p-stats.pkl")
DEFAULT_NN_ODE_CKPT = os.path.join(
    THIS_FILE_DIR,
    "ph_nn_ode_fp32/data/run_wp3d_fp32/wp3d-so3ham-rk4-5p-10000.tar")
DEFAULT_NN_ODE_STATS = os.path.join(
    THIS_FILE_DIR,
    "ph_nn_ode_fp32/data/run_wp3d_fp32/wp3d-so3ham-rk4-5p-stats.pkl")
DEFAULT_NEURAL_SDE_CKPT = os.path.join(
    THIS_FILE_DIR,
    "neural_sde/data/run_wp3d_neural_sde/wp3d-neuralSDE-5p-10000.eqx")
DEFAULT_NEURAL_SDE_STATS = os.path.join(
    THIS_FILE_DIR,
    "neural_sde/data/run_wp3d_neural_sde/wp3d-neuralSDE-5p-stats.pkl")

# Show training stats only up to this step (matches checkpoint training-step
# count). Set to None to show full curves.
STATS_MAX_STEP = 10000
DEFAULT_OUT_PDF = os.path.join(
    THIS_FILE_DIR, "comparison_gp_sde_vs_nn_ode_vs_neural_sde.pdf")

# Env params (matches latest training command:
#   --friction_coeff 0.5 --external_force_std 0.0
#   --wind_force_std 0.5 --obs_noise_std 0.01)
# Note: obs_noise_std doesn't enter the env's continuous-time SDE; it only
# perturbs dataset snapshots for training. So it's omitted here.
ENV_KW = dict(
    g=9.81, m=1.0, l=1.0, dt=0.05,
    friction_coeff=0.5, varying_friction=False,
    external_force_type="sine", external_force_std=0.0,
    wind_force_std=0.5,
)
N_SUBSTEPS = 10
N_OUTER = 200                    # 200 outer steps of dt=0.05 → 10 s
N_TRAJ_ENSEMBLE = 10             # ensemble for state / energy / SO(3) plots
N_SUB_TRAJ = 5                   # trajectories for subnet-evolution plots
GT_FRICTION = 0.5
GT_M = 1.0
GT_L = 1.0
GT_G = 9.81

# Inertia tensor used for the ensemble Hamiltonian curve (matches env's I = m*l^2 * I3)
I_PERP = GT_M * GT_L * GT_L
I_PARA = I_PERP   # spherical pendulum

GP_COLOR = "#d62728"
NN_COLOR = "#1f77b4"
NEURAL_SDE_COLOR = "#2ca02c"     # green — unstructured neural SDE
GP_GT_COLOR = "#8c2d2d"      # dark red — GP_SDE integrator + GT subnets
NN_GT_COLOR = "#0d3b66"      # dark blue — NN_ODE integrator + GT subnets


# ──────────────────────────────────────────────────────────────────────────
# GT-subnet wrappers (algorithm sanity check)
# ──────────────────────────────────────────────────────────────────────────
# These two models use the *same* integrators as ph_gp_sde / ph_nn_ode_fp32
# but inject the analytic spherical-pendulum dynamics in place of the learned
# subnets. They isolate algorithm correctness from learned-subnet quality —
# if these match the env GT, the integrator math is right and any remaining
# gap with the learned models is due to subnet learning.

class GTSDEModel:
    """Drop-in for DissipativeSO3HamSDE. Implements the same drift /
    stochastic_increment interface using the analytic port-Hamiltonian
    dynamics of the env (envs/windy_pendulum_3d::_compute_omega_rates)."""

    def __init__(self, m=1.0, g=9.81, l=1.0, friction=0.5, sigma=0.0):
        self.m = float(m); self.g = float(g); self.l = float(l)
        self.friction = float(friction)
        self._sigma = float(sigma)
        # I = m·l² · I₃  for the spherical pendulum (m=l=1 here)
        self._I = m * l * l

    def drift(self, q, q_dot, u, keys=None):
        R = q.reshape(3, 3)
        ez = jnp.array([0.0, 0.0, 1.0], dtype=q.dtype)
        r_world = self.l * (R @ ez)
        Fg = -self.m * self.g * ez
        tau_g_body = R.T @ jnp.cross(r_world, Fg)
        tau_fric = self.friction * q_dot
        tau_det = tau_g_body + u - tau_fric
        Iw = self._I * q_dot
        return (tau_det - jnp.cross(q_dot, Iw)) / self._I

    def stochastic_increment(self, q, dW, keys=None):
        if self._sigma <= 0.0:
            return jnp.zeros(3, dtype=q.dtype)
        R = q.reshape(3, 3)
        ez = jnp.array([0.0, 0.0, 1.0], dtype=q.dtype)
        r_world = self.l * (R @ ez)
        dF = self._sigma * dW
        return (R.T @ jnp.cross(r_world, dF)) / self._I

    # Optional: subnet-style accessors so eval_subnets_gp-style code keeps
    # working if anyone passes this model to it. Not used by the rollout.
    def sigma(self, q, key=None):
        return jnp.asarray(self._sigma, dtype=q.dtype)


class GTNODEModel(torch.nn.Module):
    """Drop-in for DissipativeSO3HamNODE. Forward returns dx/dt for the
    analytic spherical pendulum, in the 15-dim layout [q (9), q_dot (3), u (3)]."""

    def __init__(self, m=1.0, g=9.81, l=1.0, friction=0.5, u_dim=3, device="cpu"):
        super().__init__()
        self.m = float(m); self.g = float(g); self.l = float(l)
        self.friction = float(friction)
        self.u_dim = int(u_dim)
        self.device = device
        self._I = m * l * l
        self.nfe = 0

    def forward(self, t, x):
        # x shape (B, 15)
        bs = x.shape[0]
        q, q_dot, u = torch.split(x, [9, 3, self.u_dim], dim=1)
        R = q.view(-1, 3, 3)
        ez = torch.tensor([0.0, 0.0, 1.0], dtype=x.dtype, device=x.device)
        r_world = self.l * torch.einsum("bij,j->bi", R, ez)
        Fg = -self.m * self.g * ez                                    # (3,)
        Fg_b = Fg.unsqueeze(0).expand(bs, -1)
        tau_g_world = torch.linalg.cross(r_world, Fg_b, dim=1)
        tau_g_body = torch.einsum("bji,bj->bi", R, tau_g_world)
        tau_fric = self.friction * q_dot
        tau_det = tau_g_body + u - tau_fric
        Iw = self._I * q_dot
        ddq = (tau_det - torch.linalg.cross(q_dot, Iw, dim=1)) / self._I
        # dq = R · [ω]× expressed as 9-vec; here use the same row-cross
        # convention the learned model uses (R rows × dHdp).
        dHdp = q_dot                                                   # M⁻¹·p with M=I
        dHdp_b = dHdp.unsqueeze(1).expand(-1, 3, -1)
        dq = torch.linalg.cross(R, dHdp_b, dim=2).reshape(bs, 9)
        zero_u = torch.zeros_like(u)
        self.nfe += 1
        return torch.cat([dq, ddq, zero_u], dim=1)


# ──────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────

def _resolve_gp_ckpt(ckpt_path):
    if os.path.exists(ckpt_path):
        return ckpt_path
    d = os.path.dirname(ckpt_path)
    if not os.path.isdir(d):
        raise FileNotFoundError(ckpt_path)
    import re
    cands = []
    for fn in os.listdir(d):
        m = re.match(r"(.+?-\d+p)-(\d+)\.eqx$", fn)
        if m:
            cands.append((int(m.group(2)), os.path.join(d, fn)))
    if not cands:
        raise FileNotFoundError(ckpt_path)
    cands.sort()
    chosen = cands[-1][1]
    print(f"  ! requested ckpt missing; falling back to latest: {os.path.basename(chosen)}")
    return chosen


def _resolve_nn_ckpt(ckpt_path):
    if os.path.exists(ckpt_path):
        return ckpt_path
    d = os.path.dirname(ckpt_path)
    if not os.path.isdir(d):
        raise FileNotFoundError(ckpt_path)
    import re
    cands = []
    for fn in os.listdir(d):
        m = re.match(r"(.+?-\d+p)-(\d+)\.tar$", fn)
        if m:
            cands.append((int(m.group(2)), os.path.join(d, fn)))
    if not cands:
        raise FileNotFoundError(ckpt_path)
    cands.sort()
    chosen = cands[-1][1]
    print(f"  ! requested ckpt missing; falling back to latest: {os.path.basename(chosen)}")
    return chosen


def load_gp_sde(ckpt_path, init_sigma_const=0.1):
    """Load a trained ph_gp_sde checkpoint.

    `init_sigma_const` is now ignored — the static softplus bias on
    sigma_net was removed and σ(q) = softplus(GP_raw(q)) carries no
    template-vs-train mismatch risk. Kept as a kwarg for backward-CLI
    compat; will be removed in a future cleanup.
    """
    del init_sigma_const
    ckpt_path = _resolve_gp_ckpt(ckpt_path)
    template = DissipativeSO3HamSDE(
        key=jax.random.PRNGKey(0),
        u_dim=3, init_gain=0.5, friction=True,
        init_sigma_R=0.5, init_sigma_omega=0.5,
        init_sigma_obs_omega=0.5,
    )
    return eqx.tree_deserialise_leaves(ckpt_path, template)


def load_neural_sde(ckpt_path, hidden_dim=500):
    """Load a trained neural_sde checkpoint into a fresh NeuralSO3SDE template.

    The default `hidden_dim=500` matches the train.py default; pass a different
    value if a non-default width was used at training time.
    """
    ckpt_path = _resolve_gp_ckpt(ckpt_path)         # same '*-{step}.eqx' regex
    template = NeuralSO3SDE(
        key=jax.random.PRNGKey(0), u_dim=3,
        hidden_dim=hidden_dim, init_gain=1.0,
    )
    return eqx.tree_deserialise_leaves(ckpt_path, template)


def load_nn_ode(ckpt_path, device):
    ckpt_path = _resolve_nn_ckpt(ckpt_path)
    model = DissipativeSO3HamNODE(
        device=device, u_dim=3, init_gain=0.5, friction=True,
    ).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_stats(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────────────────────────────
# Rollouts
# ──────────────────────────────────────────────────────────────────────────

def rollout_gt(env, R0, omega0, u_const, dW_per_outer):
    """Drive the env's `_lie_heun_step` with externally supplied dW."""
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


def rollout_neural_sde(model, R0, omega0, u_const, dW_per_outer, h_sub):
    """Driver for the unstructured neural SDE — flat ℝ¹² Stratonovich Heun.

    External I/O matches `rollout_gp_sde`; internally calls `model.rollout`,
    which itself does (n_outer, n_substeps, 3) → (n_outer+1, 12).
    """
    x0 = jnp.concatenate(
        [jnp.asarray(R0).reshape(-1), jnp.asarray(omega0)]).astype(jnp.float32)
    u_jnp = jnp.asarray(u_const, dtype=jnp.float32)
    dW_jnp = jnp.asarray(dW_per_outer, dtype=jnp.float32)
    traj = model.rollout(x0, u_jnp, jnp.float32(h_sub), dW_jnp)
    return np.asarray(traj)


def rollout_nn_ode(model, R0, omega0, u_const, t_eval, device):
    x0 = np.concatenate([R0.reshape(-1), omega0, u_const]).astype(np.float32)
    x0_t = torch.tensor(x0[None, :], dtype=torch.float32,
                        device=device, requires_grad=True)
    t_t = torch.tensor(t_eval, dtype=torch.float32, device=device)
    traj = odeint(model, x0_t, t_t, method="rk4")
    return traj[:, 0, :].detach().cpu().numpy()           # (T, 15)


# ──────────────────────────────────────────────────────────────────────────
# Subnet evaluations (per-state lift)
# ──────────────────────────────────────────────────────────────────────────

def eval_subnets_gp(model, traj_12):
    qs = jnp.asarray(traj_12[:, :9], dtype=jnp.float32)

    def per_state(q):
        # σ(q) with the full pipeline: softplus(GP_raw(q) + b).
        sigma_full = model.sigma(q)
        # Raw GP scalar output — no softplus, no bias. Isolates exactly
        # what `sigma_net` learned.
        raw = model.sigma_net(q, inference_mode=True)[0]
        return (model.M_net(q, inference_mode=True),
                model.V_net(q, inference_mode=True)[0],
                model.Dw_net(q, inference_mode=True),
                model.g_net(q, inference_mode=True),
                sigma_full,
                raw)

    M_inv, V, Dw, g, sigma, sigma_raw = jax.vmap(per_state)(qs)
    T = qs.shape[0]
    return {
        "M": np.asarray(M_inv),       # (T, 3, 3)
        "V": np.asarray(V),           # (T,)
        "D": np.asarray(Dw),          # (T, 3, 3)
        "B": np.asarray(g),           # (T, 3, 3)
        "Xi": np.broadcast_to(np.asarray(sigma)[:, None], (T, 3)).copy(),
        "Xi_raw": np.broadcast_to(
            np.asarray(sigma_raw)[:, None], (T, 3)).copy(),
    }


def eval_subnets_neural_sde(model, traj_12, u_const):
    """Diffusion σ(x, u) along a trajectory.  Neural SDE has no M/V/D/g."""
    xs = jnp.asarray(traj_12, dtype=jnp.float32)
    u_jnp = jnp.broadcast_to(
        jnp.asarray(u_const, dtype=jnp.float32),
        (xs.shape[0], len(u_const)),
    )
    sigmas = jax.vmap(model.diffusion)(xs, u_jnp)             # (T, 3)
    return {"Xi": np.asarray(sigmas)}


def eval_subnets_nn(model, traj_12, device):
    qs = torch.tensor(traj_12[:, :9], dtype=torch.float32, device=device)
    with torch.no_grad():
        M_inv = model.M_net(qs).cpu().numpy()
        V = model.V_net(qs).cpu().numpy().squeeze(-1)
        D = model.Dw_net(qs).cpu().numpy()
        B = model.g_net(qs).cpu().numpy()
    return {"M": M_inv, "V": V, "D": D, "B": B}     # NN_ODE has no diffusion


def estimate_beta(comp_list, gt_m_inv_scalar=1.0 / I_PERP):
    """Estimate the port-Hamiltonian scale-invariance factor β from M⁻¹.

    The dynamics in (30/68) are invariant under (M, V, D, B) → β·(M, V, D, B),
    i.e. M_β⁻¹ = M⁻¹/β.  We fit β so that β·M_β⁻¹ ≈ M_GT⁻¹ = (1/(m·l²))·I:

        β* = argmin_β  Σ_t  ‖ M_β⁻¹(q_t) / β  −  (1/(m·l²)) · I ‖_F²

    M_GT⁻¹ = (1/(m·l²))·I and M_β⁻¹ = (1/β)·M_GT⁻¹, so the closed-form
    least-squares solution against the isotropic target is

        β* = (1/(m·l²)) / mean_t( trace(M_β⁻¹(q_t)) / 3 )

    To recover GT-equivalent values for *plotting* (not for dynamics, which
    are unchanged):
        M_GT⁻¹ ≈ β · M_β⁻¹    (multiply model M⁻¹ by β)
        V_GT   ≈ V_β / β      (divide model V by β)
        D_GT   ≈ D_β / β
        B_GT   ≈ B_β / β

    Args:
        comp_list: list of subnet dicts (one per traj), each with 'M' of
                   shape (T, 3, 3) holding the model's raw M_β⁻¹.
        gt_m_inv_scalar: the analytic diagonal entry of M_GT⁻¹, default 1/(m·l²).

    Returns:
        β scalar (= 1.0 if no M data).
    """
    traces = []
    for comp in comp_list:
        if "M" not in comp:
            continue
        M_inv = comp["M"]                                 # (T, 3, 3)
        traces.append(np.trace(M_inv, axis1=1, axis2=2) / 3.0)
    if not traces:
        return 1.0
    mean_diag_M_inv = float(np.mean(np.concatenate(traces)))
    if mean_diag_M_inv < 1e-12:
        return 1.0
    return gt_m_inv_scalar / mean_diag_M_inv


def apply_beta(comp, beta):
    """Apply scale-invariance correction to a subnet dict in-place-style.

    M⁻¹ → M⁻¹ · β    (so β·M_β⁻¹ ≈ M_GT⁻¹)
    V   → V   / β    (so V/β ≈ V_GT, since V_β = β·V_GT)
    D   → D   / β
    B   → B   / β
    Xi  unchanged    (diffusion is not part of the (M,V,D,B) gauge group)
    """
    out = dict(comp)
    if "M" in out:
        out["M"] = out["M"] * beta
    if "V" in out:
        out["V"] = out["V"] / beta
    if "D" in out:
        out["D"] = out["D"] / beta
    if "B" in out:
        out["B"] = out["B"] / beta
    return out


def eval_subnets_gt(traj_12, sigma_const):
    """Analytic port-Hamiltonian terms for the spherical pendulum.

    V uses the env's *physical* convention (V = m·g·l · R[2,2]), matching
    the gravity torque `Fg = −m·g·ê_z` in envs/windy_pendulum_3d.py and the
    Hamiltonian H = ½ pᵀM⁻¹p + V used by ph_gp_sde/network.py. This is the
    target the GP_SDE V_net is trained against, so it's the right reference
    for the subnet-evolution V(q) plot. Note: only ∂V/∂q drives dynamics,
    so the model's learned V will match the GT shape up to an additive
    constant.
    """
    T = traj_12.shape[0]
    return {
        "M": np.tile(np.eye(3) / I_PERP, (T, 1, 1)),
        "V": GT_M * GT_G * GT_L * traj_12[:, 8],
        "D": np.tile(GT_FRICTION * np.eye(3), (T, 1, 1)),
        "B": np.tile(np.eye(3), (T, 1, 1)),
        "Xi": np.full((T, 3), sigma_const, dtype=np.float64),
    }


# ──────────────────────────────────────────────────────────────────────────
# Per-trajectory diagnostics
# ──────────────────────────────────────────────────────────────────────────

def rotmat_to_euler(R_flat):
    """Vectorised batch ZYX Euler conversion, returns angles in (T, 3) rad."""
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
    """GT-style energy: PE = g·(1 − R[2,2]), KE = ½ ωᵀ I ω with I = diag(I_perp, I_perp, I_para)."""
    pe = GT_G * (1.0 - traj[:, 8])
    omega = traj[:, 9:12]
    I = np.diag([I_PERP, I_PERP, I_PARA])
    return 0.5 * np.einsum("ti,ij,tj->t", omega, I, omega) + pe


def hamiltonian_from_subnets(traj_12, sub):
    """Model-self H = ½ pᵀ M⁻¹ p + V where p = solve(M⁻¹, ω)."""
    omegas = traj_12[:, 9:12]
    M_inv = sub["M"]                     # (T, 3, 3)
    V = sub["V"]
    p = np.linalg.solve(M_inv, omegas[..., None])[..., 0]
    Mp = np.einsum("tij,tj->ti", M_inv, p)
    return 0.5 * np.einsum("ti,ti->t", p, Mp) + V


def so3_orth_residual(traj_12):
    R = traj_12[:, :9].reshape(-1, 3, 3)
    diff = np.einsum("tji,tjk->tik", R, R) - np.eye(3)[None]
    return np.linalg.norm(diff.reshape(-1, 9), axis=1)


def so3_det_residual_abs(traj_12):
    R = traj_12[:, :9].reshape(-1, 3, 3)
    return np.abs(np.linalg.det(R) - 1.0)


# ──────────────────────────────────────────────────────────────────────────
# Plot helpers (mirroring the legacy report)
# ──────────────────────────────────────────────────────────────────────────

def _smooth(y, w=21):
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < w + 1:
        return y
    cumsum = np.cumsum(np.insert(y, 0, 0.0))
    half = w // 2
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = (cumsum[hi] - cumsum[lo]) / (hi - lo)
    return out


def fig_two_curves(x_a, y_a, lbl_a, x_b, y_b, lbl_b, title, ylab,
                   logy=True, smooth=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    ya, yb = (_smooth(y_a), _smooth(y_b)) if smooth else (y_a, y_b)
    ax.plot(x_a, ya, lw=1.2, label=lbl_a, color=NN_COLOR)
    ax.plot(x_b, yb, lw=1.2, label=lbl_b, color=GP_COLOR)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("training step"); ax.set_ylabel(ylab)
    ax.set_title(title); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    return fig


def fig_n_curves(curves, title, ylab, logy=True, smooth=True):
    """Generic overlay of N (x, y, label, color) tuples on one axis.

    Drop-in generalisation of `fig_two_curves` so the same panel can show
    NN_ODE / GP_SDE / Neural_SDE training curves together.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for x, y, label, color in curves:
        if y is None or len(y) == 0:
            continue
        ys = _smooth(y) if smooth else y
        ax.plot(x, ys, lw=1.2, label=label, color=color)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("training step"); ax.set_ylabel(ylab)
    ax.set_title(title); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    return fig


def fig_one_curve(x, y, title, ylab, logy=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, _smooth(y), lw=1.2, color=GP_COLOR)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("training step"); ax.set_ylabel(ylab)
    ax.set_title(title); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_gp_log_w_covar(model):
    """Visualise the variational posterior log-σ_w of every GP subnet.

    Each GP_Model carries a `log_w_covar` array of shape (D_feat, output_dim)
    parameterising the variational posterior over GP feature weights:
        q(w) = N(w_mean, diag(exp(2·log_w_covar))).
    The prior is N(0, I), so a fully-collapsed posterior (no learning signal)
    sits at log_w_covar ≈ 0  (σ_w = 1, KL = 0). The init in this codebase is
    −2.0 (σ_w ≈ 0.135). Values driven *down* by training mean the posterior
    has tightened (more confident); values pulled toward 0 mean KL is winning.
    """
    def _get_lwc(sub):
        if hasattr(sub, "log_w_covar"):
            return np.asarray(sub.log_w_covar).flatten()
        if hasattr(sub, "gp_model"):
            return np.asarray(sub.gp_model.log_w_covar).flatten()
        return None

    panels = [
        ("M_net",  _get_lwc(model.M_net),  "#1f77b4"),
        ("V_net",  _get_lwc(model.V_net),  "#ff7f0e"),
        ("Dw_net", _get_lwc(model.Dw_net), "#2ca02c"),
        ("g_net",  _get_lwc(model.g_net),  "#d62728"),
    ]
    panels = [(n, v, c) for n, v, c in panels if v is not None]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (name, vals, col) in zip(axes.flat, panels):
        ax.hist(vals, bins=40, color=col, alpha=0.85, edgecolor="black", lw=0.4)
        ax.axvline(-2.0, color="grey", linestyle=":", lw=1.0,
                   label="init (−2.0)")
        ax.axvline(0.0, color="black", linestyle="--", lw=1.0,
                   label="prior σ=1 (0.0)")
        ax.axvline(float(np.mean(vals)), color=col, linestyle="-", lw=1.5,
                   label=f"mean={np.mean(vals):+.3f}")
        ax.set_title(f"{name}.gp_model.log_w_covar  "
                     f"(N={vals.size},  σ_w∈[{np.exp(vals.min()):.2e}, "
                     f"{np.exp(vals.max()):.2e}])",
                     fontsize=10)
        ax.set_xlabel("log_w_covar")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="x-small")

    # Top-level observation-noise scalars
    ls_R = float(model.log_sigma_R)
    ls_w = float(model.log_sigma_omega)
    fig.suptitle(
        f"GP_SDE variational log_w_covar  |  "
        f"log σ_R = {ls_R:+.3f}  (σ_R = {np.exp(ls_R):.3f}),  "
        f"log σ_ω = {ls_w:+.3f}  (σ_ω = {np.exp(ls_w):.3f})",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.0, 1, 0.95))
    return fig


def fig_kl_breakdown(stats_gp):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(stats_gp["train_kl_M"]))
    for k, color in zip(
        ["train_kl_M", "train_kl_V", "train_kl_Dw", "train_kl_g", "train_kl_sigma"],
        ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
    ):
        if k in stats_gp and len(stats_gp[k]) > 0:
            ax.plot(x, _smooth(stats_gp[k]), lw=1.0,
                    label=k.replace("train_kl_", ""), color=color)
    ax.set_yscale("log")
    ax.set_xlabel("training step"); ax.set_ylabel("KL")
    ax.set_title("Per-subnet KL (GP_SDE)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    return fig


def fig_energy_ensemble(t_eval, gt_e_all, model_e):
    """gt_e_all: (N, T) GT energies. model_e: dict[name -> (mean, std, color)]."""
    fig, ax = plt.subplots(figsize=(10, 6))
    gt_m, gt_s = gt_e_all.mean(0), gt_e_all.std(0)
    ax.plot(t_eval, gt_m, "k-", lw=2, label="GT Mean")
    ax.fill_between(t_eval, gt_m - 2 * gt_s, gt_m + 2 * gt_s,
                    color="black", alpha=0.15, label="GT ±2σ")
    for name, (em, es, col) in model_e.items():
        ax.plot(t_eval, em, color=col, lw=2, label=name)
        ax.fill_between(t_eval, em - 2 * es, em + 2 * es,
                        color=col, alpha=0.2, label=f"{name} ±2σ")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Energy (J)")
    ax.legend(fontsize="x-small"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_energy_single(t_eval, gt_single_e, model_single_e):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_eval, gt_single_e, "k-", lw=2, label="GT")
    for name, (e, col) in model_single_e.items():
        ax.plot(t_eval, e, color=col, lw=2, label=name)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Energy (J)")
    ax.legend(fontsize="x-small"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _geodesic_sq(R_pred_flat, R_gt_flat):
    """Squared geodesic angle between two batches of flattened rotmats.
    Inputs (N, T, 9) → output (N, T).
    θ = arccos( clip( (trace(R_predᵀ R_gt) − 1) / 2, −1, 1 ) )
    """
    R_pred = R_pred_flat.reshape(*R_pred_flat.shape[:-1], 3, 3)
    R_gt = R_gt_flat.reshape(*R_gt_flat.shape[:-1], 3, 3)
    M = np.einsum("...ji,...jk->...ik", R_pred, R_gt)        # Rᵀ R_gt
    cos_t = (np.trace(M, axis1=-2, axis2=-1) - 1.0) / 2.0
    cos_t = np.clip(cos_t, -1.0, 1.0)
    theta = np.arccos(cos_t)
    return theta ** 2


def fig_traj_mse(t_eval, gt_trajs_all, model_trajs):
    """Two pages: geodesic error (geodesic²) and MSE ω, model rollouts vs GT,
    averaged across the trajectory ensemble (mean ±2σ band).
    `model_trajs`: dict[name -> (trajs (N,T,12+), color)]."""
    R_gt = gt_trajs_all[..., :9]
    om_gt = gt_trajs_all[..., 9:12]

    per_model = {}
    for name, (trajs, col) in model_trajs.items():
        R = trajs[..., :9]
        om = trajs[..., 9:12]
        per_model[name] = {
            "geo": _geodesic_sq(R, R_gt),
            "l2":  np.sum((om - om_gt) ** 2, axis=-1),
            "color": col,
        }

    figs = []
    for key, ylab, title in (
        ("geo", "geodesic² (rad²)",
         "Trajectory geodesic error: model vs GT (10-traj ensemble)"),
        ("l2", "‖Δω‖² (rad²/s²)",
         "Trajectory MSE angular velocity: model vs GT (10-traj ensemble)"),
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, d in per_model.items():
            arr = d[key]
            m, s = arr.mean(0), arr.std(0)
            ax.plot(t_eval, m, color=d["color"], lw=2, label=f"{name} mean")
            ax.fill_between(t_eval, np.maximum(m - 2 * s, 1e-30),
                            m + 2 * s, color=d["color"], alpha=0.18)
        ax.set_yscale("log")
        ax.set_xlabel("Time (s)"); ax.set_ylabel(ylab)
        ax.set_title(title); ax.grid(True, alpha=0.3)
        ax.legend(fontsize="x-small")
        fig.tight_layout()
        figs.append(fig)
    return figs


def fig_so3_violation(t_eval, gt_metric_all, model_metric, ylab, title):
    """`model_metric`: dict[name -> (mean_metric, single_metric, color)]."""
    fig, ax = plt.subplots(figsize=(10, 6))
    if gt_metric_all is not None:
        ax.plot(t_eval, gt_metric_all.mean(0), "k--", lw=2, label="GT (mean)")
    for name, (mm, ms, col) in model_metric.items():
        ax.plot(t_eval, mm, color=col, label=f"{name} (mean)")
        ax.plot(t_eval, ms, color=col, linestyle=":", label=f"{name} (single)")
    ax.set_yscale("log"); ax.set_xlabel("Time (s)"); ax.set_ylabel(ylab)
    ax.set_title(title); ax.legend(fontsize="x-small"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_state_ensemble(t_eval, gt_eul_all, gt_om_all, model_states):
    """3×2 grid: rows = roll/pitch/yaw or ωx/ωy/ωz, cols = euler / omega.
    gt_eul_all: (N, T, 3); gt_om_all: (N, T, 3)
    model_states: dict[name -> (eul_all, om_all, color)] each (N, T, 3)
    """
    labels_ang = ["Roll (rad)", "Pitch (rad)", "Yaw (rad)"]
    labels_om = ["Omega X", "Omega Y", "Omega Z"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    gt_eul_m, gt_eul_s = gt_eul_all.mean(0), gt_eul_all.std(0)
    gt_om_m, gt_om_s = gt_om_all.mean(0), gt_om_all.std(0)

    for i in range(3):
        for ni in range(gt_eul_all.shape[0]):
            axes[i, 0].plot(t_eval, gt_eul_all[ni, :, i],
                            color="black", alpha=0.15, lw=0.8)
            axes[i, 1].plot(t_eval, gt_om_all[ni, :, i],
                            color="black", alpha=0.15, lw=0.8)
        axes[i, 0].plot(t_eval, gt_eul_m[:, i], "k-", lw=2,
                        label="GT Mean" if i == 0 else None)
        axes[i, 0].fill_between(t_eval, gt_eul_m[:, i] - 2 * gt_eul_s[:, i],
                                gt_eul_m[:, i] + 2 * gt_eul_s[:, i],
                                color="black", alpha=0.15,
                                label="GT ±2σ" if i == 0 else None)
        axes[i, 1].plot(t_eval, gt_om_m[:, i], "k-", lw=2,
                        label="GT Mean" if i == 0 else None)
        axes[i, 1].fill_between(t_eval, gt_om_m[:, i] - 2 * gt_om_s[:, i],
                                gt_om_m[:, i] + 2 * gt_om_s[:, i],
                                color="black", alpha=0.15,
                                label="GT ±2σ" if i == 0 else None)

    for name, (eul_all, om_all, col) in model_states.items():
        em, es = eul_all.mean(0), eul_all.std(0)
        om, os_ = om_all.mean(0), om_all.std(0)
        for i in range(3):
            for ni in range(eul_all.shape[0]):
                axes[i, 0].plot(t_eval, eul_all[ni, :, i],
                                color=col, alpha=0.15, lw=0.8)
                axes[i, 1].plot(t_eval, om_all[ni, :, i],
                                color=col, alpha=0.15, lw=0.8)
            axes[i, 0].plot(t_eval, em[:, i], color=col, lw=2,
                            label=name if i == 0 else None)
            axes[i, 1].plot(t_eval, om[:, i], color=col, lw=2,
                            label=name if i == 0 else None)
            axes[i, 0].fill_between(t_eval, em[:, i] - 2 * es[:, i],
                                    em[:, i] + 2 * es[:, i],
                                    color=col, alpha=0.2,
                                    label=f"{name} ±2σ" if i == 0 else None)
            axes[i, 1].fill_between(t_eval, om[:, i] - 2 * os_[:, i],
                                    om[:, i] + 2 * os_[:, i],
                                    color=col, alpha=0.2,
                                    label=f"{name} ±2σ" if i == 0 else None)

    for i in range(3):
        axes[i, 0].set_ylabel(labels_ang[i])
        axes[i, 1].set_ylabel(labels_om[i])
        axes[i, 0].grid(True, alpha=0.3); axes[i, 1].grid(True, alpha=0.3)
    axes[2, 0].set_xlabel("Time (s)"); axes[2, 1].set_xlabel("Time (s)")
    axes[0, 0].legend(fontsize="x-small"); axes[0, 1].legend(fontsize="x-small")
    fig.suptitle("State Trajectories (10-traj ensemble)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.0, 1, 0.97))
    return fig


def fig_state_single(t_eval, gt_traj, model_trajs_single):
    labels_ang = ["Roll (rad)", "Pitch (rad)", "Yaw (rad)"]
    labels_om = ["Omega X", "Omega Y", "Omega Z"]
    gt_eul = rotmat_to_euler(gt_traj[:, :9])
    gt_om = gt_traj[:, 9:12]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    for i in range(3):
        axes[i, 0].plot(t_eval, gt_eul[:, i], "k-", lw=2,
                        label="GT" if i == 0 else None)
        axes[i, 1].plot(t_eval, gt_om[:, i], "k-", lw=2,
                        label="GT" if i == 0 else None)
    for name, (tr, col) in model_trajs_single.items():
        eul = rotmat_to_euler(tr[:, :9])
        for i in range(3):
            axes[i, 0].plot(t_eval, eul[:, i], color=col, lw=2,
                            label=name if i == 0 else None)
            axes[i, 1].plot(t_eval, tr[:, 9 + i], color=col, lw=2,
                            label=name if i == 0 else None)
    for i in range(3):
        axes[i, 0].set_ylabel(labels_ang[i])
        axes[i, 1].set_ylabel(labels_om[i])
        axes[i, 0].grid(True, alpha=0.3); axes[i, 1].grid(True, alpha=0.3)
    axes[2, 0].set_xlabel("Time (s)"); axes[2, 1].set_xlabel("Time (s)")
    axes[0, 0].legend(fontsize="x-small"); axes[0, 1].legend(fontsize="x-small")
    fig.suptitle("1 Trajectory State Trajectories", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.0, 1, 0.97))
    return fig


def fig_phase_portraits(gt_eul_all, gt_om_all, model_states):
    """3 subplots: roll-vs-ωx, pitch-vs-ωy, yaw-vs-ωz overlaying all members."""
    labels_ang = ["Roll (rad)", "Pitch (rad)", "Yaw (rad)"]
    labels_om = ["Omega X", "Omega Y", "Omega Z"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        for ni in range(gt_eul_all.shape[0]):
            axes[i].plot(gt_eul_all[ni, :, i], gt_om_all[ni, :, i],
                         color="black", alpha=0.15, lw=0.8)
        axes[i].plot(gt_eul_all.mean(0)[:, i], gt_om_all.mean(0)[:, i],
                     "k-", lw=2, label="GT Mean" if i == 0 else None)
        for name, (eul_all, om_all, col) in model_states.items():
            for ni in range(eul_all.shape[0]):
                axes[i].plot(eul_all[ni, :, i], om_all[ni, :, i],
                             color=col, alpha=0.15, lw=0.8)
            axes[i].plot(eul_all.mean(0)[:, i], om_all.mean(0)[:, i],
                         color=col, lw=2, label=name if i == 0 else None)
        axes[i].set_xlabel(labels_ang[i]); axes[i].set_ylabel(labels_om[i])
        axes[i].grid(True, alpha=0.3)
    axes[0].legend(fontsize="x-small")
    fig.suptitle("Phase Portraits (Angle vs Omega)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.0, 1, 0.97))
    return fig


# ── Subnet evolution along multiple trajectories ──────────────────────────

def plot_matrix_3x3_multi(t, comp_multi, comp_key, gt_flat, model_meta):
    """comp_multi[name] = list of N_SUB_TRAJ subnet dicts (one per traj).
    model_meta: dict[name -> color]."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for idx in range(9):
        r, c = divmod(idx, 3)
        ax = axes[r, c]
        if gt_flat is not None:
            ax.axhline(y=gt_flat[idx], color="k", linestyle="--", lw=1.5,
                       label="GT" if idx == 0 else None)
        for name, col in model_meta.items():
            first = True
            for ti, comp in enumerate(comp_multi[name]):
                if comp_key not in comp:
                    break
                arr = comp[comp_key]                     # (T, 3, 3)
                if arr.ndim != 3:
                    break
                lbl = name if (idx == 0 and first) else None
                ax.plot(t, arr.reshape(arr.shape[0], 9)[:, idx],
                        color=col, alpha=0.7, lw=1.2, label=lbl)
                first = False
        ax.set_xlabel("Time (s)"); ax.set_title(f"({r},{c})")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize="x-small")
    return fig


def plot_potential_multi(t, comp_multi, gt_v_all, model_meta):
    fig, ax = plt.subplots(figsize=(10, 6))
    for ti in range(gt_v_all.shape[0]):
        ax.plot(t, gt_v_all[ti], "k--", lw=1.0, alpha=0.5,
                label="GT" if ti == 0 else None)
    for name, col in model_meta.items():
        first = True
        for ti, comp in enumerate(comp_multi[name]):
            if "V" not in comp:
                break
            v = comp["V"].flatten()
            ax.plot(t, v, color=col, alpha=0.7, lw=1.2,
                    label=name if first else None)
            first = False
    ax.set_xlabel("Time (s)"); ax.set_ylabel("V(q)")
    ax.legend(fontsize="x-small"); ax.grid(True, alpha=0.3)
    return fig


def plot_xi_multi(t, comp_multi, model_meta, proc_noise):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for j in range(3):
        axes[j].axhline(y=proc_noise, color="k", linestyle="--", lw=1.5,
                        label="GT" if j == 0 else None)
        for name, col in model_meta.items():
            first = True
            for ti, comp in enumerate(comp_multi[name]):
                if "Xi" not in comp:
                    break
                axes[j].plot(t, comp["Xi"][:, j], color=col, alpha=0.7,
                             lw=1.2, label=name if (j == 0 and first) else None)
                first = False
        axes[j].set_xlabel("Time (s)"); axes[j].set_ylabel(f"Xi[{j}]")
        axes[j].grid(True, alpha=0.3)
    axes[0].legend(fontsize="x-small")
    return fig


def plot_xi_beta_multi(t, comp_multi, model_meta, proc_noise, betas):
    """Per-model σ(q)/β along 5 trajectories.

    Treats σ as part of the (M, V, D, g) gauge orbit (σ → α·σ under p → α·p),
    so the gauge-corrected comparison to GT is σ_model / β. Only meaningful
    if the training likelihood is in ω-space (where σ is gauge-free); see
    discussion in network.py docstring.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for j in range(3):
        axes[j].axhline(y=proc_noise, color="k", linestyle="--", lw=1.5,
                        label="GT" if j == 0 else None)
        for name, col in model_meta.items():
            beta = float(betas.get(name, 1.0))
            first = True
            for ti, comp in enumerate(comp_multi[name]):
                if "Xi" not in comp:
                    break
                axes[j].plot(t, comp["Xi"][:, j] / beta, color=col, alpha=0.7,
                             lw=1.2, label=name if (j == 0 and first) else None)
                first = False
        axes[j].set_xlabel("Time (s)"); axes[j].set_ylabel(f"σ/β [{j}]")
        axes[j].grid(True, alpha=0.3)
    axes[0].legend(fontsize="x-small")
    return fig


def _compute_table_metrics(gt_all, pred_all):
    """Compute per-trajectory metrics, aggregated as mean ± std over the
    N-trajectory ensemble.

    gt_all, pred_all: (N, T, 12).  N=10 for our ensembles.
    Returns: dict {metric_name: "{mean:.3e} ± {std:.3e}"}.
    """
    N, T, _ = gt_all.shape
    R_gt  = gt_all[..., :9].reshape(N, T, 3, 3)
    R_pr  = pred_all[..., :9].reshape(N, T, 3, 3)
    om_gt = gt_all[..., 9:12]
    om_pr = pred_all[..., 9:12]

    # geodesic² between predicted and GT rotations, per (n, t)
    M = np.einsum("ntji,ntjk->ntik", R_pr, R_gt)              # R_prᵀ R_gt
    cos_t = np.clip((np.trace(M, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    geo_sq = np.arccos(cos_t) ** 2                            # (N, T)
    om_sq  = np.sum((om_pr - om_gt) ** 2, axis=-1)            # (N, T)

    geo_traj_mean = geo_sq.mean(axis=1)   # (N,) per-traj time-mean geo²
    om_traj_mean  = om_sq.mean(axis=1)    # (N,) per-traj time-mean ‖Δω‖²
    geo_final = geo_sq[:, -1]             # (N,) final-step geodesic²
    om_final  = om_sq[:, -1]              # (N,) final-step ‖Δω‖²

    # SO(3) violation maxima per traj
    det_resid  = np.abs(np.linalg.det(R_pr) - 1.0)            # (N, T)
    orth_resid = np.linalg.norm(
        (np.einsum("ntji,ntjk->ntik", R_pr, R_pr)
         - np.eye(3)[None, None]).reshape(N, T, 9),
        axis=-1,
    )                                                          # (N, T)
    det_max  = det_resid.max(axis=1)
    orth_max = orth_resid.max(axis=1)

    # Energy: per-traj mean |E_pred − E_GT|
    e_gt = np.array([get_energy(gt_all[n])   for n in range(N)])
    e_pr = np.array([get_energy(pred_all[n]) for n in range(N)])
    e_err = np.abs(e_pr - e_gt).mean(axis=1)                  # (N,)

    def ms(x):
        return f"{float(np.mean(x)):.3e}  ±  {float(np.std(x)):.3e}"

    return {
        "Traj-mean geodesic² (rad²)":   ms(geo_traj_mean),
        "Traj-mean ‖Δω‖² (rad²/s²)":    ms(om_traj_mean),
        "Final-step geodesic² (rad²)": ms(geo_final),
        "Final-step ‖Δω‖² (rad²/s²)":  ms(om_final),
        "Mean |ΔE| over time (J)":      ms(e_err),
        "Max |det(R)−1|":               ms(det_max),
        "Max ‖RᵀR−I‖_F":                ms(orth_max),
    }


def fig_comparison_table(gt_trajs_all, models):
    """First-page summary: trajectory-aggregate metrics per model.

    `models`: dict[name -> trajs_all (N, T, 12)].
    """
    per_model = {name: _compute_table_metrics(gt_trajs_all, trajs)
                 for name, trajs in models.items()}
    metric_names = list(next(iter(per_model.values())).keys())
    model_names  = list(per_model.keys())

    cell_text = [
        [per_model[m][metric] for m in model_names] for metric in metric_names
    ]

    fig, ax = plt.subplots(figsize=(14, 1.5 + 0.55 * len(metric_names)))
    ax.axis("off")
    title = (f"Trajectory comparison — {gt_trajs_all.shape[0]}-rollout ensemble vs GT\n"
             "(per-trajectory aggregates over 10 trajs: mean ± std)")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)

    table = ax.table(
        cellText=cell_text,
        rowLabels=metric_names,
        colLabels=model_names,
        cellLoc="center",
        rowLoc="left",
        loc="center",
        colWidths=[0.30] * len(model_names),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.7)
    # Bold the header row
    for j in range(len(model_names)):
        cell = table[0, j]
        cell.set_text_props(weight="bold")

    if "NN_ODE" in model_names:
        fig.text(
            0.5, 0.02,
            "Note: NN_ODE is deterministic — its 10 ensemble copies are "
            "identical, so its std-columns are 0.\n"
            "GP_SDE has stochastic rollouts; std reflects diffusion-driven "
            "spread across the 10 dW samples (same dW sequences as GT).",
            ha="center", fontsize=9, style="italic", color="gray",
        )

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gp_sde_ckpt", default=DEFAULT_GP_SDE_CKPT)
    ap.add_argument("--gp_sde_stats", default=DEFAULT_GP_SDE_STATS)
    ap.add_argument("--nn_ode_ckpt", default=DEFAULT_NN_ODE_CKPT)
    ap.add_argument("--nn_ode_stats", default=DEFAULT_NN_ODE_STATS)
    ap.add_argument("--neural_sde_ckpt", default=DEFAULT_NEURAL_SDE_CKPT)
    ap.add_argument("--neural_sde_stats", default=DEFAULT_NEURAL_SDE_STATS)
    ap.add_argument("--neural_sde_hidden_dim", type=int, default=500,
                    help="hidden width used at neural_sde train time (default 500)")
    ap.add_argument("--out_pdf", default=DEFAULT_OUT_PDF)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--u", type=float, nargs=3, default=(0.0, 0.0, 0.0),
        help="constant body-frame torque held across rollouts (default 0,0,0)")
    ap.add_argument(
        "--gp_sde_max_step", type=int, default=None,
        help="clip GP_SDE training stats to this step (auto-detected from "
             "--gp_sde_ckpt filename '*-{step}.eqx' if omitted)")
    ap.add_argument(
        "--nn_ode_max_step", type=int, default=None,
        help="clip NN_ODE training stats to this step (auto-detected from "
             "--nn_ode_ckpt filename '*-{step}.tar' if omitted)")
    ap.add_argument(
        "--neural_sde_max_step", type=int, default=None,
        help="clip Neural_SDE training stats to this step (auto-detected from "
             "--neural_sde_ckpt filename '*-{step}.eqx' if omitted)")
    ap.add_argument(
        "--no_neural_sde", action="store_true",
        help="skip the neural_sde model (use the legacy 2-model GP_SDE vs "
             "NN_ODE comparison)")
    ap.add_argument(
        "--include_gt_models", action="store_true",
        help="also roll out GP_SDE_GT and NN_ODE_GT (same integrators with "
             "analytic GT subnets) for algorithm-vs-subnet sanity checking. "
             "WARNING: GP_SDE_GT path is currently broken (GTSDEModel lacks "
             "drift_p/stochastic_increment_p/M_inv after the (q,p) switch).")
    ap.add_argument(
        "--gp_sde_only", action="store_true",
        help="skip all NN_ODE loading, rollouts, and plots — produce a "
             "GP_SDE-only PDF. Use when no NN_ODE checkpoint is available.")
    ap.add_argument(
        "--init_sigma_const", type=float, default=0.1,
        help="DEPRECATED / IGNORED. The static softplus bias on sigma_net "
             "was removed; σ(q) = softplus(GP_raw(q)) now needs no "
             "template-side init. Kept only for backward CLI compat.")
    args = ap.parse_args()

    # Auto-detect step from checkpoint filename if --*_max_step omitted.
    import re
    def _step_from_ckpt(path, ext):
        m = re.search(rf"-(\d+)\.{ext}$", os.path.basename(path))
        return int(m.group(1)) if m else None
    if args.gp_sde_max_step is None:
        args.gp_sde_max_step = _step_from_ckpt(args.gp_sde_ckpt, "eqx")
    if args.nn_ode_max_step is None:
        args.nn_ode_max_step = _step_from_ckpt(args.nn_ode_ckpt, "tar")
    if args.neural_sde_max_step is None:
        args.neural_sde_max_step = _step_from_ckpt(args.neural_sde_ckpt, "eqx")

    device = torch.device(args.device)
    print(f"Loading GP_SDE  : {args.gp_sde_ckpt}")
    gp_sde = load_gp_sde(args.gp_sde_ckpt)
    print(f"Loading GP_SDE stats: {args.gp_sde_stats}")
    stats_gp = load_stats(args.gp_sde_stats)
    if not args.gp_sde_only:
        print(f"Loading NN_ODE  : {args.nn_ode_ckpt}")
        nn_ode = load_nn_ode(args.nn_ode_ckpt, device)
        print(f"Loading NN_ODE stats: {args.nn_ode_stats}")
        stats_nn = load_stats(args.nn_ode_stats)
    else:
        nn_ode = None
        stats_nn = None
        print("  --gp_sde_only: skipping NN_ODE load")

    use_neural_sde = (not args.gp_sde_only) and (not args.no_neural_sde)
    if use_neural_sde:
        try:
            print(f"Loading Neural_SDE : {args.neural_sde_ckpt}")
            neural_sde = load_neural_sde(
                args.neural_sde_ckpt, hidden_dim=args.neural_sde_hidden_dim)
            print(f"Loading Neural_SDE stats: {args.neural_sde_stats}")
            stats_neural = load_stats(args.neural_sde_stats)
        except FileNotFoundError as e:
            print(f"  ! Neural_SDE assets missing ({e}); skipping Neural_SDE")
            use_neural_sde = False
            neural_sde = None
            stats_neural = None
    else:
        neural_sde = None
        stats_neural = None

    # ── Initial condition: env.reset(seed=42) → fixed (R0, ω0) ──
    env_setup = windy_pendulum_3d(seed=args.seed, **ENV_KW)
    env_setup.reset(seed=args.seed)
    R0 = env_setup.R.copy()
    omega0 = env_setup.omega.copy()
    u_const = np.asarray(args.u, dtype=np.float64)

    # ── Pre-sample dW for each ensemble member ──
    dt_sub = ENV_KW["dt"] / N_SUBSTEPS
    sqrt_h = float(np.sqrt(dt_sub))
    rng = np.random.default_rng(args.seed)
    dW_ensemble = [
        rng.normal(0.0, sqrt_h, size=(N_OUTER, N_SUBSTEPS, 3))
        for _ in range(N_TRAJ_ENSEMBLE)
    ]
    t_eval = np.arange(N_OUTER + 1) * ENV_KW["dt"]

    # ── 10 GT trajectories with the same (R0, ω0) but different dW ──
    print(f"Rolling out {N_TRAJ_ENSEMBLE} GT trajectories ...")
    gt_trajs_all = np.zeros((N_TRAJ_ENSEMBLE, N_OUTER + 1, 12))
    for ti in range(N_TRAJ_ENSEMBLE):
        env = windy_pendulum_3d(seed=args.seed + ti, **ENV_KW)
        env.reset(seed=args.seed + ti)   # advances RNG; we override with our dW anyway
        gt_trajs_all[ti] = rollout_gt(env, R0, omega0, u_const, dW_ensemble[ti])

    # ── 10 GP_SDE trajectories with the *same* dW sequences as GT ──
    print(f"Rolling out {N_TRAJ_ENSEMBLE} GP_SDE trajectories ...")
    gp_trajs_all = np.zeros((N_TRAJ_ENSEMBLE, N_OUTER + 1, 12))
    for ti in range(N_TRAJ_ENSEMBLE):
        gp_trajs_all[ti] = rollout_gp_sde(
            gp_sde, R0, omega0, u_const, dW_ensemble[ti], dt_sub)

    # ── 10 NN_ODE trajectories — deterministic, so same trajectory each time;
    #    we still sample 10 to have matched ensemble shape ──
    if not args.gp_sde_only:
        print(f"Rolling out NN_ODE trajectory (deterministic, replicated x{N_TRAJ_ENSEMBLE})...")
        nn_single = rollout_nn_ode(nn_ode, R0, omega0, u_const, t_eval, device)[:, :12]
        nn_trajs_all = np.broadcast_to(
            nn_single[None, ...], (N_TRAJ_ENSEMBLE, N_OUTER + 1, 12)).copy()
    else:
        nn_trajs_all = None

    # ── 10 Neural_SDE trajectories (Stratonovich Heun on flat ℝ¹², same dW
    #    sequence as GT/GP_SDE so the comparison is matched) ──
    if use_neural_sde:
        print(f"Rolling out {N_TRAJ_ENSEMBLE} Neural_SDE trajectories ...")
        neural_trajs_all = np.zeros((N_TRAJ_ENSEMBLE, N_OUTER + 1, 12))
        for ti in range(N_TRAJ_ENSEMBLE):
            neural_trajs_all[ti] = rollout_neural_sde(
                neural_sde, R0, omega0, u_const, dW_ensemble[ti], dt_sub)
    else:
        neural_trajs_all = None

    # ── GT-subnet sanity checks: same integrators, analytic dynamics ──
    if args.include_gt_models:
        print(f"Rolling out {N_TRAJ_ENSEMBLE} GP_SDE_GT trajectories "
              f"(Lie-Heun + analytic subnets)...")
        gp_gt_model = GTSDEModel(
            m=GT_M, g=GT_G, l=GT_L, friction=GT_FRICTION,
            sigma=ENV_KW["wind_force_std"])
        gp_gt_trajs_all = np.zeros((N_TRAJ_ENSEMBLE, N_OUTER + 1, 12))
        for ti in range(N_TRAJ_ENSEMBLE):
            gp_gt_trajs_all[ti] = rollout_gp_sde(
                gp_gt_model, R0, omega0, u_const, dW_ensemble[ti], dt_sub)

        print("Rolling out NN_ODE_GT trajectory (RK4 + analytic dynamics)...")
        nn_gt_model = GTNODEModel(
            m=GT_M, g=GT_G, l=GT_L, friction=GT_FRICTION, u_dim=3, device=device,
        ).to(device)
        nn_gt_single = rollout_nn_ode(nn_gt_model, R0, omega0, u_const, t_eval, device)[:, :12]
        nn_gt_trajs_all = np.broadcast_to(
            nn_gt_single[None, ...], (N_TRAJ_ENSEMBLE, N_OUTER + 1, 12)).copy()
    else:
        gp_gt_trajs_all = None
        nn_gt_trajs_all = None

    # ── Pre-compute Euler / omega / energy for each ensemble ──
    def euler_om_energy(trajs):
        eul = np.array([rotmat_to_euler(trajs[ti, :, :9]) for ti in range(trajs.shape[0])])
        om = trajs[:, :, 9:12]
        en = np.array([get_energy(trajs[ti]) for ti in range(trajs.shape[0])])
        return eul, om, en

    gt_eul_all, gt_om_all, gt_e_all = euler_om_energy(gt_trajs_all)
    gp_eul_all, gp_om_all, gp_e_all = euler_om_energy(gp_trajs_all)
    if not args.gp_sde_only:
        nn_eul_all, nn_om_all, nn_e_all = euler_om_energy(nn_trajs_all)
    if use_neural_sde:
        ns_eul_all, ns_om_all, ns_e_all = euler_om_energy(neural_trajs_all)
    if args.include_gt_models:
        gp_gt_eul_all, gp_gt_om_all, gp_gt_e_all = euler_om_energy(gp_gt_trajs_all)
        nn_gt_eul_all, nn_gt_om_all, nn_gt_e_all = euler_om_energy(nn_gt_trajs_all)

    # ── Subnet evaluations along each of N_SUB_TRAJ GT trajectories ──
    print(f"Evaluating subnets along {N_SUB_TRAJ} GT trajectories ...")
    comp_multi = {"GP_SDE": []}
    if not args.gp_sde_only:
        comp_multi["NN_ODE"] = []
    if use_neural_sde:
        comp_multi["Neural_SDE"] = []
    for ti in range(N_SUB_TRAJ):
        if not args.gp_sde_only:
            comp_multi["NN_ODE"].append(eval_subnets_nn(nn_ode, gt_trajs_all[ti], device))
        comp_multi["GP_SDE"].append(eval_subnets_gp(gp_sde, gt_trajs_all[ti]))
        if use_neural_sde:
            comp_multi["Neural_SDE"].append(
                eval_subnets_neural_sde(neural_sde, gt_trajs_all[ti], u_const))

    # ── Port-Hamiltonian scale-invariance correction ──
    # The dynamics (30/68) are invariant under (M, V, D, B) → β·(M, V, D, B).
    # Without an absolute reference for momentum p in the dataset, each model
    # learns its subnets up to this gauge β. Calibrate β from M⁻¹ vs the
    # analytic GT (M_GT⁻¹ = (1/(m·l²))·I), then apply consistently to all
    # four subnets so the plotted values match the GT scale. Xi is excluded.
    gt_m_inv_scalar = 1.0 / I_PERP
    betas = {}
    for name, comps in comp_multi.items():
        beta = estimate_beta(comps, gt_m_inv_scalar=gt_m_inv_scalar)
        betas[name] = beta
        comp_multi[name] = [apply_beta(c, beta) for c in comps]
    print("  scale-invariance β per model: " +
          ", ".join(f"{n}={b:.4f}" for n, b in betas.items()))

    # GT subnet references
    sub_gt_single = eval_subnets_gt(gt_trajs_all[0], ENV_KW["wind_force_std"])

    # ── GT constants for component grids ──
    gt_m_flat = (np.eye(3) / I_PERP).flatten()
    fric_vec = np.array([GT_FRICTION] * 3)
    gt_d_flat = (np.diag(fric_vec)).flatten()    # env applies fric*omega per axis
    gt_b_flat = np.eye(3).flatten()

    # GT V(q) along each subnet trajectory (env's physical convention,
    # matching what V_net was trained against — see eval_subnets_gt).
    gt_v_all = np.array([
        GT_M * GT_G * GT_L * gt_trajs_all[ti, :, 8]
        for ti in range(N_SUB_TRAJ)
    ])

    # ── Build PDF ──
    print(f"Writing PDF: {args.out_pdf}")
    os.makedirs(os.path.dirname(args.out_pdf) or ".", exist_ok=True)
    with PdfPages(args.out_pdf) as pdf:
        # ── Page 1: comparison summary table ─────────────────────────
        table_models = {"GP_SDE": gp_trajs_all}
        if not args.gp_sde_only:
            # Insert NN_ODE first so it appears in the leftmost column.
            table_models = {"NN_ODE": nn_trajs_all, "GP_SDE": gp_trajs_all}
        if use_neural_sde:
            table_models["Neural_SDE"] = neural_trajs_all
        pdf.savefig(fig_comparison_table(gt_trajs_all, table_models))
        plt.close()

        # ── A. Loss curves ────────────────────────────────────────────
        # Per-model training-step caps from CLI / ckpt filename. Train series
        # are 1-per-step → simple slice. Test/eval series are 1-per-eval_step
        # → mask by eval_step value.
        def _train_slice(arr, max_step):
            arr = np.asarray(arr)
            return arr if max_step is None else arr[: max_step + 1]

        def _eval_mask(eval_steps, max_step):
            eval_steps = np.asarray(eval_steps)
            if max_step is None:
                return slice(None), eval_steps
            mask = eval_steps <= max_step
            return mask, eval_steps[mask]

        gp_max = args.gp_sde_max_step
        nn_max = args.nn_ode_max_step
        ns_max = args.neural_sde_max_step
        if gp_max is not None or nn_max is not None or ns_max is not None:
            print(f"  clipping stats: GP_SDE→step {gp_max}  "
                  f"NN_ODE→step {nn_max}  Neural_SDE→step {ns_max}")

        mse_R_gp = _train_slice(stats_gp["train_mse_R"], gp_max)
        mse_w_gp = _train_slice(stats_gp["train_mse_omega"], gp_max)
        x_gp = np.arange(len(mse_R_gp))
        gp_mask, eval_x_gp = _eval_mask(stats_gp["eval_step"], gp_max)
        test_R_gp = np.asarray(stats_gp["test_mse_R"])[gp_mask]
        test_w_gp = np.asarray(stats_gp["test_mse_omega"])[gp_mask]

        # Per-train-key tuples for fig_n_curves: (x, y, label, color).
        train_R_curves    = [(x_gp, mse_R_gp,           "GP_SDE", GP_COLOR)]
        train_w_curves    = [(x_gp, mse_w_gp,           "GP_SDE", GP_COLOR)]
        train_RW_curves   = [(x_gp, mse_R_gp + mse_w_gp,"GP_SDE", GP_COLOR)]
        test_R_curves     = [(eval_x_gp, test_R_gp,            "GP_SDE", GP_COLOR)]
        test_w_curves     = [(eval_x_gp, test_w_gp,            "GP_SDE", GP_COLOR)]
        test_RW_curves    = [(eval_x_gp, test_R_gp + test_w_gp,"GP_SDE", GP_COLOR)]

        if not args.gp_sde_only:
            mse_R_nn = _train_slice(stats_nn["train_geo_loss"], nn_max)
            mse_w_nn = _train_slice(stats_nn["train_l2_loss"], nn_max)
            x_nn = np.arange(len(mse_R_nn))
            nn_mask, eval_x_nn = _eval_mask(stats_nn["eval_step"], nn_max)
            test_R_nn = np.asarray(stats_nn["test_geo_loss"])[nn_mask]
            test_w_nn = np.asarray(stats_nn["test_l2_loss"])[nn_mask]

            train_R_curves.insert(0,  (x_nn, mse_R_nn,            "NN_ODE", NN_COLOR))
            train_w_curves.insert(0,  (x_nn, mse_w_nn,            "NN_ODE", NN_COLOR))
            train_RW_curves.insert(0, (x_nn, mse_R_nn + mse_w_nn, "NN_ODE", NN_COLOR))
            test_R_curves.insert(0,   (eval_x_nn, test_R_nn,            "NN_ODE", NN_COLOR))
            test_w_curves.insert(0,   (eval_x_nn, test_w_nn,            "NN_ODE", NN_COLOR))
            test_RW_curves.insert(0,  (eval_x_nn, test_R_nn + test_w_nn,"NN_ODE", NN_COLOR))

        if use_neural_sde:
            mse_R_ns = _train_slice(stats_neural["train_geo_loss"], ns_max)
            mse_w_ns = _train_slice(stats_neural["train_l2_loss"], ns_max)
            x_ns = np.arange(len(mse_R_ns))
            ns_mask, eval_x_ns = _eval_mask(stats_neural["eval_step"], ns_max)
            test_R_ns = np.asarray(stats_neural["test_geo_loss"])[ns_mask]
            test_w_ns = np.asarray(stats_neural["test_l2_loss"])[ns_mask]

            train_R_curves.append( (x_ns, mse_R_ns,            "Neural_SDE", NEURAL_SDE_COLOR))
            train_w_curves.append( (x_ns, mse_w_ns,            "Neural_SDE", NEURAL_SDE_COLOR))
            train_RW_curves.append((x_ns, mse_R_ns + mse_w_ns, "Neural_SDE", NEURAL_SDE_COLOR))
            test_R_curves.append(  (eval_x_ns, test_R_ns,            "Neural_SDE", NEURAL_SDE_COLOR))
            test_w_curves.append(  (eval_x_ns, test_w_ns,            "Neural_SDE", NEURAL_SDE_COLOR))
            test_RW_curves.append( (eval_x_ns, test_R_ns + test_w_ns,"Neural_SDE", NEURAL_SDE_COLOR))

        if not args.gp_sde_only:
            pdf.savefig(fig_n_curves(train_RW_curves,
                "Train MSE (rotation + ω)", "MSE")); plt.close()
            pdf.savefig(fig_n_curves(train_R_curves,
                "Train geodesic error (geodesic²)", "MSE")); plt.close()
            pdf.savefig(fig_n_curves(train_w_curves,
                "Train MSE angular velocity", "MSE")); plt.close()
            pdf.savefig(fig_n_curves(test_RW_curves,
                "Test MSE (rotation + ω)", "MSE", smooth=False)); plt.close()
            pdf.savefig(fig_n_curves(test_R_curves,
                "Test geodesic error (geodesic²)", "MSE", smooth=False)); plt.close()
            pdf.savefig(fig_n_curves(test_w_curves,
                "Test MSE angular velocity", "MSE", smooth=False)); plt.close()
        else:
            pdf.savefig(fig_one_curve(x_gp, mse_R_gp + mse_w_gp,
                                      "Train MSE (rotation + ω) — GP_SDE", "MSE")); plt.close()
            pdf.savefig(fig_one_curve(x_gp, mse_R_gp,
                                      "Train geodesic error (geodesic²) — GP_SDE", "MSE")); plt.close()
            pdf.savefig(fig_one_curve(x_gp, mse_w_gp,
                                      "Train MSE angular velocity — GP_SDE", "MSE")); plt.close()
            pdf.savefig(fig_one_curve(eval_x_gp, test_R_gp + test_w_gp,
                                      "Test MSE (rotation + ω) — GP_SDE", "MSE")); plt.close()
            pdf.savefig(fig_one_curve(eval_x_gp, test_R_gp,
                                      "Test geodesic error (geodesic²) — GP_SDE", "MSE")); plt.close()
            pdf.savefig(fig_one_curve(eval_x_gp, test_w_gp,
                                      "Test MSE angular velocity — GP_SDE", "MSE")); plt.close()
        pdf.savefig(fig_one_curve(x_gp, _train_slice(stats_gp["train_nll"], gp_max),
                                  "Total train NLL (GP_SDE)", "NLL", logy=False)); plt.close()
        pdf.savefig(fig_one_curve(x_gp, _train_slice(stats_gp["train_kl_total"], gp_max),
                                  "Total train KL (GP_SDE)", "KL")); plt.close()
        # Clip per-subnet KL series to gp_max for the breakdown page.
        stats_gp_clipped = dict(stats_gp)
        for k in ("train_kl_M", "train_kl_V", "train_kl_Dw",
                  "train_kl_g", "train_kl_sigma"):
            if k in stats_gp_clipped:
                stats_gp_clipped[k] = _train_slice(stats_gp_clipped[k], gp_max)
        pdf.savefig(fig_kl_breakdown(stats_gp_clipped)); plt.close()
        pdf.savefig(fig_gp_log_w_covar(gp_sde)); plt.close()

        # ── A2. Per-trajectory MSE vs GT (rotation + ω) ───────────────
        traj_mse_models = {"GP_SDE": (gp_trajs_all, GP_COLOR)}
        if not args.gp_sde_only:
            traj_mse_models["NN_ODE"] = (nn_trajs_all, NN_COLOR)
        if use_neural_sde:
            traj_mse_models["Neural_SDE"] = (neural_trajs_all, NEURAL_SDE_COLOR)
        if args.include_gt_models:
            if not args.gp_sde_only:
                traj_mse_models["NN_ODE_GT"] = (nn_gt_trajs_all, NN_GT_COLOR)
            traj_mse_models["GP_SDE_GT"] = (gp_gt_trajs_all, GP_GT_COLOR)
        for f in fig_traj_mse(t_eval, gt_trajs_all, traj_mse_models):
            pdf.savefig(f); plt.close(f)

        # ── B. Ensemble dynamics ──────────────────────────────────────
        model_e = {"GP_SDE": (gp_e_all.mean(0), gp_e_all.std(0), GP_COLOR)}
        if not args.gp_sde_only:
            model_e["NN_ODE"] = (nn_e_all.mean(0), nn_e_all.std(0), NN_COLOR)
        if use_neural_sde:
            model_e["Neural_SDE"] = (ns_e_all.mean(0), ns_e_all.std(0), NEURAL_SDE_COLOR)
        if args.include_gt_models:
            if not args.gp_sde_only:
                model_e["NN_ODE_GT"] = (nn_gt_e_all.mean(0), nn_gt_e_all.std(0), NN_GT_COLOR)
            model_e["GP_SDE_GT"] = (gp_gt_e_all.mean(0), gp_gt_e_all.std(0), GP_GT_COLOR)
        pdf.savefig(fig_energy_ensemble(t_eval, gt_e_all, model_e)); plt.close()

        single_e = {"GP_SDE": (gp_e_all[0], GP_COLOR)}
        if not args.gp_sde_only:
            single_e["NN_ODE"] = (nn_e_all[0], NN_COLOR)
        if use_neural_sde:
            single_e["Neural_SDE"] = (ns_e_all[0], NEURAL_SDE_COLOR)
        if args.include_gt_models:
            if not args.gp_sde_only:
                single_e["NN_ODE_GT"] = (nn_gt_e_all[0], NN_GT_COLOR)
            single_e["GP_SDE_GT"] = (gp_gt_e_all[0], GP_GT_COLOR)
        pdf.savefig(fig_energy_single(t_eval, gt_e_all[0], single_e)); plt.close()

        # SO(3) violation (det)
        def _det_all(trajs):
            return np.array([so3_det_residual_abs(trajs[ti])
                             for ti in range(trajs.shape[0])])
        gt_det = _det_all(gt_trajs_all)
        gp_det = _det_all(gp_trajs_all)
        det_metrics = {"GP_SDE": (gp_det.mean(0), gp_det[0], GP_COLOR)}
        if not args.gp_sde_only:
            nn_det = _det_all(nn_trajs_all)
            det_metrics["NN_ODE"] = (nn_det.mean(0), nn_det[0], NN_COLOR)
        if use_neural_sde:
            ns_det = _det_all(neural_trajs_all)
            det_metrics["Neural_SDE"] = (ns_det.mean(0), ns_det[0], NEURAL_SDE_COLOR)
        if args.include_gt_models:
            gp_gt_det = _det_all(gp_gt_trajs_all)
            det_metrics["GP_SDE_GT"] = (gp_gt_det.mean(0), gp_gt_det[0], GP_GT_COLOR)
            if not args.gp_sde_only:
                nn_gt_det = _det_all(nn_gt_trajs_all)
                det_metrics["NN_ODE_GT"] = (nn_gt_det.mean(0), nn_gt_det[0], NN_GT_COLOR)
        pdf.savefig(fig_so3_violation(t_eval, gt_det, det_metrics,
                                      "|det(R) − 1|",
                                      "SO(3) Violation - Determinant")); plt.close()

        # SO(3) violation (orth)
        def _orth_all(trajs):
            return np.array([so3_orth_residual(trajs[ti])
                             for ti in range(trajs.shape[0])])
        gt_orth = _orth_all(gt_trajs_all)
        gp_orth = _orth_all(gp_trajs_all)
        orth_metrics = {"GP_SDE": (gp_orth.mean(0), gp_orth[0], GP_COLOR)}
        if not args.gp_sde_only:
            nn_orth = _orth_all(nn_trajs_all)
            orth_metrics["NN_ODE"] = (nn_orth.mean(0), nn_orth[0], NN_COLOR)
        if use_neural_sde:
            ns_orth = _orth_all(neural_trajs_all)
            orth_metrics["Neural_SDE"] = (ns_orth.mean(0), ns_orth[0], NEURAL_SDE_COLOR)
        if args.include_gt_models:
            gp_gt_orth = _orth_all(gp_gt_trajs_all)
            orth_metrics["GP_SDE_GT"] = (gp_gt_orth.mean(0), gp_gt_orth[0], GP_GT_COLOR)
            if not args.gp_sde_only:
                nn_gt_orth = _orth_all(nn_gt_trajs_all)
                orth_metrics["NN_ODE_GT"] = (nn_gt_orth.mean(0), nn_gt_orth[0], NN_GT_COLOR)
        pdf.savefig(fig_so3_violation(t_eval, gt_orth, orth_metrics,
                                      "‖RᵀR − I‖_F",
                                      "SO(3) Violation - Orthogonality")); plt.close()

        # State trajectories ensemble
        model_states = {"GP_SDE": (gp_eul_all, gp_om_all, GP_COLOR)}
        if not args.gp_sde_only:
            model_states["NN_ODE"] = (nn_eul_all, nn_om_all, NN_COLOR)
        if use_neural_sde:
            model_states["Neural_SDE"] = (ns_eul_all, ns_om_all, NEURAL_SDE_COLOR)
        if args.include_gt_models:
            if not args.gp_sde_only:
                model_states["NN_ODE_GT"] = (nn_gt_eul_all, nn_gt_om_all, NN_GT_COLOR)
            model_states["GP_SDE_GT"] = (gp_gt_eul_all, gp_gt_om_all, GP_GT_COLOR)
        pdf.savefig(fig_state_ensemble(t_eval, gt_eul_all, gt_om_all, model_states))
        plt.close()

        single_trajs = {"GP_SDE": (gp_trajs_all[0], GP_COLOR)}
        if not args.gp_sde_only:
            single_trajs["NN_ODE"] = (nn_trajs_all[0], NN_COLOR)
        if use_neural_sde:
            single_trajs["Neural_SDE"] = (neural_trajs_all[0], NEURAL_SDE_COLOR)
        if args.include_gt_models:
            if not args.gp_sde_only:
                single_trajs["NN_ODE_GT"] = (nn_gt_trajs_all[0], NN_GT_COLOR)
            single_trajs["GP_SDE_GT"] = (gp_gt_trajs_all[0], GP_GT_COLOR)
        pdf.savefig(fig_state_single(t_eval, gt_trajs_all[0], single_trajs)); plt.close()

        pdf.savefig(fig_phase_portraits(gt_eul_all, gt_om_all, model_states))
        plt.close()

        # ── C. Subnet evolution along 5 GT trajectories ───────────────
        def _finalise(fig, title):
            fig.suptitle(title, fontsize=14, fontweight="bold")
            fig.tight_layout(rect=(0, 0.0, 1, 0.97))
            pdf.savefig(fig); plt.close(fig)

        model_meta = {"GP_SDE": GP_COLOR}
        if not args.gp_sde_only:
            model_meta["NN_ODE"] = NN_COLOR
        beta_str = "  (β: " + ", ".join(
            f"{n}={betas[n]:.3f}" for n in model_meta) + ")"
        _finalise(
            plot_matrix_3x3_multi(t_eval, comp_multi, "M", gt_m_flat, model_meta),
            "Inverse Mass β·M⁻¹(q) Along 5 Trajectories" + beta_str)
        _finalise(
            plot_matrix_3x3_multi(t_eval, comp_multi, "D", gt_d_flat, model_meta),
            "Dissipation Dw(q)/β Along 5 Trajectories" + beta_str)
        _finalise(
            plot_matrix_3x3_multi(t_eval, comp_multi, "B", gt_b_flat, model_meta),
            "Control gain g(q)/β Along 5 Trajectories" + beta_str)
        _finalise(
            plot_potential_multi(t_eval, comp_multi, gt_v_all, model_meta),
            "Potential Energy V(q)/β Along 5 Trajectories" + beta_str)
        # σ subnet page — both GP_SDE and Neural_SDE have a diffusion head
        # (NN_ODE has no diffusion).
        sigma_meta = {"GP_SDE": GP_COLOR}
        if use_neural_sde:
            sigma_meta["Neural_SDE"] = NEURAL_SDE_COLOR
        _finalise(
            plot_xi_multi(t_eval, comp_multi, sigma_meta,
                          ENV_KW["wind_force_std"]),
            "Diffusion σ(q) Along 5 Trajectories")
        # Gauge-corrected σ/β page — σ is part of the (M, V, D, g) gauge
        # orbit (σ → α·σ under p → α·p), so σ_model / β is the right
        # comparison to GT when the likelihood is in ω-space.  Neural_SDE
        # has β = 1 (no port-Hamiltonian gauge), so its σ/β = σ.
        _finalise(
            plot_xi_beta_multi(t_eval, comp_multi, sigma_meta,
                               ENV_KW["wind_force_std"], betas),
            "Diffusion σ(q)/β Along 5 Trajectories" + beta_str)

    print(f"Done: {args.out_pdf}")


if __name__ == "__main__":
    main()
