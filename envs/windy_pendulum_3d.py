from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ──────────────────────────────────────────────────────────────────────────────
# SO(3) Lie group utilities
# ──────────────────────────────────────────────────────────────────────────────

def _hat(w: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix (hat map) for w in R^3.  hat: R^3 -> so(3)."""
    wx, wy, wz = w
    return np.array([[0.0, -wz,  wy],
                     [wz,  0.0, -wx],
                     [-wy, wx,  0.0]], dtype=np.float64)


def _vee(W: np.ndarray) -> np.ndarray:
    """Inverse hat map (vee).  vee: so(3) -> R^3."""
    return np.array([W[2, 1], W[0, 2], W[1, 0]], dtype=np.float64)


def _exp_so3(phi: np.ndarray) -> np.ndarray:
    """Matrix exponential on so(3) via Rodrigues' formula.
    
    Given phi in R^3, computes exp([phi]_x) in SO(3).
    
    exp([phi]_x) = I + (sin(theta)/theta) [phi]_x 
                     + ((1 - cos(theta))/theta^2) [phi]_x^2
    
    where theta = ||phi||.  Uses Taylor expansions for small theta
    to avoid division by zero.
    """
    theta_sq = np.dot(phi, phi)
    theta = np.sqrt(theta_sq)
    Phi = _hat(phi)

    if theta < 1e-10:
        # Taylor: sin(t)/t ≈ 1 - t²/6,  (1-cos(t))/t² ≈ 1/2 - t²/24
        A = 1.0 - theta_sq / 6.0
        B = 0.5 - theta_sq / 24.0
    else:
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / theta_sq

    return np.eye(3) + A * Phi + B * (Phi @ Phi)


def _log_so3(R: np.ndarray) -> np.ndarray:
    """Logarithmic map on SO(3).  log: SO(3) -> R^3.
    
    Returns phi such that R = exp([phi]_x).
    """
    cos_theta = 0.5 * (np.trace(R) - 1.0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-10:
        # Near identity: log(R) ≈ (R - R^T)/2
        return _vee(0.5 * (R - R.T))
    elif abs(theta - np.pi) < 1e-6:
        # Near pi: need special handling
        # Find the column of (R + I) with largest norm
        M = R + np.eye(3)
        norms = np.linalg.norm(M, axis=0)
        k = np.argmax(norms)
        v = M[:, k] / norms[k]
        return v * theta
    else:
        return _vee(theta / (2.0 * np.sin(theta)) * (R - R.T))


def _project_to_so3(R: np.ndarray) -> np.ndarray:
    """Project a 3x3 matrix to the nearest rotation matrix via polar decomposition.
    (Kept as a safety net, but should rarely be needed with the exp map integrator.)
    """
    U, _, Vt = np.linalg.svd(R)
    Rproj = U @ Vt
    if np.linalg.det(Rproj) < 0:
        U[:, -1] *= -1.0
        Rproj = U @ Vt
    return Rproj


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation matrix via random unit quaternion (Shoemake)."""
    u1, u2, u3 = rng.random(3)
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    x, y, z, w = q1, q2, q3, q4
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R


# ──────────────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────────────

class windy_pendulum_3d(gym.Env):
    """3D spherical pendulum on SO(3) with wind, friction, and stochastic forcing.
    
    Uses a geometrically exact Lie group integrator:
      - Rotation updates use the exponential map (Rodrigues), so R stays on SO(3)
        by construction — no SVD projection needed in the integration loop.
      - Stratonovich SDE is integrated via Heun's method on the Lie algebra.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        g: float = 9.81,
        m: float = 1.0,
        l: float = 1.0,
        dt: float = 0.05,
        max_torque: float = 2.0,
        max_speed: float = 8.0,
        friction_coeff: Union[float, Tuple[float, float, float]] = 0.1,
        varying_friction: bool = True,
        external_force_type: str = "sine",
        external_force_std: float = 1.0,
        external_force_direction: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        wind_force_std: float = 0.0,
        ori_rep: str = "rotmat",
        seed: Optional[int] = None,
    ):
        super().__init__()

        if ori_rep != "rotmat":
            raise ValueError("Only ori_rep='rotmat' is supported for the 3D pendulum env.")

        self.render_mode = render_mode
        self.g = float(g)
        self.m = float(m)
        self.l = float(l)
        self.dt = float(dt)

        self.max_torque = float(max_torque)
        self.max_speed = float(max_speed)

        self.friction_coeff = np.broadcast_to(
            np.asarray(friction_coeff, dtype=np.float64), (3,)
        ).copy()
        self.varying_friction = bool(varying_friction)

        self.external_force_type = str(external_force_type)
        self.external_force_std = float(external_force_std)
        wdir = np.array(external_force_direction, dtype=np.float64)
        if np.linalg.norm(wdir) < 1e-12:
            raise ValueError("external_force_direction must be non-zero.")
        self.external_force_direction = wdir / np.linalg.norm(wdir)

        self.wind_force_std = float(wind_force_std)
        self.ori_rep = ori_rep

        # Inertia tensor (isotropic for spherical pendulum)
        I_perp = self.m * (self.l ** 2)
        I_para = I_perp
        self.I = np.diag([I_perp, I_perp, I_para]).astype(np.float64)
        self.I_inv = np.linalg.inv(self.I)

        # NOTE: action_space uses self.max_torque to define the API bounds for
        # the agent's torque inputs. Kept active because gym.Env requires a
        # well-defined action_space.
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(3,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

        self._np_rng = np.random.default_rng(seed)

        self.t = 0.0
        self.last_u = np.zeros(3, dtype=np.float64)
        self.last_w = 0.0

        self.R = np.eye(3, dtype=np.float64)
        self.omega = np.zeros(3, dtype=np.float64)

        self._fig = None
        self._ax = None

    # ── Seeding & state access ────────────────────────────────────────────

    def seed(self, seed: Optional[int] = None):
        self._np_rng = np.random.default_rng(seed)

    def get_state(self):
        return self.R.copy(), self.omega.copy()

    # ── Wind model ────────────────────────────────────────────────────────

    def update_wind(self, t: float) -> float:
        if self.external_force_type == "sine":
            return self.external_force_std * np.sin(2 * np.pi * 0.5 * t)
        if self.external_force_type == "square":
            return self.external_force_std * (1.0 if np.sin(2 * np.pi * 0.5 * t) >= 0 else -1.0)
        if self.external_force_type == "random":
            return float(self._np_rng.normal(loc=0, scale=self.external_force_std))
        if self.external_force_type == "constant":
            return float(self.external_force_std)
        raise ValueError(f"Unknown external_force_type: {self.external_force_type}")

    # ── State-dependent friction ──────────────────────────────────────────

    def _variable_friction(self, R: np.ndarray, omega: np.ndarray) -> np.ndarray:
        if not self.varying_friction:
            return self.friction_coeff
        ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        bob_dir = R @ ez
        height_term = 0.5 * (1.0 - bob_dir[2])
        speed_term = np.tanh(np.linalg.norm(omega))
        return self.friction_coeff * (1.0 + 0.5 * height_term + 0.5 * speed_term)

    # ── Observation ───────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        obs = np.hstack((self.R.reshape(-1), self.omega)).astype(np.float32)
        return obs

    # ── Dynamics: compute omega_dot and stochastic increment ──────────────

    def _compute_omega_rates(
        self, R: np.ndarray, omega: np.ndarray,
        w: float, u: np.ndarray, dW: np.ndarray, sigma: float
    ):
        """Compute deterministic omega_dot and stochastic dOmega for given state.
        
        Returns:
            omega_dot_det : R^3  (deterministic angular acceleration in body frame)
            dOmega_stoch  : R^3  (stochastic angular velocity increment)
        """
        ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # Bob position in world frame
        r_world = self.l * (R @ ez)

        # Gravity torque  (world -> body)
        Fg = -self.m * self.g * ez
        tau_g_body = R.T @ np.cross(r_world, Fg)

        # Deterministic wind torque  (world -> body)
        Fw_det = w * self.external_force_direction
        tau_w_body = R.T @ np.cross(r_world, Fw_det)

        # Friction torque  (body frame, opposes omega)
        fric = self._variable_friction(R, omega)
        tau_fric = fric * omega

        # Control torque is already in body frame
        tau_det = tau_g_body + tau_w_body + u - tau_fric

        # Euler's rigid body equation: I w_dot = tau - w x (I w)
        Iw = self.I @ omega
        omega_dot_det = self.I_inv @ (tau_det - np.cross(omega, Iw))

        # Stochastic forcing
        if sigma > 0.0:
            dF_stoch = sigma * dW
            tau_stoch_body = R.T @ np.cross(r_world, dF_stoch)
            dOmega_stoch = self.I_inv @ tau_stoch_body
        else:
            dOmega_stoch = np.zeros(3)

        return omega_dot_det, dOmega_stoch

    # ── Lie group Heun integrator (Stratonovich) ──────────────────────────

    def _lie_heun_step(
        self, R: np.ndarray, omega: np.ndarray,
        w: float, u: np.ndarray, h: float, sigma: float,
        dW: np.ndarray
    ):
        """One substep of Stratonovich Heun integration on SO(3) x R^3.
        
        Rotation update uses the exponential map so R stays on SO(3)
        by construction.  The Heun scheme averages two Lie algebra
        elements before exponentiating, giving second-order accuracy
        in the deterministic part.
        
        Algorithm:
        ---------
        1. At (R_n, omega_n), compute omega_dot_1 and the Lie algebra
           element phi_1 = omega_n * h  (body angular displacement).
        
        2. Predictor:
             R_pred    = R_n * exp([phi_1]_x)
             omega_pred = omega_n + omega_dot_1 * h + dOmega_stoch_1
        
        3. At (R_pred, omega_pred), compute omega_dot_2 and
           phi_2 = omega_pred * h.
        
        4. Corrector (average in Lie algebra, then exponentiate once):
             phi_avg   = (phi_1 + phi_2) / 2
             R_{n+1}   = R_n * exp([phi_avg]_x)
             omega_{n+1} = omega_n + (omega_dot_1 + omega_dot_2)/2 * h
                           + (dOmega_stoch_1 + dOmega_stoch_2) / 2
        """
        # ── Stage 1: evaluate at current state ──
        omega_dot_1, dOmega_stoch_1 = self._compute_omega_rates(
            R, omega, w, u, dW, sigma
        )
        phi_1 = omega * h  # Lie algebra element for rotation

        # ── Stage 2: predictor (Euler on the manifold) ──
        R_pred = R @ _exp_so3(phi_1)
        omega_pred = omega + omega_dot_1 * h + dOmega_stoch_1

        # ── Stage 3: evaluate at predicted state (reuse same dW) ──
        omega_dot_2, dOmega_stoch_2 = self._compute_omega_rates(
            R_pred, omega_pred, w, u, dW, sigma
        )
        phi_2 = omega_pred * h

        # ── Stage 4: corrector (average in Lie algebra, single exp) ──
        phi_avg = 0.5 * (phi_1 + phi_2)
        R_new = R @ _exp_so3(phi_avg)

        omega_new = (
            omega
            + 0.5 * (omega_dot_1 + omega_dot_2) * h
            + 0.5 * (dOmega_stoch_1 + dOmega_stoch_2)
        )

        return R_new, omega_new

    # ── Step ──────────────────────────────────────────────────────────────

    def step(self, u):
        u = np.asarray(u, dtype=np.float64).reshape(3)
        # NOTE: torque clipping by self.max_torque disabled — keeping the
        # parameter available but no longer constraining the action.
        # u = np.clip(u, -self.max_torque, self.max_torque)
        self.last_u = u.copy()

        self.t += self.dt
        w = self.update_wind(self.t)
        self.last_w = float(w)

        # Substep settings
        n_substeps = 10
        dt_sub = self.dt / n_substeps
        sigma = self.wind_force_std

        for _ in range(n_substeps):
            # Sample Wiener increment once per substep
            if sigma > 0.0:
                dW = self._np_rng.normal(0.0, np.sqrt(dt_sub), size=3)
            else:
                dW = np.zeros(3)

            # Lie group Heun step
            self.R, self.omega = self._lie_heun_step(
                self.R, self.omega, w, u, dt_sub, sigma, dW
            )

        # NOTE: angular velocity clipping by self.max_speed disabled —
        # keeping the parameter available but no longer constraining omega.
        # self.omega = np.clip(self.omega, -self.max_speed, self.max_speed)

        # Periodic re-orthogonalization as a safety net
        # (numerical drift from floating-point accumulation over many steps)
        if abs(np.linalg.det(self.R) - 1.0) > 1e-8 or \
           np.linalg.norm(self.R.T @ self.R - np.eye(3)) > 1e-8:
            self.R = _project_to_so3(self.R)

        # ── Reward ──
        ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        bob_dir = self.R @ ez
        angle_cost = 1.0 - float(bob_dir[2])
        vel_cost = 0.1 * float(np.dot(self.omega, self.omega))
        act_cost = 0.001 * float(np.dot(u, u))
        cost = angle_cost + vel_cost + act_cost

        obs = self._get_obs()
        reward = -cost
        terminated = False
        truncated = False
        info = {"wind": self.last_w}

        return obs, reward, terminated, truncated, info

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        self.t = 0.0
        self.last_u = np.zeros(3, dtype=np.float64)
        self.last_w = 0.0

        if options is None:
            options = {}

        if "R_init" in options:
            R0 = np.asarray(options["R_init"], dtype=np.float64).reshape(3, 3)
            R0 = _project_to_so3(R0)
        else:
            R0 = _random_rotation(self._np_rng)

        if "omega_init" in options:
            w0 = np.asarray(options["omega_init"], dtype=np.float64).reshape(3)
        else:
            w0 = self._np_rng.uniform(low=-1.0, high=1.0, size=3)

        # NOTE: initial-omega clipping by self.max_speed disabled — keeping the
        # parameter available but no longer constraining the initial state.
        # w0 = np.clip(w0, -self.max_speed, self.max_speed)

        self.R = R0
        self.omega = w0

        return self._get_obs(), {}

    # ── Rendering ─────────────────────────────────────────────────────────

    def _init_render(self):
        import matplotlib
        if self.render_mode == "rgb_array":
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self._fig = plt.figure(figsize=(7, 7))
        self._ax = self._fig.add_subplot(111, projection="3d")

    def render(self):
        if self.render_mode is None:
            return

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        if self._fig is None or not plt.fignum_exists(self._fig.number):
            self._init_render()

        ax = self._ax
        ax.cla()

        origin = np.zeros(3)
        ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        bob = self.l * (self.R @ ez)

        ax.plot(
            [origin[0], bob[0]],
            [origin[1], bob[1]],
            [origin[2], bob[2]],
            color="#555555", linewidth=3, solid_capstyle="round",
        )

        ax.scatter(*bob, color="#cc4d4d", s=120, depthshade=True, zorder=5)
        ax.scatter(*origin, color="black", s=60, depthshade=True, zorder=5)

        axis_len = 0.35 * self.l
        axis_colors = ["#e74c3c", "#2ecc71", "#3498db"]
        axis_labels = ["x_b", "y_b", "z_b"]
        for i in range(3):
            e_body = np.zeros(3)
            e_body[i] = 1.0
            tip = bob + axis_len * (self.R @ e_body)
            ax.quiver(
                bob[0], bob[1], bob[2],
                tip[0] - bob[0], tip[1] - bob[1], tip[2] - bob[2],
                color=axis_colors[i], linewidth=2, arrow_length_ratio=0.15,
            )
            ax.text(tip[0], tip[1], tip[2], f" {axis_labels[i]}",
                    color=axis_colors[i], fontsize=8, fontweight="bold")

        if abs(self.last_w) > 1e-6:
            w_vec = self.last_w * self.external_force_direction
            w_scale = 0.5 * self.l
            ax.quiver(
                0, 0, 0,
                w_vec[0] * w_scale, w_vec[1] * w_scale, w_vec[2] * w_scale,
                color="#00bcd4", linewidth=2.5, arrow_length_ratio=0.18,
                label=f"wind = {self.last_w:+.2f}",
            )

        g_len = 0.4 * self.l
        ax.quiver(
            0, 0, 0,
            0, 0, -g_len,
            color="#999999", linewidth=1.5, arrow_length_ratio=0.15,
            linestyle="dashed", label="gravity",
        )

        theta = np.linspace(0, 2 * np.pi, 60)
        r_circle = 1.2 * self.l
        xc = r_circle * np.cos(theta)
        yc = r_circle * np.sin(theta)
        zc = np.zeros_like(theta)
        verts = [list(zip(xc, yc, zc))]
        ground = Poly3DCollection(
            verts, alpha=0.08, facecolor="#b0bec5",
            edgecolor="#78909c", linewidth=0.8,
        )
        ax.add_collection3d(ground)

        lim = 1.5 * self.l
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel("X", fontsize=9)
        ax.set_ylabel("Y", fontsize=9)
        ax.set_zlabel("Z", fontsize=9)
        ax.set_title("3D Pendulum on SO(3) — Lie Group Integrator", fontsize=12, fontweight="bold")
        ax.set_box_aspect([1, 1, 1])

        omega_mag = float(np.linalg.norm(self.omega))
        hud = (
            f"t = {self.t:.2f}s    "
            f"wind = {self.last_w:+.2f}    "
            f"|omega| = {omega_mag:.2f}"
        )
        ax.text2D(
            0.02, 0.96, hud, transform=ax.transAxes,
            fontsize=9, fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

        if self.render_mode == "human":
            plt.draw()
            plt.pause(0.001)
        else:
            self._fig.canvas.draw()
            buf = self._fig.canvas.buffer_rgba()
            img = np.asarray(buf)[:, :, :3].copy()
            return img

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
        self._fig = None
        self._ax = None


# ──────────────────────────────────────────────────────────────────────────────
# Demo / video generation
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    N_STEPS = 500
    SAVE_PATH = "videos/windy_pendulum_3d_lie_group.mp4"

    env = windy_pendulum_3d(
        g=9.81,
        m=1.0,
        l=1.0,
        dt=0.05,
        varying_friction=False,
        friction_coeff=[0.5, 0.5, 0.5],
        external_force_type="sine",
        external_force_std=0,
        render_mode="rgb_array",
        wind_force_std=5,
        seed=42,
    )
    obs, _ = env.reset(seed=42)

    frames = []
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    for step in range(N_STEPS):
        action = [0.0, 0.0, 0.0]
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if step % 20 == 0:
            R, omega = env.get_state()
            det_R = np.linalg.det(R)
            orth_err = np.linalg.norm(R.T @ R - np.eye(3))
            print(
                f"Step {step:3d}/{N_STEPS}  |  reward={reward:.3f}  |  "
                f"wind={info['wind']:+.2f}  |  det(R)={det_R:.8f}  |  "
                f"orth_err={orth_err:.2e}"
            )

    env.close()

    print(f"\nSaving {len(frames)} frames to {SAVE_PATH} ...")

    import os
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    fig_vid, ax_vid = plt.subplots(figsize=(7, 7))
    ax_vid.axis("off")
    im = ax_vid.imshow(frames[0])

    def _update(i):
        im.set_data(frames[i])
        return [im]

    anim = FuncAnimation(fig_vid, _update, frames=len(frames), interval=1000 / 30, blit=True)
    anim.save(SAVE_PATH, writer="ffmpeg", fps=30, dpi=100)
    plt.close(fig_vid)

    print(f"Done! Video saved to: {SAVE_PATH}")






















































# from __future__ import annotations

# from typing import Optional, Tuple, Union

# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces


# # ──────────────────────────────────────────────────────────────────────────────
# # SO(3) Lie group utilities
# # ──────────────────────────────────────────────────────────────────────────────

# def _hat(w: np.ndarray) -> np.ndarray:
#     """Skew-symmetric matrix (hat map) for w in R^3.  hat: R^3 -> so(3)."""
#     wx, wy, wz = w
#     return np.array([[0.0, -wz,  wy],
#                      [wz,  0.0, -wx],
#                      [-wy, wx,  0.0]], dtype=np.float64)


# def _vee(W: np.ndarray) -> np.ndarray:
#     """Inverse hat map (vee).  vee: so(3) -> R^3."""
#     return np.array([W[2, 1], W[0, 2], W[1, 0]], dtype=np.float64)


# def _exp_so3(phi: np.ndarray) -> np.ndarray:
#     """Matrix exponential on so(3) via Rodrigues' formula.
    
#     Given phi in R^3, computes exp([phi]_x) in SO(3).
    
#     exp([phi]_x) = I + (sin(theta)/theta) [phi]_x 
#                      + ((1 - cos(theta))/theta^2) [phi]_x^2
    
#     where theta = ||phi||.  Uses Taylor expansions for small theta
#     to avoid division by zero.
#     """
#     theta_sq = np.dot(phi, phi)
#     theta = np.sqrt(theta_sq)
#     Phi = _hat(phi)

#     if theta < 1e-10:
#         # Taylor: sin(t)/t ≈ 1 - t²/6,  (1-cos(t))/t² ≈ 1/2 - t²/24
#         A = 1.0 - theta_sq / 6.0
#         B = 0.5 - theta_sq / 24.0
#     else:
#         A = np.sin(theta) / theta
#         B = (1.0 - np.cos(theta)) / theta_sq

#     return np.eye(3) + A * Phi + B * (Phi @ Phi)


# def _log_so3(R: np.ndarray) -> np.ndarray:
#     """Logarithmic map on SO(3).  log: SO(3) -> R^3.
    
#     Returns phi such that R = exp([phi]_x).
#     """
#     cos_theta = 0.5 * (np.trace(R) - 1.0)
#     cos_theta = np.clip(cos_theta, -1.0, 1.0)
#     theta = np.arccos(cos_theta)

#     if theta < 1e-10:
#         # Near identity: log(R) ≈ (R - R^T)/2
#         return _vee(0.5 * (R - R.T))
#     elif abs(theta - np.pi) < 1e-6:
#         # Near pi: need special handling
#         # Find the column of (R + I) with largest norm
#         M = R + np.eye(3)
#         norms = np.linalg.norm(M, axis=0)
#         k = np.argmax(norms)
#         v = M[:, k] / norms[k]
#         return v * theta
#     else:
#         return _vee(theta / (2.0 * np.sin(theta)) * (R - R.T))


# def _project_to_so3(R: np.ndarray) -> np.ndarray:
#     """Project a 3x3 matrix to the nearest rotation matrix via polar decomposition.
#     (Kept as a safety net, but should rarely be needed with the exp map integrator.)
#     """
#     U, _, Vt = np.linalg.svd(R)
#     Rproj = U @ Vt
#     if np.linalg.det(Rproj) < 0:
#         U[:, -1] *= -1.0
#         Rproj = U @ Vt
#     return Rproj


# def _random_rotation(rng: np.random.Generator) -> np.ndarray:
#     """Uniform random rotation matrix via random unit quaternion (Shoemake)."""
#     u1, u2, u3 = rng.random(3)
#     q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
#     q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
#     q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
#     q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
#     x, y, z, w = q1, q2, q3, q4
#     R = np.array([
#         [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
#         [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
#         [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
#     ], dtype=np.float64)
#     return R


# # ──────────────────────────────────────────────────────────────────────────────
# # Environment
# # ──────────────────────────────────────────────────────────────────────────────

# class windy_pendulum_3d(gym.Env):
#     """3D spherical pendulum on SO(3) with wind, friction, and stochastic forcing.
    
#     Uses a geometrically exact Lie group integrator:
#       - Rotation updates use the exponential map (Rodrigues), so R stays on SO(3)
#         by construction — no SVD projection needed in the integration loop.
#       - Stratonovich SDE is integrated via Heun's method on the Lie algebra.
#     """

#     metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

#     def __init__(
#         self,
#         render_mode: Optional[str] = None,
#         g: float = 9.81,
#         m: float = 1.0,
#         l: float = 1.0,
#         dt: float = 0.05,
#         max_torque: float = 2.0,
#         max_speed: float = 8.0,
#         friction_coeff: Union[float, Tuple[float, float, float]] = 0.1,
#         varying_friction: bool = True,
#         wind_type: str = "sine",
#         max_wind: float = 1.0,
#         external_force_direction: Tuple[float, float, float] = (1.0, 0.0, 0.0),
#         process_noise_std: float = 0.0,
#         ori_rep: str = "rotmat",
#         seed: Optional[int] = None,
#     ):
#         super().__init__()

#         if ori_rep != "rotmat":
#             raise ValueError("Only ori_rep='rotmat' is supported for the 3D pendulum env.")

#         self.render_mode = render_mode
#         self.g = float(g)
#         self.m = float(m)
#         self.l = float(l)
#         self.dt = float(dt)

#         self.max_torque = float(max_torque)
#         self.max_speed = float(max_speed)

#         self.friction_coeff = np.broadcast_to(
#             np.asarray(friction_coeff, dtype=np.float64), (3,)
#         ).copy()
#         self.varying_friction = bool(varying_friction)

#         self.wind_type = str(wind_type)
#         self.max_wind = float(max_wind)
#         wdir = np.array(external_force_direction, dtype=np.float64)
#         if np.linalg.norm(wdir) < 1e-12:
#             raise ValueError("external_force_direction must be non-zero.")
#         self.external_force_direction = wdir / np.linalg.norm(wdir)

#         self.process_noise_std = float(process_noise_std)
#         self.ori_rep = ori_rep

#         # Inertia tensor (isotropic for spherical pendulum)
#         I_perp = self.m * (self.l ** 2)
#         I_para = I_perp
#         self.I = np.diag([I_perp, I_perp, I_para]).astype(np.float64)
#         self.I_inv = np.linalg.inv(self.I)

#         self.action_space = spaces.Box(
#             low=-self.max_torque, high=self.max_torque, shape=(3,), dtype=np.float32
#         )
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
#         )

#         self._np_rng = np.random.default_rng(seed)

#         self.t = 0.0
#         self.last_u = np.zeros(3, dtype=np.float64)
#         self.last_w = 0.0

#         self.R = np.eye(3, dtype=np.float64)
#         self.omega = np.zeros(3, dtype=np.float64)

#         self._fig = None
#         self._ax = None

#     # ── Seeding & state access ────────────────────────────────────────────

#     def seed(self, seed: Optional[int] = None):
#         self._np_rng = np.random.default_rng(seed)

#     def get_state(self):
#         return self.R.copy(), self.omega.copy()

#     # ── Wind model ────────────────────────────────────────────────────────

#     def update_wind(self, t: float) -> float:
#         if self.wind_type == "sine":
#             return self.max_wind * np.sin(2 * np.pi * 0.5 * t)
#         if self.wind_type == "square":
#             return self.max_wind * (1.0 if np.sin(2 * np.pi * 0.5 * t) >= 0 else -1.0)
#         if self.wind_type == "random":
#             return float(self._np_rng.normal(loc=0, scale=self.max_wind))
#         if self.wind_type == "constant":
#             return float(self.max_wind)
#         raise ValueError(f"Unknown wind_type: {self.wind_type}")

#     # ── State-dependent friction ──────────────────────────────────────────

#     def _variable_friction(self, R: np.ndarray, omega: np.ndarray) -> np.ndarray:
#         if not self.varying_friction:
#             return self.friction_coeff
#         ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)
#         bob_dir = R @ ez
#         height_term = 0.5 * (1.0 - bob_dir[2])
#         speed_term = np.tanh(np.linalg.norm(omega))
#         return self.friction_coeff * (1.0 + 0.5 * height_term + 0.5 * speed_term)

#     # ── Observation ───────────────────────────────────────────────────────

#     def _get_obs(self) -> np.ndarray:
#         obs = np.hstack((self.R.reshape(-1), self.omega)).astype(np.float32)
#         return obs

#     # ── Dynamics: compute omega_dot and stochastic increment ──────────────

#     def _compute_omega_rates(
#         self, R: np.ndarray, omega: np.ndarray,
#         w: float, u: np.ndarray, dW: np.ndarray, sigma: float
#     ):
#         """Compute deterministic omega_dot and stochastic dOmega for given state.
        
#         Returns:
#             omega_dot_det : R^3  (deterministic angular acceleration in body frame)
#             dOmega_stoch  : R^3  (stochastic angular velocity increment)
#         """
#         ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)

#         # Bob position in world frame
#         r_world = self.l * (R @ ez)

#         # Gravity torque  (world -> body)
#         Fg = -self.m * self.g * ez
#         tau_g_body = R.T @ np.cross(r_world, Fg)

#         # Deterministic wind torque  (world -> body)
#         Fw_det = w * self.external_force_direction
#         tau_w_body = R.T @ np.cross(r_world, Fw_det)

#         # Friction torque  (body frame, opposes omega)
#         fric = self._variable_friction(R, omega)
#         tau_fric = fric * omega

#         # Control torque is already in body frame
#         tau_det = tau_g_body + tau_w_body + u - tau_fric

#         # Euler's rigid body equation: I w_dot = tau - w x (I w)
#         Iw = self.I @ omega
#         omega_dot_det = self.I_inv @ (tau_det - np.cross(omega, Iw))

#         # Stochastic forcing
#         if sigma > 0.0:
#             dF_stoch = sigma * dW
#             tau_stoch_body = R.T @ np.cross(r_world, dF_stoch)
#             dOmega_stoch = self.I_inv @ tau_stoch_body
#         else:
#             dOmega_stoch = np.zeros(3)

#         return omega_dot_det, dOmega_stoch

#     # ── Lie group Heun integrator (Stratonovich) ──────────────────────────

#     def _lie_heun_step(
#         self, R: np.ndarray, omega: np.ndarray,
#         w: float, u: np.ndarray, h: float, sigma: float,
#         dW: np.ndarray
#     ):
#         """One substep of Stratonovich Heun integration on SO(3) x R^3.
        
#         Rotation update uses the exponential map so R stays on SO(3)
#         by construction.  The Heun scheme averages two Lie algebra
#         elements before exponentiating, giving second-order accuracy
#         in the deterministic part.
        
#         Algorithm:
#         ---------
#         1. At (R_n, omega_n), compute omega_dot_1 and the Lie algebra
#            element phi_1 = omega_n * h  (body angular displacement).
        
#         2. Predictor:
#              R_pred    = R_n * exp([phi_1]_x)
#              omega_pred = omega_n + omega_dot_1 * h + dOmega_stoch_1
        
#         3. At (R_pred, omega_pred), compute omega_dot_2 and
#            phi_2 = omega_pred * h.
        
#         4. Corrector (average in Lie algebra, then exponentiate once):
#              phi_avg   = (phi_1 + phi_2) / 2
#              R_{n+1}   = R_n * exp([phi_avg]_x)
#              omega_{n+1} = omega_n + (omega_dot_1 + omega_dot_2)/2 * h
#                            + (dOmega_stoch_1 + dOmega_stoch_2) / 2
#         """
#         # ── Stage 1: evaluate at current state ──
#         omega_dot_1, dOmega_stoch_1 = self._compute_omega_rates(
#             R, omega, w, u, dW, sigma
#         )
#         phi_1 = omega * h  # Lie algebra element for rotation

#         # ── Stage 2: predictor (Euler on the manifold) ──
#         R_pred = R @ _exp_so3(phi_1)
#         omega_pred = omega + omega_dot_1 * h + dOmega_stoch_1

#         # ── Stage 3: evaluate at predicted state (reuse same dW) ──
#         omega_dot_2, dOmega_stoch_2 = self._compute_omega_rates(
#             R_pred, omega_pred, w, u, dW, sigma
#         )
#         phi_2 = omega_pred * h

#         # ── Stage 4: corrector (average in Lie algebra, single exp) ──
#         phi_avg = 0.5 * (phi_1 + phi_2)
#         R_new = R @ _exp_so3(phi_avg)

#         omega_new = (
#             omega
#             + 0.5 * (omega_dot_1 + omega_dot_2) * h
#             + 0.5 * (dOmega_stoch_1 + dOmega_stoch_2)
#         )

#         return R_new, omega_new

#     # ── Step ──────────────────────────────────────────────────────────────

#     def step(self, u):
#         u = np.asarray(u, dtype=np.float64).reshape(3)
#         u = np.clip(u, -self.max_torque, self.max_torque)
#         self.last_u = u.copy()

#         self.t += self.dt
#         w = self.update_wind(self.t)
#         self.last_w = float(w)

#         # Substep settings
#         n_substeps = 10
#         dt_sub = self.dt / n_substeps
#         sigma = self.process_noise_std

#         for _ in range(n_substeps):
#             # Sample Wiener increment once per substep
#             if sigma > 0.0:
#                 dW = self._np_rng.normal(0.0, np.sqrt(dt_sub), size=3)
#             else:
#                 dW = np.zeros(3)

#             # Lie group Heun step
#             self.R, self.omega = self._lie_heun_step(
#                 self.R, self.omega, w, u, dt_sub, sigma, dW
#             )

#         # Clip angular velocity for safety
#         self.omega = np.clip(self.omega, -self.max_speed, self.max_speed)

#         # Periodic re-orthogonalization as a safety net
#         # (numerical drift from floating-point accumulation over many steps)
#         if abs(np.linalg.det(self.R) - 1.0) > 1e-8 or \
#            np.linalg.norm(self.R.T @ self.R - np.eye(3)) > 1e-8:
#             self.R = _project_to_so3(self.R)

#         # ── Reward ──
#         ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)
#         bob_dir = self.R @ ez
#         angle_cost = 1.0 - float(bob_dir[2])
#         vel_cost = 0.1 * float(np.dot(self.omega, self.omega))
#         act_cost = 0.001 * float(np.dot(u, u))
#         cost = angle_cost + vel_cost + act_cost

#         obs = self._get_obs()
#         reward = -cost
#         terminated = False
#         truncated = False
#         info = {"wind": self.last_w}

#         return obs, reward, terminated, truncated, info

#     # ── Reset ─────────────────────────────────────────────────────────────

#     def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
#         super().reset(seed=seed)
#         if seed is not None:
#             self.seed(seed)

#         self.t = 0.0
#         self.last_u = np.zeros(3, dtype=np.float64)
#         self.last_w = 0.0

#         if options is None:
#             options = {}

#         if "R_init" in options:
#             R0 = np.asarray(options["R_init"], dtype=np.float64).reshape(3, 3)
#             R0 = _project_to_so3(R0)
#         else:
#             R0 = _random_rotation(self._np_rng)

#         if "omega_init" in options:
#             w0 = np.asarray(options["omega_init"], dtype=np.float64).reshape(3)
#         else:
#             w0 = self._np_rng.uniform(low=-1.0, high=1.0, size=3)

#         w0 = np.clip(w0, -self.max_speed, self.max_speed)

#         self.R = R0
#         self.omega = w0

#         return self._get_obs(), {}

#     # ── Rendering ─────────────────────────────────────────────────────────

#     def _init_render(self):
#         import matplotlib
#         if self.render_mode == "rgb_array":
#             matplotlib.use("Agg")
#         import matplotlib.pyplot as plt

#         self._fig = plt.figure(figsize=(7, 7))
#         self._ax = self._fig.add_subplot(111, projection="3d")

#     def render(self):
#         if self.render_mode is None:
#             return

#         import matplotlib.pyplot as plt
#         from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#         if self._fig is None or not plt.fignum_exists(self._fig.number):
#             self._init_render()

#         ax = self._ax
#         ax.cla()

#         origin = np.zeros(3)
#         ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)
#         bob = self.l * (self.R @ ez)

#         ax.plot(
#             [origin[0], bob[0]],
#             [origin[1], bob[1]],
#             [origin[2], bob[2]],
#             color="#555555", linewidth=3, solid_capstyle="round",
#         )

#         ax.scatter(*bob, color="#cc4d4d", s=120, depthshade=True, zorder=5)
#         ax.scatter(*origin, color="black", s=60, depthshade=True, zorder=5)

#         axis_len = 0.35 * self.l
#         axis_colors = ["#e74c3c", "#2ecc71", "#3498db"]
#         axis_labels = ["x_b", "y_b", "z_b"]
#         for i in range(3):
#             e_body = np.zeros(3)
#             e_body[i] = 1.0
#             tip = bob + axis_len * (self.R @ e_body)
#             ax.quiver(
#                 bob[0], bob[1], bob[2],
#                 tip[0] - bob[0], tip[1] - bob[1], tip[2] - bob[2],
#                 color=axis_colors[i], linewidth=2, arrow_length_ratio=0.15,
#             )
#             ax.text(tip[0], tip[1], tip[2], f" {axis_labels[i]}",
#                     color=axis_colors[i], fontsize=8, fontweight="bold")

#         if abs(self.last_w) > 1e-6:
#             w_vec = self.last_w * self.external_force_direction
#             w_scale = 0.5 * self.l
#             ax.quiver(
#                 0, 0, 0,
#                 w_vec[0] * w_scale, w_vec[1] * w_scale, w_vec[2] * w_scale,
#                 color="#00bcd4", linewidth=2.5, arrow_length_ratio=0.18,
#                 label=f"wind = {self.last_w:+.2f}",
#             )

#         g_len = 0.4 * self.l
#         ax.quiver(
#             0, 0, 0,
#             0, 0, -g_len,
#             color="#999999", linewidth=1.5, arrow_length_ratio=0.15,
#             linestyle="dashed", label="gravity",
#         )

#         theta = np.linspace(0, 2 * np.pi, 60)
#         r_circle = 1.2 * self.l
#         xc = r_circle * np.cos(theta)
#         yc = r_circle * np.sin(theta)
#         zc = np.zeros_like(theta)
#         verts = [list(zip(xc, yc, zc))]
#         ground = Poly3DCollection(
#             verts, alpha=0.08, facecolor="#b0bec5",
#             edgecolor="#78909c", linewidth=0.8,
#         )
#         ax.add_collection3d(ground)

#         lim = 1.5 * self.l
#         ax.set_xlim(-lim, lim)
#         ax.set_ylim(-lim, lim)
#         ax.set_zlim(-lim, lim)
#         ax.set_xlabel("X", fontsize=9)
#         ax.set_ylabel("Y", fontsize=9)
#         ax.set_zlabel("Z", fontsize=9)
#         ax.set_title("3D Pendulum on SO(3) — Lie Group Integrator", fontsize=12, fontweight="bold")
#         ax.set_box_aspect([1, 1, 1])

#         omega_mag = float(np.linalg.norm(self.omega))
#         hud = (
#             f"t = {self.t:.2f}s    "
#             f"wind = {self.last_w:+.2f}    "
#             f"|omega| = {omega_mag:.2f}"
#         )
#         ax.text2D(
#             0.02, 0.96, hud, transform=ax.transAxes,
#             fontsize=9, fontfamily="monospace",
#             verticalalignment="top",
#             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
#         )

#         ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

#         if self.render_mode == "human":
#             plt.draw()
#             plt.pause(0.001)
#         else:
#             self._fig.canvas.draw()
#             buf = self._fig.canvas.buffer_rgba()
#             img = np.asarray(buf)[:, :, :3].copy()
#             return img

#     def close(self):
#         if self._fig is not None:
#             import matplotlib.pyplot as plt
#             plt.close(self._fig)
#         self._fig = None
#         self._ax = None


# # ──────────────────────────────────────────────────────────────────────────────
# # Demo / video generation
# # ──────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
#     from matplotlib.animation import FuncAnimation

#     N_STEPS = 500
#     SAVE_PATH = "videos/windy_pendulum_3d_lie_group.mp4"

#     env = windy_pendulum_3d(
#         g=9.81,
#         m=1.0,
#         l=1.0,
#         dt=0.05,
#         varying_friction=False,
#         friction_coeff=[0.5, 0.5, 0.5],
#         wind_type="sine",
#         max_wind=0,
#         render_mode="rgb_array",
#         process_noise_std=5,
#         seed=42,
#     )
#     obs, _ = env.reset(seed=42)

#     frames = []
#     frame = env.render()
#     if frame is not None:
#         frames.append(frame)

#     for step in range(N_STEPS):
#         action = [0.0, 0.0, 0.0]
#         obs, reward, terminated, truncated, info = env.step(action)
#         frame = env.render()
#         if frame is not None:
#             frames.append(frame)
#         if step % 20 == 0:
#             R, omega = env.get_state()
#             det_R = np.linalg.det(R)
#             orth_err = np.linalg.norm(R.T @ R - np.eye(3))
#             print(
#                 f"Step {step:3d}/{N_STEPS}  |  reward={reward:.3f}  |  "
#                 f"wind={info['wind']:+.2f}  |  det(R)={det_R:.8f}  |  "
#                 f"orth_err={orth_err:.2e}"
#             )

#     env.close()

#     print(f"\nSaving {len(frames)} frames to {SAVE_PATH} ...")

#     import os
#     os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

#     fig_vid, ax_vid = plt.subplots(figsize=(7, 7))
#     ax_vid.axis("off")
#     im = ax_vid.imshow(frames[0])

#     def _update(i):
#         im.set_data(frames[i])
#         return [im]

#     anim = FuncAnimation(fig_vid, _update, frames=len(frames), interval=1000 / 30, blit=True)
#     anim.save(SAVE_PATH, writer="ffmpeg", fps=30, dpi=100)
#     plt.close(fig_vid)

#     print(f"Done! Video saved to: {SAVE_PATH}")