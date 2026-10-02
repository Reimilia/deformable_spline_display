import jax
import jax.numpy as jnp
import equinox as eqx
import math
from typing import Optional

class MaternFeatures(eqx.Module):
    base_rotations_flat: jnp.ndarray
    omega_angles: jnp.ndarray
    omega_vels: jnp.ndarray
    phases: jnp.ndarray
    scale: jnp.ndarray
    nu:           float = eqx.field(static=True)
    ell:          float = eqx.field(static=True)
    n_features:   int   = eqx.field(static=True)
    has_velocity: bool  = eqx.field(static=True)

    def __init__(self, key, input_dim: int, n_features: int, nu: float = 2.5, ell: float = 1.0):
        self.nu = nu
        self.ell = ell
        self.n_features = n_features
        # If input is 12D, it includes angular velocity (TSO3 manifold)
        self.has_velocity = (input_dim == 12) 
        
        k_rot, k_z, k_s, k_p = jax.random.split(key, 4)
        
        # 1. Sample VALID random SO(3) base rotations
        def get_rand_rot(k):
            u1, u2, u3 = jax.random.uniform(k, (3,))
            q1 = jnp.sqrt(1 - u1) * jnp.sin(2 * jnp.pi * u2)
            q2 = jnp.sqrt(1 - u1) * jnp.cos(2 * jnp.pi * u2)
            q3 = jnp.sqrt(u1) * jnp.sin(2 * jnp.pi * u3)
            q4 = jnp.sqrt(u1) * jnp.cos(2 * jnp.pi * u3)
            x, y, z, w = q1, q2, q3, q4
            return jnp.array([
                [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
                [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
                [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
            ]).flatten()
            
        self.base_rotations_flat = jax.vmap(get_rand_rot)(jax.random.split(k_rot, self.n_features))
        
        # 2. Sample 1D Matérn frequency scales for the Geodesic Angles
        df = 2.0 * self.nu
        s = jax.random.gamma(k_s, df / 2.0, shape=(self.n_features,)) / 0.5
        z_angles = jax.random.normal(k_z, (self.n_features,))
        self.omega_angles = z_angles / jnp.sqrt(s / df) / self.ell
        
        # 3. If velocity exists, sample standard Euclidean frequencies for it
        if self.has_velocity:
            k_v = jax.random.fold_in(k_z, 1)
            z_vels = jax.random.normal(k_v, (self.n_features, 3))
            self.omega_vels = z_vels / jnp.sqrt(s[:, None] / df) / self.ell
        else:
            self.omega_vels = jnp.zeros((self.n_features, 3))
            
        self.phases = jax.random.uniform(k_p, (self.n_features,)) * 2.0 * math.pi
        self.scale = jnp.array(math.sqrt(2.0 / self.n_features))

    def __call__(self, x):
        # x shape: (..., 9) or (..., 12)
        rot_part = x[..., :9]
        
        # 1. Dot product of flattened matrices is identical to Trace(W^T R)
        traces = rot_part @ self.base_rotations_flat.T
        
        # 2. Convert Trace to Geodesic Angle (distance on manifold).
        # Use the atan2 form θ = arctan2(√(1−c²), c) instead of arccos —
        # the older clip-to-(−1+1e-4, 1−1e-4) + arccos floored angle precision
        # at ~0.014 rad (~0.8°), the same fp32 floor that loss_utils_jax.py
        # explicitly fixed.
        #
        # Backward-mode safety: ∂√x/∂x = 1/(2√x) is singular at x=0 in
        # reverse-mode AD even when forward uses jnp.maximum(0, ...) — JAX's
        # adjoint propagates through the unselected branch of `maximum`. The
        # **double-where idiom** below (placeholder inside sqrt, true value
        # outside) makes both forward and backward finite at θ=0 with a true
        # zero gradient, mirroring the fix in loss_utils_jax.py.
        cos_theta = jnp.clip((traces - 1.0) / 2.0, -1.0, 1.0)
        sin_sq_raw = 1.0 - cos_theta * cos_theta
        nonzero = sin_sq_raw > 0.0
        sin_sq_safe = jnp.where(nonzero, sin_sq_raw, 1.0)
        sin_theta = jnp.where(nonzero, jnp.sqrt(sin_sq_safe), 0.0)
        angles = jnp.arctan2(sin_theta, cos_theta)
        
        # 3. Base feature projection from physical angles
        proj = angles * self.omega_angles
        
        # 4. Add velocity projection if present (Euclidean part of tangent bundle)
        if self.has_velocity:
            vel_part = x[..., 9:]
            proj_vel = vel_part @ self.omega_vels.T
            proj = proj + proj_vel
            
        return self.scale * jnp.cos(proj + self.phases)

class PeriodicFeatures(eqx.Module):
    c0:  jnp.ndarray
    ck:  jnp.ndarray
    m_k: jnp.ndarray
    period: float = eqx.field(static=True)
    ell:    float = eqx.field(static=True)
    m_max:  int   = eqx.field(static=True)

    def __init__(self, period: float = 2.0 * math.pi, ell: float = 1.0, m_max: int = 5):
        self.period = period
        self.ell = ell
        self.m_max = m_max
        
        m = jnp.arange(0, self.m_max + 1)
        a0 = 1.0
        a_m = jnp.exp(-2.0 * (math.pi**2) * (self.ell**2) * (m**2) / (self.period**2))
        
        # JAX arrays are immutable, use .at[].set for updates during creation logic
        a_m = a_m.at[0].set(a0)
        a_m = a_m / jnp.sum(a_m)
        
        self.c0 = jnp.sqrt(a_m[0:1])
        self.ck = jnp.sqrt(2.0 * a_m[1:])
        self.m_k = jnp.arange(1, self.m_max + 1, dtype=float)

    def __call__(self, x_scalar):
        # x_scalar: (..., )
        x = x_scalar[..., None]
        args = (2.0 * math.pi / self.period) * x * self.m_k
        cos_part = self.ck * jnp.cos(args)
        sin_part = self.ck * jnp.sin(args)
        harmonics = jnp.stack([cos_part, sin_part], axis=-1).reshape(*x.shape[:-1], -1)
        c0_part = jnp.broadcast_to(self.c0, x.shape)
        return jnp.concatenate([c0_part, harmonics], axis=-1)

class GP_Model(eqx.Module):
    matern: MaternFeatures
    periodic: PeriodicFeatures
    w_mean:      jnp.ndarray
    log_w_covar: jnp.ndarray
    Dm:           int   = eqx.field(static=True)
    Dp:           int   = eqx.field(static=True)
    output_dim:   int   = eqx.field(static=True)
    periodic_dim: int   = eqx.field(static=True)
    weight_shape: tuple = eqx.field(static=True)

    def __init__(self, key, input_dim: int, output_dim: int = 1, n_matern_features: int = 64, 
                 nu: float = 2.5, ell_m: float = 1.0, period: float = 2*math.pi, 
                 ell_p: float = 0.5, m_max: int = 5, periodic_dim: int = 0):
        
        k1, k2 = jax.random.split(key)
        self.periodic_dim = periodic_dim
        self.matern = MaternFeatures(k1, input_dim, n_matern_features, nu, ell_m)
        self.periodic = PeriodicFeatures(period, ell_p, m_max)

        self.Dm = int(n_matern_features)
        self.Dp = 1 + 2 * int(m_max)
        D_feat = self.Dm * self.Dp
        self.output_dim = output_dim
        self.weight_shape = (D_feat, output_dim)

        scale_factor = 1.0 / (self.Dm * self.Dp)

        self.w_mean = scale_factor * jax.random.normal(k2, self.weight_shape)
        # log σ_w init at −2.0 (σ_w ≈ 0.135) instead of the previous −4.0
        # (σ_w ≈ 0.018).  The tighter init produced a large, slow-decaying
        # KL floor relative to the prior N(0, 1) and biased early training
        # toward posterior collapse once β annealed in.
        self.log_w_covar = jnp.full(self.weight_shape, -2.0)

    def __call__(self, x, key: Optional[jax.random.PRNGKey] = None, inference_mode: bool = False):
        matern_f = self.matern(x)
        x_per = x[..., self.periodic_dim]
        per_f = self.periodic(x_per)

        if inference_mode:
            w = self.w_mean
        else:
            if key is None:
                raise ValueError("Key required for training mode GP sampling")
            w_std = jnp.exp(self.log_w_covar)
            epsilon = jax.random.normal(key, self.w_mean.shape)
            w = self.w_mean + w_std * epsilon

        w_reshaped = w.reshape(self.Dm, self.Dp, self.output_dim)
        return jnp.einsum('...m, ...p, mpo -> ...o', matern_f, per_f, w_reshaped)

    def weight_kl_loss(self):
        var = jnp.exp(2.0 * self.log_w_covar)
        mean_sq = jnp.square(self.w_mean)
        log_var = 2.0 * self.log_w_covar
        kl_div = 0.5 * jnp.sum(var + mean_sq - 1.0 - log_var)
        return kl_div

class GP_MatrixNet(eqx.Module):
    gp_model: GP_Model
    shape: tuple = eqx.field(static=True)

    def __init__(self, key, input_dim, hidden_dim, output_dim, shape=(2,2)):
        self.shape = shape
        self.gp_model = GP_Model(
            key,
            input_dim=input_dim, 
            output_dim=output_dim, 
            n_matern_features=hidden_dim,
            m_max=5
        )

    def __call__(self, x, key=None, inference_mode=False):
        flattened = self.gp_model(x, key=key, inference_mode=inference_mode)
        target_shape = flattened.shape[:-1] + self.shape
        return flattened.reshape(target_shape)

    def weight_kl_loss(self):
        return self.gp_model.weight_kl_loss()
    


class PSD_GP_Model(eqx.Module):
    gp_model: GP_Model
    diag_dim:     int   = eqx.field(static=True)
    off_diag_dim: int   = eqx.field(static=True)
    epsilon:      float = eqx.field(static=True)

    def __init__(
        self, 
        key, 
        input_dim, 
        hidden_dim, 
        diag_dim, 
        epsilon=0.0,
        # GP specific args passed through
        nu=2.5, ell_m=1.0, period=2*math.pi, m_max=5
    ):
        self.diag_dim = diag_dim
        self.epsilon = epsilon

        # 1. Determine output dimension based on PSD matrix size
        if diag_dim == 1:
            self.off_diag_dim = 0
            total_output_dim = 1
            
        else:
            self.off_diag_dim = int(diag_dim * (diag_dim - 1) / 2)
            total_output_dim = diag_dim + self.off_diag_dim

        # 2. Initialize the GP Model instead of MLP layers
        # hidden_dim is used as n_matern_features
        self.gp_model = GP_Model(
            key=key,
            input_dim=input_dim,
            output_dim=total_output_dim,
            n_matern_features=hidden_dim,
            nu=nu,
            ell_m=ell_m,
            period=period,
            m_max=m_max
        )

    def _construct_matrix(self, vector):
        """
        Helper function to reconstruct (N, N) matrix from a single vector.
        We separate this to easily vmap it over batches.
        """
        # 1. Split into diagonal and off-diagonal parts
        # vector shape: (total_output_dim,)
        diag_raw, off_diag = jnp.split(vector, [self.diag_dim], axis=-1)

        diag_raw = diag_raw + self.epsilon

        # # 2. Enforce Strict Positivity on Diagonal
        # # Softplus ensures > 0. Epsilon ensures >= 1e-6 (prevents singularity)
        # diag = jax.nn.softplus(diag_raw) 

        # # 3. Construct Lower Triangular Matrix L
        # L = jnp.zeros((self.diag_dim, self.diag_dim))
        
        # # Fill diagonal
        # diag_idx = jnp.diag_indices(self.diag_dim)
        # L = L.at[diag_idx].set(diag)
        
        # # Fill off-diagonal (strictly lower triangle)
        # if self.off_diag_dim > 0:
        #     rows, cols = jnp.tril_indices(self.diag_dim, k=-1)
        #     L = L.at[rows, cols].set(off_diag)

        # # 4. Compute PSD matrix D = L @ L.T
        # return jnp.matmul(L, L.T)
        
        max_L_val = 1.0 
        diag = max_L_val * jnp.tanh(jax.nn.softplus(diag_raw) / max_L_val) + self.epsilon
        diag = max_L_val * jnp.tanh(jax.nn.softplus(diag_raw) / max_L_val) + self.epsilon

        # diag = jax.nn.softplus(diag_raw) + self.epsilon


        L = jnp.zeros((self.diag_dim, self.diag_dim))
        diag_idx = jnp.diag_indices(self.diag_dim)
        L = L.at[diag_idx].set(diag)
        
        if self.off_diag_dim > 0:
            rows, cols = jnp.tril_indices(self.diag_dim, k=-1)
            # Optional: Tanh cap off-diagonals too if needed
            L = L.at[rows, cols].set(off_diag)

        return jnp.matmul(L, L.T)




    def __call__(self, x, key: Optional[jax.random.PRNGKey] = None, inference_mode: bool = False):
        # 1. Get raw predictions from GP (shape: [..., total_output_dim])
        h = self.gp_model(x, key=key, inference_mode=inference_mode)

        # 2. Case: 1D Output (Scalar PSD)
        if self.diag_dim == 1:
            # Squeeze to ensure scalar shape if needed, then square for positivity
            val = jnp.squeeze(h, axis=-1)
            return jax.nn.softplus(val) + self.epsilon

        # 3. Case: nD Matrix Output (Cholesky Decomposition)
        else:
            if h.ndim == 1:
                return self._construct_matrix(h)
            else:
                # Use vmap to apply matrix construction to every sample in the batch
                return jax.vmap(self._construct_matrix)(h)

    def weight_kl_loss(self):
        # Forward the KL loss from the internal GP model
        return self.gp_model.weight_kl_loss()