"""Per-subnetwork physics-target MSE diagnostics for the SO(3) Hamiltonian NODE.

Evaluates M_net, V_net, Dw_net, g_net on the model's own predicted q values
(from odeint output, excluding the ground-truth initial condition) and compares
to the true physical targets known from the env:

    M_true(q)  = m·l²·I₃                            (constant for spherical pendulum)
    V_true(q)  = m·g·l·(R·e_z)_z = m·g·l·q[8]       (centered: V is gauge-free up to const)
    Dw_true(q) = friction · I₃                      (or varying-friction formula)
    g_true(q)  = I₃                                 (body-frame torque)

Returns a dict of 4 scalar mean-MSE values per call. ~1 ms per call.
"""
import torch


def _unwrap(model):
    """Get the underlying nn.Module from a possibly torch.compile-wrapped model."""
    return model._orig_mod if hasattr(model, '_orig_mod') else model


@torch.no_grad()
def subnet_physics_mse(
    model,
    x_hat,
    *,
    m: float = 1.0,
    l: float = 1.0,
    g: float = 9.81,
    friction_coeff=0.5,
    varying_friction: bool = False,
):
    """Compute mean MSE between each subnetwork's outputs and the true physics.

    Args:
        model: DissipativeSO3HamNODE (or torch.compile wrapper).
        x_hat: odeint output, shape (T, B, 15) — initial frame is dropped.
        m, l, g: pendulum constants (env defaults: m=l=1, g=9.81).
        friction_coeff: scalar or len-3 vector (env spec).
        varying_friction: env flag.

    Returns:
        dict with scalar floats: {'M_loss', 'V_loss', 'Dw_loss', 'g_loss'}.
    """
    inner = _unwrap(model)
    device = x_hat.device
    dtype = x_hat.dtype

    # Use only predicted timesteps (drop the ground-truth initial condition)
    pred = x_hat[1:]                                # (T-1, B, 15)
    T1, B, _ = pred.shape
    flat = pred.reshape(T1 * B, 15)
    q = flat[:, :9]                                 # (N, 9)
    q_dot = flat[:, 9:12]                           # (N, 3)
    N = q.shape[0]

    # ── Subnet outputs ─────────────────────────────────────────────────
    M_pred  = inner.M_net(q)                        # (N, 3, 3)
    V_pred  = inner.V_net(q).squeeze(-1)            # (N,)
    # Dw_net takes (q, p); p = solve(M⁻¹(q), q_dot) using the model's M_net.
    # Fall back to (q,) for legacy single-arg variants.
    try:
        p_diag = torch.linalg.solve(M_pred, q_dot.unsqueeze(-1)).squeeze(-1)  # (N, 3)
        Dw_pred = inner.Dw_net(torch.cat((q, p_diag), dim=1))   # (N, 3, 3)
    except RuntimeError:
        Dw_pred = inner.Dw_net(q)                               # legacy
    g_pred  = inner.g_net(q)                        # (N, 3, 3)

    # ── Ground-truth targets ───────────────────────────────────────────
    I3 = torch.eye(3, device=device, dtype=dtype)
    M_tgt = (m * l * l) * I3.unsqueeze(0).expand(N, 3, 3)

    # V is gauge-free up to a constant — center both before MSE.
    # q[:, 8] is the (3,3) entry of R = (R·e_z)_z = bob z-component.
    V_tgt_raw = (m * g * l) * q[:, 8]
    V_pred_c = V_pred - V_pred.mean()
    V_tgt_c  = V_tgt_raw - V_tgt_raw.mean()

    # Dw target — diagonal with friction_coeff entries, optionally state-modulated
    fc = torch.as_tensor(friction_coeff, device=device, dtype=dtype)
    if fc.ndim == 0:
        fc = fc.expand(3)
    Dw_diag = torch.diag(fc).unsqueeze(0).expand(N, 3, 3)
    if varying_friction:
        height_term = 0.5 * (1.0 - q[:, 8])                     # (N,)
        speed_term  = torch.tanh(torch.linalg.norm(q_dot, dim=-1))
        mult = (1.0 + 0.5 * height_term + 0.5 * speed_term).view(N, 1, 1)
        Dw_tgt = Dw_diag * mult
    else:
        Dw_tgt = Dw_diag

    g_tgt = I3.unsqueeze(0).expand(N, 3, 3)

    # ── Mean MSE per subnet ────────────────────────────────────────────
    M_loss  = (M_pred  - M_tgt ).pow(2).mean().item()
    V_loss  = (V_pred_c - V_tgt_c).pow(2).mean().item()
    Dw_loss = (Dw_pred - Dw_tgt).pow(2).mean().item()
    g_loss  = (g_pred  - g_tgt ).pow(2).mean().item()

    return {'M_loss': M_loss, 'V_loss': V_loss,
            'Dw_loss': Dw_loss, 'g_loss': g_loss}
