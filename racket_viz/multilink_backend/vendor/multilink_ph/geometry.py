from __future__ import annotations
import math
import torch


def wrap_angle(theta: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(theta), torch.cos(theta))


def _sinc(theta: torch.Tensor) -> torch.Tensor:
    # torch.sinc(x) = sin(pi x)/(pi x)
    return torch.sinc(theta / math.pi)


def _cosc(theta: torch.Tensor) -> torch.Tensor:
    # (1-cos(theta))/theta with stable series near zero.
    small = theta.abs() < 1e-6
    series = theta / 2.0 - theta**3 / 24.0 + theta**5 / 720.0
    safe = (1.0 - torch.cos(theta)) / torch.where(small, torch.ones_like(theta), theta)
    return torch.where(small, series, safe)


def se2_exp_increment(xi: torch.Tensor, dt: float | torch.Tensor = 1.0) -> torch.Tensor:
    """Exponential coordinates to pose increment [dx_body, dy_body, dtheta].

    xi has (...,3) ordering [v_x, v_y, omega].
    """
    dt_t = torch.as_tensor(dt, dtype=xi.dtype, device=xi.device)
    th = xi[..., 2] * dt_t
    vdt = xi[..., :2] * dt_t
    A = _sinc(th)
    B = _cosc(th)
    dx = A * vdt[..., 0] - B * vdt[..., 1]
    dy = B * vdt[..., 0] + A * vdt[..., 1]
    return torch.stack([dx, dy, th], dim=-1)


def se2_compose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compose planar poses a*b represented as [x,y,theta]."""
    ca, sa = torch.cos(a[..., 2]), torch.sin(a[..., 2])
    bx, by = b[..., 0], b[..., 1]
    x = a[..., 0] + ca * bx - sa * by
    y = a[..., 1] + sa * bx + ca * by
    th = wrap_angle(a[..., 2] + b[..., 2])
    return torch.stack([x, y, th], dim=-1)


def se2_inverse(a: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(a[..., 2]), torch.sin(a[..., 2])
    x = -c * a[..., 0] - s * a[..., 1]
    y = s * a[..., 0] - c * a[..., 1]
    return torch.stack([x, y, wrap_angle(-a[..., 2])], dim=-1)


def se2_relative(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return a^{-1} b."""
    return se2_compose(se2_inverse(a), b)


def se2_log(pose: torch.Tensor) -> torch.Tensor:
    """Logarithm of SE(2) pose as integrated twist coordinates [vxdt,vydt,theta]."""
    th = wrap_angle(pose[..., 2])
    A = _sinc(th)
    B = _cosc(th)
    den = A * A + B * B
    x, y = pose[..., 0], pose[..., 1]
    vx = (A * x + B * y) / den.clamp_min(1e-12)
    vy = (-B * x + A * y) / den.clamp_min(1e-12)
    return torch.stack([vx, vy, th], dim=-1)


def se2_endpoint_error(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Lie-algebra residual Log(target^{-1} current)."""
    return se2_log(se2_relative(target, current))


def integrate_body_twists(xi: torch.Tensor, dt: float, pose0: torch.Tensor | None = None) -> torch.Tensor:
    """Integrate body twists xi (...,N,3) by right multiplication.

    Returns (...,N+1,3).
    """
    batch_shape = xi.shape[:-2]
    if pose0 is None:
        pose = torch.zeros(*batch_shape, 3, dtype=xi.dtype, device=xi.device)
    else:
        pose = pose0
    poses = [pose]
    for k in range(xi.shape[-2]):
        inc = se2_exp_increment(xi[..., k, :], dt)
        pose = se2_compose(pose, inc)
        poses.append(pose)
    return torch.stack(poses, dim=-2)


def se2_coadjoint_term(xi: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """ad^*_xi mu for xi=[v_x,v_y,w], mu=[p_x,p_y,l].

    Convention matches dot(mu)=ad^*_xi mu + ... in the left-trivialized
    equations used by the accompanying note.  The term is power-orthogonal:
    xi dot ad^*_xi mu = 0 when xi and mu are dual through the rigid block.
    """
    vx, vy, w = xi.unbind(-1)
    px, py, _ = mu.unbind(-1)
    # J p = [-py, px],  p dot J v = -px*vy + py*vx
    out_x = -w * py
    out_y = w * px
    out_l = -px * vy + py * vx
    return torch.stack([out_x, out_y, out_l], dim=-1)
