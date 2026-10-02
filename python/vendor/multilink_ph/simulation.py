from __future__ import annotations
import torch
from .geometry import se2_compose, se2_exp_increment
from .chain import PlanarMultiLinkChain


def modes_to_joints(c: torch.Tensor, Ksp: torch.Tensor) -> torch.Tensor:
    """c (...,T,modes) -> q (...,T,joints)."""
    return torch.einsum("jm,...tm->...tj", Ksp, c)


def geometric_rollout(
    chain: PlanarMultiLinkChain,
    Ksp: torch.Tensor,
    c: torch.Tensor,
    cdot: torch.Tensor,
    dt: float,
    pose0: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Dissipation-dominated horizontal reconstruction from a modal path."""
    q = modes_to_joints(c, Ksp)
    qdot = modes_to_joints(cdot, Ksp)
    batch = q.shape[:-2]
    if pose0 is None:
        pose = torch.zeros(*batch, 3, dtype=q.dtype, device=q.device)
    else:
        pose = pose0
    poses = [pose]
    xis = []
    powers = []
    # use left endpoint on each of N intervals
    for k in range(q.shape[-2] - 1):
        power, xi = chain.geometric_power(q[..., k, :], qdot[..., k, :])
        pose = se2_compose(pose, se2_exp_increment(xi, dt))
        poses.append(pose)
        xis.append(xi)
        powers.append(power)
    return {
        "c": c,
        "cdot": cdot,
        "q": q,
        "qdot": qdot,
        "pose": torch.stack(poses, dim=-2),
        "xi": torch.stack(xis, dim=-2),
        "power": torch.stack(powers, dim=-1),
        "energy": 0.5 * dt * torch.stack(powers, dim=-1).sum(dim=-1),
    }


def dynamic_outer_dissipation(chain: PlanarMultiLinkChain, q: torch.Tensor, nu: torch.Tensor, dt: float) -> torch.Tensor:
    vals = []
    for k in range(nu.shape[-2]):
        vals.append(chain.full_outer_power(q[..., k, :], nu[..., k, :]))
    return 0.5 * dt * torch.stack(vals, dim=-1).sum(dim=-1)
