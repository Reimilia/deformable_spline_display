from __future__ import annotations

from dataclasses import dataclass
import torch

from .chain import PlanarMultiLinkChain


@dataclass
class InformationGeometryConfig:
    """Configuration- and momentum-space information geometry parameters.

    ``fim_eps`` and ``sf_eps`` are numerical SPD regularizers.  The FIM is a
    pullback of body-frame link-center/angle measurements to joint coordinates.
    The Souriau--Fisher object used here is an *empirical local covariance proxy*
    of the planar momentum map [Lz, Px, Py] over a small tangent-space ensemble.
    It is intended as a numerical conditioning/diagnostic metric, rather than a
    claim that a finite deterministic ensemble equals the exact Souriau Gibbs
    family.
    """

    fim_eps: float = 1e-4
    fim_angle_weight: float = 0.05
    fim_normalize_trace: bool = True
    sf_eps: float = 5e-4
    sf_q_perturb: float = 2e-2
    sf_v_perturb: float = 4e-2
    sf_normalize_trace: bool = True


def configuration_fim(
    chain: PlanarMultiLinkChain,
    q: torch.Tensor,
    cfg: InformationGeometryConfig | None = None,
) -> torch.Tensor:
    """Pull back body-frame kinematic measurements to joint space.

    The observation is the concatenation of all link-center coordinates and a
    weak link-orientation channel.  If ``J_c`` is the stacked center Jacobian and
    ``H`` maps joint rates to link angular rates, then

        I_F = J_c^T J_c / M + w_beta H^T H / M + eps I.

    This is invariant to a common world SE(2) transform because the measurement
    is expressed in the chain body frame.
    """

    cfg = cfg or InformationGeometryConfig()
    kin = chain.kinematics(q)
    J = kin["Jshape"]  # (..., M, 2, n_joints)
    H = kin["h"]       # (..., M, n_joints)
    fim = torch.einsum("...mai,...maj->...ij", J, J) / float(chain.n_links)
    fim = fim + cfg.fim_angle_weight * torch.einsum("...mi,...mj->...ij", H, H) / float(chain.n_links)
    eye = torch.eye(chain.n_joints, dtype=q.dtype, device=q.device)
    fim = fim + cfg.fim_eps * eye
    if cfg.fim_normalize_trace:
        tr = torch.diagonal(fim, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
        fim = fim * (chain.n_joints / tr.clamp_min(cfg.fim_eps)).unsqueeze(-1)
    return fim


def souriau_momentum_map(pi: torch.Tensor) -> torch.Tensor:
    """Planar momentum map in [Lz, Px, Py] ordering from generalized momentum."""
    if pi.shape[-1] < 3:
        raise ValueError("generalized momentum must contain the 3 planar base components")
    return torch.stack([pi[..., 2], pi[..., 0], pi[..., 1]], dim=-1)


def generalized_momentum(chain: PlanarMultiLinkChain, q: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    M = chain.assemble(q)["M"]
    return (M @ nu.unsqueeze(-1)).squeeze(-1)


def empirical_souriau_fisher_metric(
    chain: PlanarMultiLinkChain,
    q: torch.Tensor,
    nu: torch.Tensor,
    cfg: InformationGeometryConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Empirical local covariance of the planar momentum map.

    A deterministic +/- tangent ensemble perturbs the internal joint position
    and velocity coordinates.  The returned tuple is ``(metric, mean, samples)``.
    This function expects one state (1-D q and nu) to keep the diagnostic clear.
    """

    cfg = cfg or InformationGeometryConfig()
    if q.ndim != 1 or nu.ndim != 1:
        raise ValueError("empirical_souriau_fisher_metric currently expects single q and nu vectors")
    samples = []
    nj = chain.n_joints
    for j in range(nj):
        for sign in (-1.0, 1.0):
            q_s = q.clone()
            nu_s = nu.clone()
            q_s[j] = q_s[j] + sign * cfg.sf_q_perturb
            nu_s[3 + j] = nu_s[3 + j] + sign * cfg.sf_v_perturb
            pi_s = generalized_momentum(chain, q_s, nu_s)
            samples.append(souriau_momentum_map(pi_s))
    # Also retain the unperturbed state so the covariance is anchored locally.
    pi0 = generalized_momentum(chain, q, nu)
    samples.append(souriau_momentum_map(pi0))
    S = torch.stack(samples, dim=0)
    # Long settling-horizon ablations can intentionally expose unstable models.
    # Keep the local SF *metric diagnostic* numerically defined even when a
    # failing rollout produces very large momentum.  Trace normalization makes
    # the absolute sample scale irrelevant, so rescale before forming covariance.
    S = torch.nan_to_num(S, nan=0.0, posinf=1e12, neginf=-1e12)
    mean = S.mean(dim=0)
    centered = S - mean
    scale = centered.detach().abs().max().clamp_min(1.0)
    centered_scaled = centered / scale
    denom = max(S.shape[0] - 1, 1)
    cov = centered_scaled.transpose(0, 1) @ centered_scaled / float(denom)
    eye3 = torch.eye(3, dtype=q.dtype, device=q.device)
    cov = cov + cfg.sf_eps * eye3
    if cfg.sf_normalize_trace:
        cov = cov * (3.0 / torch.trace(cov).clamp_min(cfg.sf_eps))
    # Re-regularize *after* trace normalization.  Extreme high-energy states can
    # otherwise scale the pre-normalization epsilon below machine resolution,
    # making the empirical metric numerically singular during long-horizon tests.
    cov = 0.5 * (cov + cov.transpose(0, 1)) + cfg.sf_eps * eye3
    return cov, mean, S


def trajectory_information_metrics(
    chain: PlanarMultiLinkChain,
    q: torch.Tensor,
    nu: torch.Tensor,
    cfg: InformationGeometryConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Compute FIM/SF metric sequences and eigenspectra for a single rollout."""

    cfg = cfg or InformationGeometryConfig()
    if q.ndim != 2 or nu.ndim != 2:
        raise ValueError("q and nu must have shapes [T,nj] and [T,nv]")
    T = min(q.shape[0], nu.shape[0])
    fim_list = []
    sf_list = []
    J_list = []
    for k in range(T):
        F = configuration_fim(chain, q[k], cfg)
        SF, _, _ = empirical_souriau_fisher_metric(chain, q[k], nu[k], cfg)
        pi = generalized_momentum(chain, q[k], nu[k])
        fim_list.append(F)
        sf_list.append(SF)
        J_list.append(souriau_momentum_map(pi))
    fim = torch.stack(fim_list)
    sf = torch.stack(sf_list)
    mom = torch.stack(J_list)
    return {
        "fim": fim,
        "fim_eigs": torch.linalg.eigvalsh(fim),
        "sf": sf,
        "sf_eigs": torch.linalg.eigvalsh(sf),
        "momentum_map": mom,
    }
