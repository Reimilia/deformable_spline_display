from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class ChainParams:
    n_links: int = 6
    total_length: float = 1.0
    total_mass: float = 0.3
    c_parallel: float = 0.25
    c_perp: float = 1.0
    drag_scale: float = 4.0
    joint_damping: float = 0.08
    joint_stiffness: float = 3.0
    joint_limit: float = 1.25
    mass_regularization: float = 1e-5
    rigid_drag_regularization: float = 1e-5


class PlanarMultiLinkChain:
    """Differentiable planar M-link chain mechanics and anisotropic drag."""

    def __init__(
        self,
        params: ChainParams,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> None:
        self.params = params
        self.dtype = dtype
        self.device = torch.device(device)
        M = params.n_links
        self.n_links = M
        self.n_joints = M - 1
        self.lengths = torch.full((M,), params.total_length / M, dtype=dtype, device=self.device)
        self.masses = torch.full((M,), params.total_mass / M, dtype=dtype, device=self.device)
        # Slender uniform links about their centers.
        self.inertias = self.masses * self.lengths**2 / 12.0
        self.drag_weights = torch.full((M,), params.drag_scale / M, dtype=dtype, device=self.device)

    def to(self, device: str | torch.device | None = None, dtype: torch.dtype | None = None) -> "PlanarMultiLinkChain":
        """Move the chain's persistent tensors in-place and return ``self``.

        The mechanics methods are also robust to mixed-device inputs, but this
        helper is useful for long evaluation loops where avoiding repeated local
        ``Tensor.to`` views is preferable.
        """
        if device is None:
            device = self.device
        device = torch.device(device)
        if dtype is None:
            dtype = self.dtype
        self.device = device
        self.dtype = dtype
        self.lengths = self.lengths.to(device=device, dtype=dtype)
        self.masses = self.masses.to(device=device, dtype=dtype)
        self.inertias = self.inertias.to(device=device, dtype=dtype)
        self.drag_weights = self.drag_weights.to(device=device, dtype=dtype)
        return self

    @staticmethod
    def _Jvec(v: torch.Tensor) -> torch.Tensor:
        return torch.stack([-v[..., 1], v[..., 0]], dim=-1)

    def kinematics(self, q: torch.Tensor) -> dict[str, torch.Tensor]:
        """Body-frame geometry and shape Jacobians.

        q: (..., M-1)
        returns centers (...,M,2), tangents (...,M,2),
        Jshape (...,M,2,M-1), h (...,M,M-1), joint_positions (...,M+1,2).
        """
        if q.shape[-1] != self.n_joints:
            raise ValueError(f"expected q last dim {self.n_joints}, got {q.shape[-1]}")
        batch = q.shape[:-1]
        # Keep chain parameters usable from plotting/evaluation code even when
        # the chain object and the trajectory tensors were created on different
        # devices.  This commonly happens after CUDA training when diagnostics
        # reconstruct a CPU chain or load CPU metadata.  The local views below
        # do not mutate the chain and are no-ops when device/dtype already match.
        lengths = self.lengths.to(device=q.device, dtype=q.dtype)
        zero = torch.zeros(*batch, 1, dtype=q.dtype, device=q.device)
        beta = torch.cat([zero, torch.cumsum(q, dim=-1)], dim=-1)  # (...,M)
        tangents = torch.stack([torch.cos(beta), torch.sin(beta)], dim=-1)
        edge_vec = tangents * lengths.view(*([1] * len(batch)), self.n_links, 1)
        origin = torch.zeros(*batch, 1, 2, dtype=q.dtype, device=q.device)
        joint_positions = torch.cat([origin, torch.cumsum(edge_vec, dim=-2)], dim=-2)
        centers = joint_positions[..., :-1, :] + 0.5 * edge_vec

        Jshape = torch.zeros(*batch, self.n_links, 2, self.n_joints, dtype=q.dtype, device=q.device)
        h = torch.zeros(*batch, self.n_links, self.n_joints, dtype=q.dtype, device=q.device)
        Jt = self._Jvec(tangents)
        for i in range(self.n_links):
            for j in range(min(i, self.n_joints)):
                # q_j rotates links a >= j+1.  For center of link i, contributions
                # from full intermediate links plus half of link i.
                val = torch.zeros(*batch, 2, dtype=q.dtype, device=q.device)
                if j + 1 <= i - 1:
                    for a in range(j + 1, i):
                        val = val + lengths[a] * Jt[..., a, :]
                val = val + 0.5 * lengths[i] * Jt[..., i, :]
                Jshape[..., i, :, j] = val
                h[..., i, j] = 1.0

        return {
            "beta": beta,
            "tangents": tangents,
            "normals": self._Jvec(tangents),
            "joint_positions": joint_positions,
            "centers": centers,
            "Jshape": Jshape,
            "h": h,
        }

    def assemble(self, q: torch.Tensor, reduced: bool = False) -> dict[str, torch.Tensor]:
        kin = self.kinematics(q)
        centers = kin["centers"]
        tangents = kin["tangents"]
        normals = kin["normals"]
        Jshape = kin["Jshape"]
        h = kin["h"]
        batch = q.shape[:-1]
        # See ``kinematics``: dynamics may be queried from a trajectory living
        # on another device (especially by plotting/checkpoint replay code).
        masses = self.masses.to(device=q.device, dtype=q.dtype)
        inertias = self.inertias.to(device=q.device, dtype=q.dtype)
        drag_weights = self.drag_weights.to(device=q.device, dtype=q.dtype)
        n = 3 + self.n_joints
        Mmat = torch.zeros(*batch, n, n, dtype=q.dtype, device=q.device)
        A = torch.zeros(*batch, 3, 3, dtype=q.dtype, device=q.device)
        B = torch.zeros(*batch, 3, self.n_joints, dtype=q.dtype, device=q.device)
        C = torch.zeros(*batch, self.n_joints, self.n_joints, dtype=q.dtype, device=q.device)

        eye2 = torch.eye(2, dtype=q.dtype, device=q.device)
        for i in range(self.n_links):
            ci = centers[..., i, :]
            Jci = Jshape[..., i, :, :]
            Jc = self._Jvec(ci)
            G = torch.zeros(*batch, 2, 3, dtype=q.dtype, device=q.device)
            G[..., :2, :2] = eye2
            G[..., :, 2] = Jc
            H = torch.cat([G, Jci], dim=-1)  # (...,2,n)

            omega = torch.zeros(*batch, n, dtype=q.dtype, device=q.device)
            omega[..., 2] = 1.0
            omega[..., 3:] = h[..., i, :]

            Mmat = Mmat + masses[i] * (H.transpose(-1, -2) @ H)
            Mmat = Mmat + inertias[i] * torch.einsum("...i,...j->...ij", omega, omega)

            ti = tangents[..., i, :]
            ni = normals[..., i, :]
            Di = drag_weights[i] * (
                self.params.c_parallel * torch.einsum("...i,...j->...ij", ti, ti)
                + self.params.c_perp * torch.einsum("...i,...j->...ij", ni, ni)
            )
            A = A + G.transpose(-1, -2) @ Di @ G
            B = B + G.transpose(-1, -2) @ Di @ Jci
            C = C + Jci.transpose(-1, -2) @ Di @ Jci

        I_n = torch.eye(n, dtype=q.dtype, device=q.device)
        I3 = torch.eye(3, dtype=q.dtype, device=q.device)
        Mmat = Mmat + self.params.mass_regularization * I_n
        A = A + self.params.rigid_drag_regularization * I3
        Dj = self.params.joint_damping * torch.eye(self.n_joints, dtype=q.dtype, device=q.device)
        Dfull = torch.zeros_like(Mmat)
        Dfull[..., :3, :3] = A
        Dfull[..., :3, 3:] = B
        Dfull[..., 3:, :3] = B.transpose(-1, -2)
        Dfull[..., 3:, 3:] = C + Dj

        out = {**kin, "M": Mmat, "A": A, "B": B, "C_out": C, "D": Dfull}
        if reduced:
            connection = torch.linalg.solve(A, B)
            Rin = self.params.joint_damping * torch.eye(self.n_joints, dtype=q.dtype, device=q.device)
            Gshape = C + Rin - B.transpose(-1, -2) @ connection
            out["connection"] = connection
            out["G_shape"] = Gshape
        return out

    def local_connection_twist(self, q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        ops = self.assemble(q, reduced=False)
        rhs = -(ops["B"] @ qdot.unsqueeze(-1)).squeeze(-1)
        return torch.linalg.solve(ops["A"], rhs.unsqueeze(-1)).squeeze(-1)

    def reduced_shape_metric(self, q: torch.Tensor) -> torch.Tensor:
        return self.assemble(q, reduced=True)["G_shape"]

    def geometric_power(self, q: torch.Tensor, qdot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reduced dissipation power and reconstructed rigid twist."""
        ops = self.assemble(q, reduced=True)
        xi = -(ops["connection"] @ qdot.unsqueeze(-1)).squeeze(-1)
        power = torch.einsum("...i,...ij,...j->...", qdot, ops["G_shape"], qdot)
        return power, xi

    def full_outer_power(self, q: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        """nu=[xi,qdot], return nu^T D_environment nu (including joint damping)."""
        D = self.assemble(q)["D"]
        return torch.einsum("...i,...ij,...j->...", nu, D, nu)

    def spring_effort(self, q: torch.Tensor, q_ref: torch.Tensor, stiffness: torch.Tensor | None = None) -> torch.Tensor:
        if stiffness is None:
            return self.params.joint_stiffness * (q - q_ref)
        return stiffness * (q - q_ref)
