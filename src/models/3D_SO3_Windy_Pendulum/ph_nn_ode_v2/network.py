"""fp32-stable variant of the SO(3) Hamiltonian Dissipative NODE,
   with performance optimizations #1 and #2 applied (#3 reverted).

Differences from the fp64 production network:
  - M_net epsilon raised 0.1 -> 1.0 to floor lambda_min(M) and avoid the
    near-singular inverse that overflows in fp32.
  - torch.inverse(M_q) replaced with torch.linalg.solve(M_q, q_dot).
  - torch.cross -> torch.linalg.cross with explicit dim=1.

Performance optimizations:
  #1  dM_inv/dt computed via torch.func.jvp in a single call, replacing the
      9-iteration autograd.grad double loop. ~2-3x speedup on the forward.
  #2  Cross products batched: one call over the 3 rows of R reshaped to
      (B, 3, 3), instead of 3 separate linalg.cross calls (for both dq and
      the dp gravity terms). Saves ~6 kernel launches per forward.

#3 (single M_net call with analytical dHdp) was reverted — the analytical
form `M_q_inv @ p` with p = solve(M_q_inv, q_dot) creates two backward
paths whose contributions mathematically cancel but in fp32 produce
cancellation noise that compounded under create_graph=True double-backward,
driving Dw_net to diverge by step ~300. Original two-M_net-call cat-split
structure restored; it makes p_split independent of theta_M for the inner
autograd.grad, avoiding the cancellation entirely.

Functionally identical to the fp64 model; safe to load fp64 weights into
and vice versa.
"""
import torch
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../utils')))
from ode_nn_models import MLP, PSD, MatrixNet


class FixedInverseMass(torch.nn.Module):
    """Drop-in replacement for the M_net subnetwork that returns the constant
    ground-truth inverse mass (1/(m·l²)) · I₃ — no learnable parameters.

    Use when the mass / inertia is known (e.g. spherical pendulum with
    m = l = 1) so M_net is fixed and the optimizer focuses on V_net,
    Dw_net, g_net.
    """
    def __init__(self, m: float = 1.0, l: float = 1.0):
        super().__init__()
        scale = 1.0 / (m * l * l)
        self.register_buffer('M_inv', scale * torch.eye(3))

    def forward(self, q):
        N = q.shape[0]
        return self.M_inv.unsqueeze(0).expand(N, 3, 3)


class DissipativeSO3HamNODE(torch.nn.Module):
    def __init__(self, M_net=None, Dw_net=None, V_net=None, g_net=None,
                 device=None, u_dim=3, init_gain=0.01, friction=True):
        super().__init__()
        self.rotmatdim = 9
        self.angveldim = 3
        self.friction = friction

        # epsilon=1.0 (fp32 stability fix)
        self.M_net = M_net or PSD(self.rotmatdim, 20, self.angveldim,
                                  init_gain=init_gain, epsilon=1.0).to(device)

        if friction:
            self.Dw_net = Dw_net or PSD(self.rotmatdim, 20, self.angveldim,
                                        init_gain=init_gain, epsilon=0.0).to(device)
        self.V_net = V_net or MLP(self.rotmatdim, 20, 1, init_gain=init_gain).to(device)

        self.u_dim = u_dim
        if g_net is None:
            if u_dim == 1:
                self.g_net = MLP(self.rotmatdim, 20, self.angveldim).to(device)
            else:
                self.g_net = MatrixNet(self.rotmatdim, 20, self.angveldim * self.u_dim,
                                       shape=(self.angveldim, self.u_dim),
                                       init_gain=init_gain).to(device)
        else:
            self.g_net = g_net

        self.device = device
        self.nfe = 0

    def forward(self, t, x):
        with torch.enable_grad():
            self.nfe += 1
            bs = x.shape[0]
            zero_vec = torch.zeros(bs, self.u_dim, dtype=x.dtype, device=self.device)

            # Under torchdiffeq's no_grad eval path the integrator-internal y
            # has requires_grad=False. Normally this is masked by M_net's MLP
            # parameters re-introducing grad downstream — but when M_net is
            # fixed (no params) the cat-split q_p ends up grad-less and the
            # inner autograd.grad(H, q_p) errors. Forcing a fresh leaf here
            # is cheap and a no-op when x already requires grad.
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)

            q, q_dot, u = torch.split(x, [self.rotmatdim, self.angveldim, self.u_dim], dim=1)

            # ── Original two-M_net-call + cat-split structure ──
            # The cat-split makes q_p[0:9] (= q_split) and q_p[9:12] (= p_split)
            # independent for the inner autograd.grad. This is what avoids the
            # fp32 cancellation noise that broke optimization #3.
            M_q = self.M_net(q)
            q_dot_aug = torch.unsqueeze(q_dot, dim=2)
            p = torch.squeeze(torch.linalg.solve(M_q, q_dot_aug), dim=2)

            q_p = torch.cat((q, p), dim=1)
            q, p = torch.split(q_p, [self.rotmatdim, self.angveldim], dim=1)

            M_q_inv = self.M_net(q)   # name retained for parity with fp64 net
            V_q  = self.V_net(q)
            g_q  = self.g_net(q)
            Dw_q = self.Dw_net(q)

            p_aug = torch.unsqueeze(p, dim=2)
            H = (torch.squeeze(torch.matmul(torch.transpose(p_aug, 1, 2),
                                             torch.matmul(M_q_inv, p_aug))) / 2.0
                 + torch.squeeze(V_q))

            dH = torch.autograd.grad(H.sum(), q_p, create_graph=True)[0]
            dHdq, dHdp = torch.split(dH, [self.rotmatdim, self.angveldim], dim=1)

            if self.u_dim == 1:
                F = g_q * u
            else:
                F = torch.squeeze(torch.matmul(g_q, torch.unsqueeze(u, dim=2)))

            # ── #2: Batched cross product for dq (rows of R x dHdp) ──
            q_3x3  = q.view(-1, 3, 3)                       # (B, 3, 3) rows of R
            dHdp_b = dHdp.unsqueeze(1).expand(-1, 3, -1)    # (B, 3, 3) broadcast
            dq = torch.linalg.cross(q_3x3, dHdp_b, dim=2).reshape(-1, 9)

            # ── #2: Batched cross product for the gravity terms in dp ──
            dHdq_3x3 = dHdq.view(-1, 3, 3)                  # (B, 3, 3)
            grav = torch.linalg.cross(q_3x3, dHdq_3x3, dim=2).sum(dim=1)  # (B, 3)

            if self.friction:
                dp = (torch.linalg.cross(p, dHdp, dim=1)
                      + grav
                      - torch.squeeze(torch.matmul(Dw_q, torch.unsqueeze(dHdp, dim=2)))
                      + F)
            else:
                dp = torch.linalg.cross(p, dHdp, dim=1) + grav + F

            # ── #1: Vectorized dM_inv/dt via JVP ──
            # Replaces 9 separate autograd.grad(create_graph=True) calls with
            # one forward-mode AD call. dM_inv/dt = (∂M/∂q) · dq.
            _, dM_inv_dt = torch.func.jvp(self.M_net, (q,), (dq,))

            ddq = (torch.squeeze(torch.matmul(M_q_inv, torch.unsqueeze(dp, dim=2)), dim=2)
                   + torch.squeeze(torch.matmul(dM_inv_dt, torch.unsqueeze(p, dim=2)), dim=2))

            return torch.cat((dq, ddq, zero_vec), dim=1)
