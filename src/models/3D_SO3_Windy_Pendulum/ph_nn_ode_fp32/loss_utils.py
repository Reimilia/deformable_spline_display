"""fp32-stable losses.

Mirrors src/utils/ode_utils.py but applies a strict-interior clamp before
torch.acos so the backward pass through arccos cannot produce NaN at cos = +/- 1
(where the analytic gradient -1/sqrt(1-cos^2) is infinite).
"""
import os, sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../utils')))
from ode_utils import (
    L2_loss,
    compute_rotation_matrix_from_unnormalized_rotmat,
)


# Strict-interior clamp epsilon for arccos.
# At cos = 1 - eps the gradient |d/dcos arccos| = 1/sqrt(2*eps - eps^2),
# so eps=1e-6 gives a max gradient of ~707, large but finite.
ACOS_EPS = 1e-6


def compute_geodesic_distance_from_two_matrices_safe(m1, m2):
    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))
    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2

    upper = (1.0 - ACOS_EPS) * torch.ones(batch, device=cos.device, dtype=cos.dtype)
    lower = (-1.0 + ACOS_EPS) * torch.ones(batch, device=cos.device, dtype=cos.dtype)
    cos = torch.min(cos, upper)
    cos = torch.max(cos, lower)

    theta = torch.acos(cos)
    return theta


def compute_geodesic_loss_safe(gt_r_matrix, out_r_matrix):
    theta = compute_geodesic_distance_from_two_matrices_safe(gt_r_matrix, out_r_matrix)
    theta = theta ** 2
    error = theta.mean()
    return error, theta


def rotmat_L2_geodesic_loss_safe(u, u_hat, split):
    # L2 is on q_dot (= ω) only — the control channel is identical between
    # target and prediction. Convention matches `ph_gp_sde` exactly:
    #   MSE_ω = mean over N of ‖Δω‖²    (sum over 3 components, mean over N)
    # so that the printed `test_l2_loss` here equals the GP_SDE trainer's
    # `MSE_ω` printed value on the same data.
    q_hat, q_dot_hat, _u_hat = torch.split(u_hat, split, dim=2)
    q,     q_dot,     _u     = torch.split(u,     split, dim=2)

    diff_omega = q_dot - q_dot_hat                                       # (T, B, 3)
    l2_loss = torch.mean(torch.sum(diff_omega * diff_omega, dim=-1))     # scalar

    q_hat = q_hat.flatten(start_dim=0, end_dim=1)
    q     = q.flatten(start_dim=0, end_dim=1)
    R_hat = compute_rotation_matrix_from_unnormalized_rotmat(q_hat)
    R     = compute_rotation_matrix_from_unnormalized_rotmat(q)
    geo_loss, _ = compute_geodesic_loss_safe(R, R_hat)
    return l2_loss + geo_loss, l2_loss, geo_loss


def _rotmat_L2_geodesic_diff_safe(u, u_hat, split):
    # Same change as above: drop the `u` channel from the L2 residual.
    q_hat, q_dot_hat, _u_hat = torch.split(u_hat, split, dim=1)
    q,     q_dot,     _u     = torch.split(u,     split, dim=1)
    l2_diff = torch.sum((q_dot - q_dot_hat) ** 2, dim=1)
    R_hat = compute_rotation_matrix_from_unnormalized_rotmat(q_hat)
    R     = compute_rotation_matrix_from_unnormalized_rotmat(q)
    _, geo_diff = compute_geodesic_loss_safe(R, R_hat)
    return l2_diff + geo_diff, l2_diff, geo_diff


def traj_rotmat_L2_geodesic_loss_safe(traj, traj_hat, split):
    total_loss = l2_loss = geo_loss = None
    for t in range(traj.shape[0]):
        u = traj[t, :, :]
        u_hat = traj_hat[t, :, :]
        if total_loss is None:
            total_loss, l2_loss, geo_loss = _rotmat_L2_geodesic_diff_safe(u, u_hat, split=split)
            total_loss = total_loss.unsqueeze(0)
            l2_loss = l2_loss.unsqueeze(0)
            geo_loss = geo_loss.unsqueeze(0)
        else:
            t_total, t_l2, t_geo = _rotmat_L2_geodesic_diff_safe(u, u_hat, split=split)
            total_loss = torch.cat((total_loss, t_total.unsqueeze(0)), dim=0)
            l2_loss = torch.cat((l2_loss, t_l2.unsqueeze(0)), dim=0)
            geo_loss = torch.cat((geo_loss, t_geo.unsqueeze(0)), dim=0)
    return total_loss, l2_loss, geo_loss
