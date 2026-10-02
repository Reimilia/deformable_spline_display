"""JAX/Equinox port of src/utils/ode_nn_models.py.

MLP, PSD, MatrixNet — same architectures and orthogonal init as the PyTorch
versions used by ph_nn_ode_fp32. All modules operate on a *single* sample
(no batch dim); use jax.vmap to batch externally.
"""
from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np


def choose_nonlinearity(name: str) -> Callable:
    if name == 'tanh':
        return jnp.tanh
    if name == 'relu':
        return jax.nn.relu
    if name == 'sigmoid':
        return jax.nn.sigmoid
    if name == 'softplus':
        return jax.nn.softplus
    if name == 'selu':
        return jax.nn.selu
    if name == 'elu':
        return jax.nn.elu
    if name == 'swish':
        return jax.nn.swish
    raise ValueError(f"nonlinearity {name!r} not recognized")


def _orthogonal_linear(in_dim: int, out_dim: int, gain: float, *,
                       key, use_bias: bool = True) -> eqx.nn.Linear:
    """eqx.nn.Linear with weight reinitialised orthogonally (gain=`gain`).

    Mirrors `torch.nn.init.orthogonal_(l.weight, gain=init_gain)`.
    eqx.nn.Linear initialises weights uniformly by default; we overwrite.
    Bias keeps Equinox's default uniform init.
    """
    k_layer, k_init = jax.random.split(key)
    layer = eqx.nn.Linear(in_dim, out_dim, use_bias=use_bias, key=k_layer)
    init_fn = jax.nn.initializers.orthogonal(scale=gain)
    new_weight = init_fn(k_init, (out_dim, in_dim), dtype=layer.weight.dtype)
    return eqx.tree_at(lambda l: l.weight, layer, new_weight)


class MLP(eqx.Module):
    """3-layer MLP with tanh activations; single-sample input/output.

    Mirrors src/utils/ode_nn_models.py::MLP — same shapes, same orthogonal
    init with the supplied gain, no activation on the final layer.
    """
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    linear3: eqx.nn.Linear
    nonlinearity: Callable = eqx.field(static=True)

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 nonlinearity: str = 'tanh', bias_bool: bool = True,
                 init_gain: float = 1.0, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.linear1 = _orthogonal_linear(input_dim,  hidden_dim, init_gain, key=k1)
        self.linear2 = _orthogonal_linear(hidden_dim, hidden_dim, init_gain, key=k2)
        self.linear3 = _orthogonal_linear(hidden_dim, output_dim, init_gain,
                                          key=k3, use_bias=bias_bool)
        self.nonlinearity = choose_nonlinearity(nonlinearity)

    def __call__(self, x):
        h = self.nonlinearity(self.linear1(x))
        h = self.nonlinearity(self.linear2(h))
        return self.linear3(h)


class PSD(eqx.Module):
    """Positive semi-definite matrix L L^T + epsilon·I (via lower-triangular L).

    Mirrors src/utils/ode_nn_models.py::PSD with diag_dim > 1 (the only branch
    used by the SO(3) windy pendulum: M_net and Dw_net both use diag_dim=3).

    Input: q (single sample, shape (input_dim,))
    Output: D ∈ ℝ^(diag_dim × diag_dim), symmetric PSD.
    """
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    linear3: eqx.nn.Linear
    linear4: eqx.nn.Linear
    nonlinearity: Callable = eqx.field(static=True)
    diag_dim: int = eqx.field(static=True)
    off_diag_dim: int = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    _tril_rows: Tuple[int, ...] = eqx.field(static=True)
    _tril_cols: Tuple[int, ...] = eqx.field(static=True)

    def __init__(self, input_dim: int, hidden_dim: int, diag_dim: int,
                 nonlinearity: str = 'tanh', init_gain: float = 0.0,
                 epsilon: float = 0.0, *, key):
        assert diag_dim > 1, "PSD JAX port only implements the diag_dim>1 branch."
        self.diag_dim = diag_dim
        self.off_diag_dim = diag_dim * (diag_dim - 1) // 2
        self.epsilon = float(epsilon)

        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.linear1 = _orthogonal_linear(input_dim,  hidden_dim, init_gain, key=k1)
        self.linear2 = _orthogonal_linear(hidden_dim, hidden_dim, init_gain, key=k2)
        self.linear3 = _orthogonal_linear(hidden_dim, hidden_dim, init_gain, key=k3)
        self.linear4 = _orthogonal_linear(hidden_dim, self.diag_dim + self.off_diag_dim,
                                          init_gain, key=k4)

        self.nonlinearity = choose_nonlinearity(nonlinearity)

        rows, cols = np.tril_indices(self.diag_dim, k=-1)
        self._tril_rows = tuple(int(r) for r in rows)
        self._tril_cols = tuple(int(c) for c in cols)

    def __call__(self, q):
        h = self.nonlinearity(self.linear1(q))
        h = self.nonlinearity(self.linear2(h))
        h = self.nonlinearity(self.linear3(h))
        out = self.linear4(h)
        diag, off_diag = out[:self.diag_dim], out[self.diag_dim:]

        diag = diag + jnp.sqrt(self.epsilon)
        L = jnp.diag(diag)
        L = L.at[jnp.array(self._tril_rows), jnp.array(self._tril_cols)].set(off_diag)
        return L @ L.T


class MatrixNet(eqx.Module):
    """MLP whose flat output is reshaped to a matrix (single-sample)."""
    mlp: MLP
    shape: Tuple[int, int] = eqx.field(static=True)

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 nonlinearity: str = 'tanh', bias_bool: bool = True,
                 shape: Tuple[int, int] = (2, 2), init_gain: float = 1.0, *, key):
        assert output_dim == shape[0] * shape[1], (
            f"MatrixNet output_dim={output_dim} must equal prod(shape={shape})")
        self.mlp = MLP(input_dim, hidden_dim, output_dim,
                       nonlinearity=nonlinearity, bias_bool=bias_bool,
                       init_gain=init_gain, key=key)
        self.shape = tuple(shape)

    def __call__(self, x):
        flat = self.mlp(x)
        return flat.reshape(self.shape)
