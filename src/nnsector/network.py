"""Feedforward neural networks for nnsector."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .activations import get_activation


class Network:
    """Feedforward network with the same activation in every hidden layer.

    The network is::

        omega_0 = y,   omega_i = f(W_i omega_{i-1} + b_i),   i = 1, ..., q,
        pi(y)   = W_{q+1} omega_q + b_{q+1},

    with ``W_i`` of shape ``(l_i, l_{i-1})``, the layout of
    ``torch.nn.Linear.weight``, and ``b_i`` of shape ``(l_i,)``.

    Parameters
    ----------
    weights : sequence of 2-D arrays
        ``[W_1, ..., W_{q+1}]``.
    biases : sequence of 1-D arrays, optional
        ``[b_1, ..., b_{q+1}]``; ``None`` (or a ``None`` entry) means zero.
    activation : str
        ``'tanh'`` (default), ``'sigmoid'``, ``'relu'``, ``'leakyrelu'`` or
        ``'softplus'``.
    alpha : float, optional
        Slope of the leaky ReLU for ``x < 0`` (default 0.01). Only used by
        ``'leakyrelu'``.
    """

    def __init__(self, weights: Sequence, biases: Optional[Sequence] = None,
                 activation: str = "tanh", alpha: Optional[float] = None):
        if isinstance(weights, np.ndarray) or len(weights) == 0:
            raise ValueError("weights must be a list [W_1, ..., W_{q+1}] of 2-D arrays.")
        W = []
        for i, Wi in enumerate(weights, start=1):
            Wi = np.array(Wi, dtype=float)
            if Wi.ndim != 2 or Wi.size == 0 or not np.all(np.isfinite(Wi)):
                raise ValueError(f"W_{i} must be a nonempty finite 2-D array.")
            if W and Wi.shape[1] != W[-1].shape[0]:
                raise ValueError(f"W_{i} has {Wi.shape[1]} columns, but W_{i-1} "
                                 f"has {W[-1].shape[0]} rows.")
            W.append(Wi)

        if biases is None:
            biases = [None] * len(W)
        if len(biases) != len(W):
            raise ValueError(f"biases must have {len(W)} entries, like weights.")
        b = []
        for i, (Wi, bi) in enumerate(zip(W, biases), start=1):
            bi = np.zeros(Wi.shape[0]) if bi is None else np.array(bi, dtype=float).ravel()
            if bi.size != Wi.shape[0] or not np.all(np.isfinite(bi)):
                raise ValueError(f"b_{i} must have {Wi.shape[0]} finite entries.")
            b.append(bi)

        activation = str(activation).lower()
        if activation == "leakyrelu":
            alpha = 0.01 if alpha is None else alpha
        elif alpha is not None:
            raise ValueError("alpha is used only by the 'leakyrelu' activation.")
        self._act = get_activation(activation, alpha)     # checks name and alpha
        self.weights = W
        self.biases = b
        self.activation = activation
        self.alpha = self._act.alpha

    # ---- sizes --------------------------------------------------------------

    @property
    def p(self) -> int:
        """Input size."""
        return self.weights[0].shape[1]

    @property
    def m(self) -> int:
        """Output size."""
        return self.weights[-1].shape[0]

    @property
    def depth(self) -> int:
        """Number of hidden layers q."""
        return len(self.weights) - 1

    @property
    def widths(self) -> list:
        """Layer sizes [l_0, l_1, ..., l_{q+1}]."""
        return [self.p] + [Wi.shape[0] for Wi in self.weights]

    # ---- evaluation ---------------------------------------------------------

    def forward(self, y) -> np.ndarray:
        """Evaluate the network.

        ``y`` has shape ``(p,)`` (one input) or ``(N, p)`` (one input per row);
        the output has shape ``(m,)`` or ``(N, m)``. For ``p = 1`` a scalar is
        also accepted.
        """
        Y = np.asarray(y, dtype=float)
        single = Y.ndim <= 1
        Y = Y.reshape(1, -1) if single else Y
        if Y.ndim != 2 or Y.shape[1] != self.p:
            raise ValueError(f"y must have shape ({self.p},) or (N, {self.p}).")
        Z = Y.T
        for Wi, bi in zip(self.weights[:-1], self.biases[:-1]):
            Z = self._act.f(Wi @ Z + bi[:, None])
        out = (self.weights[-1] @ Z + self.biases[-1][:, None]).T
        return out[0] if single else out

    __call__ = forward

    def __repr__(self) -> str:
        extra = f", alpha={self.alpha:g}" if self.alpha is not None else ""
        return f"Network(widths={self.widths}, activation='{self.activation}'{extra})"

    # ---- PyTorch ------------------------------------------------------------

    @classmethod
    def from_torch(cls, model) -> "Network":
        """Build a Network from a ``torch.nn.Sequential``.

        The model may hold ``nn.Linear`` layers and one kind of activation:
        ``nn.Tanh``, ``nn.Sigmoid``, ``nn.ReLU``, ``nn.LeakyReLU`` or
        ``nn.Softplus`` (with ``beta = 1``). It must start and end with
        ``nn.Linear``. Nested ``nn.Sequential`` blocks are opened, two
        ``nn.Linear`` layers in a row are multiplied into one, and
        ``nn.Identity``, ``nn.Dropout`` and ``nn.Flatten`` are skipped (they do
        nothing at evaluation on a vector input). Weights are copied in double
        precision.
        """
        try:
            from torch import nn
        except ImportError as err:                                 # pragma: no cover
            raise ImportError("Network.from_torch needs PyTorch: pip install torch") from err
        if not isinstance(model, nn.Sequential):
            raise TypeError("from_torch takes a torch.nn.Sequential of Linear and "
                            "activation layers.")

        def layers(seq):
            for layer in seq:
                if isinstance(layer, nn.Sequential):
                    yield from layers(layer)
                elif not isinstance(layer, (nn.Identity, nn.Dropout, nn.Flatten)):
                    yield layer

        weights, biases, acts = [], [], []
        W = b = None                                   # pending affine map
        for layer in layers(model):
            if isinstance(layer, nn.Linear):
                Wl = layer.weight.detach().cpu().double().numpy()
                bl = (np.zeros(Wl.shape[0]) if layer.bias is None
                      else layer.bias.detach().cpu().double().numpy())
                W, b = (Wl, bl) if W is None else (Wl @ W, Wl @ b + bl)
                continue
            if W is None:
                raise ValueError(f"{type(layer).__name__} must come after an nn.Linear layer.")
            acts.append(_torch_activation(layer, nn))
            weights.append(W)
            biases.append(b)
            W = b = None
        if W is None:
            raise ValueError("The model must end with an nn.Linear layer "
                             "(an output activation is not supported).")
        weights.append(W)
        biases.append(b)
        if len(set(acts)) > 1:
            raise ValueError("All hidden layers must use the same activation; "
                             f"this model has {sorted(set(a[0] for a in acts))}.")
        name, alpha = acts[0] if acts else ("tanh", None)
        return cls(weights, biases, activation=name, alpha=alpha)


def _torch_activation(layer, nn):
    """(name, alpha) of a torch activation layer."""
    if isinstance(layer, nn.Tanh):
        return ("tanh", None)
    if isinstance(layer, nn.Sigmoid):
        return ("sigmoid", None)
    if isinstance(layer, nn.ReLU):
        return ("relu", None)
    if isinstance(layer, nn.LeakyReLU):
        return ("leakyrelu", float(layer.negative_slope))
    if isinstance(layer, nn.Softplus):
        if layer.beta != 1:
            raise ValueError("nn.Softplus with beta != 1 is not supported yet.")
        return ("softplus", None)
    raise ValueError(f"Layer {type(layer).__name__} is not supported. Use nn.Linear "
                     "with nn.Tanh, nn.Sigmoid, nn.ReLU, nn.LeakyReLU or nn.Softplus.")
