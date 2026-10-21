"""A small network shipped with the package, for examples and tests."""

from __future__ import annotations

from importlib import resources

import numpy as np

from .network import Network


def demo_network() -> Network:
    """A small tanh network trained to act like the controller u = -tanh(2y) - 0.5y.

    One input, two hidden layers of 8 tanh neurons, one output, with biases.
    It was fitted on ``-2 <= y <= 2`` and stays within 0.007 of the
    controller there. Its local sector bound about ``y* = 0`` on ``[-1, 1]``
    is ``[-2.786, -1.200]``.
    """
    data = resources.files("nnsector") / "data"
    weights, biases = [], []
    for i in (1, 2, 3):
        with (data / f"demo_layer{i}.csv").open("r") as fh:
            M = np.loadtxt(fh, delimiter=",", ndmin=2)  # one row per neuron: weights, then bias
        weights.append(M[:, :-1])
        biases.append(M[:, -1])
    return Network(weights, biases, activation="tanh")
