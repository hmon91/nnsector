"""Secant slope ranges of the activations, and one-neuron networks."""

import numpy as np
import pytest

import nnsector as ns

ACTIVATIONS = [("tanh", None), ("sigmoid", None), ("relu", None), ("leakyrelu", 0.01),
               ("leakyrelu", 0.3), ("leakyrelu", 2.0), ("softplus", None)]


@pytest.mark.parametrize("name, alpha", ACTIVATIONS)
def test_secant_range_contains_and_is_tight(name, alpha):
    act = ns.get_activation(name, alpha)
    rng = np.random.default_rng(1)
    gap = 0.0
    for _ in range(400):
        a = 0.0 if rng.random() < 0.15 else 2 * rng.standard_normal()
        x = np.sort(a + 3 * rng.standard_normal(2))
        if rng.random() < 0.25:                      # an end point at a
            x[rng.integers(2)] = a
            x.sort()
        s_inf, s_sup = act.secant_range(np.array([a]), x[:1], x[1:])
        xs = np.linspace(x[0], x[1], 2001)
        xs = xs[np.abs(xs - a) > 1e-5 * (1 + abs(a))]
        if xs.size == 0:
            continue
        s = (act.f(xs) - act.f(a)) / (xs - a)
        assert s_inf[0] <= s.min() + 1e-9 and s.max() <= s_sup[0] + 1e-9
        gap = max(gap, s.min() - s_inf[0], s_sup[0] - s.max())
    assert gap < 1e-3


def softplus(x):
    return np.log1p(np.exp(x))


@pytest.mark.parametrize("name, alpha, box, sector", [
    ("relu", None, (-1, 2), (0.0, 1.0)),
    ("leakyrelu", 0.1, (-1, 2), (0.1, 1.0)),
    ("sigmoid", None, (-1, 1), (1 / (1 + np.exp(-1)) - 0.5, 0.25)),
    ("softplus", None, (-1, 1), (softplus(0) - softplus(-1), softplus(1) - softplus(0))),
])
def test_one_neuron_gives_the_exact_sector(name, alpha, box, sector):
    net = ns.Network([[[1.0]], [[1.0]]], activation=name, alpha=alpha)
    sb = ns.local_bound(net, *box)
    assert sb.gamma1.item() == pytest.approx(sector[0], abs=1e-12)
    assert sb.gamma2.item() == pytest.approx(sector[1], abs=1e-12)


def test_relu_case3_is_tight():
    # 0 <= relu(y1 - y2) <= y1 on [0, 1]^2: in Case 3 the slope of ReLU left
    # of nu* = 0 is 0, and max(U, 0) drops y2, which can only lower it
    net = ns.Network([[[1.0, -1.0]], [[1.0]]], activation="relu")
    sb = ns.local_bound(net, [0, 0], [1, 1])
    assert np.array_equal(sb.gamma1, [[0.0, 0.0]])
    assert np.array_equal(sb.gamma2, [[1.0, 0.0]])


def test_sigmoid_global_bound_uses_one_quarter():
    tanh_net = ns.demo_network()
    net = ns.Network(tanh_net.weights, activation="sigmoid")
    gb = ns.global_bound(net)
    assert gb.c == 0.25
    assert gb.Gamma2.item() == pytest.approx(0.25 ** 2 * ns.global_bound(tanh_net).Gamma2.item())


def test_unsupported_activation_and_bad_alpha():
    with pytest.raises(ValueError, match="not supported"):
        ns.Network([[[1.0]]], activation="swish")
    with pytest.raises(ValueError, match="alpha"):
        ns.Network([[[1.0]]], activation="leakyrelu", alpha=-0.1)
    with pytest.raises(ValueError, match="only by the 'leakyrelu'"):
        ns.Network([[[1.0]]], activation="relu", alpha=0.2)
