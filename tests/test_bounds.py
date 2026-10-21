"""Bounds on random networks, special cases and input errors."""

import numpy as np
import pytest

import nnsector as ns

ACTIVATIONS = [("tanh", None), ("sigmoid", None), ("relu", None), ("leakyrelu", 0.01),
               ("leakyrelu", 0.3), ("leakyrelu", 2.0), ("softplus", None)]


def random_network(rng, name, alpha):
    p, m = rng.integers(1, 4), rng.integers(1, 3)
    widths = [p, *rng.integers(2, 11, rng.integers(1, 4)), m]
    W = [rng.standard_normal((widths[k + 1], widths[k])) for k in range(len(widths) - 1)]
    b = [rng.standard_normal(widths[k + 1]) for k in range(len(widths) - 1)]
    return ns.Network(W, b, activation=name, alpha=alpha)


@pytest.mark.parametrize("name, alpha", ACTIVATIONS)
def test_random_networks_with_biases(name, alpha):
    rng = np.random.default_rng(ACTIVATIONS.index((name, alpha)))
    for t in range(30):
        net = random_network(rng, name, alpha)
        lo = -2 + 2 * rng.random(net.p)
        hi = lo + 0.1 + 2 * rng.random(net.p)
        ys = lo if t % 5 == 0 else lo + (hi - lo) * rng.random(net.p)
        for intervals in ("intersect", "ibp"):
            sb = ns.local_bound(net, lo, hi, ystar=ys, intervals=intervals)
            assert ns.check(net, sb, samples=3000).passed
        gb = ns.global_bound(net)
        assert ns.check(net, gb, samples=3000, box=(lo, hi), ystar=ys).passed


def test_linear_network_is_exact():
    W = np.array([[1.0, -2.0], [3.0, 0.5]])
    net = ns.Network([W], [[1.0, -1.0]])
    sb = ns.local_bound(net, [-1, 0], [1, 2], ystar=[0, 1])
    assert len(sb.pieces) == 4
    for P in sb.pieces:
        assert np.array_equal(P.L, P.U) and np.array_equal(P.L, W * P.signs)


def test_one_point_box_gives_the_derivative():
    net = ns.demo_network()
    h = 1e-6
    d = (net(1 + h) - net(1 - h)).item() / (2 * h)
    sb = ns.local_bound(net, 1, 1, ystar=1)
    assert sb.gamma1.item() == pytest.approx(d, abs=1e-6)
    assert sb.gamma2.item() == pytest.approx(d, abs=1e-6)


@pytest.mark.parametrize("name, alpha", ACTIVATIONS)
def test_monotone_network_has_no_case3(name, alpha):
    # every neuron is monotone in y, so no interval straddles nu*, and
    # rounding must not put a neuron in Case 3
    rng = np.random.default_rng(5)
    for _ in range(20):
        s1 = rng.choice([-1.0, 1.0], 6)
        r3 = rng.choice([-1.0, 1.0], 4)
        W = [s1[:, None] * np.abs(rng.standard_normal((6, 1))),
             np.abs(rng.standard_normal((5, 6))) * s1,              # each row follows s1
             r3[:, None] * np.abs(rng.standard_normal((4, 5))),     # one sign per row
             rng.standard_normal((1, 4))]
        b = [rng.standard_normal(k) for k in (6, 5, 4, 1)]
        net = ns.Network(W, b, activation=name, alpha=alpha)
        for intervals in ("intersect", "ibp"):
            sb = ns.local_bound(net, -1, 1.5, ystar=0.3, intervals=intervals)
            assert all(c == 0 for P in sb.pieces for c in P.case3)


def test_two_inputs_with_ystar_inside_give_pieces():
    rng = np.random.default_rng(3)
    net = ns.Network([rng.standard_normal((8, 2)), rng.standard_normal((1, 8))],
                     [rng.standard_normal(8), [0.0]])
    sb = ns.local_bound(net, [0, 0], [2, 1], ystar=[1, 0.5])
    assert len(sb.pieces) == 4 and sb.gamma1 is None
    assert ns.check(net, sb).passed
    sb = ns.local_bound(net, [0, 0], [2, 1], ystar=[0, 0])
    assert len(sb.pieces) == 1 and sb.gamma1.shape == (1, 2)


def test_forward_shapes():
    net = ns.demo_network()
    assert net(0.5).shape == (1,)
    assert net(np.zeros((7, 1))).shape == (7, 1)
    with pytest.raises(ValueError):
        net(np.zeros((7, 2)))


def test_input_errors():
    net = ns.demo_network()
    with pytest.raises(ValueError, match="ystar must lie in the box"):
        ns.local_bound(net, 0, 2, ystar=3)
    with pytest.raises(ValueError, match="ystar must lie in the box"):
        ns.local_bound(net, 1, 2)                      # default y* = 0
    with pytest.raises(ValueError, match="y_lo <= y_hi"):
        ns.local_bound(net, 2, 0, ystar=1)
    with pytest.raises(ValueError, match="intervals"):
        ns.local_bound(net, 0, 2, intervals="crown")
    with pytest.raises(ValueError, match="columns"):
        ns.Network([np.ones((3, 2)), np.ones((1, 4))])
    with pytest.raises(ValueError, match="b_1"):
        ns.Network([np.ones((3, 2))], [np.ones(2)])
    with pytest.raises(ValueError, match="box"):
        ns.check(net, ns.global_bound(net))
