"""The demo network that ships with the package."""

import numpy as np
import pytest

import nnsector as ns


@pytest.fixture(scope="module")
def net():
    return ns.demo_network()


def secant_slopes(net, ystar, lo, hi, n=200001):
    """Range of (pi(y) - pi(y*)) / (y - y*) on a fine grid of [lo, hi]."""
    y = np.linspace(lo, hi, n)
    y = y[np.abs(y - ystar) > 1e-6]
    r = (net(y[:, None])[:, 0] - net(ystar)[0]) / (y - ystar)
    return r.min(), r.max()


def test_demo_network_acts_like_its_controller(net):
    assert net.widths == [1, 8, 8, 1] and net.activation == "tanh"
    y = np.linspace(-2, 2, 2001)
    assert np.abs(net(y[:, None])[:, 0] - (-np.tanh(2 * y) - 0.5 * y)).max() < 0.01


def test_local_bound(net):
    sb = ns.local_bound(net, -1, 1)
    assert len(sb.pieces) == 2
    assert sb.gamma1.item() == pytest.approx(-2.78625, abs=1e-5)
    assert sb.gamma2.item() == pytest.approx(-1.20015, abs=1e-5)
    assert all(P.case3 == [0, 0] for P in sb.pieces)


def test_bound_contains_the_exact_slopes(net):
    sb = ns.local_bound(net, -1, 1)
    lo, hi = secant_slopes(net, 0.0, -1, 1)
    assert sb.gamma1.item() <= lo and hi <= sb.gamma2.item()
    assert lo == pytest.approx(-2.51995, abs=1e-5)
    assert hi == pytest.approx(-1.46305, abs=1e-5)


def test_plain_ibp_intervals(net):
    sb = ns.local_bound(net, -1, 1, intervals="ibp")
    assert sb.gamma1.item() == pytest.approx(-2.87818, abs=1e-5)
    assert sb.gamma2.item() == pytest.approx(0.25138, abs=1e-5)
    assert all(P.case3 == [0, 8] for P in sb.pieces)


def test_global_bound(net):
    gb = ns.global_bound(net)
    W1, W2, W3 = net.weights
    assert np.allclose(gb.Gamma2, np.abs(W3) @ np.abs(W2) @ np.abs(W1))
    assert gb.Gamma2.item() == pytest.approx(4.65881, abs=1e-5)
    assert np.array_equal(gb.Gamma1, -gb.Gamma2)
    assert ns.check(net, gb, box=(-20, 20)).passed


def test_ystar_inside_the_box(net):
    sb = ns.local_bound(net, -1, 1, ystar=0.5)
    assert len(sb.pieces) == 2
    lo, hi = secant_slopes(net, 0.5, -1, 1)
    assert sb.gamma1.item() <= lo and hi <= sb.gamma2.item()
    assert sb.gamma1.item() == pytest.approx(-2.37991, abs=1e-5)
    assert sb.gamma2.item() == pytest.approx(-0.75116, abs=1e-5)
    assert ns.check(net, sb).passed


def test_ystar_at_the_ends_of_the_box(net):
    sb = ns.local_bound(net, -1, 1, ystar=1)
    assert len(sb.pieces) == 1 and sb.pieces[0].signs[0] == -1
    assert ns.check(net, sb).passed
    sb = ns.local_bound(net, -1, 1, ystar=-1)
    assert len(sb.pieces) == 1 and sb.pieces[0].signs[0] == 1
    assert ns.check(net, sb).passed
