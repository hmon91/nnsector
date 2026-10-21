"""Local and global sector bounds of feedforward networks.

The local bound follows the network layer by layer from the point y*: it
carries linear bounds on how much each preactivation can change, and replaces
each activation by two slopes that hold on the neuron's interval. The global
bound multiplies the absolute values of the weight matrices.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .network import Network


@dataclass
class Piece:
    """Bound on one piece of the box, where every ``y_k - y*_k`` has the sign ``signs[k]``.

    On the piece::

        L @ |y - y*|  <=  pi(y) - pi(y*)  <=  U @ |y - y*|.
    """

    signs: np.ndarray       #: (p,) entries +1 or -1
    z_hi: np.ndarray        #: (p,) largest |y - y*| on the piece
    L: np.ndarray           #: (m, p)
    U: np.ndarray           #: (m, p)
    case3: List[int]        #: neurons per hidden layer in Case 3 (the loosest case)


@dataclass
class LocalBound:
    """Result of :func:`local_bound`.

    * One input (``p = 1``): for every ``y != y*`` in the box, row by row,
      ``gamma1 <= (pi(y) - pi(y*)) / (y - y*) <= gamma2``.
    * ``y* = y_lo`` (y* at the lower corner of the box), any ``p``:
      ``gamma1 @ (y - y*) <= pi(y) - pi(y*) <= gamma2 @ (y - y*)``.
    * Otherwise ``gamma1`` and ``gamma2`` are ``None``; use ``pieces``.

    ``gamma1`` and ``gamma2`` have shape ``(m, p)``.
    """

    gamma1: Optional[np.ndarray]
    gamma2: Optional[np.ndarray]
    ystar: np.ndarray
    pistar: np.ndarray
    y_lo: np.ndarray
    y_hi: np.ndarray
    pieces: List[Piece]
    intervals: str
    activation: str

    def __str__(self) -> str:
        lines = [f"LocalBound ({self.activation}, {len(self.pieces)} piece(s), "
                 f"intervals='{self.intervals}')",
                 f"  y*    = {self.ystar}",
                 f"  box   = {self.y_lo} .. {self.y_hi}"]
        if self.gamma1 is not None:
            lines += [f"  gamma1 = {np.array2string(self.gamma1, precision=4)}",
                      f"  gamma2 = {np.array2string(self.gamma2, precision=4)}"]
        else:
            lines.append("  y* is inside the box in several inputs: see .pieces")
        return "\n".join(lines)


@dataclass
class GlobalBound:
    """Result of :func:`global_bound`.

    For every ``y`` and ``y*``:
    ``-Gamma2 @ |y - y*| <= pi(y) - pi(y*) <= Gamma2 @ |y - y*|``.
    """

    Gamma1: np.ndarray      #: (m, p), equal to -Gamma2
    Gamma2: np.ndarray      #: (m, p)
    c: float                #: largest secant slope of the activation
    activation: str

    def __str__(self) -> str:
        return (f"GlobalBound ({self.activation}, c = {self.c:g})\n"
                f"  Gamma2 = {np.array2string(self.Gamma2, precision=4)}\n"
                f"  Gamma1 = -Gamma2")


def local_bound(net: Network, y_lo, y_hi, ystar=None, intervals: str = "intersect") -> LocalBound:
    """Local sector bound of ``net`` about ``ystar`` on the box ``[y_lo, y_hi]``.

    Bounds the change ``pi(y) - pi(y*)`` for ``y_lo <= y <= y_hi``; the box
    must contain ``y*`` (default ``y* = 0``). The box is cut at ``y*`` into
    pieces on which every ``y_k - y*_k`` keeps one sign, and the bound is
    computed on each piece. See :class:`LocalBound` for what the result means.

    Parameters
    ----------
    net : Network
    y_lo, y_hi : float or array of shape (p,)
        Corners of the box (a scalar is used for every input).
    ystar : float or array of shape (p,), optional
        The point the sector is taken about (default 0).
    intervals : {'intersect', 'ibp'}
        How each preactivation interval is found: the intersection of its
        interval bound propagation (IBP) interval with the range of its linear
        bounds over the box (default), or the IBP interval alone.
    """
    p = net.p
    y_lo = _vector(y_lo, p, "y_lo")
    y_hi = _vector(y_hi, p, "y_hi")
    ystar = _vector(0.0 if ystar is None else ystar, p, "ystar")
    if np.any(y_lo > y_hi):
        raise ValueError("The box needs y_lo <= y_hi.")
    if np.any((ystar < y_lo) | (ystar > y_hi)):
        raise ValueError("ystar must lie in the box [y_lo, y_hi] (the default ystar is 0).")
    if intervals not in ("intersect", "ibp"):
        raise ValueError("intervals must be 'intersect' or 'ibp'.")

    # signs of the pieces: +1 if the box reaches above y*_k, -1 if below,
    # and +1 alone if the box is the single point y*_k in that coordinate
    options = []
    for k in range(p):
        sk = []
        if y_hi[k] > ystar[k] or y_lo[k] == y_hi[k]:
            sk.append(1.0)
        if y_lo[k] < ystar[k]:
            sk.append(-1.0)
        options.append(sk)
    count = int(np.prod([len(sk) for sk in options]))
    if count > 1024:
        warnings.warn(f"ystar cuts the box into {count} pieces, which can be slow.",
                      stacklevel=2)

    pieces = []
    for signs in itertools.product(*options):
        s = np.array(signs)
        z_hi = np.where(s > 0, y_hi - ystar, ystar - y_lo)
        W = list(net.weights)
        b = list(net.biases)
        b[0] = W[0] @ ystar + b[0]            # the same network, seen from z,
        W[0] = W[0] * s[None, :]              # where y = y* + diag(s) z
        L, U, case3 = _incremental_bound(W, b, net._act, z_hi, intervals == "intersect")
        pieces.append(Piece(s, z_hi, L, U, case3))

    gamma1 = gamma2 = None
    if p == 1:                                # slopes of the secants through y*
        lo = [P.L[:, 0] if P.signs[0] > 0 else -P.U[:, 0] for P in pieces]
        hi = [P.U[:, 0] if P.signs[0] > 0 else -P.L[:, 0] for P in pieces]
        gamma1 = np.min(lo, axis=0)[:, None]
        gamma2 = np.max(hi, axis=0)[:, None]
    elif len(pieces) == 1 and np.all(pieces[0].signs > 0):
        gamma1, gamma2 = pieces[0].L, pieces[0].U

    return LocalBound(gamma1, gamma2, ystar, net(ystar), y_lo, y_hi, pieces,
                      intervals, net.activation)


def global_bound(net: Network) -> GlobalBound:
    """Global sector bound of ``net``.

    ``Gamma2 = c^q |W_{q+1}| ... |W_1|`` and ``Gamma1 = -Gamma2``, where q is
    the number of hidden layers and c the largest secant slope of the
    activation (1/4 for sigmoid, ``max(alpha, 1)`` for the leaky ReLU, 1 for
    the others). With or without biases, for every ``y`` and ``y*``::

        -Gamma2 @ |y - y*| <= pi(y) - pi(y*) <= Gamma2 @ |y - y*|,

    since no layer can grow a change by more than ``c |W_i|``. With no
    biases and ``f(0) = 0`` (tanh, ReLU, leaky ReLU), ``pi(0) = 0``, and for
    every ``y >= 0`` this gives ``Gamma1 @ y <= pi(y) <= Gamma2 @ y``.

    The product is accumulated from the input side, so the running product
    keeps ``l_0`` columns: cost ``O(l_0 sum_i l_{i+1} l_i)``.
    """
    c = max(abs(net._act.slope[0]), abs(net._act.slope[1]))
    P = np.abs(net.weights[0])
    for Wi in net.weights[1:]:
        P = np.abs(Wi) @ P
    G2 = c ** net.depth * P
    return GlobalBound(-G2, G2, float(c), net.activation)


# ---- the layer-by-layer recursion -------------------------------------------

def _incremental_bound(W, b, act, z_hi, intersect):
    """The local bound on the box 0 <= z <= z_hi, about z* = 0.

    Returns L, U with  L z <= pi(z) - pi(0) <= U z,  and the number of Case 3
    neurons per hidden layer. Step S1 carries linear bounds on the changes
    nu~ = nu - nu* of the preactivations through the weights. Step S2 bounds
    f(nu) - f(nu*) on each neuron's interval by two slopes, in three cases:
    the interval lies above nu* (Case 1), below it (Case 2), or contains it
    (Case 3). The biases cancel in nu~ and enter only through the nominal
    preactivations nu*.

    The concrete intervals (interval bound propagation) are carried on the
    changes as well: d_lo <= nu~ <= d_hi. This is the same interval as
    nu* + [d_lo, d_hi], but every term of d_lo is <= 0 and every term of d_hi
    is >= 0. So when an interval ends at nu* (Cases 1 and 2), d_lo or d_hi is
    exactly 0, and rounding cannot move the neuron to Case 3.
    """
    p = W[0].shape[1]
    nL = len(W)
    z_lo = np.zeros(p)

    nu_s, w_s = [], [z_lo]                     # nominal forward pass at z = 0
    for i in range(nL - 1):
        nu_s.append(W[i] @ w_s[i] + b[i])
        w_s.append(act.f(nu_s[i]))

    l = np.eye(p)                              # the input changes by z
    u = np.eye(p)
    dw_lo, dw_hi = z_lo, z_hi                  # concrete interval on that change
    case3_count = []
    for i in range(nL):
        Wp = np.maximum(W[i], 0.0)
        Wm = np.minimum(W[i], 0.0)
        Lpre = Wp @ l + Wm @ u                 # S1: Lpre z <= nu~ <= Upre z
        Upre = Wp @ u + Wm @ l
        if i == nL - 1:                        # linear output layer: nu~ = pi(z) - pi(0)
            return Lpre, Upre, case3_count

        # concrete interval d_lo <= nu~ <= d_hi (interval bound propagation)
        d_lo, d_hi = _box_range(W[i], dw_lo, dw_hi)
        if intersect:                          # nu~ also lies in [Lpre z, Upre z]
            r_lo, _ = _box_range(Lpre, z_lo, z_hi)
            _, r_hi = _box_range(Upre, z_lo, z_hi)
            d_lo = np.maximum(d_lo, r_lo)
            d_hi = np.minimum(d_hi, r_hi)

        # S2: lower and upper slopes dL, dU of each neuron
        a = nu_s[i]
        nu_lo, nu_hi = a + d_lo, a + d_hi
        s_inf, s_sup = act.secant_range(a, nu_lo, nu_hi)
        case3 = (d_lo < 0) & (d_hi > 0)        # straddles nu*
        case2 = (d_lo < 0) & (d_hi <= 0)       # below nu*
        dL = np.array(s_inf, dtype=float)      # Case 1: nu >= nu*
        dU = np.array(s_sup, dtype=float)
        dL[case2] = s_sup[case2]               # Case 2: nu <= nu*
        dU[case2] = s_inf[case2]
        l = dL[:, None] * Lpre                 # bounds on the change of the layer output
        u = dU[:, None] * Upre
        if np.any(case3):
            # Case 3: nu* is inside the interval. f never decreases, so
            # f(nu) - f(nu*) lies between k_neg min(nu~, 0) and
            # k_pos max(nu~, 0), with k_neg and k_pos the largest secant slopes
            # left and right of nu*. And z >= 0 gives
            # min(nu~, 0) >= min(Lpre, 0) z and max(nu~, 0) <= max(Upre, 0) z.
            _, k_neg = act.secant_range(a[case3], nu_lo[case3], a[case3])
            _, k_pos = act.secant_range(a[case3], a[case3], nu_hi[case3])
            l[case3] = k_neg[:, None] * np.minimum(Lpre[case3], 0.0)
            u[case3] = k_pos[:, None] * np.maximum(Upre[case3], 0.0)

        w_lo, w_hi = act.interval(nu_lo, nu_hi)
        dw_lo = np.minimum(w_lo - w_s[i + 1], 0.0)   # the interval always holds 0
        dw_hi = np.maximum(w_hi - w_s[i + 1], 0.0)
        case3_count.append(int(np.count_nonzero(case3)))
    raise AssertionError("unreachable")       # pragma: no cover


def _box_range(M, x_lo, x_hi):
    """Componentwise range of M @ x for x in the box [x_lo, x_hi]."""
    Mp = np.maximum(M, 0.0)
    Mm = np.minimum(M, 0.0)
    return Mp @ x_lo + Mm @ x_hi, Mp @ x_hi + Mm @ x_lo


def _vector(v, p, name):
    """v as an array of shape (p,); a scalar is repeated."""
    v = np.asarray(v, dtype=float)
    if v.size == 1:
        v = np.full(p, float(v.reshape(-1)[0]))
    if v.size != p or not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be a scalar or have {p} finite entries.")
    return v.reshape(p)
