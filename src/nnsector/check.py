"""Sampling test of a sector bound."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from .bounds import GlobalBound, LocalBound, _vector
from .network import Network


@dataclass
class CheckResult:
    """Result of :func:`check`."""

    max_violation: float    #: largest amount by which a sample breaks the bound (<= 0: none)
    passed: bool            #: True if max_violation <= tol
    tol: float              #: rounding tolerance
    samples: int            #: number of sample points


def check(net: Network, bound: Union[LocalBound, GlobalBound], samples: int = 20000,
          box=None, ystar=None, seed: Optional[int] = 0) -> CheckResult:
    """Test a sector bound on sample points.

    For a :class:`LocalBound`, the samples are random points of its box,
    points close to ``y*``, the box corners (``p <= 12``) and, for one input,
    a fine grid. For a :class:`GlobalBound`, give the box as
    ``box=(y_lo, y_hi)`` and optionally ``ystar`` (default 0).

    A sampling test cannot prove a bound; it can only catch a wrong one.
    """
    p = net.p
    if isinstance(bound, LocalBound):
        y_lo, y_hi, ys = bound.y_lo, bound.y_hi, bound.ystar
    elif isinstance(bound, GlobalBound):
        if box is None:
            raise ValueError("Give the box for a global bound: box=(y_lo, y_hi).")
        y_lo = _vector(box[0], p, "box[0]")
        y_hi = _vector(box[1], p, "box[1]")
        ys = _vector(0.0 if ystar is None else ystar, p, "ystar")
    else:
        raise TypeError("bound must come from local_bound or global_bound.")

    rng = np.random.default_rng(seed)
    w = y_hi - y_lo
    Y = [y_lo + w * rng.random((int(samples), p)),                   # random points
         ys + 1e-3 * w * (2 * rng.random((1000, p)) - 1)]            # close to y*
    if p <= 12:                                                      # box corners
        Y.append(y_lo + w * np.array(list(itertools.product((0.0, 1.0), repeat=p))))
    if p == 1:                                                       # fine grid
        Y.append(np.linspace(y_lo[0], y_hi[0], 100001)[:, None])
    Y = np.clip(np.vstack(Y), y_lo, y_hi)

    D = net(Y) - net(ys)                       # pi(y) - pi(y*), shape (N, m)
    Yt = Y - ys                                # y - y*, shape (N, p)
    errors = []
    if isinstance(bound, LocalBound):
        for P in bound.pieces:
            inside = np.all(P.signs * Yt >= 0, axis=1)
            Z = np.abs(Yt[inside])
            errors += [Z @ P.L.T - D[inside], D[inside] - Z @ P.U.T]
        if p == 1 and bound.gamma1 is not None:                      # slope form
            far = np.abs(Yt[:, 0]) > 1e-4 * max(w[0], np.finfo(float).eps)
            R = D[far] / Yt[far]
            errors += [bound.gamma1[:, 0] - R, R - bound.gamma2[:, 0]]
    else:
        Z = np.abs(Yt)
        errors += [-(Z @ bound.Gamma2.T) - D, D - Z @ bound.Gamma2.T]
    worst = max((float(E.max()) for E in errors if E.size), default=-np.inf)
    tol = 1e-9 * (1.0 + float(np.abs(D).max()))
    return CheckResult(worst, worst <= tol, tol, Y.shape[0])
