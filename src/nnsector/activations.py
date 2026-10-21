"""Activation functions known to nnsector.

Every activation here never decreases, which the local bound relies on.
For each one the bounds need:

* ``f``: the activation, applied elementwise;
* ``slope``: an interval ``(smin, smax)`` that contains every secant slope
  ``(f(x) - f(a)) / (x - a)`` (used by the global bound);
* ``secant_range(a, x_lo, x_hi)``: the smallest and largest secant slope
  ``(f(x) - f(a)) / (x - a)`` for ``x != a`` in ``[x_lo, x_hi]``. At an end
  point equal to ``a`` the slope is the one-sided derivative from inside the
  interval (used by the local bound).

To add an activation that never decreases, write its ``secant_range`` and add
a branch to :func:`get_activation`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Tuple

import numpy as np

SUPPORTED_ACTIVATIONS = ("tanh", "sigmoid", "relu", "leakyrelu", "softplus")


@dataclass(frozen=True)
class Activation:
    """What the bounds need to know about an activation (see the module docstring)."""

    name: str
    f: Callable[[np.ndarray], np.ndarray]
    slope: Tuple[float, float]
    secant_range: Callable[..., Tuple[np.ndarray, np.ndarray]]
    alpha: Optional[float] = None

    def interval(self, lo, hi):
        """Range of ``f`` over ``[lo, hi]``, elementwise (``f`` never decreases)."""
        return self.f(lo), self.f(hi)


def get_activation(name: str, alpha: Optional[float] = None) -> Activation:
    """Return the :class:`Activation` called ``name``.

    ``name`` is one of ``'tanh'``, ``'sigmoid'``, ``'relu'``, ``'leakyrelu'``
    and ``'softplus'``. ``alpha`` is the slope of the leaky ReLU for ``x < 0``
    (default 0.01) and is ignored by the other activations.
    """
    key = str(name).lower()
    if key == "tanh":
        return Activation("tanh", np.tanh, (0.0, 1.0), _tanh_secant_range)
    if key == "sigmoid":
        return Activation("sigmoid", _sigmoid, (0.0, 0.25), _sigmoid_secant_range)
    if key == "relu":
        return Activation("relu", _relu, (0.0, 1.0), partial(_leaky_secant_range, alpha=0.0))
    if key == "leakyrelu":
        alpha = 0.01 if alpha is None else alpha
        if not (np.isscalar(alpha) and np.isreal(alpha) and np.isfinite(alpha) and alpha >= 0):
            raise ValueError("The leaky ReLU slope alpha must be a real number >= 0.")
        alpha = float(alpha)
        return Activation("leakyrelu", partial(_leaky_relu, alpha=alpha),
                          (min(alpha, 1.0), max(alpha, 1.0)),
                          partial(_leaky_secant_range, alpha=alpha), alpha)
    if key == "softplus":
        return Activation("softplus", _softplus, (0.0, 1.0), _softplus_secant_range)
    raise ValueError(f"Activation '{name}' is not supported. "
                     f"Supported: {', '.join(SUPPORTED_ACTIVATIONS)}.")


def _floats(*xs):
    return np.broadcast_arrays(*[np.asarray(x, dtype=float) for x in xs])


# ---- tanh -------------------------------------------------------------------

def _tanh_secant(x, a):
    """s = (tanh(x) - tanh(a)) / (x - a), and s = sech(a)^2 at x = a.

    Near x = a the identity tanh(x) - tanh(a) = sinh(x-a) / (cosh(x) cosh(a))
    avoids the cancellation in the difference quotient.
    """
    x, a = _floats(x, a)
    d = x - a
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        s = (np.tanh(x) - np.tanh(a)) / d
        sinhc = np.ones_like(d)                 # sinh(d)/d, equal to 1 at d = 0
        nz = d != 0
        sinhc[nz] = np.sinh(d[nz]) / d[nz]
        near = sinhc / (np.cosh(x) * np.cosh(a))
    return np.where(np.abs(d) < 1, near, s)


def _tanh_secant_peak(a):
    """Peak x_c of the secant slope of tanh about a.

    The tangent of tanh at x_c passes through (a, tanh(a)), i.e. x_c is the
    root of g(x) = sech(x)^2 (x - a) - (tanh(x) - tanh(a)). x_c = 0 for a = 0,
    x_c has the sign opposite to a, and x_c(-a) = -x_c(a). The root is found
    for -|a| on x >= 0, where g falls from |a| - tanh|a| >= 0 to
    -(1 + tanh|a|) < 0, by bisection on all neurons at once.
    """
    a = np.asarray(a, dtype=float)
    b = np.abs(a)

    def g(x):
        with np.errstate(over="ignore"):
            return (x + b) / np.cosh(x) ** 2 - np.tanh(x) - np.tanh(b)

    lo = np.zeros_like(b)
    hi = np.ones_like(b)
    while True:
        k = g(hi) > 0
        if not np.any(k):
            break
        hi = np.where(k, 2 * hi, hi)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        k = g(mid) > 0
        lo = np.where(k, mid, lo)
        hi = np.where(k, hi, mid)
    return -np.sign(a) * 0.5 * (lo + hi)


def _tanh_secant_range(a, x_lo, x_hi):
    """Secant slope range of tanh about a on [x_lo, x_hi].

    tanh is convex for x < 0 and concave for x > 0, so the secant slope rises
    up to a single peak x_c and falls after it. Hence the infimum is at an end
    point, and the supremum is at an end point or at x_c.
    """
    a, x_lo, x_hi = _floats(a, x_lo, x_hi)
    s_lo = _tanh_secant(x_lo, a)
    s_hi = _tanh_secant(x_hi, a)
    s_inf = np.minimum(s_lo, s_hi)
    s_sup = np.maximum(s_lo, s_hi)
    xc = _tanh_secant_peak(a)
    inside = (x_lo <= xc) & (xc <= x_hi)
    return s_inf, np.where(inside, _tanh_secant(xc, a), s_sup)


# ---- sigmoid ----------------------------------------------------------------

def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))


def _sigmoid_secant_range(a, x_lo, x_hi):
    """sigmoid(x) = (1 + tanh(x/2))/2, so its secant slope about a at x is a
    quarter of the secant slope of tanh about a/2 at x/2."""
    a, x_lo, x_hi = _floats(a, x_lo, x_hi)
    s_inf, s_sup = _tanh_secant_range(0.5 * a, 0.5 * x_lo, 0.5 * x_hi)
    return 0.25 * s_inf, 0.25 * s_sup


# ---- ReLU and leaky ReLU ----------------------------------------------------

def _relu(x):
    return np.maximum(np.asarray(x, dtype=float), 0.0)


def _leaky_relu(x, alpha):
    x = np.asarray(x, dtype=float)
    return np.maximum(x, 0.0) + alpha * np.minimum(x, 0.0)


def _leaky_secant(x, a, alpha):
    """Secant slope (f(x) - f(a)) / (x - a) of the leaky ReLU for x != a,
    written so that no difference of nearly equal numbers is formed."""
    x, a = _floats(x, a)
    s = np.ones(x.shape)                                    # x and a both >= 0
    s[(x <= 0) & (a <= 0)] = alpha                          # x and a both <= 0
    k = (a > 0) & (x < 0)
    s[k] = (a[k] - alpha * x[k]) / (a[k] - x[k])
    k = (a < 0) & (x > 0)
    s[k] = (x[k] - alpha * a[k]) / (x[k] - a[k])
    return s


def _leaky_secant_range(a, x_lo, x_hi, alpha):
    """f(x) = max(x,0) + alpha*min(x,0) is convex for alpha <= 1 and concave for
    alpha >= 1, so its secant slope about a only rises (or only falls) with x,
    and the extremes are at the end points. At an end point equal to a, the
    slope is the one-sided derivative from inside the interval: from the right
    at x_lo, from the left at x_hi. ReLU is alpha = 0."""
    a, x_lo, x_hi = _floats(a, x_lo, x_hi)
    s_lo = _leaky_secant(x_lo, a, alpha)
    s_hi = _leaky_secant(x_hi, a, alpha)
    k = x_lo == a
    s_lo[k] = alpha + (1.0 - alpha) * (a[k] >= 0)           # right derivative
    k = x_hi == a
    s_hi[k] = alpha + (1.0 - alpha) * (a[k] > 0)            # left derivative
    return np.minimum(s_lo, s_hi), np.maximum(s_lo, s_hi)


# ---- softplus ---------------------------------------------------------------

def _softplus(x):
    """log(1 + exp(x)) without overflow for large x."""
    x = np.asarray(x, dtype=float)
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def _softplus_secant(x, a):
    """s = (softplus(x) - softplus(a)) / (x - a), and s = sigmoid(a) at x = a.

    Near x = a, softplus(x) - softplus(a) = log1p(sigmoid(a) * expm1(x - a))
    avoids the cancellation in the difference quotient.
    """
    x, a = _floats(x, a)
    d = x - a
    sa = _sigmoid(a)                                        # softplus'(a)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        s = (_softplus(x) - _softplus(a)) / d
        near = np.log1p(sa * np.expm1(d)) / d
    near = np.where(d == 0, sa, near)
    return np.where(np.abs(d) < 1, near, s)


def _softplus_secant_range(a, x_lo, x_hi):
    """softplus is convex, so its secant slope about a rises with x, and the
    extremes are at the end points."""
    s_lo = _softplus_secant(x_lo, a)
    s_hi = _softplus_secant(x_hi, a)
    return np.minimum(s_lo, s_hi), np.maximum(s_lo, s_hi)
