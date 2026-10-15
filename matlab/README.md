# nnsector for MATLAB

Sector bounds for feedforward neural networks, in MATLAB and GNU Octave.
The Python package with the same method is described in the
[main README](../README.md).

You give the network, a point `y*` and a box that contains it. You get two
lines (slopes) through the point `(y*, pi(y*))` that the network stays
between on the box. For how such local sector bounds are used to study
stability and the region of attraction of neural network feedback systems,
see the paper in the [main README](../README.md#citing).

Supported activations: `tanh`, `sigmoid`, `relu`, `leakyrelu` and `softplus`.

## Setup

```matlab
addpath('path/to/nnsector/matlab')     % the folder that contains +nnsector
```

Needs MATLAB R2016b or newer. It also runs in GNU Octave (tested with 8.4).

## Quick start

```matlab
W   = {W1, W2, W3};             % weights: W{i} is l_i x l_{i-1}
b   = {b1, b2, b3};             % biases (leave out for none)
net = nnsector.network(W, b);                          % tanh
net = nnsector.network(W, b, 'Activation', 'relu');    % or another activation

sb = nnsector.local_bound(net, y_lo, y_hi);                   % about y* = 0
sb = nnsector.local_bound(net, y_lo, y_hi, 'ystar', ystar);   % about your y*

sb.gamma1, sb.gamma2
```

A small demo network comes with the library. It was trained to act like the
controller `u = -tanh(2y) - 0.5y` (1 input, two hidden layers of 8 tanh
neurons, 1 output, with biases):

```matlab
net = nnsector.demo_network();
sb  = nnsector.local_bound(net, -1, 1);     % [-2.786, -1.200]
```

The exact secant slopes of this network on `[-1, 1]` lie in
`[-2.520, -1.463]`, so the bound is close.

The network is `omega_0 = y`, `omega_i = f(W_i*omega_{i-1} + b_i)` for each
hidden layer, and `pi(y) = W_{q+1}*omega_q + b_{q+1}`. All hidden layers use
the same activation `f`.

## Activations

| `'Activation'` | f(x) | secant slopes lie in | f(0) |
|---|---|---|---|
| `'tanh'` (default) | tanh(x) | [0, 1] | 0 |
| `'sigmoid'` | 1 / (1 + e^-x) | [0, 1/4] | 1/2 |
| `'relu'` | max(x, 0) | [0, 1] | 0 |
| `'leakyrelu'` | max(x, 0) + alpha*min(x, 0) | [alpha, 1] for alpha <= 1 | 0 |
| `'softplus'` | log(1 + e^x) | [0, 1] | log 2 |

For the leaky ReLU, set the slope for x < 0 with `'Alpha'` (default 0.01):

```matlab
net = nnsector.network(W, b, 'Activation', 'leakyrelu', 'Alpha', 0.1);
```

All five never decrease, which the local bound relies on. The local bound
is about `pi(y*)`, so activations with f(0) ~= 0 (sigmoid, softplus) work
the same way as the others.

## What you get

**One input (`p = 1`).** For every `y ~= y*` in the box (one row per output):

    gamma1 <= (pi(y) - pi(y*)) / (y - y*) <= gamma2

For `y >= y*` this reads `gamma1*(y - y*) <= pi(y) - pi(y*) <= gamma2*(y - y*)`.
For `y <= y*` the two lines swap places.

**`y*` at the lower corner of the box (`y* = y_lo`), any number of inputs.**
Componentwise:

    gamma1*(y - y*) <= pi(y) - pi(y*) <= gamma2*(y - y*)

**`y*` inside the box, several inputs.** No single pair of matrices works on
all sides of `y*`. So the box is cut at `y*` into pieces on which every
`y_k - y*_k` keeps one sign (at most `2^p` pieces). On piece `k`, with its
signs in `sb.pieces(k).signs`:

    sb.pieces(k).L * |y - y*|  <=  pi(y) - pi(y*)  <=  sb.pieces(k).U * |y - y*|

In this case `gamma1` and `gamma2` are empty. The pieces are always in
`sb.pieces`, for any number of inputs.

Other fields: `ystar`, `pistar` (= `pi(y*)`), `y_lo`, `y_hi`, `intervals`,
`activation`. Each piece also has `z_hi` (the largest `|y - y*|` on it) and
`case3` (the number of neurons per hidden layer in Case 3, the loosest case).

## Functions

| Function | What it does |
|---|---|
| `nnsector.network(W, b, ...)` | Builds and checks a network. Leave `b` out for zero biases. Options: `'Activation'` (default `'tanh'`) and `'Alpha'` (leaky ReLU). |
| `nnsector.local_bound(net, y_lo, y_hi, ...)` | Local sector bound about `y*` on the box. Options: `'ystar'` (default 0) and `'Intervals'` (see below). |
| `nnsector.global_bound(net)` | Global sector bound. |
| `nnsector.check(net, bound, ...)` | Tests a bound on many sample points. `res.passed` is true if no sample breaks it. |
| `nnsector.forward(net, X)` | Evaluates the network on each column of `X`. |
| `nnsector.demo_network()` | The demo network above. |

`y_lo`, `y_hi` and `ystar` can be scalars (used for every input) or vectors.
The box does not have to lie in the nonnegative orthant.

## How the local bound is computed

It is computed once on each piece of the box:

1. Run the network at `y*` to get the nominal preactivations `nu*`.
2. Layer by layer, bound the changes `nu - nu*` by linear functions of
   `|y - y*|` (step S1). Then replace the activation by two slopes that hold
   on each neuron's interval (step S2). There are three cases: the interval
   lies above `nu*` (Case 1), below it (Case 2), or contains it (Case 3).
3. The output layer gives `L` and `U`.

The biases cancel in the changes, so networks with biases are fine.

**Case 3** (a neuron's interval contains `nu*`) uses two slopes, one for
each side of `nu*`, and only the parts of the neuron's linear bounds `L`,
`U` from step S1 that can push its change below or above 0 (`min(L, 0)` and
`max(U, 0)`). This helps most for ReLU: a neuron that is off at `y*`
(`nu* <= 0`) can only go up, so its lower slope is 0.

**`'Intervals'`.** In step S2, each neuron's interval is by default the
overlap of its interval bound propagation (IBP) interval and the range of
its linear bounds over the box (`'intersect'`). `'ibp'` uses the IBP
interval alone. For the demo network this matters: `[-2.786, -1.200]` with
the overlap and `[-2.878, 0.251]` with IBP alone. Both kinds of interval are
computed for the changes `nu - nu*`, so a neuron whose interval ends exactly
at `nu*` is never pushed into Case 3 by rounding.

## Global bound

`gb = nnsector.global_bound(net)` gives `gb.Gamma2 = c^q |W_{q+1}| ... |W_1|`
and `gb.Gamma1 = -gb.Gamma2`. Here q is the number of hidden layers and c is
the largest secant slope of the activation: 1/4 for sigmoid, and 1 for the
others (for the leaky ReLU, `max(alpha, 1)`).

* With or without biases, for every `y` and `y*`:
  `-Gamma2*|y - y*| <= pi(y) - pi(y*) <= Gamma2*|y - y*|`.
  It holds because no secant slope of the activation is larger than c.
* With no biases and f(0) = 0 (tanh, ReLU, leaky ReLU), `pi(0) = 0`, so for
  every `y >= 0`: `Gamma1*y <= pi(y) <= Gamma2*y`.

## Examples and tests

* `examples/example1.m`: the demo network, plus a bound about `y* = 0.5`.
* `examples/example2.m`: random networks with biases, `y*` inside the box, two inputs.
* `examples/example3.m`: the weights of the demo network with each activation.
* `tests/run_tests.m`: from the `tests` folder, run `run_tests`.

## Adding an activation

Everything that depends on the activation is in `+nnsector/activation.m`.
To add one that never decreases, add a `case` there that sets `f`, `slope`
and `secant_range`: the smallest and largest secant slope
`(f(x) - f(a))/(x - a)` for x in an interval, with the one-sided derivative
at an end point equal to a. A non-monotone activation such as Swish would
also need changes in `interval` and in step S2 of `local_bound.m`.
