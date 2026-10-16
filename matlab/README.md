# nnsector for MATLAB

This folder contains the MATLAB implementation of nnsector, which computes
local and global sector bounds for feedforward neural networks. It is also
compatible with GNU Octave. The method, the notation and the guarantees are
the same as in the Python package and are described in the
[main README](../README.md).

## Setup

```matlab
addpath('path/to/nnsector/matlab')     % the folder that contains +nnsector
```

The library requires MATLAB R2016b or later. It also runs in GNU Octave
(tested with version 8.4).

## Quick start

```matlab
W   = {W1, W2, W3};                     % weights: W{i} is l_i x l_{i-1}
b   = {b1, b2, b3};                     % biases (omit b for a network without biases)
net = nnsector.network(W, b);                            % tanh (default)
net = nnsector.network(W, b, 'Activation', 'relu');      % another activation

sb = nnsector.local_bound(net, y_lo, y_hi);                   % operating point y* = 0
sb = nnsector.local_bound(net, y_lo, y_hi, 'ystar', ystar);   % another operating point
gb = nnsector.global_bound(net);                              % bound for all inputs

res = nnsector.check(net, sb);          % res.passed is true if no sample violates the bound
```

**Example.** The demonstration network has one input, two hidden layers of
eight tanh neurons, one output, and biases. It was trained to approximate
the controller `u = -tanh(2y) - 0.5y`.

```matlab
net = nnsector.demo_network();
sb  = nnsector.local_bound(net, -1, 1);     % sb.gamma1 = -2.786, sb.gamma2 = -1.200
```

On `[-1, 1]`, the secant slopes of this network about `y* = 0` lie in
`[-2.520, -1.463]`, so the bound is close.

## Functions

| Function | Description |
|---|---|
| `nnsector.network(W, b, ...)` | Defines and checks a network. Options: `'Activation'` (`'tanh'` (default), `'sigmoid'`, `'relu'`, `'leakyrelu'` or `'softplus'`) and `'Alpha'` (slope of the leaky ReLU for x < 0, default 0.01). |
| `nnsector.local_bound(net, y_lo, y_hi, ...)` | Local sector bound on the box `[y_lo, y_hi]`. Options: `'ystar'` (operating point, default 0) and `'Intervals'` (`'intersect'` (default) or `'ibp'`). |
| `nnsector.global_bound(net)` | Global sector bound. |
| `nnsector.check(net, bound, ...)` | Tests a bound on sample points. Options: `'Samples'` (default 20000), and `'Box'` and `'ystar'` for a global bound. A sampling test cannot prove a bound; it can only detect an incorrect one. |
| `nnsector.forward(net, X)` | Evaluates the network on each column of `X`. |
| `nnsector.demo_network()` | The demonstration network described above. |

The arguments `y_lo`, `y_hi` and `ystar` may be scalars (used for every
input) or vectors. The box need not lie in the nonnegative orthant.

## Results

**Local bound.** The structure `sb` contains:

* `gamma1`, `gamma2` (`m x p`): the slopes. For one input, every secant
  slope `(pi(y) - pi(y*))/(y - y*)` on the box lies between them (row by
  row). If `y*` is the lower corner of the box, then
  `gamma1*(y - y*) <= pi(y) - pi(y*) <= gamma2*(y - y*)` for every `y` in
  the box. If `y*` lies inside the box and there are several inputs, both
  fields are empty.
* `pieces`: the bound on each piece of the box on which every component of
  `y - y*` has a fixed sign (`signs`). On piece `k`,
  `pieces(k).L*|y - y*| <= pi(y) - pi(y*) <= pieces(k).U*|y - y*|`. Each
  piece also contains `z_hi` (the largest `|y - y*|` on the piece) and
  `case3` (the number of neurons per hidden layer in Case 3).
* `ystar`, `pistar` (the value `pi(y*)`), `y_lo`, `y_hi`, `intervals` and
  `activation`.

**Global bound.** The structure `gb` contains
`Gamma2 = c^q |W_{q+1}| ... |W_1|` and `Gamma1 = -Gamma2`, where `q` is the
number of hidden layers and `c` is the largest secant slope of the
activation (1/4 for sigmoid, `max(alpha, 1)` for the leaky ReLU, and 1 for
the others). With or without biases, for all `y` and `y*`,
`-Gamma2*|y - y*| <= pi(y) - pi(y*) <= Gamma2*|y - y*|`. If the network has
no biases and `f(0) = 0` (tanh, ReLU, leaky ReLU), then `pi(0) = 0`, and
`Gamma1*y <= pi(y) <= Gamma2*y` for all `y >= 0`.

The computation of the local bound, including the treatment of neurons in
Case 3 and the `'Intervals'` option, is described in the
[main README](../README.md#computation-of-the-local-bound).

## Examples and tests

| File | Content |
|---|---|
| `examples/example1.m` | The demonstration network, and a bound about `y* = 0.5`. |
| `examples/example2.m` | Random networks with biases, operating points inside the box, and two inputs. |
| `examples/example3.m` | The weights of the demonstration network with each activation function. |
| `tests/run_tests.m` | The test suite. Run `run_tests` from the `tests` folder. |

## Adding an activation function

All code that depends on the activation function is in
`+nnsector/activation.m`. A nondecreasing activation function is added as a
new `case` that defines `f`, `slope` and `secant_range`. The function
`secant_range` returns the smallest and the largest secant slope
`(f(x) - f(a))/(x - a)` for `x` in an interval, where the one-sided
derivative is used at `x = a`. A non-monotone activation function, such as
Swish, would also require changes in `interval` and in step S2 of
`local_bound.m`.

## Citation

See the [main README](../README.md#citation).
