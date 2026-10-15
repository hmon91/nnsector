# nnsector

[![tests](https://github.com/hmon91/nnsector/actions/workflows/tests.yml/badge.svg)](https://github.com/hmon91/nnsector/actions/workflows/tests.yml)

**Sector bounds for feedforward neural networks**, in Python and MATLAB.

You give a trained network, a point `y*` and a box around it. You get two
slopes `gamma1 <= gamma2`: on the box, the network stays between the two lines
through `(y*, pi(y*))` with these slopes. This is the form that
absolute-stability tests need (positive Aizerman, circle criterion, Lur'e
systems), so you can certify a feedback loop with a neural network
controller, observer or sensor.

For how local sector bounds are used to study the stability and the region
of attraction of neural network feedback systems, see our
[IEEE CDC 2025 paper](#citing).

## Install

```bash
pip install git+https://github.com/hmon91/nnsector.git
```

It needs Python 3.10 or newer and NumPy. To read PyTorch models, also
install PyTorch (`pip install torch`). To work on the code:

```bash
git clone https://github.com/hmon91/nnsector.git
cd nnsector
pip install -e ".[test]"
pytest
```

## Quick start

```python
import nnsector as ns

# weights W_i have shape (l_i, l_{i-1}), like torch.nn.Linear.weight
net = ns.Network([W1, W2, W3], [b1, b2, b3], activation="tanh")

sb = ns.local_bound(net, y_lo=-1, y_hi=1)           # about y* = 0
sb = ns.local_bound(net, -1, 1, ystar=0.5)           # about your own y*
print(sb.gamma1, sb.gamma2)

print(ns.check(net, sb))                             # test it on samples
```

A small demo network ships with the package. It was trained to act like the
controller `u = -tanh(2y) - 0.5y`:

```python
>>> net = ns.demo_network()        # 1 input, two hidden layers of 8 tanh neurons, 1 output
>>> sb = ns.local_bound(net, -1, 1)
>>> sb.gamma1.item(), sb.gamma2.item()
(-2.7862..., -1.2001...)
```

The exact secant slopes of this network on `[-1, 1]` lie in
`[-2.520, -1.463]`, so the bound is close.

### From PyTorch

```python
model = torch.nn.Sequential(
    torch.nn.Linear(1, 16), torch.nn.Tanh(),
    torch.nn.Linear(16, 16), torch.nn.Tanh(),
    torch.nn.Linear(16, 1),
)
# ... train the model ...
net = ns.Network.from_torch(model)
sb = ns.local_bound(net, -1, 1, ystar=0)
```

`from_torch` reads an `nn.Sequential` of `nn.Linear` layers and one kind of
activation (`nn.Tanh`, `nn.Sigmoid`, `nn.ReLU`, `nn.LeakyReLU`,
`nn.Softplus`). It opens nested `nn.Sequential` blocks, merges two
`nn.Linear` layers in a row, and skips `nn.Identity`, `nn.Dropout` and
`nn.Flatten`.

## What you get

`local_bound(net, y_lo, y_hi, ystar)` bounds the change `pi(y) - pi(y*)` for
`y_lo <= y <= y_hi`. The box must contain `y*` (default `y* = 0`).

* **One input (`p = 1`).** For every `y != y*` in the box, row by row:

      gamma1 <= (pi(y) - pi(y*)) / (y - y*) <= gamma2

  For `y >= y*` this reads `gamma1 (y - y*) <= pi(y) - pi(y*) <= gamma2 (y - y*)`.
  For `y <= y*` the two lines swap places.

* **`y*` at the lower corner (`y* = y_lo`), any number of inputs.**
  Componentwise:

      gamma1 @ (y - y*) <= pi(y) - pi(y*) <= gamma2 @ (y - y*)

* **`y*` inside the box, several inputs.** No single pair of matrices works
  on all sides of `y*`. So the box is cut at `y*` into pieces on which every
  `y_k - y*_k` keeps one sign (at most `2^p` pieces), and `sb.pieces[k]` holds
  the bound on piece `k`:

      P.L @ |y - y*|  <=  pi(y) - pi(y*)  <=  P.U @ |y - y*|

  In this case `gamma1` and `gamma2` are `None`.

`gamma1` and `gamma2` have shape `(m, p)`. The result also holds `ystar`,
`pistar` (= `pi(y*)`), `y_lo`, `y_hi` and, for each piece, `signs`, `z_hi`
(the largest `|y - y*|` on it) and `case3` (the number of neurons per hidden
layer in the loosest case of the relaxation).

`global_bound(net)` needs no box: for every `y` and `y*`,

    -Gamma2 @ |y - y*| <= pi(y) - pi(y*) <= Gamma2 @ |y - y*|,   Gamma2 = c^q |W_{q+1}| ... |W_1|

where q is the number of hidden layers and c the largest secant slope of the
activation. With no biases and `f(0) = 0` (tanh, ReLU, leaky ReLU),
`pi(0) = 0`, so for `y >= 0` this reads `Gamma1 @ y <= pi(y) <= Gamma2 @ y`,
with `Gamma1 = -Gamma2`.

## Activations

All hidden layers use the same activation.

| `activation` | f(x) | secant slopes lie in | f(0) |
|---|---|---|---|
| `"tanh"` (default) | tanh(x) | [0, 1] | 0 |
| `"sigmoid"` | 1 / (1 + e^-x) | [0, 1/4] | 1/2 |
| `"relu"` | max(x, 0) | [0, 1] | 0 |
| `"leakyrelu"` | max(x, 0) + alpha min(x, 0) | [alpha, 1] for alpha <= 1 | 0 |
| `"softplus"` | log(1 + e^x) | [0, 1] | log 2 |

For the leaky ReLU, set the slope for `x < 0` with `alpha` (default 0.01):
`ns.Network(W, b, activation="leakyrelu", alpha=0.1)`.

## Functions

| Function | What it does |
|---|---|
| `Network(weights, biases=None, activation="tanh", alpha=None)` | A network. `net(y)` evaluates it on `y` of shape `(p,)` or `(N, p)`. |
| `Network.from_torch(model)` | A network from a `torch.nn.Sequential`. |
| `local_bound(net, y_lo, y_hi, ystar=None, intervals="intersect")` | Local sector bound about `y*` on a box. |
| `global_bound(net)` | Global sector bound. |
| `check(net, bound, samples=20000, box=None, ystar=None)` | Tests a bound on sample points. A sampling test cannot prove a bound; it can only catch a wrong one. |
| `demo_network()` | The demo network above. |

## How the local bound is computed

It is computed on each piece of the box:

1. Run the network at `y*` to get the nominal preactivations `nu*`.
2. Layer by layer, bound the changes `nu - nu*` by linear functions of
   `|y - y*|` (step S1). Then replace the activation by two slopes that hold
   on each neuron's interval (step S2). There are three cases: the interval
   lies above `nu*` (Case 1), below it (Case 2), or contains it (Case 3).
3. The output layer gives `L` and `U`.

The biases cancel in the changes, so networks with biases are fine.

* **`intervals`.** In step S2, each neuron's interval is by default the
  overlap of its interval bound propagation (IBP) interval and the range of
  its linear bounds over the box (`"intersect"`). `"ibp"` uses the IBP
  interval alone. For the demo network this matters: `[-2.786, -1.200]` with
  the overlap and `[-2.878, 0.251]` with IBP alone. Both kinds of interval
  are computed for the changes `nu - nu*`, so a neuron whose interval ends
  exactly at `nu*` is never pushed into Case 3 by rounding.
* **Case 3** (a neuron's interval contains `nu*`) uses two slopes, one for
  each side of `nu*`, and only the parts of the neuron's linear bounds `L`,
  `U` from step S1 that can push its change below or above 0 (`min(L, 0)`
  and `max(U, 0)`). This helps most for ReLU: a neuron that is off at `y*`
  (`nu* <= 0`) can only go up, so its lower slope is 0.

## Examples and tests

* `examples/example1_demo.py`: the demo network, plus a bound about `y* = 0.5`.
* `examples/example2_biases.py`: random networks with biases, `y*` inside the box, two inputs.
* `examples/example3_activations.py`: the weights of the demo network with each activation.
* `examples/example4_pytorch.py`: train a small PyTorch controller, then bound it.
* `pytest` runs the tests: the demo network against its exact secant slopes,
  secant slopes against brute force for every activation, and random networks
  with biases.

## MATLAB

The same library for MATLAB (and GNU Octave) is in [`matlab/`](matlab/README.md),
with the same functions: `nnsector.network`, `nnsector.local_bound`,
`nnsector.global_bound`, `nnsector.check`, `nnsector.forward` and
`nnsector.demo_network`.

## Citing

If you use nnsector, please cite:

H. Montazeri Hedesh, M. K. Wafi, and M. Siami, "Local Stability and Region of
Attraction Analysis for Neural Network Feedback Systems under Positivity
Constraints," in *2025 IEEE 64th Conference on Decision and Control (CDC)*,
Rio de Janeiro, Brazil, 2025.
[IEEE Xplore](https://ieeexplore.ieee.org/document/11312171),
[arXiv:2505.22889](https://arxiv.org/abs/2505.22889)

```bibtex
@inproceedings{montazerihedesh2025local,
  author    = {Montazeri Hedesh, Hamidreza and Wafi, Moh Kamalul and Siami, Milad},
  title     = {Local Stability and Region of Attraction Analysis for Neural Network
               Feedback Systems under Positivity Constraints},
  booktitle = {2025 IEEE 64th Conference on Decision and Control (CDC)},
  year      = {2025},
  doi       = {10.1109/CDC57313.2025.11312171}
}
```

## License

BSD 3-Clause, see [LICENSE](LICENSE).
