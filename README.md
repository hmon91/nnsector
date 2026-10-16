# nnsector

[![tests](https://github.com/hmon91/nnsector/actions/workflows/tests.yml/badge.svg)](https://github.com/hmon91/nnsector/actions/workflows/tests.yml)

**nnsector** computes local and global sector bounds for feedforward neural
networks. It is implemented in Python and in MATLAB (compatible with GNU
Octave).

A sector bound encloses a nonlinear map between two linear maps. Sector
bounds are the standard description of a nonlinearity in absolute stability
analysis, for example in the circle criterion and in Aizerman-type
conditions for Lur'e systems. They therefore make it possible to certify
feedback systems that contain a trained neural network as a controller, an
observer, or a sensor model. The use of local sector bounds in the stability
and region of attraction analysis of neural network feedback systems is
described in [[1]](#citation). The companion library
[nnstab](https://github.com/hmon91/nnstab) uses nnsector to verify the
exponential stability of such systems.

## Problem statement

A feedforward network with $`q`$ hidden layers maps an input
$`y \in \mathbb{R}^p`$ to an output $`\pi(y) \in \mathbb{R}^m`$:

```math
\omega_0 = y, \qquad
\omega_i = f\left(W_i\,\omega_{i-1} + b_i\right), \quad i = 1, \dots, q, \qquad
\pi(y) = W_{q+1}\,\omega_q + b_{q+1},
```

where $`W_i \in \mathbb{R}^{l_i \times l_{i-1}}`$ are the weights,
$`b_i \in \mathbb{R}^{l_i}`$ are the biases, and the activation function
$`f`$ acts entrywise and is the same in all hidden layers.

Given an operating point $`y^*`$ and a box
$`\mathcal{Y} = [\underline{y}, \overline{y}]`$ that contains it, nnsector
computes two matrices $`\gamma_1, \gamma_2 \in \mathbb{R}^{m \times p}`$
such that, on $`\mathcal{Y}`$, the network lies between the two linear maps
through the point $`(y^*, \pi(y^*))`$ with slopes $`\gamma_1`$ and
$`\gamma_2`$. The precise statements are given in
[Guarantees](#guarantees). nnsector also computes a global bound that holds
for all inputs.

## Installation

```bash
pip install git+https://github.com/hmon91/nnsector.git
```

The package requires Python 3.10 or later and NumPy. PyTorch is optional and
is needed only to import PyTorch models (`pip install torch`). To install
the package for development and run the tests:

```bash
git clone https://github.com/hmon91/nnsector.git
cd nnsector
pip install -e ".[test]"
pytest
```

## Quick start

```python
import nnsector as ns

# W_i has shape (l_i, l_{i-1}), the layout of torch.nn.Linear.weight
net = ns.Network([W1, W2, W3], [b1, b2, b3], activation="tanh")

sb = ns.local_bound(net, y_lo=-1, y_hi=1)         # operating point y* = 0 (default)
sb = ns.local_bound(net, -1, 1, ystar=0.5)         # another operating point
print(sb.gamma1, sb.gamma2)

gb = ns.global_bound(net)                          # bound for all inputs
print(ns.check(net, sb))                           # numerical check on sample points
```

**Example.** The package contains a demonstration network with one input,
two hidden layers of eight tanh neurons, and one output. It was trained to
approximate the controller $`u = -\tanh(2y) - 0.5\,y`$.

```python
>>> net = ns.demo_network()
>>> sb = ns.local_bound(net, -1, 1)
>>> sb.gamma1.item(), sb.gamma2.item()
(-2.7862..., -1.2001...)
```

On $`[-1, 1]`$, the secant slopes of this network about $`y^* = 0`$ lie in
$`[-2.520, -1.463]`$. The computed bound $`[-2.786, -1.200]`$ contains this
interval and is close to it.

**PyTorch models.** A trained model is converted with

```python
net = ns.Network.from_torch(model)
```

`Network.from_torch` accepts a `torch.nn.Sequential` of `nn.Linear` layers
with one type of activation (`nn.Tanh`, `nn.Sigmoid`, `nn.ReLU`,
`nn.LeakyReLU` or `nn.Softplus`). Nested `nn.Sequential` blocks are
flattened, consecutive `nn.Linear` layers are merged, and `nn.Identity`,
`nn.Dropout` and `nn.Flatten` are ignored. The weights are copied in double
precision.

## Guarantees

Inequalities between vectors hold entrywise, and $`\lvert \cdot \rvert`$
denotes the entrywise absolute value. The function
`local_bound(net, y_lo, y_hi, ystar)` returns a bound of one of the
following three forms.

**One input ($`p = 1`$).** For every $`y \neq y^*`$ in the box,

```math
\gamma_1 \;\le\; \frac{\pi(y) - \pi(y^*)}{y - y^*} \;\le\; \gamma_2 .
```

**Operating point at the lower corner of the box ($`y^* = \underline{y}`$),
any number of inputs.** For every $`y`$ in the box,

```math
\gamma_1\,(y - y^*) \;\le\; \pi(y) - \pi(y^*) \;\le\; \gamma_2\,(y - y^*) .
```

**Operating point inside the box, several inputs.** In this case no single
pair of matrices holds on all sides of $`y^*`$. The box is divided at
$`y^*`$ into at most $`2^p`$ pieces, on each of which every component of
$`y - y^*`$ has a fixed sign. On piece $`k`$,

```math
L_k\,\lvert y - y^* \rvert \;\le\; \pi(y) - \pi(y^*) \;\le\; U_k\,\lvert y - y^* \rvert ,
```

where $`L_k`$ and $`U_k`$ are stored in `sb.pieces[k]`, and `sb.gamma1` and
`sb.gamma2` are `None`.

**Global bound.** The function `global_bound(net)` returns, for all $`y`$
and $`y^*`$,

```math
-\Gamma_2\,\lvert y - y^* \rvert \;\le\; \pi(y) - \pi(y^*) \;\le\; \Gamma_2\,\lvert y - y^* \rvert ,
\qquad
\Gamma_2 = c^{\,q}\,\lvert W_{q+1} \rvert \cdots \lvert W_1 \rvert ,
```

where $`c`$ is the largest secant slope of the activation function (see
[Supported activations](#supported-activations)). If the network has no
biases and $`f(0) = 0`$ (tanh, ReLU, leaky ReLU), then $`\pi(0) = 0`$, and
the bound gives $`\Gamma_1\,y \le \pi(y) \le \Gamma_2\,y`$ for all
$`y \ge 0`$, with $`\Gamma_1 = -\Gamma_2`$.

The matrices $`\gamma_1, \gamma_2`$ (`gamma1`, `gamma2`) and
$`\Gamma_1, \Gamma_2`$ (`Gamma1`, `Gamma2`) have shape $`m \times p`$. The
result of `local_bound` also contains `ystar`, `pistar` (the value
$`\pi(y^*)`$), `y_lo`, `y_hi` and, for each piece, `signs`, `z_hi` (the
largest $`\lvert y - y^* \rvert`$ on the piece) and `case3` (the number of
neurons per hidden layer in Case 3; see
[Computation of the local bound](#computation-of-the-local-bound)).

## Supported activations

All hidden layers use the same activation function.

| `activation` | $`f(x)`$ | Secant slopes lie in | $`f(0)`$ |
|---|---|---|---|
| `"tanh"` (default) | $`\tanh(x)`$ | $`[0, 1]`$ | $`0`$ |
| `"sigmoid"` | $`1/(1 + e^{-x})`$ | $`[0, 1/4]`$ | $`1/2`$ |
| `"relu"` | $`\max(x, 0)`$ | $`[0, 1]`$ | $`0`$ |
| `"leakyrelu"` | $`\max(x, 0) + \alpha \min(x, 0)`$ | $`[\alpha, 1]`$ for $`\alpha \le 1`$ | $`0`$ |
| `"softplus"` | $`\log(1 + e^{x})`$ | $`[0, 1]`$ | $`\log 2`$ |

The slope of the leaky ReLU for $`x < 0`$ is set with `alpha` (default
0.01), for example `ns.Network(W, b, activation="leakyrelu", alpha=0.1)`.

## Functions

| Function | Description |
|---|---|
| `Network(weights, biases=None, activation="tanh", alpha=None)` | Defines a network. `net(y)` evaluates it for `y` of shape `(p,)` or `(N, p)`. |
| `Network.from_torch(model)` | Converts a `torch.nn.Sequential` model. |
| `local_bound(net, y_lo, y_hi, ystar=None, intervals="intersect")` | Local sector bound about `ystar` (default 0) on the box `[y_lo, y_hi]`. |
| `global_bound(net)` | Global sector bound. |
| `check(net, bound, samples=20000, box=None, ystar=None)` | Evaluates the network on sample points and reports the largest violation of a bound. A sampling test cannot prove a bound; it can only detect an incorrect one. |
| `demo_network()` | The demonstration network described above. |

## Computation of the local bound

The local bound is computed separately on each piece of the box, in three
steps.

1. The network is evaluated at $`y^*`$. This gives the nominal
   preactivations $`\nu^*`$ of all hidden neurons.
2. Layer by layer, the changes $`\nu - \nu^*`$ of the preactivations are
   bounded by linear functions of $`\lvert y - y^* \rvert`$ (step S1). The
   activation function of each neuron is then replaced by two slopes that
   are valid on the interval of its preactivation (step S2). Three cases
   occur: the interval lies above $`\nu^*`$ (Case 1), below $`\nu^*`$
   (Case 2), or contains $`\nu^*`$ (Case 3).
3. The output layer gives the matrices $`L`$ and $`U`$.

Only changes with respect to $`y^*`$ are propagated, so the biases cancel.
Networks with biases are therefore supported.

**Case 3.** If the interval of a neuron contains $`\nu^*`$, one slope is
used on each side of $`\nu^*`$. In addition, only the parts of the linear
bounds of the neuron that can make its change negative or positive are
kept ($`\min(L, 0)`$ and $`\max(U, 0)`$). This is most effective for ReLU: a
neuron that is inactive at $`y^*`$ ($`\nu^* \le 0`$) can only increase, so
its lower slope is zero.

**Intervals.** In step S2, the interval of each neuron is by default the
intersection of its interval bound propagation (IBP) interval with the
range of its linear bounds over the box (`intervals="intersect"`). The
option `intervals="ibp"` uses the IBP interval alone. For the demonstration
network the difference is large: the bound is $`[-2.786, -1.200]`$ with the
intersection and $`[-2.878, 0.251]`$ with IBP alone. Both intervals are
computed for the changes $`\nu - \nu^*`$, so that rounding errors cannot
place a neuron whose interval ends exactly at $`\nu^*`$ in Case 3.

## Examples and tests

| File | Content |
|---|---|
| `examples/example1_demo.py` | The demonstration network, and a bound about $`y^* = 0.5`$. |
| `examples/example2_biases.py` | Random networks with biases, operating points inside the box, and two inputs. |
| `examples/example3_activations.py` | The weights of the demonstration network with each activation function. |
| `examples/example4_pytorch.py` | Training of a small PyTorch controller and computation of its bound. |

The test suite (`pytest`) compares the bound of the demonstration network
with its exact secant slopes, compares the secant slope ranges of all
activation functions with brute-force evaluation, and checks the bounds of
random networks with biases on sample points.

## MATLAB

A MATLAB implementation with the same functionality is provided in
[`matlab/`](matlab/README.md). It contains the functions
`nnsector.network`, `nnsector.local_bound`, `nnsector.global_bound`,
`nnsector.check`, `nnsector.forward` and `nnsector.demo_network`. It
requires MATLAB R2016b or later and also runs in GNU Octave.

## Citation

If you use nnsector in your research, please cite [1].

[1] H. Montazeri Hedesh, M. K. Wafi, and M. Siami, "Local Stability and
Region of Attraction Analysis for Neural Network Feedback Systems under
Positivity Constraints," in *Proc. 2025 IEEE 64th Conference on Decision
and Control (CDC)*, Rio de Janeiro, Brazil, 2025.
doi: [10.1109/CDC57313.2025.11312171](https://ieeexplore.ieee.org/document/11312171).
Preprint: [arXiv:2505.22889](https://arxiv.org/abs/2505.22889).

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

nnsector is released under the BSD 3-Clause License. See [LICENSE](LICENSE).
