"""The weights and biases of the demo network with each activation of nnsector.

For y* = 0 on [-1, 1]: the local bound, the exact range of the secant slopes,
and the global bound. The weights were trained for tanh, so for the other
activations this only shows how to use the library.
"""

import numpy as np

import nnsector as ns

demo = ns.demo_network()
y = np.linspace(-1, 1, 200001)
y = y[y != 0]                                   # grid for the exact slopes

print(f"{'activation':<11} {'local bound':<19} {'exact slopes':<19} {'global bound':<19} check")
for name in ns.SUPPORTED_ACTIVATIONS:
    net = ns.Network(demo.weights, demo.biases, activation=name)    # leaky ReLU: alpha = 0.01
    sb = ns.local_bound(net, -1, 1)
    gb = ns.global_bound(net)
    r = (net(y[:, None])[:, 0] - sb.pistar[0]) / y      # (pi(y) - pi(0)) / y
    print(f"{name:<11} [{sb.gamma1.item():7.4f}, {sb.gamma2.item():7.4f}]  "
          f"[{r.min():7.4f}, {r.max():7.4f}]  "
          f"[{gb.Gamma1.item():7.4f}, {gb.Gamma2.item():7.4f}]  {ns.check(net, sb).passed}")
