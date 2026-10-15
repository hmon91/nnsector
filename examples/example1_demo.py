"""The demo network that ships with nnsector.

A small tanh network trained to act like the controller u = -tanh(2y) - 0.5y.
Computes its local sector bound about y* = 0 on [-1, 1], compares it with the
exact range of the secant slopes, and adds the global bound and a bound about
y* = 0.5. Plots them if matplotlib is installed.
"""

import numpy as np

import nnsector as ns

net = ns.demo_network()                 # 1 input, 8 and 8 tanh neurons, 1 output, with biases
print(net)

sb0 = ns.local_bound(net, -1, 1)        # about y* = 0
print(f"Local bound, y* = 0 on [-1, 1]   : [{sb0.gamma1.item():.4f}, {sb0.gamma2.item():.4f}]")

y = np.linspace(-1, 1, 200001)
y = y[y != 0]
r = (net(y[:, None])[:, 0] - sb0.pistar[0]) / y        # (pi(y) - pi(0)) / y
print(f"  exact secant slopes             : [{r.min():.4f}, {r.max():.4f}]")

sb_ibp = ns.local_bound(net, -1, 1, intervals="ibp")
print(f"  with plain IBP intervals        : [{sb_ibp.gamma1.item():.4f}, {sb_ibp.gamma2.item():.4f}]")

gb = ns.global_bound(net)
print(f"Global bound                      : [{gb.Gamma1.item():.4f}, {gb.Gamma2.item():.4f}]")

# y* inside the box: the box is cut at y* into [-1, 0.5] and [0.5, 1], and
#     gamma1 <= (pi(y) - pi(0.5)) / (y - 0.5) <= gamma2   for every y != 0.5.
sb1 = ns.local_bound(net, -1, 1, ystar=0.5)
print(f"Local bound, y* = 0.5 on [-1, 1] : [{sb1.gamma1.item():.4f}, {sb1.gamma2.item():.4f}]")
for P in sb1.pieces:
    side = "y >= y*" if P.signs[0] > 0 else "y <= y*"
    print(f"  piece {side} : {P.L.item():.4f} |y-y*| <= pi(y)-pi(y*) <= {P.U.item():.4f} |y-y*|")

checks = [ns.check(net, sb0), ns.check(net, sb1), ns.check(net, gb, box=(-20, 20))]
print("Sample checks passed              :", [c.passed for c in checks])

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("(install matplotlib to see the plot)")
else:
    y = np.linspace(-1, 1, 400)
    fig, ax = plt.subplots()
    ax.plot(y, net(y[:, None])[:, 0], "k", lw=1.5, label=r"$\pi(y)$")
    for sb, style, tag in ((sb0, "--", "0"), (sb1, ":", "0.5")):
        ys, ps = sb.ystar[0], sb.pistar[0]
        ax.plot(y, ps + sb.gamma1.item() * (y - ys), "r" + style, label=rf"$\gamma_1$, $y^*={tag}$")
        ax.plot(y, ps + sb.gamma2.item() * (y - ys), "g" + style, label=rf"$\gamma_2$, $y^*={tag}$")
    ax.plot(sb1.ystar[0], sb1.pistar[0], "ko")
    ax.set_xlabel("y")
    ax.set_ylabel("u")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    plt.show()
