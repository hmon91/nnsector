"""nnsector on networks with biases.

A one-input network about a point y* inside the box, and a two-input network,
where the bound comes in pieces. The networks are random, with a fixed seed.
"""

import numpy as np

import nnsector as ns

rng = np.random.default_rng(1)

# 1. One input, hidden layers of 10 and 5, y* = 0.5 in [-1, 2]
W = [rng.standard_normal((10, 1)), 0.5 * rng.standard_normal((5, 10)), rng.standard_normal((1, 5))]
b = [0.5 * rng.standard_normal(Wi.shape[0]) for Wi in W]
net = ns.Network(W, b)

sb = ns.local_bound(net, -1, 2, ystar=0.5)
print("One input, y* = 0.5 on [-1, 2]")
print(f"  pi(y*) = {sb.pistar[0]:.4f}")
print(f"  gamma1 <= (pi(y)-pi(y*))/(y-y*) <= gamma2 : "
      f"[{sb.gamma1.item():.4f}, {sb.gamma2.item():.4f}]")
print(f"  sample check passed: {ns.check(net, sb).passed}\n")

# 2. Two inputs, two outputs, two hidden layers of 8, y* = (1, 0.5)
W2 = [0.7 * rng.standard_normal((8, 2)), 0.5 * rng.standard_normal((8, 8)),
      0.5 * rng.standard_normal((2, 8))]
b2 = [0.3 * rng.standard_normal(8), 0.3 * rng.standard_normal(8), 0.3 * rng.standard_normal(2)]
net2 = ns.Network(W2, b2)

y_lo, y_hi, ystar = [0, 0], [2, 1], [1, 0.5]
sb2 = ns.local_bound(net2, y_lo, y_hi, ystar=ystar)
print(f"Two inputs, y* = {ystar} in [0,2] x [0,1]: {len(sb2.pieces)} pieces")
for P in sb2.pieces:
    print(f"  signs {P.signs.astype(int)}:  L |y-y*| <= pi(y)-pi(y*) <= U |y-y*|")
    print(f"    L = {np.array2string(P.L, precision=3)}".replace("\n", "\n        "))
    print(f"    U = {np.array2string(P.U, precision=3)}".replace("\n", "\n        "))
print(f"  sample check passed: {ns.check(net2, sb2).passed}\n")

# With y* at the lower corner there is one piece, and the bound has the form
#     gamma1 (y - y*) <= pi(y) - pi(y*) <= gamma2 (y - y*).
sb3 = ns.local_bound(net2, y_lo, y_hi, ystar=y_lo)
print("Two inputs, y* = y_lo:")
print(sb3)
