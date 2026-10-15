"""nnsector: sector bounds for feedforward neural networks.

Give a network, a point ``y*`` and a box that contains it, and get the slopes
of two lines through ``(y*, pi(y*))`` that the network stays between on the
box.

>>> import nnsector as ns
>>> net = ns.demo_network()
>>> sb = ns.local_bound(net, -1, 1)
>>> round(sb.gamma1.item(), 3), round(sb.gamma2.item(), 3)
(-2.786, -1.2)
"""

from .activations import SUPPORTED_ACTIVATIONS, Activation, get_activation
from .bounds import GlobalBound, LocalBound, Piece, global_bound, local_bound
from .check import CheckResult, check
from .demo import demo_network
from .network import Network

__version__ = "0.1.0"

__all__ = [
    "Activation",
    "CheckResult",
    "GlobalBound",
    "LocalBound",
    "Network",
    "Piece",
    "SUPPORTED_ACTIVATIONS",
    "check",
    "demo_network",
    "get_activation",
    "global_bound",
    "local_bound",
]
