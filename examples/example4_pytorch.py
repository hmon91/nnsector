"""Train a small PyTorch controller and compute its sector bound.

The demo network of nnsector (ns.demo_network()) was trained the same way,
with 8 neurons per hidden layer. Needs PyTorch (pip install torch).
"""

import torch
from torch import nn

import nnsector as ns

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(1, 16), nn.Tanh(),
                      nn.Linear(16, 16), nn.Tanh(),
                      nn.Linear(16, 1))

# imitate the controller u = -tanh(2y) - 0.5y on [-2, 2]
y = torch.linspace(-2, 2, 400).unsqueeze(1)
u = -torch.tanh(2 * y) - 0.5 * y
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for step in range(2000):
    opt.zero_grad()
    loss = ((model(y) - u) ** 2).mean()
    loss.backward()
    opt.step()
print(f"training loss: {loss.item():.2e}")

net = ns.Network.from_torch(model)              # weights copied in double precision
print(net)

sb = ns.local_bound(net, -1, 1, ystar=0)        # about y* = 0 on [-1, 1]
print(sb)
print(ns.check(net, sb))
