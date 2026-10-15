"""Networks from PyTorch (skipped when PyTorch is not installed)."""

import numpy as np
import pytest

import nnsector as ns

torch = pytest.importorskip("torch")
nn = torch.nn

LAYERS = [(nn.Tanh, "tanh", None), (nn.Sigmoid, "sigmoid", None), (nn.ReLU, "relu", None),
          (lambda: nn.LeakyReLU(0.05), "leakyrelu", 0.05), (nn.Softplus, "softplus", None)]


@pytest.mark.parametrize("make, name, alpha", LAYERS)
def test_from_torch_matches_the_model(make, name, alpha):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 16), make(), nn.Linear(16, 8), make(),
                          nn.Linear(8, 3)).double()
    net = ns.Network.from_torch(model)
    assert (net.activation, net.alpha, net.widths) == (name, alpha, [2, 16, 8, 3])
    y = np.random.default_rng(0).uniform(-2, 2, (100, 2))
    with torch.no_grad():
        expected = model(torch.from_numpy(y)).numpy()
    assert np.allclose(net(y), expected, atol=1e-12)
    sb = ns.local_bound(net, [-1, -1], [1, 1], ystar=[0.2, -0.3])
    assert ns.check(net, sb, samples=5000).passed


def test_float32_model_and_layers_that_do_nothing():
    torch.manual_seed(1)
    model = nn.Sequential(nn.Flatten(), nn.Linear(1, 8), nn.Identity(), nn.Linear(8, 8),
                          nn.Tanh(), nn.Dropout(0.1), nn.Sequential(nn.Linear(8, 1)))
    model.eval()
    net = ns.Network.from_torch(model)
    assert net.widths == [1, 8, 1]                     # Linear, Linear merged into one
    y = np.linspace(-1, 1, 50)[:, None]
    with torch.no_grad():
        expected = model(torch.from_numpy(y).float()).double().numpy()
    assert np.allclose(net(y), expected, atol=1e-5)


def test_models_that_are_not_supported():
    with pytest.raises(ValueError, match="not supported"):
        ns.Network.from_torch(nn.Sequential(nn.Linear(1, 4), nn.GELU(), nn.Linear(4, 1)))
    with pytest.raises(ValueError, match="end with an nn.Linear"):
        ns.Network.from_torch(nn.Sequential(nn.Linear(1, 4), nn.Tanh()))
    with pytest.raises(ValueError, match="same activation"):
        ns.Network.from_torch(nn.Sequential(nn.Linear(1, 4), nn.Tanh(), nn.Linear(4, 4),
                                            nn.ReLU(), nn.Linear(4, 1)))
    with pytest.raises(ValueError, match="beta"):
        ns.Network.from_torch(nn.Sequential(nn.Linear(1, 4), nn.Softplus(beta=2),
                                            nn.Linear(4, 1)))
    with pytest.raises(TypeError):
        ns.Network.from_torch(nn.Linear(1, 1))
