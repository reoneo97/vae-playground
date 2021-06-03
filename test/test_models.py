from models import VAE, Conv_VAE, Flatten, Stack
import torch


def test_flatten():
    test_img = torch.rand(5, 1, 28, 28)
    flatten = Flatten()
    assert flatten(test_img).shape == (5, 784), "Tensor of Wrong size"


def test_stack():
    inp = torch.rand(5, 784)
    stack = Stack(1, 28, 28)
    assert stack(inp).shape == (5, 1, 28, 28), "Tensor of Wrong size"


def test_stack3():
    inp = torch.rand(5, 2352)
    stack = Stack(3, 28, 28)
    assert stack(inp).shape == (5, 3, 28, 28), "Tensor of Wrong size"


def test_VAE():
    original_shape = (5, 1, 28, 28)
    inp = torch.rand(*original_shape)
    model = VAE(28, 1, "mnist")
    assert model(inp).shape == original_shape


def test_Conv_VAE():
    original_shape = (5, 1, 28, 28)
    inp = torch.rand(*original_shape)
    model = Conv_VAE(28, 1, "mnist")
    assert model(inp).shape == original_shape
