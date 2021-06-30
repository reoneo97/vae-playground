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

    hidden_size = 28
    batch_size = 5
    original_shape = (batch_size, 1, 28, 28)
    inp = torch.rand(*original_shape)
    model = VAE(hidden_size, 1, 0.01, batch_size)
    mu, log_var, output = model(inp)

    assert mu.shape == (batch_size, hidden_size)
    assert log_var.shape == (batch_size, hidden_size)
    assert output.shape == original_shape


def test_Conv_VAE():
    hidden_size = 28
    batch_size = 5
    channels = 1
    height = width = 28
    original_shape = (batch_size, channels, height, width)
    inp = torch.rand(*original_shape)
    model = Conv_VAE(channels, height, width, 0.01, hidden_size, 1, batch_size)

    mu, log_var, output = model(inp)

    assert mu.shape == (batch_size, hidden_size)
    assert log_var.shape == (batch_size, hidden_size)
    assert output.shape == original_shape


def test_Conv_VAE_3():
    hidden_size = 28
    batch_size = 5
    channels = 3
    height = width = 28
    original_shape = (batch_size, channels, height, width)
    inp = torch.rand(*original_shape)
    model = Conv_VAE(channels, height, width, 0.01, hidden_size, 1, batch_size)

    mu, log_var, output = model(inp)

    assert mu.shape == (batch_size, hidden_size)
    assert log_var.shape == (batch_size, hidden_size)
    assert output.shape == original_shape


def test_interpolate():
    x1 = torch.rand(1, 28, 28)
    x2 = torch.rand(1, 28, 28)
    model = VAE(28, 1, 0.01, 1)
    model.eval()  # So that batch norm does not work
    test, _, _ = model.interpolate(x1, x2)
    assert test.shape == (10, 1, 28, 28)


def test_conv_interpolate():
    x1 = torch.rand(1, 28, 28)
    x2 = torch.rand(1, 28, 28)
    model = Conv_VAE(1, 28, 28, 0.01, 28, 1, 1)
    model.eval()  # So that batch norm does not work
    test, _, _ = model.interpolate(x1, x2)
    assert test.shape == (10, 1, 28, 28)
