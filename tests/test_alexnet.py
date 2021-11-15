from alexnet.alexnet import AlexNet
import torch
import torch.nn as nn
import pytest


@pytest.fixture(scope="module")
def alexnet():
    return AlexNet(n_classes=10)


def test_alexnet_architecture_shapes(alexnet):
    """Test that an image tensor can make a pass through the network"""
    image_batch = torch.rand(1, 3, 227, 227)
    alexnet_out = alexnet(image_batch)
    assert alexnet_out.shape == (1, 10), "The model must be configured right"


def test_weight_initilization_strategy(alexnet):
    """Test that weights are initilized according to original paper"""
    layers = (m for m in alexnet.modules() if isinstance(m, nn.Conv2d))

    for layer in layers:
        assert abs(0.0 - layer.weight.mean().item()) < 0.001, (
            "Weights should be initialized Gaussian with zero mean!"
        )
        assert abs(0.1 - layer.weight.std().item()) < 0.001, (
            "Weights should be initialized Gaussian with standard deviation: 0.1!"
        )


def test_bias_initilization_strategy(alexnet):
    """Test that biases are initilized according to original paper"""
    conv_layers = [m for m in alexnet.modules() if isinstance(m, nn.Conv2d)]
    
    # Second, fourth and fifth convolutional layer should have constant bias equal to 1.
    one_constant_layers = (2, 4, 5)
    for layer_idx in one_constant_layers:
        layer = conv_layers[layer_idx - 1]
        assert (layer.bias != 1).sum().item() == 0, (
            "2., 4. and 5. convolutional layer should have constant bias == 1"
        )

    # .. the rest should have constant zero bias.
    zero_constant_layers = (1, 3)
    for layer_idx in zero_constant_layers:
        layer = conv_layers[layer_idx - 1]
        assert (layer.bias != 0).sum().item() == 0, (
            "1. and 3. convolutional layer should have constant bias == 0"
        )
