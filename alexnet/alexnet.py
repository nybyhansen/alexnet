import torch.nn as nn
from alexnet.blocks import ConvBlock, FCBlock


class AlexNet(nn.Module):
  def __init__(self, n_classes: int):
    super(AlexNet, self).__init__()
    self.features = nn.Sequential(
      ConvBlock(kernel_size=11, in_channels=3, out_channels=96, stride=4, padding=0),
      ConvBlock(kernel_size=5, in_channels=96, out_channels=256, stride=1, padding=2),
      nn.MaxPool2d(kernel_size=3, stride=2),
      ConvBlock(kernel_size=3, in_channels=256, out_channels=384, stride=1, padding=1),
      nn.MaxPool2d(kernel_size=3, stride=2),
      ConvBlock(kernel_size=3, in_channels=384, out_channels=384, stride=1, padding=1),
      ConvBlock(kernel_size=3, in_channels=384, out_channels=256, stride=1, padding=1),
      nn.MaxPool2d(kernel_size=3, stride=2),
    )
    self.fc = nn.Sequential(
      FCBlock(in_features=256 * 6 * 6, out_features=4_096, p=0.5),
      FCBlock(in_features=4_096, out_features=4096, p=0.5),
      FCBlock(in_features=4_096, out_features=n_classes, p=0.5),
    )
    self._initilize_weights()

  def _initilize_weights(self):
    """Initilize weights and biases according to original paper"""
    layers = (m for m in self.modules() if isinstance(m, (nn.Conv2d, nn.Linear)))
    conv_layers = [m for m in self.modules() if isinstance(m, nn.Conv2d)]

    for layer in layers:
      nn.init.normal_(layer.weight, mean=0.0, std=0.1)
      nn.init.constant_(layer.bias, val=0.0)

    for layer_idx in (1, 3 ,4):
      nn.init.constant_(conv_layers[layer_idx].bias, val=1.0)

  def forward(self, x):
    x = self.features(x)
    # flatten the tensor
    x = x.view(-1, 256 * 6 * 6)
    return self.fc(x)