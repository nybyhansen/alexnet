import torch.nn as nn

class ConvBlock(nn.Module):
  def __init__(self, kernel_size, in_channels: int, out_channels: int, stride: int, padding: int):
    super(ConvBlock, self).__init__()
    self.conv = nn.Conv2d(
        kernel_size=kernel_size,
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
        padding=padding,
    )
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.relu(self.conv(x))

  
class FCBlock(nn.Module):
  def __init__(self, in_features: int, out_features: int, p: float):
    super(FCBlock, self).__init__()
    self.linear = nn.Linear(in_features=in_features, out_features=out_features)
    self.dropout = nn.Dropout(p=p)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.linear(x)
    x = self.dropout(x)
    x = self.relu(x)
    return x