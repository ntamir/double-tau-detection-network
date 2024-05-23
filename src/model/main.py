from torch import nn
from torch.nn import functional as F

from model.cylinrical_conv import CylindricalConv2d

class DoubelTauRegionDetection (nn.Module):
  def __init__(self, resolution=100):
    super(DoubelTauRegionDetection, self).__init__()
    self.resolution = resolution
    self.conv_1 = CylindricalConv2d(2, 16, kernel_size=3, padding=1)
    self.conv_2 = nn.MaxPool2d(2)
    self.conv_3 = CylindricalConv2d(16, 16, kernel_size=3, padding=1)
    self.conv_4 = nn.MaxPool2d(2)
    self.conv_5 = CylindricalConv2d(16, 16, kernel_size=3, padding=1)
    self.conv_6 = nn.MaxPool2d(2)
    self.conv_7 = CylindricalConv2d(16, 16, kernel_size=3, padding=1)
    self.conv_8 = nn.MaxPool2d(2)
    self.linear_1 = nn.Linear(16 * 6 * 6, 64)
    self.linear_2 = nn.Linear(64, resolution * resolution)

    self.conv_layers = [self.conv_1, self.conv_2, self.conv_3, self.conv_4, self.conv_5, self.conv_6, self.conv_7, self.conv_8]
    self.linear_layers = [self.linear_1, F.relu, self.linear_2]

    # count and print the number of parameters in the network
    trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    print("Initialized NN with {} trainable parameters".format(trainable_params))

  def forward(self, x):
    for layer in self.conv_layers:
      x = layer(x)

    x = x.view(x.size(0), -1)
    
    for layer in self.linear_layers:
      x = layer(x)

    x = x.view(-1, self.resolution, self.resolution)
    return x