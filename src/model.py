from torch import nn
from torch.nn import functional as F

from modules.cylinrical_conv import CylindricalConv2d

# This block recieves the array of positions of the tracks and clusters and outputs the origin of the event
class OriginFindingModule (nn.Module):
  def __init__(self):
    super(OriginFindingModule, self).__init__()
    self.conv_1 = CylindricalConv2d(2, 16, kernel_size=3, padding=1)
    self.conv_2 = nn.MaxPool2d(2)
    self.conv_3 = CylindricalConv2d(16, 32, kernel_size=3, padding=1)
    self.conv_4 = nn.MaxPool2d(2)
    self.conv_5 = CylindricalConv2d(32, 64, kernel_size=3, padding=1)
    self.conv_6 = nn.MaxPool2d(2)
    self.linear_1 = nn.Linear(64 * 12 * 12, 64)
    self.linear_2 = nn.Linear(64, 2)

    self.conv_layers = [self.conv_1, self.conv_2, self.conv_3, self.conv_4, self.conv_5, self.conv_6]
    self.linear_layers = [self.linear_1, F.relu, self.linear_2]
  
  def forward(self, x):
    for layer in self.conv_layers:
      x = layer(x)

    x = x.view(x.size(0), -1)
    
    for layer in self.linear_layers:
      x = layer(x)

    return x

# A graph blcok that takes a graph of possible jets and outputs a classification
class DoubleTauClassificationModule (nn.Module):
  pass

class DoubleTauDetectionNetwork (nn.Module):
  def __init__(self):
    super(DoubleTauDetectionNetwork, self).__init__()
    self.origin_finding = OriginFindingModule()
    self.double_tau_classification = DoubleTauClassificationModule()

  def forward(self, x):
    x = self.origin_finding(x)
    x = self.double_tau_classification(x)
    return x