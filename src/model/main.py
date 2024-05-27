import torch
from torch import nn
from torch.nn import functional as F

from model.cylinrical_conv import CylindricalConv2d

class DoubelTauRegionDetection (nn.Module):
  def __init__(self, resolution=100):
    super(DoubelTauRegionDetection, self).__init__()
    self.resolution = resolution

    # seperate convolutional networks for the tracks and clusters, then mereged using a fully connected network

    self.track_layers = nn.ModuleList([
      CylindricalConv2d(2, 16, kernel_size=3, padding=1),
      nn.MaxPool2d(2),
      CylindricalConv2d(16, 16, kernel_size=3, padding=1),
      nn.MaxPool2d(2)
    ])

    self.cluster_layers = nn.ModuleList([
      CylindricalConv2d(2, 16, kernel_size=3, padding=1),
      nn.MaxPool2d(2),
      CylindricalConv2d(16, 16, kernel_size=3, padding=1),
      nn.MaxPool2d(2)
    ])

    self.linear_layers = nn.ModuleList([
      nn.Linear(32 * (resolution // 4) * (resolution // 4), 128),
      nn.ReLU(),
      nn.Linear(128, self.resolution * self.resolution)
    ])

    # count and print the number of parameters in the network
    trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    print("Initialized NN with {} trainable parameters".format(trainable_params))

  def forward(self, x):
    clusters, tracks = x

    for layer in self.track_layers:
      tracks = layer(tracks)

    for layer in self.cluster_layers:
      clusters = layer(clusters)

    clusters = clusters.view(-1, 16 * (self.resolution // 4) * (self.resolution // 4))
    tracks = tracks.view(-1, 16 * (self.resolution // 4) * (self.resolution // 4))    

    x = torch.cat([clusters, tracks], dim=1)

    for layer in self.linear_layers:
      x = layer(x)

    return x.view(-1, self.resolution, self.resolution)
