import torch
from torch import nn

from model.cylinrical_conv import CylindricalConv2d

class DoubelTauRegionDetection (nn.Module):
  def __init__(self, resolution=100, dropout_probability=0.15):
    super(DoubelTauRegionDetection, self).__init__()
    self.resolution = resolution
    self.dropout_probability = dropout_probability

    # seperate convolutional networks for the tracks and clusters, then mereged using a fully connected network
    
    def conv_module ():
      return nn.ModuleList([
        CylindricalConv2d(5, 8, kernel_size=5, padding=2),
        nn.PReLU(),
        nn.Dropout2d(self.dropout_probability),
        nn.MaxPool2d(2),
        CylindricalConv2d(8, 16, kernel_size=3, padding=1),
        nn.PReLU(),
        nn.Dropout2d(self.dropout_probability),
        nn.MaxPool2d(2),
        CylindricalConv2d(16, 16, kernel_size=3, padding=1),
        nn.PReLU(),
        nn.Dropout2d(self.dropout_probability),
        nn.MaxPool2d(5),
        CylindricalConv2d(16, 8, kernel_size=5, padding=2),
        nn.PReLU(),
        nn.Dropout2d(self.dropout_probability),
      ])

    self.track_layers = conv_module()
    self.cluster_layers = conv_module()

    # cluster layers is the same as track layers. Just copy the layers from track_layers

    self.linear_layers = nn.ModuleList([
      nn.Linear(2 * 8 * (resolution // 20) * (resolution // 20), 64),
      nn.PReLU(),
      nn.Dropout(self.dropout_probability),
      nn.Linear(64, 4)
    ])

    # count and print the number of parameters in the network
    convulational_params = sum(p.numel() for p in self.track_layers.parameters() if p.requires_grad) + sum(p.numel() for p in self.cluster_layers.parameters() if p.requires_grad)
    linear_params = sum(p.numel() for p in self.linear_layers.parameters() if p.requires_grad)
    print(f"Initialized NN.")
    print(f"Convolutional layers parameters:  {convulational_params}")
    print(f"Linear layers parameters:         {linear_params}")

  def forward(self, clusters, tracks):
    for layer in self.track_layers:
      tracks = layer(tracks)

    for layer in self.cluster_layers:
      clusters = layer(clusters)

    clusters = clusters.view(-1, 8 * (self.resolution // 20) * (self.resolution // 20))
    tracks = tracks.view(-1, 8 * (self.resolution // 20) * (self.resolution // 20))

    x = torch.cat((clusters, tracks), 1)

    for layer in self.linear_layers:
      x = layer(x)

    return x
