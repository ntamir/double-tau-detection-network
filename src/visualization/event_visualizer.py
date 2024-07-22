from matplotlib import pyplot as plt
from numpy import pi

from settings import MAP_2D_TICKS, ETA_RANGE, PHI_RANGE

class EventVisualizer:
  def __init__ (self, event, resolution = 100):
    self.event = event
    self.resolution = resolution

  def density_map (self):
    clusters_points = [cluster.position().relative() for cluster in self.event.clusters]
    tracks_points = [track.position().relative() for track in self.event.tracks]
    truth_points = [truth.visible_position().relative() for truth in self.event.truths]

    self.map([clusters_points, tracks_points], scatter=truth_points)

  def momentum_map (self):
    cluster_points = [cluster.position().relative() for cluster in self.event.clusters]
    track_points = [track.position().relative() for track in self.event.tracks]
    truth_points = [truth.visible_position().relative() for truth in self.event.truths]

    clusters_momentum = [cluster.momentum().p_t for cluster in self.event.clusters]
    tracks_momentum = [track.momentum().p_t for track in self.event.tracks]

    self.map([cluster_points, track_points], weights=[clusters_momentum, tracks_momentum], scatter=truth_points)

  def map (self, maps, weights=None, scatter=None):
    for index, map in enumerate(maps):
      if weights == None or weights[index] == None:
        plt.hist2d(*zip(*map), bins=self.resolution, range=[[0, 1], [0, 1]], cmap='Blues', alpha=0.5)
      else:
        plt.hist2d(*zip(*map), bins=self.resolution, range=[[0, 1], [0, 1]], cmap='Blues', alpha=0.5, weights=weights[index])

    if scatter != None:
      plt.scatter(*zip(*scatter), s=30, c='black', marker='x')
    # set the axis to show the full eta and phi range
    plt.xticks([i / MAP_2D_TICKS for i in range(MAP_2D_TICKS + 1)], [round(ETA_RANGE[0] + i * (ETA_RANGE[1] - ETA_RANGE[0]) / MAP_2D_TICKS) for i in range(MAP_2D_TICKS + 1)])
    plt.yticks([i / MAP_2D_TICKS for i in range(MAP_2D_TICKS + 1)], [round(PHI_RANGE[0] + i * (PHI_RANGE[1] - PHI_RANGE[0]) / MAP_2D_TICKS) for i in range(MAP_2D_TICKS + 1)])
    plt.show()

  