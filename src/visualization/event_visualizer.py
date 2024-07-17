from matplotlib import pyplot as plt
from numpy import pi

from settings import MAP_2D_TICKS, ETA_RANGE, PHI_RANGE

class EventVisualizer:
  def __init__ (self, event, resolution = 100):
    self.event = event
    self.resolution = resolution

  def map (self):
    clusters_points = [cluster.position().relative() for cluster in self.event.clusters]
    tracks_points = [track.position().relative() for track in self.event.tracks]
    truth_points = [truth.visible_position().relative() for truth in self.event.truths]

    # show a histogram of the clusters and tracks, with the truth points marked with Xs.
    plt.hist2d(*zip(*tracks_points), bins=self.resolution, range=[[0, 1], [0, 1]], cmap='Reds', alpha=1)
    plt.hist2d(*zip(*clusters_points), bins=self.resolution, range=[[0, 1], [0, 1]], cmap='Blues', alpha=0.5)
    plt.scatter(*zip(*truth_points), s=30, c='black', marker='x')
    # set the axis to show the full eta and phi range
    plt.xticks([i / MAP_2D_TICKS for i in range(MAP_2D_TICKS + 1)], [round(ETA_RANGE[0] + i * (ETA_RANGE[1] - ETA_RANGE[0]) / MAP_2D_TICKS) for i in range(MAP_2D_TICKS + 1)])
    plt.yticks([i / MAP_2D_TICKS for i in range(MAP_2D_TICKS + 1)], [round(PHI_RANGE[0] + i * (PHI_RANGE[1] - PHI_RANGE[0]) / MAP_2D_TICKS) for i in range(MAP_2D_TICKS + 1)])
    plt.show()