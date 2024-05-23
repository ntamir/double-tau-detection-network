import numpy as np

from utils import relative_position

from data.cluster import Cluster
from data.track import Track
from data.tau_truth import Truth

class Event:
  def __init__ (self, event, clusters, tracks, truth, cluster_fields, track_fields, truth_fields):
    self.average_interactions_per_crossing = event[0]
    self.clusters = [Cluster(cluster, cluster_fields) for cluster in clusters if cluster['valid']]
    self.tracks = [Track(track, track_fields) for track in tracks if track['valid']]
    self.truths = [Truth(truth, truth_fields) for truth in truth if truth['valid']]

  def clusters_and_tracks_density_map (self, resulotion):
    map = np.zeros((2, resulotion, resulotion), dtype=np.float32)
    for cluster in self.clusters:
      x, y = relative_position(cluster.position())
      if (x > 0 and x < 1 and y > 0 and y < 1):
        map[0, int(x * resulotion), int(y * resulotion)] += 1
    for track in self.tracks:
      x, y = relative_position(track.position())
      if (x > 0 and x < 1 and y > 0 and y < 1):
        map[1, int(x * resulotion), int(y * resulotion)] += 1
    return map
  
  def clusters_and_tracks_momentum_map (self, resulotion):
    map = np.zeros((2, resulotion, resulotion), dtype=np.float32)
    for cluster in self.clusters:
      x, y = relative_position(cluster.position())
      if (x > 0 and x < 1 and y > 0 and y < 1):
        momentum = cluster.momentum()
        map[0, int(x * resulotion), int(y * resulotion)] += momentum.p_t
        pass
    for track in self.tracks:
      x, y = relative_position(track.position())
      if (x > 0 and x < 1 and y > 0 and y < 1):
        map[1, int(x * resulotion), int(y * resulotion)] += track.pt
    return map
  
  def true_position (self):
    return [truth.visible_position() for truth in self.truths]
  
  def true_position_map (self, resulotion):
    map = np.zeros((resulotion, resulotion), dtype=np.float32)
    for truth in self.truths:
      x, y = relative_position(truth.visible_position())
      if (x > 0 and x < 1 and y > 0 and y < 1):
        map[int(x * resulotion), int(y * resulotion)] += 1
    return map
  
  def true_momentum_map (self, resulotion):
    map = np.zeros((resulotion, resulotion), dtype=np.float32)
    for truth in self.truths:
      x, y = relative_position(truth.visible_position())
      if (x > 0 and x < 1 and y > 0 and y < 1):
        map[int(x * resulotion), int(y * resulotion)] += truth.visible_momentum().p_t
    return map
  
  def true_four_momentum (self):
    return [truth.visible_four_momentum() for truth in self.truths]