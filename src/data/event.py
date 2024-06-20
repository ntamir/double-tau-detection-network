import numpy as np
from sklearn.preprocessing import StandardScaler

from utils import relative_position

from data.cluster import Cluster
from data.track import Track
from data.tau_truth import Truth

FIELDS_TO_NORMALIZE = {
  'clusters': ['center_mag', 'center_lambda', 'second_r', 'second_lambda'],
  'tracks': ['number_of_pixel_hits', 'number_of_sct_hits', 'number_of_trt_hits', 'q_over_p'],
}

class Event:
  def __init__ (self, event, clusters, tracks, truth, cluster_fields, track_fields, truth_fields):
    self.average_interactions_per_crossing = event[0]
    self.clusters = [Cluster(cluster, cluster_fields) for cluster in clusters if cluster['valid']]
    self.tracks = [Track(track, track_fields) for track in tracks if track['valid']]
    self.truths = [Truth(truth, truth_fields) for truth in truth if truth['valid']]
    self._calculateion_cache = {}

    self.normalize()

  def normalize (self):
    def normzlied (values):
      values = np.array(values).reshape(-1, 1)
      return StandardScaler().fit_transform(values).reshape(-1)

    # normalize clusters
    total_energy = sum([cluster.cal_e for cluster in self.clusters])
    cluster_normalized_fields = { field: normzlied([getattr(cluster, field) for cluster in self.clusters]) for field in FIELDS_TO_NORMALIZE['clusters'] }
    for index, cluster in enumerate(self.clusters):
      cluster.cal_e /= total_energy
      for field in FIELDS_TO_NORMALIZE['clusters']:
        setattr(cluster, field, cluster_normalized_fields[field][index])
    
    # normalize tracks
    total_pt = sum([track.pt for track in self.tracks])
    track_normalized_fields = { field: normzlied([getattr(track, field) for track in self.tracks]) for field in FIELDS_TO_NORMALIZE['tracks'] }
    for index, track in enumerate(self.tracks):
      track.pt /= total_pt
      for field in FIELDS_TO_NORMALIZE['tracks']:
        setattr(track, field, track_normalized_fields[field][index])

  def calculate_and_cache (self, key, calculation):
    if key not in self._calculateion_cache:
      self._calculateion_cache[key] = calculation()
    return self._calculateion_cache[key]

  # input types

  def clusters_and_tracks_density_map (self, resulotion):    
    def calculate ():
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

    return self.calculate_and_cache('clusters_and_tracks_density_map', calculate)
  
  def clusters_and_tracks_momentum_map (self, resulotion):
    def calculate ():
      map = np.zeros((2, resulotion, resulotion), dtype=np.float32)
      for cluster in self.clusters:
        x, y = relative_position(cluster.position())
        if (x > 0 and x < 1 and y > 0 and y < 1):
          momentum = cluster.momentum()
          map[0, int(x * resulotion), int(y * resulotion)] += momentum.p_t
      for track in self.tracks:
        x, y = relative_position(track.position())
        if (x > 0 and x < 1 and y > 0 and y < 1):
          map[1, int(x * resulotion), int(y * resulotion)] += track.pt
      return map

    return self.calculate_and_cache('clusters_and_tracks_momentum_map', calculate)
  
  def clusters_map (self, resulotion, channel_providers):
    def calculate ():
      map = np.zeros((len(channel_providers), resulotion, resulotion), dtype=np.float32)
      for cluster in self.clusters:
        x, y = relative_position(cluster.position())
        if (x > 0 and x < 1 and y > 0 and y < 1):
          for index, provider in enumerate(channel_providers):
            map[index, int(x * resulotion), int(y * resulotion)] += provider(cluster)
      return map

    return self.calculate_and_cache('clusters_map', calculate)
  
  def tracks_map (self, resulotion, channel_providers):
    def calculate():
      map = np.zeros((len(channel_providers), resulotion, resulotion), dtype=np.float32)
      for track in self.tracks:
        x, y = relative_position(track.position())
        if (x > 0 and x < 1 and y > 0 and y < 1):
          for index, provider in enumerate(channel_providers):
            map[index, int(x * resulotion), int(y * resulotion)] += provider(track)
      return map
    
    return self.calculate_and_cache('tracks_map', calculate)
  
  # target types
  
  def true_position (self):
    return self.calculate_and_cache('true_position', lambda: [truth.position() for truth in self.truths])
  
  def true_position_map (self, resulotion):
    def calculate ():
      map = np.zeros((resulotion, resulotion), dtype=np.float32)
      for truth in self.truths:
        x, y = relative_position(truth.visible_position())
        if (x > 0 and x < 1 and y > 0 and y < 1):
          map[int(x * resulotion), int(y * resulotion)] += 1
      return map
    
    return self.calculate_and_cache('true_position_map', calculate)
  
  def true_momentum_map (self, resulotion):
    def calculate ():
      map = np.zeros((resulotion, resulotion), dtype=np.float32)
      for truth in self.truths:
        x, y = relative_position(truth.visible_position())
        if (x > 0 and x < 1 and y > 0 and y < 1):
          map[int(x * resulotion), int(y * resulotion)] += truth.visible_momentum().p_t
      return map
    
    return self.calculate_and_cache('true_momentum_map', calculate)
  
  def true_four_momentum (self):
    return self.calculate_and_cache('true_four_momentum', lambda: [truth.visible_four_momentum() for truth in self.truths])
