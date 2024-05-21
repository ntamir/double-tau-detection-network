import h5py
from torch.utils.data import Dataset
import numpy as np

from utils import *

class EventsDataset (Dataset):
  def __init__(self, source_file, resolution=100):
    super().__init__()
    self.source_file = source_file
    self.raw_data = h5py.File(source_file, 'r')
    self.cache = {}
    self.resolution = resolution

    self._cluster_fields = [(name, python_name_from_dtype_name(name)) for name in self.raw_data['clusters'].dtype.names]
    self._track_fields = [(name, python_name_from_dtype_name(name)) for name in self.raw_data['tracks'].dtype.names]
    self._truth_fields = [(name, python_name_from_dtype_name(name)) for name in self.raw_data['truthTaus'].dtype.names]

  def get_event(self, index):
    if index in self.cache:
      return self.cache[index]
    event = self.raw_data['event'][index]
    clusters = self.raw_data['clusters'][index]
    tracks = self.raw_data['tracks'][index]
    truth = self.raw_data['truthTaus'][index]
    
    item = Event(event, clusters, tracks, truth, self._cluster_fields, self._track_fields, self._truth_fields)
    self.cache[index] = item
    return item

  def __getitem__(self, index):
    event = self.get_event(index)
    return (event.clusters_and_tracks_density_map(self.resolution), event.true_position_map(self.resolution))

  def __len__(self):
    return len(self.raw_data['event'])

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
  
  def pt_map (self, resulotion):
    map = np.zeros((resulotion, resulotion), dtype=np.float32)
    for cluster in self.clusters:
      x, y = relative_position(cluster.position())
      if (x > 0 and x < 1 and y > 0 and y < 1):
        map[int(x * resulotion), int(y * resulotion)] += cluster.pt
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
  
  def true_four_momentum (self):
    return [truth.visible_four_momentum() for truth in self.truths]

class Cluster:
  def __init__ (self, cluster, fields):
    for name, python_name in fields:
      setattr(self, python_name, cluster[name])

  def position (self):
    return np.array([self.cal_eta, self.cal_phi])
  
class Track:
  def __init__ (self, track, fields):
    for name, python_name in fields:
      setattr(self, python_name, track[name])

  def position (self):
    return np.array([self.eta, self.phi])

class Truth:
  def __init__ (self, truth, fields):
    for name, python_name in fields:
      setattr(self, python_name, truth[name])

  def visible_position (self):
    return np.array([self.eta_vis, self.phi_vis])
  
  def invisible_position (self):
    return np.array([self.eta_invis, self.phi_invis])
  
  def visible_four_momentum (self):
    return np.array([self.pt_vis, self.eta_vis, self.phi_vis, self.m_vis])
  
  def invisible_four_momentum (self):
    return np.array([self.pt_invis, self.eta_invis, self.phi_invis, self.m_invis])
  
  def total_four_momentum (self):
    return self.visible_four_momentum() + self.invisible_four_momentum()
