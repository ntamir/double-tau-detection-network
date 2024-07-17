import h5py
from progress.bar import IncrementalBar
from torch.utils.data import Dataset
import numpy as np

from data.event import Event
from utils import *

class EventsDataset (Dataset):
  def __init__(self, source_file, resolution=100):
    super().__init__()
    self.load(source_file)
    self.cache = {}
    self.resolution = resolution

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

    cluster_channel_providers = [
      lambda cluster: cluster.momentum().p_t,
      lambda cluster: cluster.center_mag,
      lambda cluster: cluster.center_lambda,
      lambda cluster: cluster.second_r,
      lambda cluster: cluster.second_lambda
    ]

    track_channel_providers = [
      lambda track: track.pt,
      lambda track: track.number_of_pixel_hits,
      lambda track: track.number_of_sct_hits,
      lambda track: track.number_of_trt_hits,
      lambda track: track.q_over_p
    ]

    clusters_map = event.clusters_map(self.resolution, cluster_channel_providers)
    tracks_map = event.tracks_map(self.resolution, track_channel_providers)
    inputs = (np.concatenate([clusters_map, tracks_map], axis=0),)
    target = np.array(event.true_position()).flatten()[:4]

    return inputs, target

  def __len__(self):
    return len(self.raw_data['event'])
  
  # io operations

  def save (self, filename):
    with h5py.File(filename, 'w') as f:
      for key in self.raw_data:
        f.create_dataset(key, data=self.raw_data[key])

  def load(self, source_file):
    self.source_file = source_file
    self.raw_data = h5py.File(source_file, 'r')

    self._cluster_fields = [(name, python_name_from_dtype_name(name)) for name in self.raw_data['clusters'].dtype.names]
    self._track_fields = [(name, python_name_from_dtype_name(name)) for name in self.raw_data['tracks'].dtype.names]
    self._truth_fields = [(name, python_name_from_dtype_name(name)) for name in self.raw_data['truthTaus'].dtype.names]
