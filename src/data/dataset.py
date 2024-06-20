import h5py
from torch.utils.data import Dataset
import numpy as np

from data.event import Event
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
    
    input = (
      event.clusters_map(self.resolution, cluster_channel_providers),
      event.tracks_map(self.resolution, track_channel_providers)
    )
    
    target = event.true_momentum_map(self.resolution)

    return input, target

  def __len__(self):
    return len(self.raw_data['event'])
