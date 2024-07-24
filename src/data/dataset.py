import h5py
from progress.bar import IncrementalBar
from torch.utils.data import Dataset
import numpy as np

from data.event import Event
from utils import *
from settings import RESOLUTION, DATASET_FIELDS

class EventsDataset (Dataset):
  def __init__(self, source_file):
    super().__init__()
    self.dataset_fields = DATASET_FIELDS
    self.cache = {}
    self.load(source_file)

  def get_event(self, index):
    if index in self.cache:
      return self.cache[index]
    
    fields = [self.raw_data[field][index] for field in self.dataset_fields]

    item = Event(*fields, **self._fields)
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

    clusters_map = event.clusters_map(RESOLUTION, cluster_channel_providers)
    tracks_map = event.tracks_map(RESOLUTION, track_channel_providers)
    inputs = (np.concatenate([clusters_map, tracks_map], axis=0),)
    target = np.array([position.to_list() for position in event.true_position()], dtype=np.float32).flatten()[:4]
    
    if len(target) < 4:
      target = np.concatenate([target, np.zeros(4 - len(target), dtype=np.float32)])

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

    self._fields = { f'{field}_fields': [(name, python_name_from_dtype_name(name)) for name in self.raw_data[field].dtype.names] for field in self.dataset_fields }
