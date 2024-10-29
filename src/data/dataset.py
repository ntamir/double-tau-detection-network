import h5py
from torch.utils.data import Dataset
import torch
import numpy as np
import gc
import math
from time import time

from data.event import Event
from utils import *
from settings import RESOLUTION, DATASET_FIELDS, ETA_RANGE, PHI_RANGE

class EventsDataset (Dataset):
  def __init__(self, source_file):
    super().__init__()
    self.dataset_fields = DATASET_FIELDS
    self.use_cache = True
    self.cache = {}
    self.preloaded = False
    self.load(source_file)

    self.cluster_channel_providers = [
      lambda cluster: cluster.momentum().p_t,
    ]

    self.track_channel_providers = [
      lambda track: track.pt,
    ]

    self.input_channels = len(self.cluster_channel_providers) + len(self.track_channel_providers)

  def get_event(self, index):
    if self.use_cache and index in self.cache:
      return self.cache[index]
    
    start = time()
    fields = [(self.data if self.preloaded else self.raw_data)[field][index] for field in self.dataset_fields]
    load_time = time()
    item = Event(*fields, **self._fields)
    item_time = time()
    if self.use_cache:
      self.cache[index] = item
    cache_time = time()
    print(f'Load: {load_time - start:.4f}s, Item: {item_time - load_time:.4f}s, Cache: {cache_time - item_time:.4f}s')
    return item

  def __getitem__(self, index):
    event = self.get_event(index)

    clusters_map = event.clusters_map(RESOLUTION, self.cluster_channel_providers)
    tracks_map = event.tracks_map(RESOLUTION, self.track_channel_providers)
    input = np.concatenate([clusters_map, tracks_map], axis=0)
    target = np.array([position.to_list() for position in event.true_position()], dtype=np.float32).flatten()[:4]
    
    if len(target) < 4:
      target = np.concatenate([target, np.zeros(4 - len(target), dtype=np.float32)])
    
    # if target is all zeros, it means that there is no tau in the event, throw error
    if np.all(target == 0):
      raise ValueError('No tau in the event #{}'.format(index))
      
    # turn inputs and target into torch tensors
    input = torch.tensor(input, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)

    return input, target

  def __len__(self):
    return len(self.data['event']) if self.preloaded else len(self.raw_data['event'])
  
  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
      iter_start = 0
      iter_end = len(self)
    else:
      per_worker = int(math.ceil(len(self) / float(worker_info.num_workers)))
      worker_id = worker_info.id
      iter_start = worker_id * per_worker
      iter_end = min(iter_start + per_worker, len(self))
    return iter(range(iter_start, iter_end))
  
  def clear_cache (self):
    # release memorys
    self.cache = {}
    gc.collect()
  
  def post_processing(self, x):
    x[0::2] = transform_into_range(x[..., 0::2], ETA_RANGE)
    x[1::2] = transform_into_range(x[..., 1::2], PHI_RANGE)
    return x
  
  # io operations

  def save (self, filename):
    with h5py.File(filename, 'w') as f:
      for key in self.raw_data:
        f.create_dataset(key, data=self.raw_data[key])

  def load(self, source_file):
    self.source_file = source_file
    self.raw_data = h5py.File(source_file, 'r')

    self._fields = { f'{field}_fields': [(name, python_name_from_dtype_name(name)) for name in self.raw_data[field].dtype.names] for field in self.dataset_fields }

  def full_preload (self):
    self.preloaded = True
    self.data = {}
    def preload (next):
      for field in self.dataset_fields:
        self.data[field] = self.raw_data[field][:]
        next()
    long_operation(preload, max=len(self.dataset_fields), message='Preloading')
