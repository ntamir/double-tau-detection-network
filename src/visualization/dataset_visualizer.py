from matplotlib import pyplot as plt
import numpy as np

from settings import HISTOGRAM_BINS
from utils import long_operation

class DatasetVisualizer:
  def __init__ (self, dataset):
    self.dataset = dataset

  def histogram (self, callback):
    def load (next):
      hist = [] 
      for i in range(len(self.dataset)):
        hist.append(callback(self.dataset.get_event(i)))
        next()
      return np.array(hist).flatten().tolist()

    hist = long_operation(load, max=len(self.dataset))
    plt.hist(hist, bins=HISTOGRAM_BINS, edgecolor='black')
    plt.show()

  histogram_fields = {
    'cluster_count': lambda event: len(event.clusters),
    'track_count': lambda event: len(event.tracks),
    'truth_count': lambda event: len(event.truths),

    'cluster_pt': lambda event: [cluster.momentum().p_t for cluster in event.clusters],
    'track_pt': lambda event: [track.momentum().p_t for track in event.tracks],
    'truth_pt': lambda event: [truth.visible_momentum().p_t for truth in event.truths],
  }