from matplotlib import pyplot as plt
import numpy as np

from settings import HISTOGRAM_BINS
from utils import long_operation

class DatasetVisualizer:
  def __init__ (self, dataset):
    self.dataset = dataset

  def histogram (self, callback, subfields=None):
    def load (next):
      hist = [] if subfields is None else { subfield: [] for subfield in subfields }
      for i in range(len(self.dataset)):
        event = self.dataset.get_event(i)
        datum = callback(event)
        if subfields is None:
          hist.append(datum)
        else:
          for subfield in subfields:
            hist[subfield].extend(datum[subfield])
        next()
      return hist
    
    result = long_operation(load, max=len(self.dataset))
    if subfields is None:
      hist = np.array(result).flatten().tolist()
      plt.hist(hist, bins=HISTOGRAM_BINS, edgecolor='black')
    else:
      axes, fig = plt.subplots(len(subfields))
      for subfield in subfields:
        hist = np.array(result[subfield]).flatten().tolist()
        fig[subfields.index(subfield)].hist(hist, bins=HISTOGRAM_BINS, edgecolor='black')
        axes[subfields.index(subfield)].set_title(subfield)
    plt.show()

  def print_fields (self):
    print('Cluster fields:')
    [print(python_name) for _, python_name in self.dataset._cluster_fields]
    print()
    print('Track fields:')
    [print(python_name) for _, python_name in self.dataset._track_fields]
    print()
    print('Truth fields:')
    [print(python_name) for _, python_name in self.dataset._truth_fields]

  histogram_fields = {
    'average_interaction_per_crossing': lambda event: event.average_interactions_per_crossing,

    'cluster_count': lambda event: len(event.clusters),
    'track_count': lambda event: len(event.tracks),
    'truth_count': lambda event: len(event.truths),

    'cluster_pt': lambda event: [cluster.momentum().p_t for cluster in event.clusters],
    'track_pt': lambda event: [track.momentum().p_t for track in event.tracks],
    'truth_pt': lambda event: [truth.visible_momentum().p_t for truth in event.truths],

    'normlization_factors': lambda event: event.normalization_factors(),
  }