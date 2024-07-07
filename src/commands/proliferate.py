import numpy as np
from progress.bar import IncrementalBar

from utils import datafile_path

def proliferate (dataset, factor):
  initial_count = len(dataset)
  new_data = { 'event': [], 'clusters': [], 'tracks': [], 'truthTaus': [] }
  get = lambda index: { key: dataset.raw_data[key][index] for key in dataset.raw_data }
  add = lambda event: [new_data[key].append(event[key]) for key in new_data]

  print('starting proliferation')
  bar = IncrementalBar('Processing', max=len(dataset))
  for index in range(len(dataset)):
    original = get(index)
    add(original)
    for i in range(factor - 1):
      # deep copy of the event
      copy = { key: np.copy(original[key]) for key in original }
      # if random > 0.5 is odd, flip eta
      if np.random.rand() > 0.5:
        copy['clusters']['Clusters.calEta'] = -copy['clusters']['Clusters.calEta']
        copy['tracks']['Tracks.eta'] = -copy['tracks']['Tracks.eta']
        copy['truthTaus']['TruthTaus.eta_vis'] = -copy['truthTaus']['TruthTaus.eta_vis']
      # rotate phi by a random angle
      random_angle = np.random.rand() * 2 * np.pi
      copy['clusters']['Clusters.calPhi'] = (copy['clusters']['Clusters.calPhi'] + random_angle) % (2 * np.pi)
      copy['tracks']['Tracks.phi'] = (copy['tracks']['Tracks.phi'] + random_angle) % (2 * np.pi)
      copy['truthTaus']['TruthTaus.phi_vis'] = (copy['truthTaus']['TruthTaus.phi_vis'] + random_angle) % (2 * np.pi)

      add(copy)
    bar.next()
  bar.finish()
  dataset.raw_data = new_data
  print(f'Proliferated {initial_count} events by factor of {factor} to {len(dataset)}')
  dataset.save(datafile_path(dataset.source_file + '_x' + str(factor)))
  print(f'Saved to {datafile_path(dataset.source_file + "_x" + str(factor))}')

