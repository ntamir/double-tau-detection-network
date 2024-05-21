import sys
import matplotlib.pyplot as plt
import numpy as np
import time

from dataset import *
from utils import *

def print_fields (dataset):
  print('Cluster fields:')
  [print(python_name) for _, python_name in dataset._cluster_fields]
  print()
  print('Track fields:')
  [print(python_name) for _, python_name in dataset._track_fields]
  print()
  print('Truth fields:')
  [print(python_name) for _, python_name in dataset._truth_fields]

def plot_event_track_histogram (dataset, index, **options):
  event_data = dataset[index]
  track_positions = np.array([track.position() for track in event_data.tracks if track.eta < 2.5 and track.eta > -2.5 and track.phi < 3.2 and track.phi > -3.2])
  eta_phi_histogram(plt, track_positions, **options)

def plot_event_cluster_histogram (dataset, index, **options):
  event_data = dataset[index]
  cluster_positions = np.array([cluster.position() for cluster in event_data.clusters if cluster.cal_eta < 2.5 and cluster.cal_eta > -2.5 and cluster.cal_phi < 3.2 and cluster.cal_phi > -3.2])
  eta_phi_histogram(plt, cluster_positions, **options)

def plot_event_histrograms (dataset, index, **options):
  event_data = dataset[index]
  track_positions = np.array([track.position() for track in event_data.tracks if track.eta < 2.5 and track.eta > -2.5 and track.phi < 3.2 and track.phi > -3.2])
  cluster_positions = np.array([cluster.position() for cluster in event_data.clusters if cluster.cal_eta < 2.5 and cluster.cal_eta > -2.5 and cluster.cal_phi < 3.2 and cluster.cal_phi > -3.2])
  truth_positions = np.array([truth.visible_position() for truth in event_data.truths if truth.eta_vis < 2.5 and truth.eta_vis > -2.5 and truth.phi_vis < 3.2 and truth.phi_vis > -3.2])
  _fig, axes = plt.subplots(1, 3)
  eta_phi_histogram(axes[0], track_positions, **options)
  eta_phi_histogram(axes[1], cluster_positions, **options)
  eta_phi_histogram(axes[2], truth_positions, **options)

def plot_clusters_and_tracks_per_event (dataset, **options):
  histograms_across_events(dataset, [lambda event: len(event.clusters), lambda event: len(event.tracks)], **options)

def plot_average_interactions_per_crossing (dataset, **options):
  histograms_across_events(dataset, [lambda event: event.average_interactions_per_crossing], **options)

def histograms_across_events (dataset, callbacks, **options):
  # build a histogram for every callback function
  hists = [ [] for _ in range(len(callbacks)) ]
  for index in range(len(dataset)):
    if index % 1000 == 0:
      print(f'event {index}')
    event = dataset[index]
    for i, callback in enumerate(callbacks):
      hists[i].append(callback(event))

  # plot all histograms
  for hist in hists:
    plt.hist(hist, **options)

def histograms_across_tracks (dataset, callbacks, **options):
  # build a histogram for every callback function
  hists = [ [] for _ in range(len(callbacks))]
  for index in range(len(dataset)):
    if index % 1000 == 0:
      print(f'event {index}')
    event = dataset[index]
    for i, callback in enumerate(callbacks):
      for track in event.tracks:
        hists[i].append(callback(track))

  # plot all histograms
  for hist in hists:
    plt.hist(hist, **options)
  print('done')

def histograms_across_clusters (dataset, callbacks, **options):
  # build a histogram for every callback function
  hists = [ [] for _ in range(len(callbacks))]
  for index in range(len(dataset)):
    if index % 1000 == 0:
      print(f'event {index}')
    event = dataset[index]
    for i, callback in enumerate(callbacks):
      for cluster in event.clusters:
        hists[i].append(callback(cluster))

  # plot all histograms
  for hist in hists:
    plt.hist(hist, bins=range(0, 1000, 10), alpha=0.5, **options)

def eta_phi_histogram (axes, data, **options):
  eta_res = 100
  phi_res = 100
  hist = np.zeros([eta_res, phi_res])
  for position in data:
    eta_bin = int((position[0] + 2.5) / 5 * eta_res)
    phi_bin = int((position[1] + 3.2) / 6.4 * phi_res)
    hist[eta_bin, phi_bin] += 1

  axes.imshow(hist, origin='lower', interpolation='bilinear', **options)

  x_ticks = 5
  y_ticks = 5
  axes.set_xticks([loc * 100 / (x_ticks - 1) for loc in range(0, x_ticks)])
  axes.set_xticklabels([float(int((float(loc / (x_ticks - 1)) * 5 - 2.5) * 10)) / 10 for loc in range(0, x_ticks)])

  axes.set_yticks([loc * 100 / (y_ticks - 1) for loc in range(0, y_ticks)])
  axes.set_yticklabels([float(int((float(loc / (y_ticks - 1)) * 6.4 - 3.2) * 10)) / 10 for loc in range(0, y_ticks)])

def number_of_tracks_and_average_interactions_heatmap (dataset, **options):
  tracks_res = 120
  plot_average_interactions_per_crossing_res = 120
  hist = np.zeros([tracks_res, plot_average_interactions_per_crossing_res])
  for index, event in enumerate(dataset):
    if index % 1000 == 0:
      print(f'event {index}')
    tracks_bin = int((len(event.tracks) - 1) / 1200 * tracks_res)
    average_interactions_per_crossing_bin = int(event.average_interactions_per_crossing)
    hist[tracks_bin, average_interactions_per_crossing_bin] += 1

  _fig, ax = plt.subplots(figsize=[10,10])

  ax.imshow(hist,
            origin='lower',
            interpolation='bilinear',
            **options
            )

  y_locs, y_labels = plt.yticks()
  y_labels = [int(float(loc / 120) * 1200) for loc in y_locs if loc >= 0 and loc < 120]
  y_locs = [loc for loc in y_locs if loc >= 0 and loc < 120]
  plt.yticks(y_locs, y_labels)

if __name__ == '__main__':
  start_time = time.time()
  dataset = EventsDataset(sys.argv[1])
  print(f'Loaded dataset with {len(dataset)} events')
  # print(dataset[0].true_position())
  # print(dataset[0].map(3))
  print_fields(dataset)
  # plot_event_histrograms(dataset, 3141, cmap='hot')
  # number_of_tracks_and_average_interactions_heatmap(dataset, cmap='hot')
  # end_time = time.time()
  # print(f'Total time: {end_time - start_time}')
  # print(f'Time per event: {(end_time - start_time) / len(dataset)}')
  plt.show()