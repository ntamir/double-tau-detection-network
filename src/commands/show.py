from visualization import EventVisualizer, DatasetVisualizer

# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from datetime import datetime
# from progress.bar import IncrementalBar

# import matplotlib.colors as colors
# import matplotlib.patches as patches

# from data.dataset import *
# from data.event import *
# from utils import *

# PLOT_SIZE = 6

# def print_fields (dataset):
#   print('Cluster fields:')
#   [print(python_name) for _, python_name in dataset._cluster_fields]
#   print()
#   print('Track fields:')
#   [print(python_name) for _, python_name in dataset._track_fields]
#   print()
#   print('Truth fields:')
#   [print(python_name) for _, python_name in dataset._truth_fields]

# def field_value (event, field):
#   if field in event.__dict__:
#     return event.__dict__[field]
#   if field in event.clusters[0].__dict__:
#     return [cluster[0].__dict__[field] for cluster in event.clusters]
#   if field in event.tracks[0].__dict__:
#     return [track[0].__dict__[field] for track in event.tracks]
#   if field in event.truths[0].__dict__:
#     return [truth[0].__dict__[field] for truth in event.truths]
#   return None

# def print_sample_input_and_target (dataset, index):
#   (input, target) = dataset[index]
#   print('Input:')
#   print(input)
#   print('Target:')
#   print(target)

# def plot_event_track_histogram (dataset, index, **options):
#   event_data = dataset.get_event(index)
#   track_positions = np.array([track.position() for track in event_data.tracks if track.eta < 2.5 and track.eta > -2.5 and track.phi < 3.2 and track.phi > -3.2])
#   plt.figure(figsize=[PLOT_SIZE, PLOT_SIZE])
#   eta_phi_histogram(plt, track_positions, **options)
#   plt.show()

# def plot_event_cluster_histogram (dataset, index, **options):
#   event_data = dataset.get_event(index)
#   cluster_positions = np.array([cluster.position() for cluster in event_data.clusters if cluster.cal_eta < 2.5 and cluster.cal_eta > -2.5 and cluster.cal_phi < 3.2 and cluster.cal_phi > -3.2])
#   plt.figure(figsize=[PLOT_SIZE, PLOT_SIZE])
#   eta_phi_histogram(plt, cluster_positions, **options)
#   plt.show()

# def plot_event_pt_histogram (dataset, index, **options):
#   event_data = dataset.get_event(index)
#   (clusters_pts, track_pts) = event_data.clusters_and_tracks_momentum_map(dataset.resolution)
#   _fig, axes = plt.subplots(1, 2, figsize=[2 * PLOT_SIZE, PLOT_SIZE])

#   for index, topule in enumerate([('Clusters', clusters_pts), ('Tracks', track_pts)]):
#     (title, data) = topule
#     hist = eta_phi_multivalued_histogram(axes[index], data, **options)
#     bounding_boxes = [patches.Circle((eta, phi), 0.2, linewidth=1, edgecolor='red', facecolor='none') for (eta, phi) in event_data.true_position()]
#     for box in bounding_boxes:
#       axes[index].add_patch(box)
#     axes[index].set_title(title)
#     plt.colorbar(hist[3])

#   plt.show()

# def plot_event_histrograms (dataset, index, **options):
#   event_data = dataset.get_event(index)
#   track_positions = np.array([track.position() for track in event_data.tracks if track.eta < 2.5 and track.eta > -2.5 and track.phi < 3.2 and track.phi > -3.2])
#   cluster_positions = np.array([cluster.position() for cluster in event_data.clusters if cluster.cal_eta < 2.5 and cluster.cal_eta > -2.5 and cluster.cal_phi < 3.2 and cluster.cal_phi > -3.2])
#   truth_positions = np.array([truth.visible_position() for truth in event_data.truths if truth.eta_vis < 2.5 and truth.eta_vis > -2.5 and truth.phi_vis < 3.2 and truth.phi_vis > -3.2])
#   plt.figure(figsize=[3 * PLOT_SIZE, PLOT_SIZE])
#   _fig, axes = plt.subplots(1, 3)
#   eta_phi_histogram(axes[0], track_positions, **options)
#   eta_phi_histogram(axes[1], cluster_positions, **options)
#   eta_phi_histogram(axes[2], truth_positions, **options)
#   plt.show()

# def plot_clusters_and_tracks_per_event (dataset, **options):
#   plt.figure(figsize=[PLOT_SIZE, PLOT_SIZE])
#   histograms_across_events(dataset, [lambda event: len(event.clusters), lambda event: len(event.tracks)], **options)
#   plt.show()

# def plot_average_interactions_per_crossing (dataset, **options):
#   plt.figure(figsize=[PLOT_SIZE, PLOT_SIZE])
#   histograms_across_events(dataset, [lambda event: event.average_interactions_per_crossing], **options)
#   plt.show()

# def histograms_across_events (dataset, callbacks, **options):
#   # build a histogram for every callback function
#   hists = [ [] for _ in range(len(callbacks)) ]
#   bar = IncrementalBar('Processing', max=len(dataset))
#   for index in range(len(dataset)):
#     bar.next()
#     event = dataset.get_event(index)
#     for i, callback in enumerate(callbacks):
#       hists[i].append(callback(event))

#   # plot all histograms
#   for hist in hists:
#     plt.hist(hist, **options)

# def histograms_across_tracks (dataset, callbacks, **options):
#   # build a histogram for every callback function
#   hists = [ [] for _ in range(len(callbacks))]
#   bar = IncrementalBar('Processing', max=len(dataset))
#   for index in range(len(dataset)):
#     bar.next()
#     event = dataset.get_event(index)
#     for i, callback in enumerate(callbacks):
#       for track in event.tracks:
#         hists[i].append(callback(track))

#   # plot all histograms
#   for hist in hists:
#     plt.hist(hist, **options)
#   print('done')

# def histograms_across_clusters (dataset, callbacks, **options):
#   # build a histogram for every callback function
#   hists = [ [] for _ in range(len(callbacks))]
#   bar = IncrementalBar('Processing', max=len(dataset))
#   for index in range(len(dataset)):
#     bar.next()
#     event = dataset.get_event(index)
#     for i, callback in enumerate(callbacks):
#       for cluster in event.clusters:
#         hists[i].append(callback(cluster))

#   # plot all histograms
#   for hist in hists:
#     plt.hist(hist, bins=range(0, 1000, 10), alpha=0.5, **options)

# def eta_phi_histogram (axes, data, **options):
#   eta_res = 100
#   phi_res = 100
#   hist = np.zeros([eta_res, phi_res])
#   for position in data:
#     eta_bin = int((position[0] + 2.5) / 5 * eta_res)
#     phi_bin = int((position[1] + 3.2) / 6.4 * phi_res)
#     hist[eta_bin, phi_bin] += 1

#   return eta_phi_multivalued_histogram(axes, hist, **options)

# def eta_phi_multivalued_histogram (axes, data, **options):
#   etas = []
#   phis = []
#   weights = []
#   for line_index, line in enumerate(data):
#     for cell_index, cell in enumerate(line):
#       if not cell == 0:
#         etas.append(line_index / 100 * 5 - 2.5)
#         phis.append(cell_index / 100 * 6.4 - 3.2)
#         weights.append(cell)
#   return axes.hist2d(etas, phis, weights=weights, bins=100, range=[[-2.5,2.5], [-3.2,3.2]], norm=colors.PowerNorm(0.5), **options)

# def number_of_tracks_and_average_interactions_heatmap (dataset, **options):
#   tracks_res = 120
#   plot_average_interactions_per_crossing_res = 120
#   hist = np.zeros([tracks_res, plot_average_interactions_per_crossing_res])
#   bar = IncrementalBar('Processing', max=len(dataset))
#   for index in range(len(dataset)):
#     bar.next()
#     event = dataset.get_event(index)
#     tracks_bin = int((len(event.tracks) - 1) / 1200 * tracks_res)
#     average_interactions_per_crossing_bin = int(event.average_interactions_per_crossing)
#     hist[tracks_bin, average_interactions_per_crossing_bin] += 1

#   _fig, ax = plt.subplots(figsize=[10,10])

#   ax.imshow(hist,
#             origin='lower',
#             interpolation='bilinear',
#             **options
#             )

#   y_locs, y_labels = plt.yticks()
#   y_labels = [int(float(loc / 120) * 1200) for loc in y_locs if loc >= 0 and loc < 120]
#   y_locs = [loc for loc in y_locs if loc >= 0 and loc < 120]
#   plt.yticks(y_locs, y_labels)
#   plt.show()

# def clusters_per_pixel (dataset, **options):
#   resolution = dataset.resolution
#   hist = np.zeros(len(dataset) * resolution * resolution)
#   bar = IncrementalBar('Processing', max=len(dataset))
#   for index in range(len(dataset)):
#     bar.next()
#     event = dataset.get_event(index)
#     for cluster in event.clusters:
#       x, y = relative_position(cluster.position())
#       if (x > 0 and x < 1 and y > 0 and y < 1):
#         hist[index * resolution * resolution + int(x * resolution) * resolution + int(y * resolution)] += 1
  
#   print('show histogram')
#   plt.hist(hist, bins=range(0, 10, 1), **options)
#   plt.show()

# def show_normalization_histograms (dataset, **options):
#   normalization_factors_length = len(dataset.get_event(0).normalization_factors())
#   hists = np.zeros((normalization_factors_length, len(dataset)))
#   bar = IncrementalBar('Processing', max=len(dataset))
#   for index in range(len(dataset)):
#     bar.next()
#     event = dataset.get_event(index)
#     for i, factor in enumerate(event.normalization_factors()):
#       hists[i, index] = factor
#   # create multiple histograms
#   fig, axes = plt.subplots(2, 8, figsize=[PLOT_SIZE, PLOT_SIZE * normalization_factors_length])
#   clusters_titles = [f"{title} mean" for title in FIELDS_TO_NORMALIZE['clusters']] + [f"{title} variance" for title in FIELDS_TO_NORMALIZE['clusters']]
#   tracks_titles = [f"{title} mean" for title in FIELDS_TO_NORMALIZE['tracks']] + [f"{title} variance" for title in FIELDS_TO_NORMALIZE['tracks']]
#   titles = clusters_titles + tracks_titles
#   for index, hist in enumerate(hists):
#     axes[int(index / 8)][index % 8].hist(hist, bins=100, **options)
#     axes[int(index / 8)][index % 8].set_title(titles[index])
#     axes[int(index / 8)][index % 8].figure.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', f'{titles[index]}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.png'))
#   plt.show()

# def show (dataset, graph, params):
#   if graph == 'fields':
#     print_fields(dataset)
#     return
  
#   if graph == 'event_track_histogram':
#     plot_event_track_histogram(dataset, int(params[0]))
#     return
  
#   if graph == 'event_cluster_histogram':
#     plot_event_cluster_histogram(dataset, int(params[0]))
#     return
  
#   if graph == 'event_histograms':
#     plot_event_histrograms(dataset, int(params[0]))
#     return
  
#   if graph == 'clusters_and_tracks_per_event':
#     plot_clusters_and_tracks_per_event(dataset, log=True)
#     return
  
#   if graph == 'average_interactions_per_crossing':
#     plot_average_interactions_per_crossing(dataset, log=True)
#     return
  
#   if graph == 'number_of_tracks_and_average_interactions_heatmap':
#     number_of_tracks_and_average_interactions_heatmap(dataset)
#     return
  
#   if graph == 'clusters_per_pixel':
#     clusters_per_pixel(dataset, log=True)
#     return
  
#   if graph == 'event_pt_histogram':
#     plot_event_pt_histogram(dataset, int(params[0]))
#     return
  
#   if graph == 'sample_input_and_target':
#     print_sample_input_and_target(dataset, int(params[0]))
#     return
  
#   if graph == 'field_value':
#     print(field_value(dataset.get_event(int(params[1])), params[0]))
    
#   if graph == 'field_across_events':
#     field = params[0]
#     histograms_across_events(dataset, [lambda event: field_value(event, field)], log=True)
#     plt.show()
#     return
  
#   if graph == 'normalization_histograms':
#     show_normalization_histograms(dataset)
#     return
  
#   print(f'Unknown graph: {graph}')

def show (dataset, scope, params):
  if scope == 'event':
    try:
      event = dataset.get_event(int(params[0]))
    except IndexError:
      exit('Event not found')

    if event is None:
      exit('Event not found')
    
    visualizer = EventVisualizer(event)
    if params[1] == 'map':
      visualizer.map()
      return
    
    exit(f'Unknown event command: {params[1]}')
  
  if scope == 'dataset':
    visualizer = DatasetVisualizer(dataset)
    if params[0] == 'histogram':
      visualizer.histogram(visualizer.histogram_fields[params[1]])
      return
    
    exit(f'Unknown dataset command: {params[0]}')
  
  
  exit(f'Unknown scope: {scope}')