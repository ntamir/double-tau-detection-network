import matplotlib.pyplot as plt
import numpy as np

def plot_results (outputs, targets):
  arrows_on_eta_phi_plot(outputs, targets, color='blue')

def arrows_on_eta_phi_plot (starts, ends, **kwargs):
  ax = plt.gca()
  for start, end in zip(starts, ends):
    # color the arrow based on the length of the arrow: the longer the arrow, the redder it is
    distance_normalized = min(1, max(0, 0.5 + 0.5 * (np.linalg.norm(end - start) - 0.5) / 2))
    color = (distance_normalized, 0, 1 - distance_normalized, 0.6)
    ax.arrow(start[1], start[0], end[1]-start[1], end[0]-start[0], head_width=0.05, head_length=0.1, fc=color, ec=color, **kwargs)
  plt.xlabel('phi')
  plt.ylabel('eta')
  plt.xlim(-3.2, 3.2)
  plt.ylim(-2.5, 2.5)
  plt.show()