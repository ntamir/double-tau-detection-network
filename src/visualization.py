import matplotlib.pyplot as plt
import numpy as np

def plot_results (outputs, targets):
  # draw two plots side by side
  fig, axs = plt.subplots(1, 2, figsize=(10, 5))
  arrows_on_eta_phi_plot(outputs, targets, axs[0], color='blue')
  distances_histogram(outputs, targets, axs[1])
  plt.show()

def arrows_on_eta_phi_plot (starts, ends, ax, **kwargs):
  def arrow_with_color (x, y, dx, dy, **kwargs):
    distance_normalized = min(1, max(0, 0.5 + 0.5 * np.linalg.norm([dx, dy]) / 2))
    color = (distance_normalized, 0, 1 - distance_normalized, 0.6)
    ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc=color, ec=color, **kwargs)

  for start, end in zip(starts, ends):
    arrow_with_color(start[1], start[0], end[1]-start[1], end[0]-start[0], **kwargs)

  ax.set_xlabel('phi')
  ax.set_ylabel('eta')
  ax.set_xlim(-3.2, 3.2)
  ax.set_ylim(-2.5, 2.5)

def distances_histogram (starts, ends, ax):
  def distance (start, end):
    if np.sign(end[1]) != np.sign(start[1]):
      return np.linalg.norm([start[0] - end[0], 2 * np.pi - abs(start[1] - end[1])])
    else:
      return np.linalg.norm([start[0] - end[0], start[1] - end[1]])
  distances = [distance(start, end) for start, end in zip(starts, ends)]
  ax.hist(distances, bins=100)
  ax.set_xlabel('distance')
  ax.set_ylabel('count')

if __name__ == '__main__':
  output_xs = np.random.rand(100) * (2 * np.pi - 0.1) - np.pi
  output_ys = np.random.rand(100) * 5 - 2.5
  outputs = np.array([[y, x] for x, y in zip(output_xs, output_ys)])
  target_xs = np.random.rand(100) * (2 * np.pi - 0.1) - np.pi
  target_ys = np.random.rand(100) * 5 - 2.5
  targets = np.array([[y, x] for x, y in zip(target_xs, target_ys)])
  plot_results(outputs, targets)