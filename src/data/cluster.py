import numpy as np

class Cluster:
  def __init__ (self, cluster, fields):
    for name, python_name in fields:
      setattr(self, python_name, cluster[name])

  def position (self):
    return np.array([self.cal_eta, self.cal_phi])
  