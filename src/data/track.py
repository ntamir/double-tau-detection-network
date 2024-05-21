import numpy as np

class Track:
  def __init__ (self, track, fields):
    for name, python_name in fields:
      setattr(self, python_name, track[name])

  def position (self):
    return np.array([self.eta, self.phi])