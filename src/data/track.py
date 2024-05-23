import numpy as np
from pylorentz import Momentum4

class Track:
  def __init__ (self, track, fields):
    for name, python_name in fields:
      setattr(self, python_name, track[name])

  def position (self):
    return np.array([self.eta, self.phi])
  
  def momentum (self):
    return Momentum4.m_eta_phi_pt(0, self.eta, self.phi, self.pt)