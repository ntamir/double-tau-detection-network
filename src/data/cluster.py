import numpy as np
from pylorentz import Momentum4

class Cluster:
  def __init__ (self, cluster, fields):
    for name, python_name in fields:
      setattr(self, python_name, cluster[name])

  def position (self):
    return np.array([self.cal_eta, self.cal_phi])
  
  def momentum (self):
    return Momentum4.e_m_eta_phi(self.cal_e, 0, self.cal_eta, self.cal_phi)
  