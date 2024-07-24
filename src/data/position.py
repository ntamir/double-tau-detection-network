import numpy as np
from collections.abc import Iterable

from settings import ETA_RANGE, PHI_RANGE

class Position (Iterable):
  def __init__ (self, eta, phi):
    self.eta = eta
    self.phi = phi
  
  def relative (self):
    return np.array([(self.eta - ETA_RANGE[0]) / (ETA_RANGE[1] - ETA_RANGE[0]), (self.phi - PHI_RANGE[0]) / (PHI_RANGE[1] - PHI_RANGE[0])])
  
  def in_range (self):
    rel_pos = self.relative()
    return rel_pos[0] >= 0 and rel_pos[0] <= 1 and rel_pos[1] >= 0 and rel_pos[1] <= 1
  
  def to_list (self):
    return np.array([self.eta, self.phi], dtype=np.float32)
  
  def __iter__ (self):
    return iter([self.eta, self.phi])
  
  def __str__ (self):
    return f'Position(eta={self.eta}, phi={self.phi})'
  
  def __repr__ (self):
    return str(self)
  
  @staticmethod
  def from_relative (position):
    return Position(position[0] * (ETA_RANGE[1] - ETA_RANGE[0]) + ETA_RANGE[0], position[1] * (PHI_RANGE[1] - PHI_RANGE[0]) + PHI_RANGE[0])