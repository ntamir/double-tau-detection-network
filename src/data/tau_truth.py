import numpy as np

class Truth:
  def __init__ (self, truth, fields):
    for name, python_name in fields:
      setattr(self, python_name, truth[name])

  def visible_position (self):
    return np.array([self.eta_vis, self.phi_vis])
  
  def invisible_position (self):
    return np.array([self.eta_invis, self.phi_invis])
  
  def visible_four_momentum (self):
    return np.array([self.pt_vis, self.eta_vis, self.phi_vis, self.m_vis])
  
  def invisible_four_momentum (self):
    return np.array([self.pt_invis, self.eta_invis, self.phi_invis, self.m_invis])
  
  def total_four_momentum (self):
    return self.visible_four_momentum() + self.invisible_four_momentum()
