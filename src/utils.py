import numpy as np
import matplotlib.pyplot as plt

def python_names_from_dtype(dtype):
  return [python_name_from_dtype_name(name) for name in dtype.names]

def python_name_from_dtype_name(dtype_name):
  name = dtype_name.split('.')[-1]
  if all([c.islower() or c == '_' for c in name]) or all([c.isupper() or c == '_' for c in name]):
    return name
  new_name = (name[0].lower() + ''.join([c if c.islower() else (c if next_char.isupper() else f'_{c.lower()}') if prev_char.isupper() else (f'_{c}' if next_char.isupper() else f'_{c.lower()}') for prev_char, c, next_char in zip([''] + list(name[1:]), list(name[1:]), list(name[2:]) + [''])])).split('.')[-1].replace('__', '_')
  return new_name if new_name[0] != '_' else new_name[1:]

# turns the eta position from -pi to pi into a position from 0 to 1 and the phi position from -2.5 to 2.5 into a position from 0 to 1
def relative_position(position):
  return np.array([(position[0] + np.pi) / (2 * np.pi), (position[1] + 2.5) / 5])