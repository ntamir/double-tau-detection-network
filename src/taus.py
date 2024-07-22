import sys
import time
from data.dataset import EventsDataset
from model.main import MainModel
from utils import datafile_path, modelfile_path

from commands.show import show
from commands.train import train_module
from commands.detect import detect
from commands.proliferate import proliferate

from settings import DATA_FILE

if __name__ == '__main__':
  command = sys.argv[1]
  dataset = EventsDataset(datafile_path(DATA_FILE))
  module = MainModel()

  if command == 'show':
    scope = sys.argv[2]
    params = sys.argv[3:]
    show(dataset, scope, params)
    exit()

  if command == 'train':
    output = modelfile_path(sys.argv[3]) if len(sys.argv) > 3 else modelfile_path('model_' + str(round(time.time() * 1000)))
    train_module(dataset, module, output)
    exit()

  if command == 'detect':
    model_file = modelfile_path(sys.argv[2])
    params = sys.argv[4:]
    detect(dataset, module, model_file)
    exit()

  if command == 'proliferate':
    factor = int(sys.argv[2])
    proliferate(dataset, factor)
    exit()

  print(f'Unknown command: {command}')