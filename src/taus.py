import sys
from dataset import EventsDataset
from model.main import OriginFindingModule
from utils import datafile_path

from commands.show import show
from commands.test import test_module
from commands.train import train
from commands.detect import detect
from commands.eval import evaluate

RESOLUTION = 100

MODELS = {
  'origin_finding': lambda: OriginFindingModule(RESOLUTION),
}

if __name__ == '__main__':
  command = sys.argv[1]
  filename = sys.argv[2]
  dataset = EventsDataset(datafile_path(filename), RESOLUTION)

  if command == 'show':
    graph_name = sys.argv[3]
    params = sys.argv[4:]
    show(dataset, graph_name, params)
    exit()

  if command == 'test':
    module = MODELS[sys.argv[3]]()
    params = sys.argv[4:]
    test_module(dataset, module, params)
    exit()

  if command == 'train':
    module = MODELS[sys.argv[3]]()
    output = sys.argv[4]
    params = sys.argv[5:]
    train(dataset, module, output, params)
    exit()

  if command == 'detect':
    module = MODELS[sys.argv[3]]()
    weights_file = sys.argv[4]
    params = sys.argv[5:]
    detect(dataset, module, weights_file, params)
    exit()

  if command == 'evaluate':
    module = MODELS[sys.argv[3]]()
    weights_file = sys.argv[4]
    params = sys.argv[5:]
    evaluate(dataset, module, weights_file, params)
    exit()

  print(f'Unknown command: {command}')