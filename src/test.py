import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data import random_split

from dataset import EventsDataset
from model import OriginFindingModule

# try and train the OriginFindingModule with the provided data
def train_origin_finding_module(dataset):
  print('initializing training...')
  train_size = int(0.8 * len(dataset))
  test_size = len(dataset) - train_size
  train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

  # Create the dataloaders
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
  print('done')

  # Create the model
  print('creating model...')
  model = OriginFindingModule()
  print('done')

  # Create the optimizer
  optimizer = Adam(model.parameters(), lr=0.001)

  # Create the loss function
  criterion = nn.MSELoss()

  # Train the model
  print('starting training...')
  for epoch in range(10):
    print(f'Epoch {epoch}')
    train(train_loader, model, criterion, optimizer, epoch)
  print('done')

  # Test the model
  print('starting testing...')
  test(test_loader, model, criterion)
  print('done')

# train the model
def train(train_loader, model, criterion, optimizer, epoch):
  model.train()
  for batch_idx, (input, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % 10 == 0:
      print(f'Train Epoch: {epoch} [{batch_idx * len(input)}/{len(train_loader.dataset)}'
            f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item()}')
      
# test the model
def test(test_loader, model, criterion):
  model.eval()
  test_loss = 0
  with torch.no_grad():
    for batch_idx, (input, target) in enumerate(test_loader):
      output = model(input)
      test_loss += criterion(output, target).item()
      test_loss /= len(test_loader.dataset)
  print(f'\nTest set: Average loss: {test_loss:.4f}\n')
    

if __name__ == '__main__':
  dataset = EventsDataset(sys.argv[1])
  print(f'Loaded dataset with {len(dataset)} events')
  train_origin_finding_module(dataset)