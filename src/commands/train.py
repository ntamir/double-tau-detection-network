import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data import random_split

import matplotlib.pyplot as plt

from utils import *

def train_module(dataset, model, output):
  print('initializing')
  train_size = int(0.7 * len(dataset))
  validation_size = int(0.2 * len(dataset))
  test_size = len(dataset) - train_size - validation_size
  train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

  # Create the dataloaders
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

  # Create the optimizer
  optimizer = Adam(model.parameters(), lr=0.001)

  # Create the loss function
  criterion = nn.MSELoss(reduction='sum')

  # Train the model
  print('training')
  best_validation_loss = float('inf')
  best_model = None
  losses = []
  for epoch in range(10):
    print(f'Epoch {epoch}')
    training_loss = train(train_loader, model, criterion, optimizer, epoch)
    validation_loss = validate(validation_loader, model, criterion)
    if validation_loss < best_validation_loss:
      best_validation_loss = validation_loss
      best_model = model.state_dict()
    losses.append((training_loss, validation_loss))

  # Plot the losses as a function of epoch
  print(losses)
  plt.plot(losses)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.show(block=False)

  # Load the best model
  model.load_state_dict(best_model)

  # Test the best model
  print('testing')
  test(test_loader, model, criterion)

  # Save the model
  torch.save(model.state_dict(), output)

# train the model
def train(train_loader, model, criterion, optimizer, epoch):
  total_loss = 0
  model.train()
  for batch_idx, (input, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    if batch_idx % 10 == 0:
      print(f'training [{batch_idx * len(input)}/{len(train_loader.dataset)}'
            f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(input):.4f}')
  return total_loss / len(train_loader.dataset)

# validate the model
def validate(val_loader, model, criterion):
  model.eval()
  total_loss = 0
  with torch.no_grad():
    for batch_idx, (input, target) in enumerate(val_loader):
      output = model(input)
      loss = criterion(output, target)
      total_loss += loss.item()
      if batch_idx % 10 == 0:
        print(f'validating [{batch_idx * len(input)}/{len(val_loader.dataset)}'
              f'({100. * batch_idx / len(val_loader):.0f}%)]\tLoss: {loss.item() / len(input):.4f}')
  return total_loss / len(val_loader.dataset)

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
