import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from progress.bar import IncrementalBar
import numpy as np

from visualization import plot_results

EPOCHS = 10
BATCH_SIZE = 512

TRAINING_PERCENTAGE = 0.7
VALIDATION_PERCENTAGE = 0.2
TEST_ARROWS_PERCENTAGE = 0.1 

def train_module(dataset, model, output):
  start_time = time.time()
  # Create the dataloaders
  train_loader, validation_loader, test_loader = init_dataloaders(dataset)
  print(f'training over {len(train_loader.dataset)} samples, validating over {len(validation_loader.dataset)} samples, testing over {len(test_loader.dataset)} samples')

  # Create the optimizer
  optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

  # Create the loss function
  criterion = nn.MSELoss()

  # Train the model
  print('training')
  best_validation_loss = float('inf')
  best_model = None
  losses = []
  epoch_start_times = []
  for epoch in range(EPOCHS):
    epoch_start_times.append(time.time())
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    training_loss = train(train_loader, model, criterion, optimizer)
    validation_loss = validate(validation_loader, model, criterion)
    if validation_loss < best_validation_loss:
      best_validation_loss = validation_loss
      best_model = model.state_dict()
    losses.append((training_loss, validation_loss))

  # Load the best model
  model.load_state_dict(best_model)

  # Test the best model
  test_start_time = time.time()
  if len(test_loader) > 0:
    print('testing')
    test(test_loader, model, criterion, dataset)
  else:
    print('skipping testing')

  # Save the model
  torch.save(model.state_dict(), output)

  # print summary
  print(f'Time: {time.time() - start_time} (trainig: {sum([epoch_start_times[i + 1] - epoch_start_times[i] for i in range(len(epoch_start_times) - 1)])}, testing: {time.time() - test_start_time})')
  print(f'Average Epoch Time: {sum([epoch_start_times[i + 1] - epoch_start_times[i] for i in range(len(epoch_start_times) - 1)]) / len(epoch_start_times)}')
  print(f'Best Validation Loss: {best_validation_loss}')

  # Plot the losses as a function of epoch
  plt.plot(range(EPOCHS), [loss[0] for loss in losses], label='Training Loss')
  plt.plot(range(EPOCHS), [loss[1] for loss in losses], label='Validation Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.yscale('log')
  plt.legend()
  plt.savefig(output + '.png')
  plt.show()


def init_dataloaders (dataset):
  train_size = int(len(dataset) * TRAINING_PERCENTAGE)
  validation_size = int(len(dataset) * VALIDATION_PERCENTAGE)
  test_size = len(dataset) - train_size - validation_size

  train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])
  
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
  
  return train_loader, validation_loader, test_loader

# train the model
def train(train_loader, model, criterion, optimizer):
  total_loss = 0
  model.train()
  bar = IncrementalBar('Processing', max=len(train_loader))
  bar.suffix = f'[0/{len(train_loader.dataset)} %(percent).1f%%]'
  for batch_idx, (inputs, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(*inputs)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    bar.next()
    bar.suffix = f'[{(batch_idx + 1) * BATCH_SIZE}/{len(train_loader.dataset)} %(percent).1f%%] Loss: {loss.item() / BATCH_SIZE:.4f}'
  bar.suffix = f'Loss: {total_loss / len(train_loader)}'
  bar.finish()
  return total_loss / len(train_loader)

# validate the model
def validate(val_loader, model, criterion):
  model.eval()
  total_loss = 0
  bar = IncrementalBar('Processing', max=len(val_loader))
  bar.suffix = f'[0/{len(val_loader.dataset)} %(percent).1f%%]'
  with torch.no_grad():
    for batch_idx, (inputs, target) in enumerate(val_loader):
      output = model(*inputs)
      loss = criterion(output, target)
      total_loss += loss.item()
      bar.next()
      bar.suffix = f'[{(batch_idx + 1) * BATCH_SIZE}/{len(val_loader.dataset)} %(percent).1f%%] Loss: {loss.item() / BATCH_SIZE:.4f}'
  bar.suffix = f'Loss: {total_loss / len(val_loader)}'
  bar.finish()
  return total_loss / len(val_loader)

# test the model
def test(test_loader, model, criterion, dataset):
  model.eval()
  total_loss = 0
  outputs, targets = [], []
  random_indeces = np.random.choice(len(test_loader.dataset), int(TEST_ARROWS_PERCENTAGE * len(test_loader.dataset)), replace=False)
  bar = IncrementalBar('Processing', max=len(test_loader))
  bar.suffix = f'[0/{len(test_loader.dataset)} %(percent).1f%%]'
  with torch.no_grad():
    for batch_idx, (inputs, target) in enumerate(test_loader):
      output = model(*inputs)
      loss = criterion(output, target)
      total_loss += loss.item()
      bar.next()
      bar.suffix = f'[{(batch_idx + 1) * BATCH_SIZE}/{len(test_loader.dataset)} %(percent).1f%%] Loss: {loss.item() / BATCH_SIZE:.4f}'
      for index, (output, target) in enumerate(zip(output, target)):
        if batch_idx * BATCH_SIZE + index in random_indeces:
          outputs.append(output)
          targets.append(target)
  bar.suffix = f'Loss: {total_loss / len(test_loader):.4f}'
  bar.finish()
  print(f'\nTest set average loss: {total_loss / len(test_loader):.4f}\n')
  plot_results(outputs, targets)
