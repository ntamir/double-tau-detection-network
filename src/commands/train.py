import sys

def train(train_loader, model, criterion, optimizer, epoch):
  pass

if __name__ == '__main__':
  epochs = sys.argv[1]
  train_loader = sys.argv[6]
  model = sys.argv[7]
  criterion = sys.argv[8]
  optimizer = sys.argv[9]
  for epoch in range(epochs):
    print(f'Epoch {epoch}')
    train(train_loader, model, criterion, optimizer, epoch)