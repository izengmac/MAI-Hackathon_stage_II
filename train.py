from data_preprocessing import parse_and_preprocess_openfoam_data
from model import AerodynamicDNN
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import random


def split(data_list, test_size=0.2):
  """
  Splits data into train and test sets.

  Args:
    data_list: List of data tuples.
    test_size: Proportion of data for the test set.

  Returns:
    train_data, test_data: Split data lists.
  """
  random.shuffle(data_list)
  split_index = int(len(data_list) * (1 - test_size))
  train_data, test_data = data_list[:split_index], data_list[split_index:]
  return train_data, test_data


def train(model, optimizer, train_loader):
  """
  Trains the model on a data loader.

  Args:
    model: The DNN model.
    optimizer: The optimizer for parameter updates.
    train_loader: A DataLoader object for the training data.

  Returns:
    avg_loss: Average loss over the training epoch.
  """
  model.train()
  total_loss, total_samples = 0, 0

  for data in train_loader:
    # Extract inputs and targets
    inputs, targets = data

    # Forward pass
    outputs = model(*inputs)

    # Calculate MAE loss
    mae_loss = L1Loss(reduction="mean")(outputs, targets)

    # Backpropagation and optimization
    optimizer.zero_grad()
    mae_loss.backward()
    optimizer.step()

    # Accumulate loss and samples
    total_loss += mae_loss.item()
    total_samples += len(data)

  # Calculate average loss
  avg_loss = total_loss / total_samples

  # Print and return average loss
  print(f"Average Train Loss: {avg_loss:.4f}")
  return avg_loss


def evaluate(model, test_loader):
  """
  Evaluates the model on a data loader.

  Args:
    model: The DNN model.
    test_loader: A DataLoader object for the test data.

  Returns:
    avg_loss: Average loss over the test data.
  """
  model.eval()
  total_loss, total_samples = 0, 0

  with torch.no_grad():
    for data in test_loader:
      # Extract inputs and targets
      inputs, targets = data

      # Forward pass
      outputs = model(*inputs)

      # Calculate MAE loss
      mae_loss = L1Loss(reduction="mean")(outputs, targets)

      # Accumulate loss and samples
      total_loss += mae_loss.item()
      total_samples += len(data)

  # Calculate average loss
  avg_loss = total_loss / total_samples

  # Print and return average loss
  print(f"Average Test MAE: {avg_loss:.4f}")
  return avg_loss


if __name__ == "__main__":
  # Define hyperparameters
  learning_rate = 0.001
  epochs = 100
  batch_size = 32

  # Load preprocessed data
  data_list = parse_and_preprocess_openfoam_data("/path/to/foambase/directory")

  # Split data into train and test sets
  train_data, test_data = split(data_list)

  # Create data loaders
  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

  # Define model
  model = AerodynamicDNN(input_dim=3, output_dim=3)  # Adjust input_dim based on actual features

  # Define optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  # Train and evaluate the model
  for epoch in range(epochs):
    train_loss = train(model, optimizer, train_loader)
    test_loss = evaluate(model, test_loader)
    print(f"Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test MAE: {test_loss:.4f}")

  
