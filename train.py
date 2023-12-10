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


def evaluate(model, data_loader):
  """
  Evaluates the model on a data loader.

  Args:
    model: The DNN model.
    data_loader: A DataLoader object.

  Returns:
    loss, accuracy: Average loss and accuracy across the data loader.
  """
  model.eval()
  total_loss, total_correct = 0, 0
  with torch.no_grad():
    for data in data_loader:
      # Extract inputs and targets
      inputs, targets = data

      # Forward pass
      outputs = model(*inputs)

      # Calculate loss
      loss = loss_fn(
