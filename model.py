import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import random


class AerodynamicDNN(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(AerodynamicDNN, self).__init__()
    # Define hidden layers with ReLU activation
    self.dense1 = nn.Linear(input_dim, 128)
    self.relu1 = nn.ReLU()
    self.dense2 = nn.Linear(128, 64)
    self.relu2 = nn.ReLU()
    self.dense3 = nn.Linear(64, 32)
    self.relu3 = nn.ReLU()
    # Output layer with linear activation
    self.output_layer = nn.Linear(32, output_dim)

  def forward(self, x):
    # Forward pass through hidden layers
    x = self.relu1(self.dense1(x))
    x = self.relu2(self.dense2(x))
    x = self.relu3(self.dense3(x))
    # Return output
    return self.output_layer(x)
