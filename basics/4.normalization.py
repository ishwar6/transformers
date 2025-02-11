import torch
import torch.nn as nn
import math


class LayerNormalization(nn.Module):
  def __init__(self, epsilon:float = 10**-6):
    super().__init__()
    self.epsilon = epsilon
    self.alpha = nn.Parameter(torch.ones(1)) #for multiply
    self.bias = nn.Parameter(torch.zeros(1)) # for addition

  def forward(self, x):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return self.alpha * (x - mean) / (std + self.epsilon) + self.bias
    
