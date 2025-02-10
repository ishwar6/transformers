import torch
import torch.nn as nn
import math


class InputEmbedding(nn.Module):
  def __init__(self, d_model: int, vocab_size:int):
    super().__init__()
    self.d_model = d_model
    self.vocab_size = vocab_size
    self.embedding = nn.Embedding(vocab_size, d_model)

  def forward(self, x):
    return self.embedding(x) * math.sqrt(self.d_model)



###### Explanation ####

# d_model: The dimension of the embedding vector (also the model’s hidden size).
# vocab_size: The size of the vocabulary (number of tokens).
# Embedding Layer: nn.Embedding(vocab_size, d_model) creates a lookup table that maps each token (represented by an integer index) to a dense vector of size d_model.

# Forward Pass:
# Input: x is typically a tensor of shape (batch_size, seq_length) containing token indices.
# Embedding Lookup: self.embedding(x) retrieves the dense representation for each token.
# scaling is there: 
# In the original Transformer paper "Attention Is All You Need", 
# scaling is performed to counteract the effect of the dot-product growing large in magnitude when the dimensionality is high. 
#   This helps in stabilizing gradients early in training.
                                        
