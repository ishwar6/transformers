import torch
import torch.nn as nn

class FeedForwardBlock(nn.Module):
  def __init__(self, d_model: int, d_ff: int, dropout: float) -> None: 
    super().__init__()
    self.linear_1 = nn.Linear(d_model, d_ff)
    self.dropout = nn.Dropout(dropout)
    self.linear_2 = nn.Linear(d_ff, d_model)


  def forward(self, x):
    return self.linear_2(self.dropout(torch.relu(self.linear_1(x)))
                          
d_model = 4   # Input dimension
d_ff = 8      # Expanded dimension
dropout = 0.1 # Dropout rate
 
ff_block = FeedForwardBlock(d_model, d_ff, dropout)

# Example input tensor (Batch size=2, Sequence length=3, Features=d_model)
x = torch.randn(2, 3, d_model)

# Forward pass
output = ff_block(x)

print("Input:\n", x)
print("Output:\n", output)

# tensor([[[ 0.3736, -0.1008,  0.0789, -1.1882],
#          [-0.6518,  0.8867,  1.5849,  1.4212],
#          [-0.5689, -0.4329,  0.3440, -0.2775]],

#         [[-0.4460, -0.5574, -0.8480, -0.6797],
#          [-0.9456, -0.2138,  0.9150,  0.3810],
#          [ 0.0407, -0.1553, -1.2665, -0.6900]]])
# Output:
#  tensor([[[-0.3444,  0.0599, -0.4255,  0.0169],
#          [ 0.0291,  0.0642, -0.0565,  0.3641],
#          [-0.3180,  0.2656, -0.1069,  0.2903]],

#         [[-0.3493,  0.2434, -0.2264,  0.1342],
#          [-0.2625,  0.2213, -0.0838,  0.3138],
#          [-0.6571,  0.2348, -0.4605,  0.1008]]], grad_fn=<ViewBackward0>)
