import torch
import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(seq_len, d_model):
    """
    Compute positional encoding for a given sequence length and embedding dimension.
    """
    pos = torch.arange(seq_len).unsqueeze(1)  # Shape: [seq_len, 1]
    i = torch.arange(d_model).unsqueeze(0)  # Shape: [1, d_model]

    # Compute PE using sine for even indices and cosine for odd indices
    div_term = torch.pow(10000, (2 * (i // 2)) / d_model)
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos / div_term[:, 0::2])  # Apply sin to even indices
    pe[:, 1::2] = torch.cos(pos / div_term[:, 1::2])  # Apply cos to odd indices
    
    return pe

# Generate positional encodings
seq_len = 100  # Sequence length (e.g., 100 words)
d_model = 16   # Embedding dimension (small for visualization)
pe = positional_encoding(seq_len, d_model)

# Visualize the encoding for first 16 dimensions
plt.figure(figsize=(10, 6))
plt.imshow(pe.numpy(), cmap="viridis", aspect="auto")
plt.colorbar(label="Value")
plt.xlabel("Embedding Dimension")
plt.ylabel("Word Position")
plt.title("Positional Encoding Heatmap")
plt.show()
