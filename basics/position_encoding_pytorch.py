import torch
import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_length: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(seq_length, d_model)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # Sin for even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Cos for odd indices

        pe = pe.unsqueeze(0)  # Shape: (1, seq_length, d_model)

        # Register PE as a buffer (not trainable)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)


d_model = embedding_tokens.shape[1]  # Get embedding size (12288 in GPT-3)
seq_length = embedding_tokens.shape[0]  # Sequence length (3 for "I love transformers")

# Initialize PE layer
pos_encoding_layer = PositionalEncoding(d_model, seq_length, dropout=0.1)

# Apply to embedding tokens
encoded_tokens = pos_encoding_layer(embedding_tokens)

# Print final result
print("\nFinal Embeddings After Adding PE:\n", encoded_tokens)




# [DEBUG] Input Shape: torch.Size([3, 12288])
# [DEBUG] Positional Encoding Shape: torch.Size([1, 3, 12288])
# [DEBUG] Output Shape After Adding PE: torch.Size([1, 3, 12288])

# Final Embeddings After Adding PE:
#  tensor([[[ -18.2410,   -0.0000, -106.3610,  ...,   78.7416,   16.1874,
#           -143.5334],
#          [ -67.6987,   56.4681,  -45.8034,  ..., -184.7881, -297.2361,
#           -115.9084],
#          [ 228.6305,   77.1439,  108.0852,  ...,   51.9746,   61.7931,
#             25.7511]]], grad_fn=<MulBackward0>)
