# seq: seq length
# d_model: embedding vector length
# h: no of heads in attention block: multihead attention
# d_k, d_v: key and value matrix length: d_model / h


# lets talk about input (seq_length, d_model): we will have Q, K, V each of same (seq_length, d_model)
# we multiply Q by W_q, K by W_k and V by W_v (each weights are of d_model * d_model) : we after multiply get result of (seq_length, d_model)
# we divide this new Q, new K and new V into h blocks (h of multihead attentions): we split matrix along embedding dimension: 
# which means each head get full sentence (sequence) but diff part of embedding of each word. : #attending to different parts of the input representation
# we apply attention formula to each head and finally combine by concat formula. 
# in concat we have w_o = (h * d_v, d_model)


import torch
import torch.nn as nn

class MultiHeadAttentionBlock(nn.module):
  def __init__(self, d_model: int, h: int, dropout: int) -> None:
    super().__init__()
    self.d_model = d_model
    self.h = h
    assert d_model % h == 0, "No of heads should be divisible by d_model"
    self.d_k = d_model // h
    # each weights are of (d_model * d_model)
    self.w_q = nn.Linear(d_model, d_model)
    self.w_k = nn.Linear(d_model, d_model)
    self.w_v = nn.Linear(d_model, d_model)

      
    
    
    




