class MultiHeadSelfAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadSelfAttention, self).__init__()
    #class is taking model dimension and no of heads as arguments
    self.d_model = d_model
    self.num_heads = num_heads

    self.d_head = self.d_model // self.num_heads

    # we make 3 weight layers for q, k and value. 
    self.wq = tf.keras.layers.Dense(self.d_model)
    self.wk = tf.keras.layers.Dense(self.d_model)
    self.wv = tf.keras.layers.Dense(self.d_model)

    # Linear layer to generate the final output.
    self.dense = tf.keras.layers.Dense(self.d_model)
  
  def split_heads(self, x):
    batch_size = x.shape[0]

    split_inputs = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_head))
    return tf.transpose(split_inputs, perm=[0, 2, 1, 3])
  
  def merge_heads(self, x):
    batch_size = x.shape[0]

    merged_inputs = tf.transpose(x, perm=[0, 2, 1, 3])
    return tf.reshape(merged_inputs, (batch_size, -1, self.d_model))

  def call(self, q, k, v, mask):
    qs = self.wq(q)
    ks = self.wk(k)
    vs = self.wv(v)

    qs = self.split_heads(qs)
    ks = self.split_heads(ks)
    vs = self.split_heads(vs)

    output, attn_weights = scaled_dot_product_attention(qs, ks, vs, mask)
    output = self.merge_heads(output)

    return self.dense(output), attn_weights




mhsa = MultiHeadSelfAttention(12, 3)

output, attn_weights = mhsa(x, x, x, None)
print(f"MHSA output{output.shape}:")
print(output)


# MHSA output(1, 3, 12):
# tf.Tensor(
# [[[ 0.84930104  0.5776336  -0.01806774 -1.1573353  -0.07407556
#     0.30329642 -0.5888044  -0.08995268  0.863901    0.28992242
#     0.2473642   1.0676059 ]
#   [ 0.8489535   0.58133125 -0.01876919 -1.157928   -0.07491764
#     0.30115992 -0.5942613  -0.08884156  0.8692355   0.28925964
#     0.24694338  1.0706242 ]
#   [ 0.84847426  0.57505053 -0.01739992 -1.1549186  -0.07409394
#     0.3033722  -0.59076697 -0.08866085  0.862929    0.29087895
#     0.2470599   1.0656964 ]]], shape=(1, 3, 12), dtype=float32)



import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, mask=None):
    matmul_qk = torch.matmul(q, k.transpose(-2, -1)) 
    # Transpose last two dimensions for dot product
    d_k = k.size()[-1]  # Get the dimensionality of keys
    scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    if mask is not None:
        scaled_attention_logits += (mask * -1e9) 

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)  # Multiply by values
    return output, attention_weights


# Main Multi Head Self Attention Class
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_head = self.d_model // self.num_heads

        # Initialize weights for Q, K, and V
        self.wq = nn.Linear(self.d_model, self.d_model)
        self.wk = nn.Linear(self.d_model, self.d_model)
        self.wv = nn.Linear(self.d_model, self.d_model)

        # Final linear layer
        self.dense = nn.Linear(self.d_model, self.d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.d_head)
        return x.permute(0, 2, 1, 3)  # Re-arrange the axis for multi-head attention

    def merge_heads(self, x, batch_size):
        x = x.permute(0, 2, 1, 3)  # Re-arrange axis back to the original
        return x.contiguous().view(batch_size, -1, self.d_model)  # Merge the heads

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Perform the linear operations and split into heads
        qs = self.split_heads(self.wq(q), batch_size)
        ks = self.split_heads(self.wk(k), batch_size)
        vs = self.split_heads(self.wv(v), batch_size)

        # Apply the scaled dot product attention
        output, attn_weights = scaled_dot_product_attention(qs, ks, vs, mask)

        # Concatenate heads and pass through final linear layer
        output = self.merge_heads(output, batch_size)
        return self.dense(output), attn_weights

