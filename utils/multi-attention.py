# What are the parts of the EncoderBlock?
# This box has several smaller tools inside it that help it do its job:

# MultiHeadSelfAttention (mhsa): Think of this as a set of magic glasses. When the box looks at information through these glasses, 
# it can see important parts of the information more clearly. These glasses help the box focus on what's really important in a bunch of words.

# Feed Forward Network (ffn): This is like a mini-factory inside the box. Once the magic glasses have highlighted the important words, this mini-factory changes them slightly to make them even better and easier for the computer to understand.

# Dropouts: These are like filters or sieves that randomly let some information through and block some. 
# This helps in making sure the box doesn't rely too much on any one piece of information. It's like making sure you don’t just listen to one friend but get ideas from many friends.

# LayerNormalization (layernorm): This is like a cleaner who tidies up after the first tool (magic glasses) and the mini-factory have done their work. 
# It makes sure everything is neat and in order so that the information looks nice and easy for the computer to read.

# After all this, the box gives us a new version of the sentence that is easier for the computer to understand, along with a special note (attention weights) that tells us which words were seen as most important by the magic glasses. 
# This helps the computer to later remember what was important in the sentence when it tries to use this information for tasks like answering questions or translating languages.

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



################################ Debug Mode : ON #############################

# Entering MultiHeadSelfAttention
# MHSA Output Shape: (1, 3, 12)
# Post-Dropout MHSA Output Shape: (1, 3, 12)
# Post-LayerNorm1 Output Shape: (1, 3, 12)

# Entering Feed Forward Network
# FFN Output Shape: (1, 3, 12)
# Post-Dropout FFN Output Shape: (1, 3, 12)
# Final Output Shape from Encoder Block: (1, 3, 12)
# Output from single encoder block (1, 3, 12):
# tf.Tensor(
# [[[-0.3847756  -0.33387783  0.76982546  0.8497208  -0.8858971
#    -0.48990846  0.9594057   0.9769396  -0.48029894 -0.04240228
#    -2.318786    1.3800544 ]
#   [ 0.07458647 -0.6279474   0.45643866 -1.0298812   0.4622764
#    -0.10734023  0.08881539  1.0231558   1.3502303  -0.08587222
#    -2.5226183   0.9181567 ]
#   [-0.14665551 -0.88339406  0.03083881 -1.8576676  -0.09421501
#     0.5884591   0.5067914   0.8045433   0.97845304  0.25669065
#    -1.7404709   1.5566266 ]]], shape=(1, 3, 12), dtype=float32)




############################################################################################# pytorch implementation ####################################################################################################

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

