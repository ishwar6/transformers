import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // self.num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        matmul_qk = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.depth)
        if mask is not None:
            matmul_qk = matmul_qk.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(matmul_qk, dim=-1)
        output = torch.matmul(attention_weights, v)

        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        return self.dense(output)

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout_rate):
        super(EncoderBlock, self).__init__()
        self.mhsa = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        attn_output = self.mhsa(x, x, x, mask)
        x = self.layernorm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + self.dropout2(ffn_output))
        return x

class Encoder(nn.Module):
    def __init__(self, num_blocks, d_model, num_heads, hidden_dim, src_vocab_size, max_seq_len, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.token_embed = nn.Embedding(src_vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([EncoderBlock(d_model, num_heads, hidden_dim, dropout_rate) for _ in range(num_blocks)])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input, mask=None):
        positions = torch.arange(0, input.size(1)).unsqueeze(0).repeat(input.size(0), 1).to(input.device)
        x = self.token_embed(input) + self.pos_embed(positions)
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x, mask)

        return x
 
input_tensor = torch.randint(0, 1000, (3, 10))  #   (batch_size, seq_len)
src_vocab_size = 1000
max_seq_len = 10
d_model = 12
num_heads = 3
hidden_dim = 48
num_blocks = 6
encoder = Encoder(num_blocks, d_model, num_heads, hidden_dim, src_vocab_size, max_seq_len)
output = encoder(input_tensor)
print(output.shape)  #  [3, 10, 12]
