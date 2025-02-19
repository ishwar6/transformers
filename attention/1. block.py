# seq: seq length
# d_model: embedding vector length
# h: no of heads in attention block: multihead attention
# d_k, d_v: key and value matrix length: d_model / h


# lets talk about input (seq_length, d_model): we will have Q, K, V each of same (seq_length, d_model)
# we multiply Q by W_q, K by W_k and V by W_v (each weights are of d_model * d_model) : we after multiply get result of (seq_length, d_model)
# we divide this new Q, new K and new V into h blocks (h of multihead attentions): we split matrix along embedding dimension: 
# which means each head get full sentence (sequence) but diff part of embedding of each word. 
# we apply attention formula to each head and finally combine by concat formula. 


