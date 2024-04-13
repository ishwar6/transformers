import tensorflow as tf
import numpy as np

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_blocks, d_model, num_heads, hidden_dim, src_vocab_size,
                 max_seq_len, dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embed = tf.keras.layers.Embedding(src_vocab_size, self.d_model)
        self.pos_embed = tf.keras.layers.Embedding(max_seq_len, self.d_model)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        # Creating multiple encoder blocks
        self.blocks = [EncoderBlock(d_model, num_heads, hidden_dim, dropout_rate) 
                       for _ in range(num_blocks)]
  
    def call(self, input, training, mask):
        print("Step 1: Token Embedding")
        token_embeds = self.token_embed(input)
        print("Token Embeddings Shape:", token_embeds.shape)

        print("\nStep 2: Positional Encoding")
        # Generate position indices for the input sequences
        pos_idx = tf.range(self.max_seq_len)
        pos_idx = tf.tile(pos_idx[None, :], [input.shape[0], 1])
        pos_embeds = self.pos_embed(pos_idx)
        print("Position Embeddings Shape:", pos_embeds.shape)

        print("\nStep 3: Combining Token and Position Embeddings with Dropout")
        x = token_embeds + pos_embeds
        x = self.dropout(x, training=training)
        print("Combined Embeddings Shape after Dropout:", x.shape)

        # Process input through each EncoderBlock
        for i, block in enumerate(self.blocks):
            print(f"\nStep 4.{i+1}: Processing through EncoderBlock {i+1}")
            x, weights = block(x, training, mask)
            print(f"Output Shape after EncoderBlock {i+1}:", x.shape)

        print("\nFinal Output from Encoder")
        return x, weights

input_batch = [
    "Where can I find a pizzeria?",
    "Mass hysteria over listeria.",
    "I ain't no circle back girl."
]

bpemb_en.encode(input_batch)
