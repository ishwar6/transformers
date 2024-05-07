import tensorflow as tf
import numpy as np

# Embeddings: First, each word in the sentence is converted into a numerical form that captures its meaning (token embedding). 
# At the same time, each word's position in the sentence is also converted into a numerical form (positional embedding). 
# This helps the model understand both what each word means and its relationship to other words in the sentence.


# Dropout: Before we start processing these embeddings, we randomly drop out some of this information.
# This is like intentionally forgetting some details to make sure our model doesn't rely too heavily on any particular piece of information, making it robust and generalized.

# Encoder Blocks: The embeddings then pass through a series of processing steps (encoder blocks), 
# each refining and adjusting the embeddings by considering how each word relates to others in the sentence. 
# This happens repeatedly, enhancing the understanding of the sentence.
    
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



num_encoder_blocks = 6

# d_model is the embedding dimension used throughout.
d_model = 12

num_heads = 3

# Feed-forward network hidden dimension width.
ffn_hidden_dim = 48

src_vocab_size = bpemb_vocab_size
max_input_seq_len = padded_input_seqs.shape[1]

encoder = Encoder(
    num_encoder_blocks,
    d_model,
    num_heads,
    ffn_hidden_dim,
    src_vocab_size,
    max_input_seq_len)

encoder_output, attn_weights = encoder(padded_input_seqs, training=True, 
                                       mask=enc_mask)
print(f"Encoder output {encoder_output.shape}:")
print(encoder_output)

######################################################################################################################################################
############################################################################## DEBUG ##################################################################
######################################################################################################################################################


# Step 1: Token Embedding
# Token Embeddings Shape: (3, 10, 12)

# Step 2: Positional Encoding
# Position Embeddings Shape: (3, 10, 12)

# Step 3: Combining Token and Position Embeddings with Dropout
# Combined Embeddings Shape after Dropout: (3, 10, 12)

# Step 4.1: Processing through EncoderBlock 1

# Entering MultiHeadSelfAttention
# MHSA Output Shape: (3, 10, 12)
# Post-Dropout MHSA Output Shape: (3, 10, 12)
# Post-LayerNorm1 Output Shape: (3, 10, 12)

# Entering Feed Forward Network
# FFN Output Shape: (3, 10, 12)
# Post-Dropout FFN Output Shape: (3, 10, 12)
# Final Output Shape from Encoder Block: (3, 10, 12)
# Output Shape after EncoderBlock 1: (3, 10, 12)

# Step 4.2: Processing through EncoderBlock 2

# Entering MultiHeadSelfAttention
# MHSA Output Shape: (3, 10, 12)
# Post-Dropout MHSA Output Shape: (3, 10, 12)
# Post-LayerNorm1 Output Shape: (3, 10, 12)

# Entering Feed Forward Network
# FFN Output Shape: (3, 10, 12)
# Post-Dropout FFN Output Shape: (3, 10, 12)
# Final Output Shape from Encoder Block: (3, 10, 12)
# Output Shape after EncoderBlock 2: (3, 10, 12)

# Step 4.3: Processing through EncoderBlock 3

# Entering MultiHeadSelfAttention
# MHSA Output Shape: (3, 10, 12)
# Post-Dropout MHSA Output Shape: (3, 10, 12)
# Post-LayerNorm1 Output Shape: (3, 10, 12)

# Entering Feed Forward Network
# FFN Output Shape: (3, 10, 12)
# Post-Dropout FFN Output Shape: (3, 10, 12)
# Final Output Shape from Encoder Block: (3, 10, 12)
# Output Shape after EncoderBlock 3: (3, 10, 12)

# Step 4.4: Processing through EncoderBlock 4

# Entering MultiHeadSelfAttention
# MHSA Output Shape: (3, 10, 12)
# Post-Dropout MHSA Output Shape: (3, 10, 12)
# Post-LayerNorm1 Output Shape: (3, 10, 12)

# Entering Feed Forward Network
# FFN Output Shape: (3, 10, 12)
# Post-Dropout FFN Output Shape: (3, 10, 12)
# Final Output Shape from Encoder Block: (3, 10, 12)
# Output Shape after EncoderBlock 4: (3, 10, 12)

# Step 4.5: Processing through EncoderBlock 5

# Entering MultiHeadSelfAttention
# MHSA Output Shape: (3, 10, 12)
# Post-Dropout MHSA Output Shape: (3, 10, 12)
# Post-LayerNorm1 Output Shape: (3, 10, 12)

# Entering Feed Forward Network
# FFN Output Shape: (3, 10, 12)
# Post-Dropout FFN Output Shape: (3, 10, 12)
# Final Output Shape from Encoder Block: (3, 10, 12)
# Output Shape after EncoderBlock 5: (3, 10, 12)

# Step 4.6: Processing through EncoderBlock 6

# Entering MultiHeadSelfAttention
# MHSA Output Shape: (3, 10, 12)
# Post-Dropout MHSA Output Shape: (3, 10, 12)
# Post-LayerNorm1 Output Shape: (3, 10, 12)

# Entering Feed Forward Network
# FFN Output Shape: (3, 10, 12)
# Post-Dropout FFN Output Shape: (3, 10, 12)
# Final Output Shape from Encoder Block: (3, 10, 12)
# Output Shape after EncoderBlock 6: (3, 10, 12)

# Final Output from Encoder
# Encoder output (3, 10, 12):
# tf.Tensor(
# [[[-0.6976648   0.6176258  -0.6892076   0.1214155  -1.1035869
#     1.2902294  -2.1537838  -0.33218265  0.6580796   0.14708684
#     0.714634    1.4273546 ]
#   [-1.2036239   0.19161372 -0.09209978  0.11330835 -0.8551608
#     1.121856   -1.9356893   0.21826924  0.68088895 -0.63670236
#     0.50603896  1.8913014 ]
#   [-0.7494565  -0.24898152 -0.5205772   0.14779288 -0.8750923
#     1.2724622  -2.0461488   0.18946916  0.83635414 -0.5193952
#     0.84427875  1.6692946 ]
#   [ 0.02694195 -0.00303182 -0.08785538  0.3010898  -1.19097
#     1.2190157  -2.2966444  -1.0859085   0.76479036  0.5290236
#     0.85163534  0.9719134 ]
#   [-0.46815968 -0.2752501  -0.92613167 -0.43615377 -1.2960696
#     0.83533955 -1.5958738   0.40349108  1.039336   -0.17065145
#     1.0731418   1.8169817 ]
#   [-1.0302775  -0.15703803 -0.0638044   0.085574   -1.0228367
#     1.4322385  -1.8125422   0.10282735  0.8735675  -0.4652037
#     0.19011731  1.8673781 ]
#   [-0.43441558 -0.08221813 -0.65018106  0.03478169 -1.2674133
#     1.2791834  -1.8876392  -0.09604713  0.68859226 -0.3534422
#     1.132545    1.636254  ]
#   [-0.24984005  0.48351514 -0.958089   -0.383954   -1.2204009
#     1.2336382  -1.8093215   0.04020286  0.85380846 -0.41335788
#     0.7079597   1.7158389 ]
#   [-0.20030257  0.27532506 -0.7386109   0.4380885  -1.6298859
#     0.89060134 -1.9205806   0.22267544  1.0005385  -0.4968032
#     0.6474101   1.511544  ]
#   [-0.32536212  0.39952272 -0.26370516  0.17869082 -1.2981709
#     1.2000662  -2.294625   -0.5807329   0.47941604  0.34212485
#     0.75623876  1.4065374 ]]

#  [[ 0.77390295 -0.40307185 -1.291947    0.9258603  -0.49763152
#     0.5544103  -2.3877587   0.26265287  0.6021101  -0.05516132
#     0.10099334  1.4156402 ]
#   [-0.90788305 -0.10286579 -0.55505466 -0.0503424  -0.88036156
#     1.4744253  -1.9500318   0.48915827  0.6329826  -0.04310491
#     0.03487486  1.8582033 ]
#   [-0.9374872  -0.19018051 -0.68501014 -0.29351142 -0.5363117
#     1.290908   -1.9790723   0.65271306  0.60694885 -0.505586
#     0.9126849   1.6639042 ]
#   [-0.72315335 -0.2209137  -0.80125016  0.95826435 -0.5892
#     0.7448848  -2.296494    0.13670681  0.8125009  -0.321438
#     0.7922091   1.5078832 ]
#   [-1.2844607  -0.23828687 -0.3719758  -0.11636236 -0.01978779
#     1.0606079  -2.2710602   0.52374685  0.3535039   0.5318983
#     0.05303435  1.7791424 ]
#   [-0.25335348 -0.549422   -0.9881126  -0.59559494 -0.47106743
#     0.8000251  -2.153176    0.6055966   1.0828613   0.60228634
#     0.23510815  1.6848491 ]
#   [-0.8021164  -0.22151716 -0.9484187   0.47422525 -0.6861126
#     1.0678678  -2.1961584   0.52694416  0.91816765  0.54335314
#    -0.18911155  1.5128764 ]
#   [-0.382473   -0.43739957 -0.19549176 -0.7050698  -1.0748632
#     1.4325424  -2.1142452   0.94795173  0.5718812  -0.06477632
#     0.7205815   1.301362  ]
#   [-0.64534736 -0.29020455 -0.37244242  0.02720589 -1.2255256
#     1.3600396  -1.9299134   0.9365135   0.5860293  -0.38707185
#     0.25503984  1.6856769 ]
#   [ 0.35777718 -0.11848765 -0.5650676  -0.6803168  -1.3336369
#     1.1150918  -2.19051     0.08817229  0.8291083   0.41004786
#     0.72337675  1.3644446 ]]

#  [[-0.8466607  -0.01442189 -0.3872516  -0.23790604 -2.054305
#     1.8214307  -1.043768   -0.00314935  0.5776779   0.3577458
#     0.5241457   1.3064623 ]
#   [-0.9202791  -0.39203227 -0.28136998  1.0611728  -0.9989972
#     1.2147524  -2.0824337  -0.58699703  0.8060393   0.47969973
#     0.45929655  1.241149  ]
#   [-0.5324197  -0.5474384  -0.9351132   1.216801   -0.8357488
#     1.3544     -2.000002   -0.49383456  0.8730617   0.23845416
#     0.63734496  1.0244946 ]
#   [ 0.2725349   0.07206397 -0.5668725   0.50478166 -0.9789198
#     1.1773564  -2.393645   -1.0157326   0.7377971   0.5498249
#     0.6821529   0.9586578 ]
#   [-0.20634213 -0.3703047  -0.18562114 -0.09016286 -1.7493542
#     1.596696   -1.7730552  -0.44150537  0.95714396  0.39607626
#     0.8709297   0.99550027]
#   [-0.45028254 -0.6370995  -0.3319987   0.9777134   0.6336327
#     1.1515436  -2.3467019  -1.2266243   0.68370014  0.08128837
#     0.8576053   0.60722345]
#   [-0.68316805 -1.1329768   0.10928006  0.8779144  -0.6428727
#     1.2653242  -1.9571701  -0.22333895  0.8354679  -0.588535
#     0.7309426   1.4091326 ]
#   [-0.9807099  -0.99496835  0.3269991   0.38882133 -0.08914374
#     1.3964939  -1.0981536  -1.7454505   1.2564791   0.24926817
#    -0.0959416   1.3863066 ]
#   [-0.768856   -0.47361955 -0.58865285  0.5359281  -0.3793763
#     1.7646166  -2.2697535  -0.3577915   0.72947973  0.10433532
#     0.7782382   0.92545223]
#   [-0.11703248 -0.15389149 -0.5213187   0.30587238 -0.7907997
#     2.2061827  -2.0611725  -0.73928803  0.5618439  -0.02846975
#     0.4594415   0.87863207]]], shape=(3, 10, 12), dtype=float32)
