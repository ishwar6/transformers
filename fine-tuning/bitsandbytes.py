# bitsandbytes enables accessible large language models via k-bit quantization for PyTorch. bitsandbytes provides three main features for dramatically reducing memory consumption for inference and training:

# 8-bit optimizers uses block-wise quantization to maintain 32-bit performance at a small fraction of the memory cost.
# LLM.Int() or 8-bit quantization enables large language model inference with only half the required memory and without any performance degradation.
    #This method is based on vector-wise quantization to quantize most features to 8-bits and separately treating outliers with 16-bit matrix multiplication.
# QLoRA or 4-bit quantization enables large language model training with several memory-saving techniques that don’t compromise performance. 
    #This method quantizes a model to 4-bits and inserts a small set of trainable low-rank adaptation (LoRA) weights to allow training.


# 8-bit optimizers
# With 8-bit optimizers, large models can be finetuned with 75% less GPU memory without losing any accuracy compared to training with standard 32-bit optimizers. 
# The reduced memory requirements means 8-bit optimizers are 4x faster than a standard optimizer, and no hyperparameter tuning is required.

import bitsandbytes as bnb

- adam = torch.optim.Adam(...)
+ adam = bnb.optim.Adam8bit(...)

# recommended for NLP models
- before: torch.nn.Embedding(...)
+ bnb.nn.StableEmbedding(...)


# Qlora: https://www.youtube.com/watch?v=y9PHWGOa8HA

# # In transformer Library: 
# Quantization techniques reduce memory and computational costs by representing weights and activations with lower-precision data types like 8-bit integers (int8). 
# This enables loading larger models you normally wouldn’t be able to fit into memory, and speeding up inference. 
# Transformers supports the AWQ and GPTQ quantization algorithms and it supports 8-bit and 4-bit quantization with bitsandbytes.


# Reference: https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/utils/quantization_config.py#L182
