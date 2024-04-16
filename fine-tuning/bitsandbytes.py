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

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    GenerationConfig
)
from tqdm import tqdm
from trl import SFTTrainer
import torch
import time
import pandas as pd
import numpy as np

#  to dynamically retrieves the float16 data type from the torch module
compute_dtype = getattr(torch, "float16")

# configuring an instance of BitsAndBytesConfig 
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,    # Model's weights should be loaded as 4-bit values. 
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

# load_in_8bit (bool, optional, defaults to False) — This flag is used to enable 8-bit quantization with LLM.int8().
# load_in_4bit (bool, optional, defaults to False) — This flag is used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from bitsandbytes.


# AutoModelForCausalLM.from_pretrained(): This function loads a pre-trained model given a model name. It automatically downloads the model's weights and configures it for use.

model_name='microsoft/phi-2'
device_map = {"": 0}
# device_map: This dictionary is used to control the mapping of the model's layers to specific devices (like different GPUs). 
# The key-value pair "": 0 indicates that all layers of the model should be loaded onto the device with ID 0 (typically the first GPU). 

original_model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                      device_map=device_map,
                                                      quantization_config=bnb_config,
                                                      trust_remote_code=True,
                                                      use_auth_token=False)


# Here not the quantization config: . It tells the model loading function to apply certain quantization parameters right from the start, which affects how the model weights are loaded and potentially how they are stored in memory.
# This is same we created above. This is part of optimizing the model's size and the speed of its operations.




# ####### Tokenizer #######
# This method is used to instantiate a tokenizer associated with a specific pre-trained model. 
# It automatically selects the correct tokenizer class based on the model identifier and loads its pre-trained configuration and vocabulary.
    
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True,padding_side="left",add_eos_token=True,add_bos_token=True,use_fast=False)
tokenizer.pad_token = tokenizer.eos_token



# # what is pad token?
    # This line sets the padding token (pad_token) of the tokenizer to be the same as the end-of-sequence token (eos_token). 
    # In many transformer models, especially autoregressive models like GPT, the eos_token is used to indicate the end of a sequence. By setting the pad_token to eos_token, 
    # we're ensuring that any padding added to sequences during batch preparation is treated as part of the sequence's normal flow, which can be crucial for maintaining consistency in models' expectations about input structure.










