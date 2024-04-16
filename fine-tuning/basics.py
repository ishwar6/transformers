# why? : to enhance their performance (LLMs) on specific tasks by adapting them to domain-specific data. 

# Fine-tuning LLM involves the additional training of a pre-existing model, which has previously acquired patterns and features from an extensive dataset, using a smaller, domain-specific dataset.
# Utilizing the existing knowledge embedded in the pre-trained model allows for achieving high performance on specific tasks with substantially reduced data and computational requirements.


# key steps involved in LLM Fine-tuning:

# Select a pre-trained model: For LLM Fine-tuning first step is to carefully select a base pre-trained model that aligns with our desired architecture and functionalities. 

# Gather relevant Dataset: Then we need to gather a dataset that is relevant to our task. The dataset should be labeled or structured in a way that the model can learn from it.

# Preprocess Dataset: Once the dataset is ready, we need to do some preprocessing for fine-tuning by cleaning it, splitting it into training, validation, and test sets, and ensuring it’s compatible with the model on which we want to fine-tune.

# Fine-tuning: After selecting a pre-trained model we need to fine tune it on our preprocessed relevant dataset which is more specific to the task at hand. 
The dataset which we will select might be related to a particular domain or application, allowing the model to adapt and specialize for that context.

# Task-specific adaptation: During fine-tuning, the model’s parameters are adjusted based on the new dataset, helping it better understand and generate content relevant to the specific task. 
This process retains the general language knowledge gained during pre-training while tailoring the model to the nuances of the target domain.


# #Fine-tuning methods: 
#   1. Full Fine Tuning (Instruction fine-tuning) : updates all model weights, creating a new version with improved capabilities. However, it demands sufficient memory and computational resources, similar to pre-training,
#   2. Parameter Efficient Fine-Tuning (PEFT) is a form of instruction fine-tuning that is much more efficient than full fine-tuning. PEFT addresses this by updating only a subset of parameters, effectively “freezing” the rest. 
      # This reduces the number of trainable parameters, making memory requirements more manageable and preventing catastrophic forgetting.
# There are various ways of achieving Parameter efficient fine-tuning. Low-Rank Adaptation LoRA & QLoRA are the most widely used and effective.

# LoRA: Low-Rank Adaptation of Large Language Models
# Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen

# Code example aims to fine-tune a transformer-based model using techniques like PEFT (Prompt-based Efficient Fine-tuning) and QLoRA for generating summaries from dialogue. 

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer, GenerationConfig
)
from tqdm import tqdm
from trl import SFTTrainer
import torch
import time
import pandas as pd
import numpy as np
from huggingface_hub import interpreter_login

# Authenticate to the Hugging Face API for accessing private models or datasets
interpreter_login()

import os
# Disable Weights and Biases to avoid external logging
os.environ['WANDB_DISABLED']="true"

# Load a specific dataset from the Hugging Face Hub
huggingface_dataset_name = "neil-code/dialogsum-test"
dataset = load_dataset(huggingface_dataset_name)

########################################################################################################

# Set compute data type for model training, using reduced precision (float16) for faster computation
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization for model weights
    bnb_4bit_quant_type='nf4',  # Specify quantization type
    bnb_4bit_compute_dtype=compute_dtype,  # Use float16 for computations
    bnb_4bit_use_double_quant=False,  # Disable double quantization
)

model_name='microsoft/phi-2'
device_map = {"": 0}  # Map model to specific CUDA device, "" implies CPU
original_model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map=device_map,
    quantization_config=bnb_config,  # Apply the quantization configuration
    trust_remote_code=True,
    use_auth_token=True
)


####################################### Tokenizer Configuration #######################################

tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True, padding_side="left", 
    add_eos_token=True, add_bos_token=True, use_fast=False
)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to EOS token for consistent end-of-sequence handling















