# Tokenization and Embeddings are two foundational steps in NLP tasks. 

# Tokenization
# Tokenization involves converting raw text into smaller units (tokens). 

import re

def tokenize(text):
    tokens = re.findall(r'\b\w+\b', text.lower())   
    return tokens


text = "Transformers are powerful tools for NLP tasks."
tokens = tokenize(text)
print(tokens)  # Output: ['transformers', 'are', 'powerful', 'tools', 'for', 'nlp', 'tasks']



from transformers import BertTokenizer

# Load a pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Now, Lets Tokenize text
tokens = tokenizer.encode("Transformers are powerful tools for NLP tasks.", add_special_tokens=True)
print(tokens)  # Token indices
decoded_tokens = tokenizer.convert_ids_to_tokens(tokens)
print(decoded_tokens)  # Output tokens with special tokens


# Embeddings
# Embeddings represent tokens as dense vectors in a continuous space. The following example shows how to use PyTorch’s embedding layer directly:


import torch
import torch.nn as nn

# Initialize an embedding layer
vocab_size = 1000  #  vocabulary size
embedding_dim = 50  # Dimensionality of the embeddings
embedding = nn.Embedding(vocab_size, embedding_dim)


# Token indices (batch size of 3 sentences, each with 4 tokens)
input_indices = torch.tensor([[4, 56, 5, 12], [7, 1, 45, 9], [34, 12, 23, 10]])
embedded_vectors = embedding(input_indices)
print(embedded_vectors)  # Each token index is converted to an embedding vector



# When working with pre-trained models like BERT, embeddings are integrated within the model. Here's an example with a BERT model:



from transformers import BertModel

# Load a pre-trained BERT model
model = BertModel.from_pretrained('bert-base-uncased')

# So this is time to tokenize text
text = "Transformers are powerful tools for NLP tasks."
inputs = tokenizer(text, return_tensors='pt', add_special_tokens=True)

# Now, lets pass the tokens to the model
outputs = model(**inputs)
 
embeddings = outputs.last_hidden_state
print(embeddings)  # The embeddings for each token in the input text












