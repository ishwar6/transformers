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

