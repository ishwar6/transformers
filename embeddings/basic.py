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

