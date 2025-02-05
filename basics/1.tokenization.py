from nltk.tokenize import word_tokenize

text = "I love Transformers!"
tokens = word_tokenize(text)
print(tokens)  
# Output: ['I', 'love', 'Transformers', '!']


#Tokenizing with Hugging Face's BERT Tokenizer
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("I love Transformers!")
print(tokens)  
# Output: ['i', 'love', 'transformers', '!']


