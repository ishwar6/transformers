# SimpleTokenizerV1

The `SimpleTokenizerV1` class provides a simple, custom tokenizer that maps words and characters to integer tokens and vice versa. It can be used for tasks involving Natural Language Processing (NLP), such as language modeling or sentiment analysis.

## How to Use

### Initialization
To initialize the tokenizer, provide a vocabulary dictionary mapping strings to integer tokens:

```python
vocab = {'Hello': 1, 'world': 2, ',': 3, '!': 4}
tokenizer = SimpleTokenizerV1(vocab)
```

### Encoding
To encode a text string into tokens:

```python
text = "Hello, world!"
token_ids = tokenizer.encode(text)
print(token_ids)  # Output: [1, 3, 2, 4]
```
