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

### Decoding
To decode a sequence of tokens back to a readable text string:

```python
decoded_text = tokenizer.decode(token_ids)
print(decoded_text)  # Output: "Hello, world!"
```


## Tokenization Concepts
1. Word-Based Tokenization: This involves splitting text by spaces to extract words. It's simple but can miss nuances like punctuation or contraction.
2. Character-Based Tokenization: Each character is treated as a token. Useful for text generation but computationally expensive.
3. Subword Tokenization: Splits text into meaningful subwords using techniques like Byte-Pair Encoding (BPE). This is used in state-of-the-art NLP models like GPT.
4. Regular Expression Tokenization: Allows custom splitting patterns like ([,.?_!"()\']|--|\s) in our implementation, which handles punctuations efficiently.
5. Error Handling: During encoding or decoding, errors can occur if tokens are missing from the vocabulary. The SimpleTokenizerV1 class will raise ValueError for these cases, helping debug data issues.
