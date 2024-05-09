import re

class SimpleTokenizerV1:
    def __init__(self, vocab):
        """
        Initialize the tokenizer with a vocabulary.

        Parameters:
        vocab (dict): A dictionary mapping strings to integer tokens.
        """
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
    
