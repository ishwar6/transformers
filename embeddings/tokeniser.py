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
    
    def encode(self, text):
        """
        Encode a text into a sequence of integer tokens.

        Parameters:
        text (str): The input text to encode.

        Returns:
        list: A list of integer tokens representing the encoded text.
        """
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        try:
            ids = [self.str_to_int[s] for s in preprocessed]
        except KeyError as e:
            raise ValueError(f"Token not in vocabulary: {e}")
        return ids
    
    def decode(self, ids):
        """
        Decode a sequence of integer tokens back into text.

        Parameters:
        ids (list): A list of integer tokens representing encoded text.

        Returns:
        str: The decoded text.
        """
        try:
            text = " ".join([self.int_to_str[i] for i in ids])
        except KeyError as e:
            raise ValueError(f"Token ID not found in vocabulary: {e}")
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
