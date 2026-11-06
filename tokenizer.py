"""Character-level tokenizer for MicroMind."""


class CharTokenizer:
    """Maps characters <-> integer ids. Built from the training text."""

    def __init__(self, text):
        # Step 1: collect all unique characters, sorted for determinism
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        # Step 2: build both lookup tables
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # string -> int
        self.itos = {i: ch for i, ch in enumerate(chars)}  # int -> string

    def encode(self, text):
        """Convert string -> list of integer ids."""
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        """Convert list of integer ids -> string."""
        return "".join(self.itos[i] for i in ids)