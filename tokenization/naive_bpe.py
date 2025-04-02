import regex as re
from base_tokenizer import BaseTokenizer

class NaiveBpe(BaseTokenizer):
    """
    Naive implementation of Byte Pair Encoding (BPE) tokenizer.
    """
    def __init__(self, pattern=None):
        super().__init__()
        self.vocab = None
        self.merges = None
        self.pattern = re.compile(pattern) if pattern else None

    def train(self, corpus, vocab_size):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        if self.pattern:
            text_chunks = re.findall(self.pattern, corpus)
            tokens = [list(chunk.encode("utf-8")) for chunk in text_chunks]
        else:
            tokens = list(corpus.encode("utf-8"))

        for i in range(num_merges):
            stat = stats(tokens)
            if not stat:
                break

            pair = max(stat, key=stat.get)
            idx = 256 + i
            tokens = merge(tokens, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        self.merges = merges
        self.vocab = vocab

    def encode(self, text):
        if self.pattern:
            text_chunks = re.findall(self.pattern, text)
            tokens = [list(chunk.encode("utf-8")) for chunk in text_chunks]
            tokens = [item for sublist in tokens for item in sublist]
        else:
            tokens = list(text.encode("utf-8"))

        while len(tokens) >= 2:
            stat = stats(tokens)
            pair = min(stat, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = merge(tokens, pair, idx)

        return tokens

    def decode(self, tokens):
        my_bytes = b"".join(self.vocab[i] for i in tokens)
        return my_bytes.decode("utf-8", errors="replace")




def stats(tokens):
 
    counts = {}

    if isinstance(tokens[0], list):
         for token_list in tokens:
            for pair in zip(token_list, token_list[1:]):
                counts[pair] = counts.get(pair, 0) + 1
    else:
         for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1

    return counts


def merge(text, pair, rep):
    i = 0
    new_text = []
    while i < len(text):
        if i < len(text) - 1 and text[i] == pair[0] and text[i + 1] == pair[1]:
            new_text.append(rep)
            i += 2
        else:
            new_text.append(text[i])
            i += 1
    return new_text