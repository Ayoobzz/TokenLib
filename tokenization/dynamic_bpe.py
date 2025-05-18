import regex as re
from base_tokenizer import BaseTokenizer


class DynamicBpe(BaseTokenizer):
    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = re.compile(pattern) if pattern else None

    def train(self, corpus, vocab_size):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        # Tokenization
        if self.pattern:
            text_chunks = re.findall(self.pattern, corpus)
            tokens = [list(chunk.encode("utf-8")) for chunk in text_chunks]
            tokens = [item for sublist in tokens for item in sublist]
        else:
            tokens = list(corpus.encode("utf-8"))

        for i in range(num_merges):
            stat = self._get_stats(tokens)
            if not stat:
                break

            pair = max(stat, key=stat.get)
            idx = 256 + i
            tokens = self._merge(tokens, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        self.vocab = vocab
        self.merges = merges

    def tokenize(self, text):
        """
        Tokenise avec programmation dynamique pour trouver le meilleur découpage.
        """
        if self.pattern:
            text_chunks = re.findall(self.pattern, text)
            tokens = [list(chunk.encode("utf-8")) for chunk in text_chunks]
            input_bytes = [item for sublist in tokens for item in sublist]
        else:
            input_bytes = list(text.encode("utf-8"))

        n = len(input_bytes)
        dp = [float("inf")] * (n + 1)
        path = [-1] * (n + 1)
        dp[0] = 0

        for i in range(1, n + 1):
            for j in range(max(0, i - 10), i):  # limit to reasonable token size
                chunk = input_bytes[j:i]
                token_id = self._get_token_id(chunk)
                if token_id is not None and dp[j] + 1 < dp[i]:
                    dp[i] = dp[j] + 1
                    path[i] = (j, token_id)

        # Backtrack to get tokens
        i = n
        output = []
        while i > 0:
            j, token_id = path[i]
            output.insert(0, token_id)
            i = j

        return output

    def decode(self, tokens):
        my_bytes = b"".join(self.vocab[i] for i in tokens)
        return my_bytes.decode("utf-8", errors="replace")

    def _get_token_id(self, byte_seq):
        for token_id, token_bytes in self.vocab.items():
            if token_bytes == bytes(byte_seq):
                return token_id
        return None

    def _get_stats(self, tokens):
        counts = {}
        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, tokens, pair, rep):
        i = 0
        new_text = []
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_text.append(rep)
                i += 2
            else:
                new_text.append(tokens[i])
                i += 1
        return new_text
