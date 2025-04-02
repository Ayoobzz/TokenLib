from base_tokenizer import BaseTokenizer
import regex as re

class DynamicBpe(BaseTokenizer):
    """
    Dynamic BPE tokenizer using dynamic programming to obtain the longest token.
    """
    def __init__(self, pattern=None):
        super().__init__()
        self.vocab = None
        self.merges = None
        self.pattern = re.compile(pattern) if pattern else None

    def train(self, corpus, vocab_size):
        words = corpus.split()
        vocab = set(" ".join(words))
        merges = {}

        while len(vocab) < vocab_size:
            pairs = {}
            for word in words:
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pairs[pair] = pairs.get(pair, 0) + 1
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            new_token = "".join(best_pair)
            vocab.add(new_token)
            merges[best_pair] = new_token

            words = [word.replace(best_pair[0] + best_pair[1], new_token) for word in words]

        self.vocab = vocab
        self.merges = merges

    def encode(self, text):
        if self.pattern:
            words = self.pattern.findall(text)
        else:
            words = text.split()
        
        encoded_tokens = []
        for word in words:
            dp = [None] * (len(word) + 1)
            dp[0] = []
            for i in range(1, len(word) + 1):
                for j in range(i):
                    subword = word[j:i]
                    if subword in self.vocab and dp[j] is not None:
                        candidate = dp[j] + [subword]
                        if dp[i] is None or len(candidate) > len(dp[i]):
                            dp[i] = candidate
            encoded_tokens.extend(dp[-1] if dp[-1] else [word])
        return encoded_tokens
    
    def decode(self, tokens):
        return "".join(tokens)

