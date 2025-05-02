from collections import Counter
import math
from base_tokenizer import BaseTokenizer

class UnigramBPETokenizer(BaseTokenizer):
    def __init__(self, vocab_size=1000):
        super().__init__()
        self.vocab_size = vocab_size
        self.vocab = set()
        self.word_freq = {}

    def train(self, corpus, vocab_size=None):
        if vocab_size:
            self.vocab_size = vocab_size

        corpus = self.preprocess_text(corpus)
        self.word_freq = Counter(corpus.split())
        all_subwords = set()

        # Step 1: Initialize vocabulary with all subwords from the corpus
        for word in self.word_freq:
            for start in range(len(word)):
                for end in range(start + 1, len(word) + 1):
                    all_subwords.add(word[start:end])

        self.vocab = set(all_subwords)
        total_words = sum(self.word_freq.values())

        # Step 2: Prune vocabulary down to target size based on loss
        while len(self.vocab) > self.vocab_size:
            subword_freq = Counter()
            for word, freq in self.word_freq.items():
                for i in range(len(word)):
                    for j in range(i + 1, len(word) + 1):
                        subword = word[i:j]
                        if subword in self.vocab:
                            subword_freq[subword] += freq

            if not subword_freq:
                break

            log_probs = {sw: math.log(freq / total_words) for sw, freq in subword_freq.items()}
            losses = {sw: subword_freq[sw] * log_probs[sw] for sw in self.vocab if sw in log_probs}

            if not losses:
                break

            least_useful = min(losses, key=losses.get)
            self.vocab.discard(least_useful)

    def tokenize(self, text):
        text = self.preprocess_text(text)
        words = text.split()
        tokenized_words = {word: self._tokenize_word(word) for word in words}
        stats = self.get_token_stats(tokenized_words)
        return tokenized_words, stats

    def _tokenize_word(self, word):
        tokens = []
        position = 0

        while position < len(word):
            best_match = None
            for end in range(position + 1, len(word) + 1):
                candidate = word[position:end]
                if candidate in self.vocab:
                    best_match = candidate
            if best_match:
                tokens.append(best_match)
                position += len(best_match)
            else:
                tokens.append(word[position])
                position += 1
        return tokens

    def decode(self, token_dict):
        """Decodes tokenized dictionary into string."""
        return " ".join("".join(tokens) for tokens in token_dict.values())
