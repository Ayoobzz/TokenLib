from collections import Counter
import math
from base_tokenizer import BaseTokenizer

class UnigramBPETokenizer(BaseTokenizer):

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = set()
        self.word_freq = {}
    
    def train(self, corpus):
        """Trains the tokenizer on a given corpus."""
        self.word_freq = Counter(corpus.split())
        all_subwords = set()

        for word in self.word_freq:
            for start in range(len(word)):
                for end in range(start + 1, len(word) + 1):
                    all_subwords.add(word[start:end])

        self.vocab = all_subwords  # Start with all possible subwords
        total_words = sum(self.word_freq.values())

        while len(self.vocab) > self.vocab_size:
            subword_freq = Counter()
            for word, freq in self.word_freq.items():
                for i in range(len(word)):
                    for j in range(i + 1, len(word) + 1):
                        subword = word[i:j]
                        if subword in self.vocab:
                            subword_freq[subword] += freq

            log_probs = {sw: math.log(freq / total_words) for sw, freq in subword_freq.items()}
            losses = {sw: subword_freq.get(sw, 0) * log_probs.get(sw, 0) for sw in self.vocab}

            if not losses:
                break
            
            least_useful = min(losses, key=losses.get)
            self.vocab.remove(least_useful)

    def tokenize(self, text):
        """Tokenizes input text using the trained vocabulary."""
        words = text.split()
        tokenized_words = {word: self._tokenize_word(word) for word in words}
        
        stats = self.get_token_stats(tokenized_words)
        return tokenized_words, stats

    def _tokenize_word(self, word):
        """Tokenizes a single word based on trained vocabulary."""
        tokens = []
        position = 0
        
        while position < len(word):
            best_match = None
            for end in range(position + 1, len(word) + 1):
                if word[position:end] in self.vocab:
                    best_match = word[position:end]

            if best_match:
                tokens.append(best_match)
                position += len(best_match)
            else:
                tokens.append(word[position])
                position += 1
        
        return tokens
