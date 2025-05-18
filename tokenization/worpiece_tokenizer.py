import regex as re
from typing import List, Tuple
from .base_tokenizer import BaseTokenizer
import collections

class WordPieceTokenizer(BaseTokenizer):
    def __init__(self, vocab_size: int = 5000, min_frequency: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = {"[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"}
        self.trained = False

    def train(self, corpus: List[str]) -> None:
        """
        Entraîne le WordPiece tokenizer sur un corpus de textes.
        """
        # Prétraitement et segmentation initiale en caractères + préfixe de mot
        words = []
        for text in corpus:
            text = self.preprocess_text(text)  # lowercase / punctuation options
            # isoler ponctuation
            text = re.sub(r'([.,!?;:])', r' \1 ', text)
            for w in text.split():
                words.append(['▁'] + list(w))

        # Vocab initial : caractères + tokens spéciaux
        char_vocab = {ch for word in words for ch in word}
        self.vocab = self.special_tokens.union(char_vocab)

        # Comptage initial de fréquences
        token_freqs = collections.Counter(tuple(word) for word in words)

        # Croissance par fusion de paires fréquentes
        while len(self.vocab) < self.vocab_size:
            pair_counts = collections.defaultdict(int)
            for tok_seq, freq in token_freqs.items():
                for i in range(len(tok_seq) - 1):
                    pair_counts[(tok_seq[i], tok_seq[i+1])] += freq
            if not pair_counts:
                break

            best_pair, best_freq = max(pair_counts.items(), key=lambda x: x[1])
            if best_freq < self.min_frequency:
                break

            new_token = ''.join(best_pair)
            self.vocab.add(new_token)
            # enregistrer la fusion
            self.merges[best_pair] = new_token
            # appliquer fusion sur tous les token_seq
            new_freqs = collections.Counter()
            for seq, freq in token_freqs.items():
                merged = []
                i = 0
                while i < len(seq):
                    if i < len(seq)-1 and (seq[i], seq[i+1]) == best_pair:
                        merged.append(new_token)
                        i += 2
                    else:
                        merged.append(seq[i])
                        i += 1
                new_freqs[tuple(merged)] += freq
            token_freqs = new_freqs

        self.trained = True

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenisation gloutonne mot à mot en sous-tokens WordPiece.
        """
        if not self.trained:
            raise ValueError("Le tokenizer doit être entraîné avant utilisation.")

        # Prétraitement identique à l'entraînement
        text = self.preprocess_text(text)
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        tokens: List[str] = []

        for w in text.split():
            word = '▁' + w
            i = 0
            while i < len(word):
                match = None
                # tenter la plus longue sous-chaîne
                for j in range(len(word), i, -1):
                    sub = word[i:j]
                    if sub in self.vocab:
                        match = sub
                        break
                if match:
                    tokens.append(match)
                    i += len(match)
                else:
                    tokens.append("[UNK]")
                    i += 1
                    break
        return tokens

    def decode(self, tokens: List[str]) -> str:
        """
        Reconstruit le texte brut à partir de la liste de sous-tokens.
        """
        text = ''.join(tokens)
        return text.replace('▁', ' ')
