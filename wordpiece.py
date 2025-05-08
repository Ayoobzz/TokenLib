import re
import json
import collections
from typing import Dict, List, Tuple


class WordPieceVocabBuilder:
    def __init__(self, vocab_size: int = 5000, min_frequency: int = 2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        self.vocab = {}

    def build_vocab(self, corpus: List[str]) -> Dict[str, int]:
        words = self._preprocess_corpus(corpus)
        char_vocab = self._get_initial_vocab(words)
        vocab = {token: i for i, token in enumerate(self.special_tokens + list(char_vocab))}

        word_tokens = [tuple(word) for word in words]  # Tokenisation initiale sous forme de tuples immuables
        token_freqs = collections.Counter(word_tokens)

        while len(vocab) < self.vocab_size:
            token_counts = self._count_token_pairs(token_freqs)
            if not token_counts:
                break

            best_pair, best_freq = max(token_counts.items(), key=lambda x: x[1])
            if best_freq < self.min_frequency:
                break

            new_token = ''.join(best_pair)
            vocab[new_token] = len(vocab)

            word_tokens, token_freqs = self._apply_merge(word_tokens, token_freqs, best_pair, new_token)

            print(f"Nombre de tokens dans le vocabulaire : {len(vocab)}", end="\r")  # 🔥 Affichage en direct

        print()  # Nouvelle ligne après la fin du processus
        self.vocab = {token: idx for idx, token in enumerate(vocab.keys())}

        return self.vocab

    def _preprocess_corpus(self, corpus: List[str]) -> List[List[str]]:
        words = []
        for text_block in corpus:
            text_block = text_block.lower()
            text_block = re.sub(r'([.,!?;:])', r' \1 ', text_block)
            text_block = re.sub(r'[^a-zàáâäæãåāéèêëēėęîïíīįìôöòóœōõûüùúūÿ0-9.,!?;: ]', ' ', text_block)
            words.extend([['▁'] + list(word) + ['ª'] for word in text_block.split()])  # Ajout du marqueur de fin de mot
        return words

    def get_vocab_list(self) -> List[str]:
        if not self.vocab:
            raise ValueError("Le vocabulaire doit être construit avant d'être récupéré.")

        vocab_list = [token.replace("▁", "") for token in self.vocab.keys()]
        return vocab_list

    def _get_initial_vocab(self, words: List[List[str]]) -> set:
        return {char for word in words for char in word}

    def _count_token_pairs(self, token_freqs: collections.Counter) -> Dict[Tuple[str, str], int]:
        pair_counts = collections.defaultdict(int)
        for word, freq in token_freqs.items():
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i + 1])] += freq
        return pair_counts

    def _apply_merge(self, word_tokens, token_freqs, pair, new_token):
        new_word_tokens = []
        new_token_freqs = collections.Counter()

        for word, freq in token_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            new_word_tokens.append(new_word)
            new_token_freqs[new_word] += freq

        return new_word_tokens, new_token_freqs

    def save_vocab_json(self, path: str) -> None:
        if not self.vocab:
            raise ValueError("Le vocabulaire doit être construit avant d'être sauvegardé.")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    def get_vocab_as_json(self) -> str:
        if not self.vocab:
            raise ValueError("Le vocabulaire doit être construit avant d'être converti en JSON.")
        return json.dumps(self.vocab, ensure_ascii=False, indent=2)

    def save_vocab_list_json(self, path: str) -> None:
        vocab_list = self.get_vocab_list()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_list, f, ensure_ascii=False, indent=2)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construire un vocabulaire WordPiece à partir d'un fichier texte.")
    parser.add_argument("input_file", type=str, help="Chemin du fichier texte d'entrée")
    parser.add_argument("output_file", type=str, help="Chemin du fichier JSON de sortie")
    parser.add_argument("--vocab_size", type=int, default=5000, help="Taille du vocabulaire")
    parser.add_argument("--min_frequency", type=int, default=2, help="Fréquence minimale pour fusionner des tokens")

    args = parser.parse_args()

    # Lire le contenu du fichier texte
    with open(args.input_file, "r", encoding="utf-8") as f:
        corpus = f.readlines()

    # Créer l'instance du builder et générer le vocabulaire
    builder = WordPieceVocabBuilder(vocab_size=args.vocab_size, min_frequency=args.min_frequency)
    vocab = builder.build_vocab(corpus)

    # Sauvegarder le vocabulaire en JSON
    builder.save_vocab_list_json(args.output_file)

    print(f"Vocabulaire généré et sauvegardé dans {args.output_file}")
