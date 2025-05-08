import json
import argparse
from collections import Counter


def load_vocab(file_path):
    """Charge un vocabulaire JSON sous forme d'un ensemble de tokens."""
    with open(file_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return set(vocab)


def compare_vocabs(wordpiece_vocab, bpe_vocab):
    """Compare deux vocabulaires et affiche les résultats."""

    # Intersection et différences
    common_tokens = wordpiece_vocab & bpe_vocab
    unique_wordpiece = wordpiece_vocab - bpe_vocab
    unique_bpe = bpe_vocab - wordpiece_vocab

    print("\nComparaison des vocabulaires")
    print(f"Taille des vocabulaires : {len(wordpiece_vocab)} (WordPiece) vs {len(bpe_vocab)} (BPE)")
    print(f"Tokens en commun : {len(common_tokens)} ({len(common_tokens) / len(wordpiece_vocab) * 100:.2f}%)")
    print(f"Tokens uniques à WordPiece : {len(unique_wordpiece)}")
    print(f"Tokens uniques à BPE : {len(unique_bpe)}\n")

    # Longueur moyenne des tokens
    avg_length_wordpiece = sum(len(token) for token in wordpiece_vocab) / len(wordpiece_vocab)
    avg_length_bpe = sum(len(token) for token in bpe_vocab) / len(bpe_vocab)

    print(f"Longueur moyenne des tokens : {avg_length_wordpiece:.2f} (WordPiece) vs {avg_length_bpe:.2f} (BPE)\n")

    # Analyse des préfixes et suffixes
    def get_most_common_prefix_suffix(vocab, top_n=10):
        prefixes = Counter(token[:2] for token in vocab if len(token) > 1)
        suffixes = Counter(token[-2:] for token in vocab if len(token) > 1)
        return prefixes.most_common(top_n), suffixes.most_common(top_n)

    wordpiece_prefixes, wordpiece_suffixes = get_most_common_prefix_suffix(wordpiece_vocab)
    bpe_prefixes, bpe_suffixes = get_most_common_prefix_suffix(bpe_vocab)

    print("10 préfixes les plus fréquents")
    print("WordPiece :", wordpiece_prefixes)
    print("BPE       :", bpe_prefixes, "\n")

    print("10 suffixes les plus fréquents")
    print("WordPiece :", wordpiece_suffixes)
    print("BPE       :", bpe_suffixes, "\n")

    # Affichage de quelques exemples de différences
    print("Exemples de tokens uniques à WordPiece :", list(unique_wordpiece)[:10])
    print("Exemples de tokens uniques à BPE :", list(unique_bpe)[:10])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare deux vocabulaires JSON (WordPiece vs BPE).")
    parser.add_argument("wordpiece_vocab", type=str, help="Fichier JSON du vocabulaire WordPiece")
    parser.add_argument("bpe_vocab", type=str, help="Fichier JSON du vocabulaire BPE")

    args = parser.parse_args()

    # Charger les vocabulaires
    wordpiece_vocab = load_vocab(args.wordpiece_vocab)
    bpe_vocab = load_vocab(args.bpe_vocab)

    # Comparer
    compare_vocabs(wordpiece_vocab, bpe_vocab)
