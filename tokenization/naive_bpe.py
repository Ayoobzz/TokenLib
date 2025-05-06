import regex as re
from base_tokenizer import BaseTokenizer

class NaiveBpe(BaseTokenizer):
    def __init__(self, pattern=None):
        super().__init__()  # Initialise vocab et merges depuis BaseTokenizer
        self.pattern = re.compile(pattern) if pattern else None

    def train(self, corpus, vocab_size):
        """
        Entraîne le modèle BPE sur un corpus pour générer un vocabulaire de taille cible.

        Args:
            corpus (str): texte d'entraînement
            vocab_size (int): taille du vocabulaire final (minimum 256 pour les octets de base)
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        # Encodage du corpus en UTF-8 et segmentation éventuelle par regex
        if self.pattern:
            text_chunks = re.findall(self.pattern, corpus)
            tokens = [list(chunk.encode("utf-8")) for chunk in text_chunks]
            tokens = [item for sublist in tokens for item in sublist]
        else:
            tokens = list(corpus.encode("utf-8"))

        # Apprentissage des fusions les plus fréquentes
        for i in range(num_merges):
            stat = stats(tokens)
            if not stat:
                break

            pair = max(stat, key=stat.get)
            idx = 256 + i
            tokens = merge(tokens, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        # Mise à jour des attributs hérités
        self.vocab = vocab
        self.merges = merges

    def tokenize(self, text):
        """
        Tokenise un texte en appliquant les fusions BPE apprises.

        Args:
            text (str): texte brut
        Returns:
            List[int]: liste d'IDs de tokens
        """
        if self.pattern:
            text_chunks = re.findall(self.pattern, text)
            tokens = [list(chunk.encode("utf-8")) for chunk in text_chunks]
            tokens = [item for sublist in tokens for item in sublist]
        else:
            tokens = list(text.encode("utf-8"))

        # Appliquer les merges appris dans l'ordre
        while len(tokens) >= 2:
            stat = stats(tokens)
            pair = min(stat, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = merge(tokens, pair, idx)

        return tokens

    def decode(self, tokens):
        """
        Décode une liste de tokens BPE en texte brut.

        Args:
            tokens (List[int]): liste d’IDs
        Returns:
            str: texte décodé
        """
        my_bytes = b"".join(self.vocab[i] for i in tokens)
        return my_bytes.decode("utf-8", errors="replace")


# Fonction utilitaire : calcule la fréquence des paires adjacentes de tokens
def stats(tokens):
    counts = {}
    for pair in zip(tokens, tokens[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

# Fusionne les paires trouvées dans les tokens en remplaçant par l’ID donné
def merge(tokens, pair, rep):
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
