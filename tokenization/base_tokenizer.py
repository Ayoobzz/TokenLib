import os
import json
import regex as re
import time
import matplotlib.pyplot as plt
from collections import Counter
from concurrent.futures import ThreadPoolExecutor


class BaseTokenizer:
    def __init__(self):
        # Vocabulaire sous forme d'ensemble de tokens connus
        self.vocab = set()
        # Règles de fusion (utilisées notamment par BPE)
        self.merges = {}

    def tokenize(self, text):
        # Méthode abstraite à implémenter dans les sous-classes (BPE, Unigram, etc.)
        raise NotImplementedError("Subclasses should implement this method.")

    def greedy_tokenize(self, text):
        """
        Méthode de tokenisation gloutonne (greedy) de gauche à droite.
        On choisit le plus long sous-mot possible à chaque étape.

        Args:
            text (str): texte brut à tokeniser
        Returns:
            dict: mapping mot -> liste de sous-tokens
        """
        text = self.preprocess_text(text)
        tokens = {}
        for word in text.split():
            i = 0
            word_tokens = []
            while i < len(word):
                match = None
                # On teste les sous-chaînes les plus longues en premier
                for j in range(len(word), i, -1):
                    substr = word[i:j]
                    if substr in self.vocab:
                        match = substr
                        word_tokens.append(match)
                        i = j
                        break
                if not match:
                    # Si aucun match, on découpe caractère par caractère
                    word_tokens.append(word[i])
                    i += 1
            tokens[word] = word_tokens
        return tokens

    def get_token_stats(self, tokenized_text):
        # Statistiques sur le nombre total et unique de tokens
        num_tokens = sum(len(tokens) for tokens in tokenized_text.values())
        unique_tokens = set(token for tokens in tokenized_text.values() for token in tokens)
        return {
            "total_tokens": num_tokens,
            "unique_tokens": len(unique_tokens)
        }

    def preprocess_text(self, text, lowercase=True, remove_punctuation=False):
        # Prétraitement basique du texte : passage en minuscule, suppression de la ponctuation
        if lowercase:
            text = text.lower()
        if remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        return text

    def save_as_json(self, filename):
        """
        Sauvegarde le vocabulaire et les règles de merge dans un fichier JSON.

        Args:
            filename (str): chemin du fichier de sortie
        """
        model_data = {
            "vocab": list(self.vocab),
            "merges": self.merges
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)

    def load_from_json(self, filename):
        """
        Charge le vocabulaire et les merges à partir d'un fichier JSON.

        Args:
            filename (str): chemin du fichier à lire
        """
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.vocab = set(data.get("vocab", []))
            self.merges = data.get("merges", {})

    def benchmark_tokenization(self, corpus):
        # Mesure du temps d'exécution pour la tokenisation d'un corpus
        start_time = time.time()
        self.tokenize(corpus)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Tokenization took {elapsed_time:.4f} seconds.")
        return elapsed_time

    def visualize_subword_frequency(self, subwords):
        # Affiche un graphique de fréquence des sous-mots
        subword_freq = Counter(subwords)
        plt.bar(subword_freq.keys(), subword_freq.values())
        plt.xlabel('Sous-mots')
        plt.ylabel('Fréquence')
        plt.title('Distribution des sous-mots')
        plt.xticks(rotation=90)
        plt.show()

    def batch_tokenize(self, texts):
        # Tokenisation d'une liste de textes en parallèle (multithreading)
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.tokenize, texts))
        return results
