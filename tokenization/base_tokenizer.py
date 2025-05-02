import os
import pickle
import regex as re
import time
import matplotlib.pyplot as plt
from collections import Counter
from concurrent.futures import ThreadPoolExecutor


class BaseTokenizer:
    def tokenize(self, text):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_token_stats(self, tokenized_text):
        num_tokens = sum(len(tokens) for tokens in tokenized_text.values())
        unique_tokens = set(token for tokens in tokenized_text.values() for token in tokens)
        return {
            "total_tokens": num_tokens,
            "unique_tokens": len(unique_tokens)
        }

    def preprocess_text(self, text, lowercase=True, remove_punctuation=False):
        if lowercase:
            text = text.lower()
        if remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        return text

    def save_tokenizer(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_tokenizer(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def benchmark_tokenization(self, corpus):
        start_time = time.time()
        self.tokenize(corpus)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Tokenization took {elapsed_time:.4f} seconds.")
        return elapsed_time

    def visualize_subword_frequency(self, subwords):
        subword_freq = Counter(subwords)
        plt.bar(subword_freq.keys(), subword_freq.values())
        plt.xlabel('Subwords')
        plt.ylabel('Frequency')
        plt.title('Subword Frequency Distribution')
        plt.xticks(rotation=90)
        plt.show()

    def batch_tokenize(self, texts):
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.tokenize, texts))
        return results
