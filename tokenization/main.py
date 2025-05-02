from collections import Counter
import math
from base_tokenizer import BaseTokenizer
from unigram_bpe import UnigramBPETokenizer


if __name__ == "__main__":
    corpus = "The quick brown fox jumps over the lazy dog. A journey of a thousand miles begins with a single step."

    # Create and train the tokenizer
    unigram_bpe = UnigramBPETokenizer(vocab_size=50)
    unigram_bpe.train(corpus)

    # Save and load the tokenizer
    unigram_bpe.save_tokenizer("unigram_bpe_tokenizer.pkl")
    loaded_unigram_bpe = BaseTokenizer.load_tokenizer("unigram_bpe_tokenizer.pkl")

    # Preprocess and tokenize a new text
    new_text = "hello world!"
    processed_text = unigram_bpe.preprocess_text(new_text, lowercase=True, remove_punctuation=True)
    encoded_tokens, stats = unigram_bpe.tokenize(processed_text)

    print(f"Encoded Tokens: {encoded_tokens}")
    print("Vocabulary:", unigram_bpe.vocab)

    # Benchmark tokenization
    unigram_bpe.benchmark_tokenization(corpus)

    # Subword visualization
    subwords = unigram_bpe.tokenize(corpus)[0]
    unigram_bpe.visualize_subword_frequency([token for tokens in subwords.values() for token in tokens])

    # Batch tokenization
    batch_texts = ["Sample text 1.", "Another text.", "More texts for batch processing."]
    batch_results = unigram_bpe.batch_tokenize(batch_texts)
    print(batch_results)
