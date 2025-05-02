import argparse
from naive_bpe import NaiveBpe
from dynamic_bpe import DynamicBpe
from unigram_bpe import UnigramBPETokenizer

def load_tokenizer(tokenizer_name, vocab_size=None, pattern=None):
    if tokenizer_name == "naive":
        return NaiveBpe(pattern)
    elif tokenizer_name == "dynamic":
        return DynamicBpe(pattern)
    elif tokenizer_name == "unigram":
        if vocab_size is None:
            raise ValueError("Unigram tokenizer requires vocab_size.")
        return UnigramBPETokenizer(vocab_size)
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer_name}")

def main():
    parser = argparse.ArgumentParser(description="Tokenization CLI Tool")
    parser.add_argument("tokenizer", choices=["naive", "dynamic", "unigram"], help="Tokenizer type")
    parser.add_argument("corpus_file", help="Path to text corpus")
    parser.add_argument("--vocab_size", type=int, default=300, help="Vocabulary size")
    parser.add_argument("--pattern", type=str, help="Regex pattern to split corpus (optional)")
    parser.add_argument("--test_text", type=str, help="Text to tokenize after training")
    parser.add_argument("--decode", action="store_true", help="Decode tokenized output")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark tokenization time")

    args = parser.parse_args()

    with open(args.corpus_file, "r", encoding="utf-8") as f:
        corpus = f.read()

    tokenizer = load_tokenizer(args.tokenizer, args.vocab_size, args.pattern)

    print(f"\n Training {args.tokenizer} tokenizer...")
    tokenizer.train(corpus, args.vocab_size)

    if args.benchmark:
        print("\n Benchmarking tokenization...")
        tokenizer.benchmark_tokenization(corpus)

    if args.test_text:
        print(f"\n Tokenizing: {args.test_text}")
        encoded = tokenizer.tokenize(args.test_text) if args.tokenizer != "unigram" else tokenizer.tokenize(args.test_text)[0]
        print("🧩 Tokens:", encoded)

        if args.decode:
            decoded = tokenizer.decode(encoded)
            print("Decoded:", decoded)

if __name__ == "__main__":
    main()
