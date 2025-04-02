from unigram_bpe import UnigramBPETokenizer

if __name__ == "__main__":
    corpus = "low lower lowest lowing"
    vocab_size = 10
    tokenizer = UnigramBPETokenizer(vocab_size)
    tokenizer.train(corpus)
    
    text = "lower lowest"
    tokens, stats = tokenizer.tokenize(text)

    print("Final Vocabulary:", sorted(tokenizer.vocab))
    print("Tokenized Output:", tokens)
    print("Tokenization Stats:", stats)
