import argparse
import random

from data_loader import load_corpus
from preprocessor import preprocess
from ngram_model import NGramModel
from generator import generate_text

DATA_FOLDER = "music-data"
TOKEN_LIMIT = 50000
N = 3
DEFAULT_SEEDS = [
    "hadchi li",
    "dounya ghir",
    "dima dima",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train and sample a Darija n-gram language model.")
    parser.add_argument("--data-folder", default=DATA_FOLDER, help="Folder containing corpus text files.")
    parser.add_argument("--token-limit", type=int, default=TOKEN_LIMIT, help="Maximum number of tokens to train on.")
    parser.add_argument("-n", "--ngram-size", type=int, default=N, help="N-gram size. Use 3 for a trigram model.")
    parser.add_argument("--length", type=int, default=12, help="Number of generated tokens per seed.")
    parser.add_argument("--top-k", type=int, default=5, help="Sample only from the k most common next words.")
    parser.add_argument("--random-seed", type=int, default=None, help="Make generated examples reproducible.")
    parser.add_argument("--seed", action="append", dest="seeds", help="Seed text. Can be passed more than once.")
    return parser.parse_args()


def train_model(data_folder, token_limit, n):
    text = load_corpus(data_folder)
    tokens = preprocess(text)

    if token_limit is not None:
        tokens = tokens[:token_limit]

    if len(tokens) < n:
        raise ValueError(f"Not enough tokens to train the model: need {n}, got {len(tokens)}.")

    return NGramModel(tokens, n=n), tokens


def main():
    args = parse_args()
    rng = random.Random(args.random_seed)
    seeds = args.seeds or DEFAULT_SEEDS

    model, tokens = train_model(args.data_folder, args.token_limit, args.ngram_size)

    print("Darija N-Gram Language Model")
    print("----------------------------")
    print("Data folder:", args.data_folder)
    print("N-gram size:", args.ngram_size)
    print("Total tokens used:", len(tokens))
    print("Vocabulary size:", len(model.vocab))

    print("\nModel trained successfully.")

    context = tokens[0:args.ngram_size - 1]
    word, prob = model.predict_next_word(context)

    print("\nPrediction example:")
    print("Context:", context)
    print("Predicted next word:", word)
    print("Probability:", round(prob, 6))

    print("\nGenerated text examples:")

    for seed in seeds:
        try:
            print(f"\nSeed: {seed}")
            print(generate_text(model, seed, length=args.length, top_k=args.top_k, rng=rng))
        except ValueError as exc:
            print(f"Could not generate text for seed '{seed}': {exc}")


if __name__ == "__main__":
    main()
