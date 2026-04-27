import random

from preprocessor import preprocess


def generate_text(model, seed_text, length=15, top_k=5, rng=None):
    words = preprocess(seed_text)
    rng = rng or random

    if len(words) < model.n - 1:
        raise ValueError(f"Seed text must contain at least {model.n - 1} words.")

    for _ in range(length):
        context = tuple(words[-(model.n - 1):])

        candidates = model.candidates(context, top_k=top_k)
        if not candidates:
            break

        possible_words = [word for word, count in candidates]
        weights = [count for word, count in candidates]

        next_word = rng.choices(possible_words, weights=weights, k=1)[0]
        words.append(next_word)

    return " ".join(words)
