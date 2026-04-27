from collections import Counter, defaultdict


def create_ngrams(tokens, n):
    if n <= 0:
        raise ValueError("n must be greater than 0.")

    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


class NGramModel:
    def __init__(self, tokens, n=3):
        if n < 2:
            raise ValueError("n must be at least 2.")
        if len(tokens) < n:
            raise ValueError(f"At least {n} tokens are required to train the model.")

        self.n = n
        self.tokens = tokens
        self.vocab = set(tokens)
        self.V = len(self.vocab)

        self.ngram_counts = Counter(create_ngrams(tokens, n))
        self.context_counts = Counter(create_ngrams(tokens, n - 1))

        self.context_to_words = defaultdict(Counter)

        for ngram, count in self.ngram_counts.items():
            context = ngram[:-1]
            word = ngram[-1]
            self.context_to_words[context][word] = count

    def probability(self, word, context):
        context = tuple(context)
        if len(context) != self.n - 1:
            raise ValueError(f"Context must contain exactly {self.n - 1} tokens.")

        ngram = context + (word,)

        return (self.ngram_counts[ngram] + 1) / (self.context_counts[context] + self.V)

    def candidates(self, context, top_k=None):
        context = tuple(context)
        words = self.context_to_words.get(context)
        if not words:
            return []

        return words.most_common(top_k)

    def predict_next_word(self, context):
        context = tuple(context)

        candidates = self.candidates(context)
        if not candidates:
            return None, 0

        best_word = None
        best_prob = 0

        for word, _count in candidates:
            prob = self.probability(word, context)

            if prob > best_prob:
                best_prob = prob
                best_word = word

        return best_word, best_prob
