import random
import unittest

from generator import generate_text
from ngram_model import NGramModel
from preprocessor import preprocess


class PreprocessorTests(unittest.TestCase):
    def test_keeps_arabizi_digits_inside_words(self):
        tokens = preprocess("3lach b7al 2026 http://example.com hi@example.com")

        self.assertIn("3lach", tokens)
        self.assertIn("b7al", tokens)
        self.assertNotIn("2026", tokens)


class NGramModelTests(unittest.TestCase):
    def test_predicts_most_likely_next_word(self):
        model = NGramModel("ana kanbghi lmaghrib ana kanbghi darija".split(), n=3)

        word, probability = model.predict_next_word(["ana", "kanbghi"])

        self.assertEqual(word, "lmaghrib")
        self.assertGreater(probability, 0)

    def test_generation_uses_preprocessed_seed_text(self):
        model = NGramModel("ana b7al nta ana b7al houwa".split(), n=3)

        generated = generate_text(model, "Ana b7al", length=1, rng=random.Random(0))

        self.assertTrue(generated.startswith("ana b7al"))
        self.assertEqual(len(generated.split()), 3)


if __name__ == "__main__":
    unittest.main()
