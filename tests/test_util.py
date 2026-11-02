import unittest

from beacon.core._util import cosine, jaccard, tokenize, token_set, split_sentences


class TestUtil(unittest.TestCase):
    def test_tokenize_lowercases_and_splits(self):
        self.assertEqual(tokenize("Hello World!"), ["hello", "world"])
        self.assertEqual(tokenize(""), [])

    def test_tokenize_cjk(self):
        toks = tokenize("北京 上海")
        # each CJK character becomes its own token
        self.assertIn("北", toks)
        self.assertIn("京", toks)
        self.assertIn("上", toks)
        self.assertIn("海", toks)
        self.assertNotIn("上海", toks)

    def test_cosine_identical(self):
        a = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(cosine(a, a), 1.0, places=6)

    def test_cosine_orthogonal(self):
        self.assertAlmostEqual(cosine([1.0, 0.0], [0.0, 1.0]), 0.0, places=6)

    def test_cosine_zero_vector(self):
        self.assertEqual(cosine([0.0, 0.0], [1.0, 1.0]), 0.0)

    def test_cosine_dim_mismatch(self):
        with self.assertRaises(ValueError):
            cosine([1.0], [1.0, 2.0])

    def test_jaccard(self):
        self.assertEqual(jaccard({"a", "b"}, {"a", "b"}), 1.0)
        self.assertEqual(jaccard({"a"}, {"b"}), 0.0)
        self.assertAlmostEqual(jaccard({"a", "b"}, {"b", "c"}), 1 / 3, places=6)

    def test_split_sentences(self):
        s = split_sentences("Hello world. How are you? I am fine.")
        self.assertEqual(len(s), 3)
        self.assertTrue(s[0].startswith("Hello"))


if __name__ == "__main__":
    unittest.main()
