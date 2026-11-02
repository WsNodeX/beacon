import unittest

from beacon.core.embeddings import HashEmbedder
from beacon.core._util import cosine


class TestEmbeddings(unittest.TestCase):
    def test_dim(self):
        e = HashEmbedder(dim=128)
        self.assertEqual(e.dim(), 128)

    def test_deterministic(self):
        e = HashEmbedder(dim=256, seed=1)
        a = e.encode("the quick brown fox")
        b = e.encode("the quick brown fox")
        self.assertEqual(a, b)

    def test_normalized(self):
        import math

        e = HashEmbedder(dim=256)
        v = e.encode("hello world foo bar")
        norm = math.sqrt(sum(x * x for x in v))
        self.assertAlmostEqual(norm, 1.0, places=6)

    def test_empty(self):
        e = HashEmbedder(dim=64)
        self.assertEqual(e.encode(""), [0.0] * 64)

    def test_similar_text_higher_cosine(self):
        e = HashEmbedder(dim=256, seed=0)
        same = cosine(e.encode("machine learning model"), e.encode("machine learning model"))
        diff = cosine(e.encode("machine learning model"), e.encode("banana spaceship quantum"))
        self.assertGreater(same, diff)

    def test_to_from_dict(self):
        e = HashEmbedder(dim=200, seed=7)
        e2 = HashEmbedder.from_dict(e.to_dict())
        self.assertEqual(e.dim(), e2.dim())
        self.assertEqual(e.seed, e2.seed)
        self.assertEqual(e.encode("abc"), e2.encode("abc"))


if __name__ == "__main__":
    unittest.main()
