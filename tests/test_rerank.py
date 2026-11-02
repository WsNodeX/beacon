import unittest

from beacon.core.rerank import FeatureReranker
from beacon.core.types import Chunk


class TestRerank(unittest.TestCase):
    def test_rerank_prefers_relevant(self):
        rr = FeatureReranker()
        chunks = [
            Chunk(text="the weather is sunny today", chunk_id="a"),
            Chunk(text="neural networks learn from data", chunk_id="b"),
            Chunk(text="a weather prediction model uses machine learning", chunk_id="c"),
        ]
        out = rr.rerank("weather prediction model", chunks, top_n=2)
        ids = [s.chunk.chunk_id for s in out]
        self.assertIn("c", ids)
        self.assertEqual(ids[0], "c")

    def test_top_n_limits(self):
        rr = FeatureReranker()
        chunks = [Chunk(text="item %d" % i, chunk_id="c%d" % i) for i in range(5)]
        out = rr.rerank("item", chunks, top_n=3)
        self.assertEqual(len(out), 3)

    def test_scores_descending(self):
        rr = FeatureReranker()
        chunks = [Chunk(text="alpha beta", chunk_id="a"),
                  Chunk(text="beta gamma", chunk_id="b")]
        out = rr.rerank("alpha beta gamma", chunks)
        self.assertGreaterEqual(out[0].score, out[1].score)

    def test_custom_weights(self):
        rr = FeatureReranker(weights={"semantic": 1.0, "lexical": 0.0, "length": 0.0})
        chunks = [Chunk(text="deep learning", chunk_id="a"),
                  Chunk(text="completely unrelated text here", chunk_id="b")]
        out = rr.rerank("deep learning", chunks, top_n=1)
        self.assertEqual(out[0].chunk.chunk_id, "a")


if __name__ == "__main__":
    unittest.main()
