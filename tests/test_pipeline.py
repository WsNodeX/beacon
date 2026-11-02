import os
import tempfile
import unittest

from beacon.core.pipeline import RetrievalPipeline
from beacon.core.types import Document


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            Document(text="BM25 ranks documents by term frequency and inverse "
                           "document frequency for search engines.", doc_id="bm25"),
            Document(text="A bloom filter is a probabilistic set with no false "
                           "negatives but possible false positives.", doc_id="bloom"),
            Document(text="HyperLogLog estimates set cardinality using little "
                           "memory via leading zero runs.", doc_id="hll"),
        ]
        self.pipe = RetrievalPipeline()
        self.pipe.index(self.corpus)

    def test_index_stats(self):
        self.assertEqual(self.pipe.stats()["chunks"], 3)
        self.assertTrue(self.pipe.indexed())

    def test_search_finds_bm25(self):
        res = self.pipe.search("how do search engines rank by term frequency", top_k=2)
        self.assertTrue(res)
        self.assertEqual(res[0].chunk.chunk_id, "bm25_c0")

    def test_search_finds_bloom(self):
        res = self.pipe.search("probabilistic set false negatives", top_k=1)
        self.assertEqual(res[0].chunk.chunk_id, "bloom_c0")

    def test_search_requires_index(self):
        p = RetrievalPipeline()
        with self.assertRaises(RuntimeError):
            p.search("x")

    def test_persistence_roundtrip(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "idx.json")
        self.pipe.save_json(path)
        loaded = RetrievalPipeline.load_json(path)
        self.assertTrue(loaded.indexed())
        r1 = self.pipe.search("term frequency ranking", top_k=2)
        r2 = loaded.search("term frequency ranking", top_k=2)
        self.assertEqual([s.chunk.chunk_id for s in r1],
                         [s.chunk.chunk_id for s in r2])

    def test_deterministic_results(self):
        a = self.pipe.search("cardinality estimate memory", top_k=3)
        b = self.pipe.search("cardinality estimate memory", top_k=3)
        self.assertEqual([s.chunk.chunk_id for s in a],
                         [s.chunk.chunk_id for s in b])


if __name__ == "__main__":
    unittest.main()
