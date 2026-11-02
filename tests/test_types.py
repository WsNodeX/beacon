import unittest

from beacon.core.types import Chunk, Document, ScoredChunk


class TestTypes(unittest.TestCase):
    def test_doc_id_stable(self):
        d1 = Document(text="hello world")
        d2 = Document(text="hello world")
        self.assertEqual(d1.doc_id, d2.doc_id)

    def test_doc_id_custom(self):
        d = Document(text="x", doc_id="abc")
        self.assertEqual(d.doc_id, "abc")

    def test_chunk_tokens(self):
        c = Chunk(text="The quick brown fox")
        toks = c.tokens()
        self.assertEqual(toks, ["the", "quick", "brown", "fox"])

    def test_scored_repr(self):
        sc = ScoredChunk(chunk=Chunk(text="abc"), score=0.5, stage="bm25")
        self.assertIn("bm25", repr(sc))


if __name__ == "__main__":
    unittest.main()
