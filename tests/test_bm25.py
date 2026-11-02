import unittest

from beacon.core.bm25 import BM25
from beacon.core.types import Chunk


def _chunks():
    return [
        Chunk(text="the cat sat on the mat", chunk_id="c0"),
        Chunk(text="the dog played in the park", chunk_id="c1"),
        Chunk(text="a mat is used for yoga exercise", chunk_id="c2"),
    ]


class TestBM25(unittest.TestCase):
    def test_empty_query(self):
        bm = BM25().fit(_chunks())
        self.assertEqual(bm.search("", top_k=3), [])

    def test_relevant_ranking(self):
        bm = BM25().fit(_chunks())
        res = bm.search("cat mat", top_k=3)
        self.assertTrue(res)
        self.assertEqual(res[0].chunk.chunk_id, "c0")

    def test_no_match_returns_empty(self):
        bm = BM25().fit(_chunks())
        res = bm.search("zzzqqq nonexistent term", top_k=3)
        # 'zzzqqq' and 'nonexistent' are unseen -> 0 score -> filtered out
        self.assertEqual(res, [])

    def test_top_k_limits(self):
        bm = BM25().fit(_chunks())
        res = bm.search("the", top_k=1)
        self.assertEqual(len(res), 1)

    def test_ranks_assigned(self):
        bm = BM25().fit(_chunks())
        res = bm.search("the cat", top_k=3)
        self.assertEqual(res[0].rank, 1)

    def test_to_from_dict(self):
        bm = BM25(k1=1.2, b=0.5).fit(_chunks())
        bm2 = BM25.from_dict(bm.to_dict())
        r1 = bm.search("cat mat", top_k=3)
        r2 = bm2.search("cat mat", top_k=3)
        self.assertEqual([s.chunk.chunk_id for s in r1], [s.chunk.chunk_id for s in r2])


if __name__ == "__main__":
    unittest.main()
