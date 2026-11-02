import unittest

from beacon.core.eval import mean_reciprocal_rank, recall_at_k
from beacon.core.types import Chunk, ScoredChunk


def _mk(cid, score):
    return ScoredChunk(chunk=Chunk(text=cid, chunk_id=cid), score=score)


class TestEval(unittest.TestCase):
    def test_recall_full(self):
        res = [_mk("a", 1.0), _mk("b", 0.5), _mk("c", 0.2)]
        self.assertEqual(recall_at_k(res, ["a", "b"], k=3), 1.0)

    def test_recall_partial(self):
        res = [_mk("a", 1.0), _mk("b", 0.5)]
        self.assertEqual(recall_at_k(res, ["a", "c"], k=3), 0.5)

    def test_recall_top_k(self):
        res = [_mk("a", 1.0), _mk("b", 0.5), _mk("c", 0.2)]
        self.assertEqual(recall_at_k(res, ["c"], k=1), 0.0)

    def test_mrr(self):
        res = [_mk("x", 1.0), _mk("y", 0.5)]
        self.assertAlmostEqual(mean_reciprocal_rank(res, ["y"]), 0.5, places=6)

    def test_mrr_no_hit(self):
        res = [_mk("x", 1.0)]
        self.assertEqual(mean_reciprocal_rank(res, ["z"]), 0.0)


if __name__ == "__main__":
    unittest.main()
