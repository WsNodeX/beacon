import unittest

from beacon.core.fusion import reciprocal_rank_fusion
from beacon.core.types import Chunk, ScoredChunk


class TestFusion(unittest.TestCase):
    def _mk(self, cid, score, stage):
        return ScoredChunk(chunk=Chunk(text=cid, chunk_id=cid), score=score, stage=stage)

    def test_fusion_merges(self):
        a = [self._mk("x", 0.9, "vector"), self._mk("y", 0.4, "vector")]
        b = [self._mk("y", 5.0, "bm25"), self._mk("z", 3.0, "bm25")]
        fused = reciprocal_rank_fusion([a, b], k=60)
        ids = [s.chunk.chunk_id for s in fused]
        self.assertEqual(set(ids), {"x", "y", "z"})

    def test_fusion_ranks(self):
        a = [self._mk("x", 0.9, "vector"), self._mk("y", 0.4, "vector")]
        b = [self._mk("y", 5.0, "bm25"), self._mk("z", 3.0, "bm25")]
        fused = reciprocal_rank_fusion([a, b], k=60)
        # 'y' is rank 1 in both -> highest fused score
        self.assertEqual(fused[0].chunk.chunk_id, "y")

    def test_empty_input(self):
        self.assertEqual(reciprocal_rank_fusion([]), [])

    def test_single_list(self):
        a = [self._mk("x", 0.9, "vector")]
        fused = reciprocal_rank_fusion([a])
        self.assertEqual(len(fused), 1)
        self.assertGreater(fused[0].score, 0.0)


if __name__ == "__main__":
    unittest.main()
