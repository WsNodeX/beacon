"""Reciprocal Rank Fusion (RRF).

RRF merges several ranked lists into one by scoring each item with
``1 / (k + rank)`` and summing across lists. The beauty: it needs no score
normalization because it operates purely on ranks, so a dense list whose
cosine scores live in [0,1] and a sparse BM25 list whose scores can exceed
100 fuse fairly. ``k`` (default 60) tempers the influence of top ranks.
"""

from typing import List

from beacon.core.types import ScoredChunk

DEFAULT_K = 60


def reciprocal_rank_fusion(
    rankings: List[List[ScoredChunk]], k: int = DEFAULT_K
) -> List[ScoredChunk]:
    """Fuse multiple ranked lists into a single merged ranking."""
    fused = {}
    for ranking in rankings:
        for rank, sc in enumerate(ranking, start=1):
            cid = sc.chunk.chunk_id
            if cid not in fused:
                # keep the chunk but reset the (now meaningless) score/stage
                fused[cid] = ScoredChunk(
                    chunk=sc.chunk, score=0.0, stage="fusion", rank=0
                )
            fused[cid].score += 1.0 / (k + rank)
    merged = list(fused.values())
    merged.sort(key=lambda x: x.score, reverse=True)
    for i, sc in enumerate(merged, start=1):
        sc.rank = i
    return merged
