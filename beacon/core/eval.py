"""Offline retrieval evaluation metrics.

``recall_at_k`` asks: of the gold chunks for a query, how many appear in the
top-k results? ``mean_reciprocal_rank`` asks: on average, where in the ranked
list does the first relevant hit show up? Both are computed without any
network call, so you can track retrieval quality as you tune the pipeline.
"""

from typing import Dict, List

from beacon.core.types import ScoredChunk


def recall_at_k(results: List[ScoredChunk], relevant: List[str], k: int) -> float:
    """Fraction of ``relevant`` chunk ids present in the top ``k`` results."""
    if not relevant:
        return 0.0
    top = {sc.chunk.chunk_id for sc in results[:k]}
    hits = sum(1 for r in relevant if r in top)
    return hits / len(relevant)


def mean_reciprocal_rank(results: List[ScoredChunk], relevant: List[str]) -> float:
    """Reciprocal rank of the first relevant hit, averaged over one query."""
    if not relevant:
        return 0.0
    relset = set(relevant)
    for i, sc in enumerate(results, start=1):
        if sc.chunk.chunk_id in relset:
            return 1.0 / i
    return 0.0


def evaluate(
    queries: List[str],
    relevant_map: Dict[str, List[str]],
    top_k: int = 5,
    search_fn=None,
) -> Dict[str, float]:
    """Run a battery of queries through ``search_fn`` and return aggregate
    recall@k and MRR. ``search_fn(query) -> List[ScoredChunk]``."""
    if search_fn is None:
        raise ValueError("evaluate requires a search_fn")
    recalls, mrrs = [], []
    for q in queries:
        res = search_fn(q)
        rel = relevant_map.get(q, [])
        recalls.append(recall_at_k(res, rel, top_k))
        mrrs.append(mean_reciprocal_rank(res, rel))
    n = len(queries) or 1
    return {
        "queries": len(queries),
        "recall@%d" % top_k: sum(recalls) / n,
        "mrr": sum(mrrs) / n,
    }
