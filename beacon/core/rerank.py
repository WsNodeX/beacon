"""Transparent late-interaction re-ranker.

Rather than a black-box cross-encoder, Beacon ships a *learned-style* feature
reranker whose score is an explicit, debuggable combination of: semantic
similarity (dense cosine), lexical overlap (token Jaccard), and a light
length prior. Weights are plain numbers you can tune; nothing is hidden. On
real deployments you would swap this for a fine-tuned cross-encoder, but the
interface — ``rerank(query, chunks) -> ranked`` — stays identical.
"""

from typing import Dict, List, Optional

from beacon.core._util import cosine, jaccard, token_set
from beacon.core.embeddings import Embedder, HashEmbedder
from beacon.core.types import Chunk, ScoredChunk

DEFAULT_WEIGHTS: Dict[str, float] = {
    "semantic": 0.6,
    "lexical": 0.3,
    "length": 0.1,
}


class FeatureReranker:
    def __init__(
        self,
        embedder: Embedder = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.embedder = embedder or HashEmbedder()
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

    def _features(self, query: str, chunk: Chunk):
        q_vec = self.embedder.encode(query)
        d_vec = self.embedder.encode(chunk.text)
        sem = cosine(q_vec, d_vec)
        lex = jaccard(token_set(query), token_set(chunk.text))
        # length prior: mildly prefer chunks near 200 chars (sweet spot for LLM context)
        L = len(chunk.text)
        length = max(0.0, 1.0 - abs(L - 200) / 800.0)
        return {"semantic": sem, "lexical": lex, "length": length}

    def rerank(self, query: str, chunks: List[Chunk], top_n: Optional[int] = None):
        scored = []
        for c in chunks:
            feats = self._features(query, c)
            s = sum(self.weights[k] * feats[k] for k in self.weights)
            scored.append(ScoredChunk(chunk=c, score=s, stage="rerank"))
        scored.sort(key=lambda x: x.score, reverse=True)
        for i, sc in enumerate(scored, start=1):
            sc.rank = i
        return scored[:top_n] if top_n else scored
