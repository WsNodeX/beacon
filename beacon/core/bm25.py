"""BM25 (Okapi) sparse retriever.

BM25 scores a query against documents using inverse document frequency and
a saturation term that prevents very long documents from dominating. It is
the workhorse of lexical search and complements dense embeddings: embeddings
miss exact keyword matches, BM25 nails them.
"""

import math
from typing import Dict, List

from beacon.core._util import tokenize
from beacon.core.types import Chunk, ScoredChunk


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: List[List[str]] = []
        self._chunks: List[Chunk] = []
        self._df: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._avgdl = 0.0
        self._n = 0

    # ---- training -------------------------------------------------------
    def fit(self, chunks: List[Chunk]) -> "BM25":
        self._chunks = list(chunks)
        self._docs = [c.tokens() for c in chunks]
        self._n = len(self._docs)
        df: Dict[str, int] = {}
        total = 0
        for d in self._docs:
            total += len(d)
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        self._df = df
        self._avgdl = (total / self._n) if self._n else 0.0
        for t, f in df.items():
            # Robertson-Spärck-Jones idf with the +1 smoothing variant
            self._idf[t] = math.log(1.0 + (self._n - f + 0.5) / (f + 0.5))
        return self

    # ---- inference ------------------------------------------------------
    def _score_one(self, query_tokens: List[str], doc: List[str]) -> float:
        if not doc:
            return 0.0
        dl = len(doc)
        score = 0.0
        # term frequency table for this doc
        tf: Dict[str, int] = {}
        for t in doc:
            tf[t] = tf.get(t, 0) + 1
        for q in query_tokens:
            if q not in self._idf:
                continue
            f = tf.get(q, 0)
            if f == 0:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1.0))
            score += self._idf[q] * (f * (self.k1 + 1)) / denom
        return score

    def search(self, query: str, top_k: int = 5) -> List[ScoredChunk]:
        qt = tokenize(query)
        if not self._docs or not qt:
            return []
        scored = []
        for chunk, doc in zip(self._chunks, self._docs):
            s = self._score_one(qt, doc)
            if s > 0.0:
                scored.append(ScoredChunk(chunk=chunk, score=s, stage="bm25"))
        scored.sort(key=lambda x: x.score, reverse=True)
        for i, sc in enumerate(scored[:top_k], start=1):
            sc.rank = i
        return scored[:top_k]

    # ---- (de)serialization ---------------------------------------------
    def to_dict(self):
        return {
            "k1": self.k1,
            "b": self.b,
            "chunks": [
                {
                    "text": c.text,
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "start": c.start,
                    "end": c.end,
                    "metadata": c.metadata,
                }
                for c in self._chunks
            ],
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls(k1=d.get("k1", 1.5), b=d.get("b", 0.75))
        from beacon.core.types import Chunk

        chunks = [
            Chunk(
                text=c["text"],
                chunk_id=c.get("chunk_id", ""),
                doc_id=c.get("doc_id", ""),
                start=c.get("start", 0),
                end=c.get("end", 0),
                metadata=c.get("metadata", {}),
            )
            for c in d.get("chunks", [])
        ]
        obj.fit(chunks)
        return obj
