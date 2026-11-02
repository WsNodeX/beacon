"""In-memory dense vector store with exact cosine search.

Embeddings are computed once at index time and cached, so repeated queries
are a single cosine sweep. Good enough for thousands of chunks on a laptop;
the same API can back a disk or ANN index later without changing callers.
"""

from typing import List

from beacon.core._util import cosine
from beacon.core.embeddings import Embedder, HashEmbedder
from beacon.core.types import Chunk, ScoredChunk


class VectorStore:
    def __init__(self, embedder: Embedder = None):
        self.embedder = embedder or HashEmbedder()
        self._chunks: List[Chunk] = []
        self._vectors = []

    def add(self, chunks: List[Chunk]) -> "VectorStore":
        for c in chunks:
            self._chunks.append(c)
            self._vectors.append(self.embedder.encode(c.text))
        return self

    def search(self, query: str, top_k: int = 5) -> List[ScoredChunk]:
        if not self._chunks:
            return []
        q = self.embedder.encode(query)
        scored = []
        for chunk, vec in zip(self._chunks, self._vectors):
            s = cosine(q, vec)
            if s > 0.0:
                scored.append(ScoredChunk(chunk=chunk, score=s, stage="vector"))
        scored.sort(key=lambda x: x.score, reverse=True)
        for i, sc in enumerate(scored[:top_k], start=1):
            sc.rank = i
        return scored[:top_k]

    # ---- (de)serialization ---------------------------------------------
    def to_dict(self):
        return {
            "embedder": self.embedder.to_dict(),
            "entries": [
                {
                    "chunk": {
                        "text": c.text,
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "start": c.start,
                        "end": c.end,
                        "metadata": c.metadata,
                    },
                    "vector": v,
                }
                for c, v in zip(self._chunks, self._vectors)
            ],
        }

    @classmethod
    def from_dict(cls, d):
        from beacon.core.types import Chunk

        emb = Embedder.from_dict(d.get("embedder", {"type": "HashEmbedder", "dim": 256}))
        obj = cls(embedder=emb)
        chunks = []
        vecs = []
        for e in d.get("entries", []):
            c = e["chunk"]
            chunks.append(
                Chunk(
                    text=c["text"],
                    chunk_id=c.get("chunk_id", ""),
                    doc_id=c.get("doc_id", ""),
                    start=c.get("start", 0),
                    end=c.get("end", 0),
                    metadata=c.get("metadata", {}),
                )
            )
            vecs.append(list(e["vector"]))
        obj._chunks = chunks
        obj._vectors = vecs
        return obj
