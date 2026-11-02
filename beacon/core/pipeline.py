"""The orchestrating retrieval pipeline.

A :class:`RetrievalPipeline` wires the primitives together:

    documents ──splitter──▶ chunks ──┬─▶ VectorStore (dense)
                                     └─▶ BM25         (sparse)
    query ─▶ VectorStore.search() ┐
            BM25.search()        ┘─▶ reciprocal_rank_fusion ─▶ FeatureReranker

The dense and sparse branches each return a ranked list; RRF merges them into
a single candidate set (robust to either branch failing on a given query),
and the reranker re-scores the merged candidates with an explainable model.
"""

import json
from typing import Dict, List, Optional

from beacon.core.bm25 import BM25
from beacon.core.embeddings import Embedder, HashEmbedder
from beacon.core.fusion import reciprocal_rank_fusion
from beacon.core.rerank import FeatureReranker
from beacon.core.textsplit import RecursiveCharacterSplitter, Splitter
from beacon.core.types import Document, ScoredChunk
from beacon.core.vectorstore import VectorStore


class RetrievalPipeline:
    def __init__(
        self,
        embedder: Embedder = None,
        splitter: Splitter = None,
        vector_top_k: int = 10,
        bm25_top_k: int = 10,
        fusion_k: int = 60,
        rerank_top_n: int = 5,
        rerank_weights: Optional[Dict[str, float]] = None,
    ):
        self.embedder = embedder or HashEmbedder()
        self.splitter = splitter or RecursiveCharacterSplitter()
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.fusion_k = fusion_k
        self.rerank_top_n = rerank_top_n
        self.rerank_weights = rerank_weights
        self.vectorstore = VectorStore(self.embedder)
        self.bm25 = BM25()
        self._chunks = []
        self._indexed = False

    # ---- indexing -------------------------------------------------------
    def index(self, documents: List[Document]) -> "RetrievalPipeline":
        self._chunks = []
        for doc in documents:
            self._chunks.extend(self.splitter.split_document(doc))
        self.bm25.fit(self._chunks)
        self.vectorstore.add(self._chunks)
        self._indexed = True
        return self

    def indexed(self) -> bool:
        return self._indexed

    def stats(self) -> Dict[str, int]:
        return {
            "documents": 0,  # filled by caller metadata if needed
            "chunks": len(self._chunks),
            "dim": self.embedder.dim(),
        }

    # ---- querying -------------------------------------------------------
    def search(self, query: str, top_k: Optional[int] = None) -> List[ScoredChunk]:
        if not self._indexed:
            raise RuntimeError("index() must be called before search()")
        k = top_k if top_k is not None else self.rerank_top_n
        dense = self.vectorstore.search(query, top_k=self.vector_top_k)
        sparse = self.bm25.search(query, top_k=self.bm25_top_k)
        fused = reciprocal_rank_fusion([dense, sparse], k=self.fusion_k)
        # de-duplicate chunks already fused; rerank on the unique chunk set
        unique_chunks = []
        seen = set()
        for sc in fused:
            cid = sc.chunk.chunk_id
            if cid in seen:
                continue
            seen.add(cid)
            unique_chunks.append(sc.chunk)
        reranker = FeatureReranker(self.embedder, self.rerank_weights)
        return reranker.rerank(query, unique_chunks, top_n=k)

    # ---- persistence ----------------------------------------------------
    def to_dict(self):
        return {
            "config": {
                "embedder": self.embedder.to_dict(),
                "splitter": self.splitter.__class__.__name__,
                "vector_top_k": self.vector_top_k,
                "bm25_top_k": self.bm25_top_k,
                "fusion_k": self.fusion_k,
                "rerank_top_n": self.rerank_top_n,
                "rerank_weights": self.rerank_weights,
            },
            "vectorstore": self.vectorstore.to_dict(),
            "bm25": self.bm25.to_dict(),
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

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d):
        from beacon.core.types import Chunk

        cfg = d.get("config", {})
        emb = Embedder.from_dict(cfg.get("embedder", {"type": "HashEmbedder", "dim": 256}))
        obj = cls(
            embedder=emb,
            vector_top_k=cfg.get("vector_top_k", 10),
            bm25_top_k=cfg.get("bm25_top_k", 10),
            fusion_k=cfg.get("fusion_k", 60),
            rerank_top_n=cfg.get("rerank_top_n", 5),
            rerank_weights=cfg.get("rerank_weights"),
        )
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
        obj._chunks = chunks
        obj.vectorstore = VectorStore.from_dict(d.get("vectorstore", {}))
        obj.bm25 = BM25.from_dict(d.get("bm25", {}))
        obj._indexed = bool(chunks)
        return obj

    @classmethod
    def load_json(cls, path: str):
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
