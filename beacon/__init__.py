"""Beacon — a zero-dependency hybrid retrieval & RAG toolkit for Python.

Beacon turns a pile of documents into a queryable index that combines
dense (embedding) retrieval with sparse (BM25) retrieval, fuses the two
with Reciprocal Rank Fusion, and re-ranks the merged candidates with a
transparent feature model. Everything runs offline on a stock Python 3.9+
interpreter — no model downloads, no API keys, no native extensions.
"""

from beacon.core.types import Chunk, Document, ScoredChunk
from beacon.core.embeddings import Embedder, HashEmbedder
from beacon.core.bm25 import BM25
from beacon.core.vectorstore import VectorStore
from beacon.core.textsplit import (
    Splitter,
    RecursiveCharacterSplitter,
    SentenceSplitter,
    TokenSplitter,
)
from beacon.core.fusion import reciprocal_rank_fusion
from beacon.core.rerank import FeatureReranker
from beacon.core.loaders import load_text, load_markdown, load_jsonl, load_dir
from beacon.core.pipeline import RetrievalPipeline
from beacon.core.eval import recall_at_k, mean_reciprocal_rank

__version__ = "1.0.0"

__all__ = [
    "Chunk",
    "Document",
    "ScoredChunk",
    "Embedder",
    "HashEmbedder",
    "BM25",
    "VectorStore",
    "Splitter",
    "RecursiveCharacterSplitter",
    "SentenceSplitter",
    "TokenSplitter",
    "reciprocal_rank_fusion",
    "FeatureReranker",
    "load_text",
    "load_markdown",
    "load_jsonl",
    "load_dir",
    "RetrievalPipeline",
    "recall_at_k",
    "mean_reciprocal_rank",
    "__version__",
]
