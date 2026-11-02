"""Core data containers used across the toolkit."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    """A raw document fed into the pipeline before splitting."""

    text: str
    doc_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.doc_id == "":
            # Stable short id derived from content so re-indexing is idempotent.
            import hashlib

            self.doc_id = "doc_" + hashlib.sha1(self.text.encode("utf-8")).hexdigest()[:12]


@dataclass
class Chunk:
    """A contiguous slice of a document produced by a splitter."""

    text: str
    chunk_id: str = ""
    doc_id: str = ""
    start: int = 0
    end: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def tokens(self) -> List[str]:
        from beacon.core._util import tokenize

        return tokenize(self.text)


@dataclass
class ScoredChunk:
    """A chunk together with a relevance score and the stage that produced it."""

    chunk: Chunk
    score: float
    stage: str = ""
    rank: int = 0

    def __repr__(self):
        return "<ScoredChunk %s score=%.4f stage=%s>" % (
            self.chunk.chunk_id or self.chunk.text[:20],
            self.score,
            self.stage,
        )
