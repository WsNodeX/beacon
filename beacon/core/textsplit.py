"""Text splitters that turn long documents into retrievable chunks.

Three strategies are provided:

* :class:`RecursiveCharacterSplitter` — recursively cuts on a hierarchy of
  separators (paragraph -> line -> sentence -> space) until each piece fits
  ``chunk_size``. Best general purpose splitter.
* :class:`SentenceSplitter` — groups whole sentences up to a char budget.
  Keeps natural sentence boundaries intact.
* :class:`TokenSplitter` — groups tokenized units with a configurable
  overlap; useful when downstream models are token-bounded.

All splitters are overlap-aware: neighbouring chunks share a tail so that a
fact split across a boundary is still retrievable.
"""

from typing import List

from beacon.core._util import split_sentences, tokenize, norm_text
from beacon.core.types import Chunk, Document


class Splitter:
    def split(self, text: str) -> List[str]:  # pragma: no cover - overridden
        raise NotImplementedError

    def split_document(self, doc: Document) -> List[Chunk]:
        chunks: List[Chunk] = []
        for i, piece in enumerate(self.split(doc.text)):
            start = doc.text.find(piece)
            end = start + len(piece) if start >= 0 else 0
            meta = dict(doc.metadata)
            meta.setdefault("index", i)
            chunks.append(
                Chunk(
                    text=piece,
                    chunk_id="%s_c%d" % (doc.doc_id or "doc", i),
                    doc_id=doc.doc_id,
                    start=start,
                    end=end,
                    metadata=meta,
                )
            )
        return chunks


class RecursiveCharacterSplitter(Splitter):
    """Recursively split on ``separators`` until chunks fit ``chunk_size``."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64,
                 separators=None):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split(self, text: str) -> List[str]:
        text = norm_text(text)
        raw = self._split(text, self.separators)
        return self._apply_overlap(raw)

    def _split(self, text: str, seps: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        sep = seps[0] if seps else ""
        if sep == "":
            # hard cut by characters
            return self._cut_hard(text)
        parts = text.split(sep)
        merged: List[str] = []
        cur = ""
        for p in parts:
            candidate = (cur + sep + p) if cur else p
            if len(candidate) <= self.chunk_size:
                cur = candidate
            else:
                if cur:
                    merged.append(cur)
                # recurse on the oversized piece if more separators remain
                if len(seps) > 1:
                    merged.extend(self._split(p, seps[1:]))
                else:
                    merged.extend(self._cut_hard(p))
                cur = ""
        if cur:
            merged.append(cur)
        # NOTE: overlap is applied exactly once, in split(), not here, so it
        # does not compound across recursion levels.
        return merged

    def _cut_hard(self, text: str) -> List[str]:
        out = []
        for i in range(0, len(text), self.chunk_size):
            piece = text[i : i + self.chunk_size]
            if piece.strip():
                out.append(piece)
        return out

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks
        out = []
        for i, c in enumerate(chunks):
            if i == 0:
                out.append(c)
                continue
            tail = chunks[i - 1][-self.chunk_overlap :]
            out.append(tail + c if tail else c)
        return out


class SentenceSplitter(Splitter):
    """Group whole sentences up to ``max_chars`` with a tail overlap."""

    def __init__(self, max_chars: int = 512, overlap: int = 64):
        self.max_chars = max_chars
        self.overlap = overlap

    def split(self, text: str) -> List[str]:
        text = norm_text(text)
        sents = split_sentences(text)
        groups: List[str] = []
        cur = ""
        for s in sents:
            if cur and len(cur) + len(s) + 1 > self.max_chars:
                groups.append(cur)
                cur = s
            else:
                cur = (cur + " " + s).strip() if cur else s
        if cur:
            groups.append(cur)
        if self.overlap > 0 and len(groups) > 1:
            out = []
            for i, g in enumerate(groups):
                if i == 0:
                    out.append(g)
                else:
                    tail = groups[i - 1][-self.overlap :]
                    out.append((tail + " " + g).strip() if tail.strip() else g)
            return out
        return groups


class TokenSplitter(Splitter):
    """Split into token windows of ``max_tokens`` with token overlap."""

    def __init__(self, max_tokens: int = 128, overlap_tokens: int = 16):
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def split(self, text: str) -> List[str]:
        text = norm_text(text)
        toks = tokenize(text)
        if not toks:
            return []
        step = self.max_tokens - self.overlap_tokens
        out = []
        for i in range(0, len(toks), max_tokens if step <= 0 else step):
            window = toks[i : i + self.max_tokens]
            if window:
                out.append(" ".join(window))
            if i + self.max_tokens >= len(toks):
                break
        return out
