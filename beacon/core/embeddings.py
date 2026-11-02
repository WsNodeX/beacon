"""Embedding backends.

The default :class:`HashEmbedder` maps text to a fixed-dimensional vector
with the hashing trick (feature hashing). It is fully deterministic, needs
no model files, and produces embeddings whose cosine similarity tracks
lexical overlap well enough to power a real retrieval demo. Swap in any
callable ``encode(text) -> list[float]`` backend via the ``Embedder``
protocol (e.g. a remote model) without touching the rest of the toolkit.
"""

import hashlib
import math
from typing import List

from beacon.core._util import tokenize


class Embedder:
    """Protocol-ish base class every embedding backend inherits from."""

    def dim(self) -> int:  # pragma: no cover - overridden
        raise NotImplementedError

    def encode(self, text) -> List[float]:  # pragma: no cover - overridden
        raise NotImplementedError

    def to_dict(self):
        return {"type": self.__class__.__name__, "dim": self.dim()}

    @classmethod
    def from_dict(cls, data):
        name = data.get("type")
        if name == "HashEmbedder":
            return HashEmbedder(dim=data.get("dim", 256), seed=data.get("seed", 0))
        raise ValueError("unknown embedder type: %r" % name)


class HashEmbedder(Embedder):
    """Deterministic embedding via signed feature hashing.

    Each token is hashed to a bucket; occurrences are accumulated with a
    sign derived from the same hash, then the vector is L2-normalized.
    Identical text always yields an identical vector, so every result in
    this project is reproducible across machines and runs.
    """

    def __init__(self, dim: int = 256, seed: int = 0):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim
        self.seed = seed

    def dim(self) -> int:
        return self._dim

    def encode(self, text) -> List[float]:
        vec = [0.0] * self._dim
        tokens = tokenize(text)
        if not tokens:
            return vec
        salt = ("%d:" % self.seed).encode("utf-8")
        for tok in tokens:
            digest = hashlib.md5(salt + tok.encode("utf-8")).digest()
            h = int.from_bytes(digest[:4], "little")
            idx = h % self._dim
            sign = 1.0 if (h & 1) else -1.0
            # frequency-weighted accumulation
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        return vec

    def to_dict(self):
        d = super().to_dict()
        d["seed"] = self.seed
        return d
