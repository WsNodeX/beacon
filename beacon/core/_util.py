"""Shared helpers: tokenization, vector math, sentence splitting.

Everything here is dependency free so the whole toolkit runs on a bare
Python 3.9+ interpreter.
"""

import math
import re
import unicodedata

# Latin/number runs are one token, each CJK ideograph is its own token.
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]", re.UNICODE)


def tokenize(text):
    """Normalize ``text`` to a flat list of lower-cased tokens.

    ASCII words and digit groups become single tokens; every CJK character
    is treated as an individual token so mixed Chinese/English corpora work
    without an external segmenter.
    """
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def token_set(text):
    """Return the *set* of unique tokens in ``text``."""
    return set(tokenize(text))


def cosine(a, b):
    """Cosine similarity of two equal-length vectors. Returns 0.0 for a
    zero vector to keep callers deterministic."""
    if len(a) != len(b):
        raise ValueError("vector dimension mismatch: %d vs %d" % (len(a), len(b)))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def jaccard(a, b):
    """Jaccard similarity of two token sets."""
    if not a and not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def split_sentences(text):
    """Split ``text`` into sentences on ASCII/CJK sentence terminators
    while keeping the trailing punctuation attached."""
    if not text:
        return []
    parts = re.split(r"(?<=[.!?。！？\n])\s*", text)
    out = [p.strip() for p in parts if p and p.strip()]
    return out


def norm_text(s):
    """Light Unicode normalization used before tokenization."""
    return unicodedata.normalize("NFKC", s)
