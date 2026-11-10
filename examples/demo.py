#!/usr/bin/env python3
"""End-to-end demonstration of the hybrid retrieval pipeline.

Runs fully offline: the default HashEmbedder needs no model download, and BM25 is
implemented in the package. Prints the fused ranking with each hit's per-stage
scores so you can see which side of the hybrid actually found it.

    python examples/demo.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beacon import Document, RetrievalPipeline

CORPUS = [
    Document(
        "BM25 is a ranking function used by search engines to score documents "
        "against a query using term frequency saturation and length normalisation.",
        doc_id="bm25",
    ),
    Document(
        "Reciprocal rank fusion combines several ranked lists by summing 1/(k+rank) "
        "for each item, which needs no score normalisation across retrievers.",
        doc_id="rrf",
    ),
    Document(
        "A vector store keeps dense embeddings and answers approximate nearest "
        "neighbour queries over them.",
        doc_id="vector",
    ),
    Document(
        "Garbage collection in the JVM reclaims objects that are no longer reachable "
        "from any root, using generational or region-based collectors.",
        doc_id="jvm-gc",
    ),
]

QUERIES = [
    "how do search engines rank documents with term frequency",
    "how does the JVM reclaim unreachable objects",
]


def main() -> int:
    pipeline = RetrievalPipeline()
    pipeline.index(CORPUS)
    print(f"Indexed {len(CORPUS)} documents.\n")

    for query in QUERIES:
        print(f"Query: {query}")
        for hit in pipeline.search(query, top_k=2):
            print(f"  [{hit.score:.4f}] ({hit.stage}) {hit.chunk.doc_id}: "
                  f"{hit.chunk.text[:70]}...")
        print()

    # The whole index round-trips through one JSON file — no database, no service.
    out = os.path.join(tempfile.mkdtemp(), "beacon_index.json")
    pipeline.save_json(out)
    restored = RetrievalPipeline.load_json(out)
    same = [h.chunk.doc_id for h in restored.search(QUERIES[0], top_k=2)]
    print(f"Persisted to {out} ({os.path.getsize(out)} bytes); "
          f"reloaded and re-queried -> {same}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
