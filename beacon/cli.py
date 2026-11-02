"""Command line interface for Beacon.

Usage examples
--------------
    python -m beacon.cli demo
    python -m beacon.cli index  --dir ./docs --store ./index.json
    python -m beacon.cli query  --store ./index.json --text "what is a bloom filter?"
    python -m beacon.cli eval   --store ./index.json --top-k 5
"""

import argparse
import json
import os
import sys
from typing import List

from beacon.core.loaders import load_dir
from beacon.core.pipeline import RetrievalPipeline
from beacon.core.types import Document


def _demo(args=None):
    corpus = [
        Document(
            text="A bloom filter is a probabilistic data structure that tests "
            "whether an element is a member of a set. False positives are "
            "possible, but false negatives are not.",
            doc_id="bloom",
        ),
        Document(
            text="HyperLogLog estimates the cardinality of a multiset using "
            "very little memory by observing the longest run of leading zeros "
            "in hashed values.",
            doc_id="hll",
        ),
        Document(
            text="BM25 is a ranking function used by search engines to score "
            "documents against a query using term frequency and inverse "
            "document frequency.",
            doc_id="bm25",
        ),
        Document(
            text="Reciprocal rank fusion combines several ranked lists by "
            "summing 1/(k+rank) for each item, which avoids needing to "
            "normalize raw scores between systems.",
            doc_id="rrf",
        ),
    ]
    pipe = RetrievalPipeline()
    pipe.index(corpus)
    query = "how do search engines rank documents with term frequency?"
    print("Query: %s\n" % query)
    for sc in pipe.search(query, top_k=3):
        print("  [%.4f] %s" % (sc.score, sc.chunk.text[:90]))
    print("\nIndexed %d chunks." % pipe.stats()["chunks"])


def _index(args):
    docs = load_dir(args.dir, recursive=not args.no_recursive)
    if not docs:
        print("no .txt/.md/.jsonl files found in %s" % args.dir, file=sys.stderr)
        return 1
    pipe = RetrievalPipeline()
    pipe.index(docs)
    pipe.save_json(args.store)
    print("indexed %d documents / %d chunks -> %s" % (
        len(docs), pipe.stats()["chunks"], args.store))
    return 0


def _query(args):
    pipe = RetrievalPipeline.load_json(args.store)
    results = pipe.search(args.text, top_k=args.top_k)
    if not results:
        print("no results")
        return 0
    for sc in results:
        print("  [%.4f] %s" % (sc.score, sc.chunk.text[:120]))
    return 0


def _eval(args):
    pipe = RetrievalPipeline.load_json(args.store)
    # Lightweight self-evaluation: each chunk is its own gold query target.
    rel_map = {}
    queries: List[str] = []
    for c in pipe._chunks:
        # use the first sentence of the chunk as a probe query
        q = c.text.split(". ")[0]
        if len(q) < 8:
            continue
        queries.append(q)
        rel_map[q] = [c.chunk_id]
    if not queries:
        print("nothing to evaluate")
        return 0

    def search_fn(q):
        return pipe.search(q, top_k=args.top_k)

    from beacon.core.eval import evaluate

    report = evaluate(queries, rel_map, top_k=args.top_k, search_fn=search_fn)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="beacon", description="Hybrid retrieval toolkit")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("demo", help="run an end-to-end demo").set_defaults(func=_demo)

    pi = sub.add_parser("index", help="build an index from a directory")
    pi.add_argument("--dir", required=True)
    pi.add_argument("--store", default="./index.json")
    pi.add_argument("--no-recursive", action="store_true")
    pi.set_defaults(func=_index)

    pq = sub.add_parser("query", help="query an existing index")
    pq.add_argument("--store", default="./index.json")
    pq.add_argument("--text", required=True)
    pq.add_argument("--top-k", type=int, default=5)
    pq.set_defaults(func=_query)

    pe = sub.add_parser("eval", help="self-evaluate the index")
    pe.add_argument("--store", default="./index.json")
    pe.add_argument("--top-k", type=int, default=5)
    pe.set_defaults(func=_eval)
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        # default to demo for a friendly first-run experience
        return _demo()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
