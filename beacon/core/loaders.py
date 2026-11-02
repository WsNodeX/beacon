"""Document loaders for common text formats."""

import json
import os
from typing import Dict, List

from beacon.core.types import Document


def load_text(path: str) -> Document:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    name = os.path.basename(path)
    return Document(text=text, doc_id="file_" + name, metadata={"source": path, "kind": "text"})


def load_markdown(path: str) -> Document:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    name = os.path.basename(path)
    title = ""
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break
    meta = {"source": path, "kind": "markdown", "title": title}
    return Document(text=text, doc_id="md_" + name, metadata=meta)


def load_jsonl(path: str) -> List[Document]:
    """Each JSON object becomes a Document. ``text`` and optional ``id`` /
    ``metadata`` keys are honored."""
    docs: List[Document] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text") or obj.get("content") or ""
            did = obj.get("id") or obj.get("doc_id") or ""
            meta = obj.get("metadata", {})
            docs.append(Document(text=text, doc_id=did, metadata=meta))
    return docs


def load_dir(path: str, recursive: bool = True) -> List[Document]:
    """Load every ``.txt`` / ``.md`` / ``.jsonl`` file in a directory."""
    docs: List[Document] = []
    walker = os.walk(path) if recursive else [(path, [], os.listdir(path))]
    for root, _dirs, files in walker:
        for fn in sorted(files):
            full = os.path.join(root, fn)
            if fn.endswith(".txt"):
                docs.append(load_text(full))
            elif fn.endswith(".md"):
                docs.append(load_markdown(full))
            elif fn.endswith(".jsonl"):
                docs.extend(load_jsonl(full))
    return docs
