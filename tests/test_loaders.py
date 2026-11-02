import json
import os
import tempfile
import unittest

from beacon.core.loaders import load_dir, load_jsonl, load_markdown, load_text


class TestLoaders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.txt = os.path.join(self.tmp, "a.txt")
        with open(self.txt, "w", encoding="utf-8") as fh:
            fh.write("hello file content")
        self.md = os.path.join(self.tmp, "b.md")
        with open(self.md, "w", encoding="utf-8") as fh:
            fh.write("# My Title\n\nsome body text here.")
        self.jl = os.path.join(self.tmp, "c.jsonl")
        with open(self.jl, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"text": "json line one", "id": "j1"}) + "\n")
            fh.write(json.dumps({"text": "json line two"}) + "\n")

    def test_load_text(self):
        d = load_text(self.txt)
        self.assertEqual(d.text, "hello file content")
        self.assertEqual(d.metadata["kind"], "text")

    def test_load_markdown_title(self):
        d = load_markdown(self.md)
        self.assertEqual(d.metadata["title"], "My Title")

    def test_load_jsonl(self):
        docs = load_jsonl(self.jl)
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].doc_id, "j1")

    def test_load_dir(self):
        # a.txt (1) + b.md (1) + c.jsonl (2 objects) => 4 documents
        docs = load_dir(self.tmp, recursive=False)
        self.assertEqual(len(docs), 4)


if __name__ == "__main__":
    unittest.main()
