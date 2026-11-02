import unittest

from beacon.core.textsplit import (
    RecursiveCharacterSplitter,
    SentenceSplitter,
    TokenSplitter,
)
from beacon.core.types import Document


class TestSplitters(unittest.TestCase):
    def test_recursive_fits_size(self):
        sp = RecursiveCharacterSplitter(chunk_size=20, chunk_overlap=4)
        doc = Document(text="alpha. " * 20)
        chunks = sp.split_document(doc)
        self.assertTrue(len(chunks) > 1)
        for c in chunks:
            self.assertLessEqual(len(c.text), 20 + 4)  # overlap may pad slightly

    def test_recursive_overlap_validation(self):
        with self.assertRaises(ValueError):
            RecursiveCharacterSplitter(chunk_size=10, chunk_overlap=10)

    def test_recursive_no_empty(self):
        sp = RecursiveCharacterSplitter(chunk_size=50, chunk_overlap=8)
        doc = Document(text="word " * 100)
        chunks = sp.split_document(doc)
        for c in chunks:
            self.assertTrue(c.text.strip())

    def test_sentence_splitter(self):
        sp = SentenceSplitter(max_chars=200, overlap=0)
        doc = Document(text="First sentence here. Second one there. Third at end.")
        chunks = sp.split_document(doc)
        self.assertEqual(len(chunks), 1)  # fits in one group
        self.assertIn("First", chunks[0].text)

    def test_token_splitter_overlap(self):
        sp = TokenSplitter(max_tokens=10, overlap_tokens=2)
        doc = Document(text=" ".join("w%d" % i for i in range(40)))
        chunks = sp.split_document(doc)
        self.assertTrue(len(chunks) > 1)
        # overlap: window size never exceeds max_tokens
        for c in chunks:
            self.assertLessEqual(len(c.tokens()), 10)

    def test_token_splitter_validation(self):
        with self.assertRaises(ValueError):
            TokenSplitter(max_tokens=5, overlap_tokens=5)

    def test_chunk_ids_unique(self):
        sp = RecursiveCharacterSplitter(chunk_size=30, chunk_overlap=4)
        doc = Document(text="content " * 60)
        chunks = sp.split_document(doc)
        ids = [c.chunk_id for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
