import unittest

from beacon.core.vectorstore import VectorStore
from beacon.core.types import Chunk


class TestVectorStore(unittest.TestCase):
    def test_search_returns_relevant(self):
        vs = VectorStore()
        vs.add([
            Chunk(text="apple banana fruit", chunk_id="a"),
            Chunk(text="car engine motor", chunk_id="b"),
            Chunk(text="fruit apple orchard", chunk_id="c"),
        ])
        res = vs.search("apple fruit", top_k=2)
        ids = [s.chunk.chunk_id for s in res]
        self.assertIn("a", ids)
        self.assertIn("c", ids)
        self.assertNotIn("b", ids)

    def test_top_k(self):
        vs = VectorStore()
        vs.add([Chunk(text="x %d" % i, chunk_id="c%d" % i) for i in range(5)])
        self.assertEqual(len(vs.search("x", top_k=3)), 3)

    def test_empty_store(self):
        vs = VectorStore()
        self.assertEqual(vs.search("anything"), [])

    def test_to_from_dict_roundtrip(self):
        vs = VectorStore()
        vs.add([Chunk(text="hello world", chunk_id="h")])
        vs2 = VectorStore.from_dict(vs.to_dict())
        r1 = vs.search("hello world", top_k=1)
        r2 = vs2.search("hello world", top_k=1)
        self.assertEqual(r1[0].chunk.chunk_id, r2[0].chunk.chunk_id)


if __name__ == "__main__":
    unittest.main()
