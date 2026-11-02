import os
import sys
import tempfile
import unittest

from beacon import cli


class TestCLI(unittest.TestCase):
    def test_demo_runs(self):
        # should not raise
        self.assertEqual(cli._demo(), None)

    def test_index_and_query(self):
        tmp = tempfile.mkdtemp()
        docdir = os.path.join(tmp, "docs")
        os.makedirs(docdir)
        with open(os.path.join(docdir, "a.txt"), "w", encoding="utf-8") as fh:
            fh.write("Search engines use BM25 to rank documents by term frequency.")
        store = os.path.join(tmp, "index.json")
        rc = cli._index(cli.build_parser().parse_args(
            ["index", "--dir", docdir, "--store", store]))
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(store))
        # query
        args = cli.build_parser().parse_args(
            ["query", "--store", store, "--text", "how to rank documents", "--top-k", "3"])
        self.assertEqual(cli._query(args), 0)

    def test_eval_runs(self):
        tmp = tempfile.mkdtemp()
        docdir = os.path.join(tmp, "docs")
        os.makedirs(docdir)
        with open(os.path.join(docdir, "a.txt"), "w", encoding="utf-8") as fh:
            fh.write("Reciprocal rank fusion merges ranked lists by rank.")
        store = os.path.join(tmp, "index.json")
        cli._index(cli.build_parser().parse_args(
            ["index", "--dir", docdir, "--store", store]))
        args = cli.build_parser().parse_args(["eval", "--store", store, "--top-k", "3"])
        self.assertEqual(cli._eval(args), 0)


if __name__ == "__main__":
    unittest.main()
