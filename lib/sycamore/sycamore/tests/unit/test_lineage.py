import glob
from pathlib import Path
import tempfile
import unittest
import uuid

from pyarrow import fs

import sycamore
from sycamore.context import ExecMode
from sycamore.data import Document, MetadataDocument


class LocalRenameFilesystem(fs.LocalFileSystem):
    def __init__(self, ext):
        super().__init__()
        self.extension = ext

    def open_output_stream(self, path):
        return super().open_output_stream(path + self.extension)


class TestLineage(unittest.TestCase):
    # Needed until we don't have a global context
    def setUp(self):
        sycamore.shutdown()

    def tearDown(self):
        sycamore.shutdown()

    def make_docs(self, num):
        docs = []
        for i in range(num):
            doc = Document({"doc_id": f"doc_{i}"})
            docs.append(doc)

        docs.append(
            MetadataDocument(
                lineage_links={"from_ids": ["root:" + str(uuid.uuid4())], "to_ids": [d.lineage_id for d in docs]}
            )
        )

        return docs

    @staticmethod
    def noop_fn(d):
        return d

    def test_noop(self):
        ctx = sycamore.init(exec_mode=ExecMode.LOCAL)
        assert ctx.exec_mode == ExecMode.LOCAL
        ctx.read.document(self.make_docs(3)).map(self.noop_fn).materialize().execute()

    def test_write_str(self):
        ctx = sycamore.init(exec_mode=ExecMode.LOCAL)
        ds = ctx.read.document(self.make_docs(3)).map(self.noop_fn)

        with tempfile.TemporaryDirectory() as tmpdir:
            ds.materialize(path=str(tmpdir)).execute()
            self.check_files(tmpdir)

    def check_files(self, tmpdir, ext=""):
        docs = glob.glob(tmpdir + "/doc_*" + ext)  # doc_id  is doc_#
        assert len(docs) == 3
        mds = glob.glob(tmpdir + "/md-*" + ext)
        # MD#1 = the manual one from make_docs
        # MD#2 = one auto-metadata from all docs to all docs becaues we're currently processing
        # all docs as a single batch
        assert len(mds) == 2

    def test_write_path(self):
        ctx = sycamore.init(exec_mode=ExecMode.LOCAL)
        ds = ctx.read.document(self.make_docs(3)).map(self.noop_fn)

        with tempfile.TemporaryDirectory() as tmpdir:
            ds.materialize(path=Path(tmpdir)).execute()
            self.check_files(tmpdir)

    def test_write_dict(self):
        from sycamore.lineage import Materialize

        ctx = sycamore.init(exec_mode=ExecMode.LOCAL)
        ds = ctx.read.document(self.make_docs(3)).map(self.noop_fn)

        with tempfile.TemporaryDirectory() as tmpdir:
            ds.materialize(path={"root": tmpdir}).execute()
            self.check_files(tmpdir)

            # Also tests the cleanup logic
            ds.materialize(path={"root": Path(tmpdir)}).execute()
            self.check_files(tmpdir)

            ds.materialize(path={"root": tmpdir, "fs": LocalRenameFilesystem(".test")}).execute()
            self.check_files(tmpdir, ext=".test")

            def doc_to_name2(doc):
                return Materialize.doc_to_name(doc) + ".test2"

            ds.materialize(path={"root": tmpdir, "name": doc_to_name2}).execute()
            self.check_files(tmpdir, ext=".test2")

            def doc_to_name3(doc):
                return Materialize.doc_to_name(doc) + ".test3"

            ds.materialize(path={"root": tmpdir, "name": doc_to_name3, "clean": False}).execute()
            # did not clean, both of these should pass
            self.check_files(tmpdir, ext=".test2")
            self.check_files(tmpdir, ext=".test3")

            files = glob.glob(tmpdir + "/*")
            assert len(files) == 10

    def test_to_binary(self):
        docs = self.make_docs(3)
        ctx = sycamore.init(exec_mode=ExecMode.LOCAL)
        ds = ctx.read.document(docs).map(self.noop_fn)

        def tobin(d):
            if isinstance(d, MetadataDocument):
                return b"md"
            else:
                return d.doc_id.encode("utf-8")

        def onlydoc(d):
            if isinstance(d, MetadataDocument):
                return None
            else:
                return d.doc_id.encode("utf-8")

        with tempfile.TemporaryDirectory() as tmpdir:
            ds.materialize(path={"root": tmpdir, "tobin": tobin}).execute()
            self.check_files(tmpdir)
            for d in docs:
                if isinstance(d, MetadataDocument):
                    continue
                with open(tmpdir + "/" + d.doc_id, "r") as f:
                    bits = f.read()
                    assert bits == d.doc_id

            with self.assertRaises(AssertionError):
                ds.materialize(path={"root": tmpdir, "tobin": lambda d: "abc"}).execute()

        with tempfile.TemporaryDirectory() as tmpdir:
            ds.materialize(path={"root": tmpdir, "tobin": onlydoc}).execute()
            docs = glob.glob(tmpdir + "/doc_*")  # doc_id  is doc_#
            assert len(docs) == 3
            mds = glob.glob(tmpdir + "/md-*")
            assert len(mds) == 0
