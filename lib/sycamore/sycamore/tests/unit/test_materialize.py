import glob
import logging
from pathlib import Path
import pytest
import re
import shutil
import tempfile
import unittest
import uuid

from pyarrow import fs

import sycamore
from sycamore.context import ExecMode
from sycamore.data import Document, MetadataDocument
from sycamore.materialize import AutoMaterialize, Materialize
from sycamore.tests.unit.inmempyarrowfs import InMemPyArrowFileSystem


def tobin(d):
    if isinstance(d, MetadataDocument):
        return b"md"
    else:
        return d.doc_id.encode("utf-8")


class LocalRenameFilesystem(fs.LocalFileSystem):
    def __init__(self, ext):
        super().__init__()
        self.extension = ext

    def open_output_stream(self, path):
        if "/materialize." in path:  # don't rewrite these, otherwise exists tests fail
            return super().open_output_stream(path)
        return super().open_output_stream(path + self.extension)


def make_docs(num):
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


def noop_fn(d):
    return d


class TestMaterializeWrite(unittest.TestCase):
    def test_tonoop(self):
        ctx = sycamore.init(exec_mode=ExecMode.LOCAL)
        assert ctx.exec_mode == ExecMode.LOCAL
        ctx.read.document(make_docs(3)).map(noop_fn).materialize().execute()

    def test_write_str(self):
        ctx = sycamore.init(exec_mode=ExecMode.LOCAL)
        ds = ctx.read.document(make_docs(3)).map(noop_fn)

        with tempfile.TemporaryDirectory() as tmpdir:
            ds.materialize(path=str(tmpdir)).execute()
            self.check_files(tmpdir)

    def check_files(self, tmpdir, ext=""):
        docs = glob.glob(tmpdir + "/doc-doc_*" + ext)  # doc_id  is doc_#; default naming sticks a doc- prefix on
        assert len(docs) == 3
        mds = glob.glob(tmpdir + "/md-*" + ext)
        # MD#1 = the manual one from make_docs
        # MD#2 = one auto-metadata from all docs to all docs becaues we're currently processing
        # all docs as a single batch
        assert len(mds) == 2

    def test_write_path(self):
        ctx = sycamore.init(exec_mode=ExecMode.LOCAL)
        ds = ctx.read.document(make_docs(3)).map(noop_fn)

        with tempfile.TemporaryDirectory() as tmpdir:
            ds.materialize(path=Path(tmpdir)).execute()
            self.check_files(tmpdir)

    def test_cleanup(self):
        ctx = sycamore.init(exec_mode=ExecMode.LOCAL)
        ds = ctx.read.document(make_docs(3)).map(noop_fn)

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "doc_x").touch()
            ds.materialize(path=Path(tmpdir)).execute()
            self.check_files(tmpdir)

    def test_write_dict(self):
        ctx = sycamore.init(exec_mode=ExecMode.LOCAL)
        ds = ctx.read.document(make_docs(3)).map(noop_fn)

        with tempfile.TemporaryDirectory() as tmpdir:
            ds.materialize(path={"root": tmpdir}).execute()
            self.check_files(tmpdir)

            # Also tests the cleanup logic
            ds.materialize(path={"root": Path(tmpdir)}).execute()
            self.check_files(tmpdir)

            ds.materialize(path={"root": tmpdir, "fs": LocalRenameFilesystem(".test")}).execute()
            self.check_files(tmpdir, ext=".test")

            def doc_to_name2(doc, bin):
                return Materialize.doc_to_name(doc, bin) + ".test2"

            ds.materialize(path={"root": tmpdir, "name": doc_to_name2}).execute()
            self.check_files(tmpdir, ext=".test2")

            def doc_to_name3(doc, bin):
                return Materialize.doc_to_name(doc, bin) + ".test3"

            ds.materialize(path={"root": tmpdir, "name": doc_to_name3, "clean": False}).execute()
            # did not clean, both of these should pass
            self.check_files(tmpdir, ext=".test2")
            self.check_files(tmpdir, ext=".test3")

            files = glob.glob(tmpdir + "/*")
            assert len(files) == 12

    def test_to_binary(self):
        docs = make_docs(3)
        ctx = sycamore.init(exec_mode=ExecMode.LOCAL)
        ds = ctx.read.document(docs).map(noop_fn)

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
                files = glob.glob(tmpdir + f"/doc-{d.doc_id}:*.pickle")
                assert len(files) == 1
                with open(files[0], "r") as f:
                    bits = f.read()
                    assert bits == d.doc_id

            with self.assertRaises(AssertionError):
                ds.materialize(path={"root": tmpdir, "tobin": lambda d: "abc"}).execute()

        with tempfile.TemporaryDirectory() as tmpdir:
            ds.materialize(path={"root": tmpdir, "tobin": onlydoc}).execute()
            docs = glob.glob(tmpdir + "/doc-doc_*")  # doc_id  is doc_#; default naming sticks a doc- prefix on
            assert len(docs) == 3
            mds = glob.glob(tmpdir + "/md-*")
            assert len(mds) == 0


class TestAutoMaterialize(unittest.TestCase):
    # Needed until we don't have a global context
    def setUp(self):
        sycamore.shutdown()

    def tearDown(self):
        sycamore.shutdown()

    def test_setdirname(self):
        docs = make_docs(3)
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = sycamore.init(exec_mode=ExecMode.LOCAL, rewrite_rules=[AutoMaterialize(tmpdir)])
            ctx.read.document(docs).map(noop_fn).execute()

            files = [f for f in Path(tmpdir).rglob("*")]
            logging.info(f"Found {files}")
            assert len([f for f in files if "DocScan.0/doc" in str(f)]) == 3
            assert len([f for f in files if "DocScan.0/md-" in str(f)]) == 1
            assert len([f for f in files if "Map.0/doc" in str(f)]) == 3
            assert len([f for f in files if "Map.0/md-" in str(f)]) == 2

    def test_autodirname(self):
        docs = make_docs(3)
        a = AutoMaterialize()
        try:
            ctx = sycamore.init(exec_mode=ExecMode.LOCAL, rewrite_rules=[a])
            ctx.read.document(docs).map(noop_fn).execute()

            files = [f for f in Path(a._directory).rglob("*")]
            logging.info(f"Found {files}")
            assert len([f for f in files if "DocScan.0/doc" in str(f)]) == 3
            assert len([f for f in files if "DocScan.0/md-" in str(f)]) == 1
            assert len([f for f in files if "Map.0/doc" in str(f)]) == 3
            assert len([f for f in files if "Map.0/md-" in str(f)]) == 2
            assert re.match(".*materialize\\.[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}", str(a._directory))
        finally:
            if a._directory is not None:
                shutil.rmtree(a._directory)

    def test_dupnodename(self):
        docs = make_docs(3)
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = sycamore.init(exec_mode=ExecMode.LOCAL, rewrite_rules=[AutoMaterialize(tmpdir)])
            ctx.read.document(docs).map(noop_fn).execute()

            files = [f for f in Path(tmpdir).rglob("*")]
            logging.info(f"DupNode Found-1 {files}")
            # counts are docs + md + success/clean file
            assert len([f for f in files if "DocScan.0/" in str(f)]) == 3 + 1 + 2
            assert len([f for f in files if "Map.0/" in str(f)]) == 3 + 2 + 2

            # This is a new pipeline so should get new names
            ctx.read.document(docs).map(noop_fn).execute()
            files = [f for f in Path(tmpdir).rglob("*")]
            logging.info(f"DupNode Found-2 {files}")
            assert len([f for f in files if "DocScan.0/" in str(f)]) == 3 + 1 + 2
            assert len([f for f in files if "Map.0/" in str(f)]) == 3 + 2 + 2
            assert len([f for f in files if "DocScan.1/" in str(f)]) == 3 + 1 + 2
            assert len([f for f in files if "Map.1/" in str(f)]) == 3 + 2 + 2

    def test_forcenodename(self):
        docs = make_docs(3)
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = sycamore.init(exec_mode=ExecMode.LOCAL, rewrite_rules=[AutoMaterialize(tmpdir)])
            ds = ctx.read.document(docs, materialize={"name": "reader"}).map(noop_fn, materialize={"name": "noop"})

            ds.execute()

            files = [f for f in Path(tmpdir).rglob("*")]
            logging.info(f"DupNode Found-1 {files}")
            assert len([f for f in files if "reader/" in str(f)]) == 3 + 1 + 2
            assert len([f for f in files if "noop/" in str(f)]) == 3 + 2 + 2

    def test_overrides(self):
        def doc_to_name4(doc, bin):
            return Materialize.doc_to_name(doc, bin) + ".test4"

        docs = make_docs(3)
        with tempfile.TemporaryDirectory() as tmpdir:
            a = AutoMaterialize(path={"root": tmpdir, "name": doc_to_name4, "clean": False, "tobin": tobin})
            ctx = sycamore.init(exec_mode=ExecMode.LOCAL, rewrite_rules=[a])

            ds = ctx.read.document(docs).map(noop_fn)
            ds.execute()

            files = [f for f in Path(tmpdir).rglob("*")]
            assert len([f for f in files if ".test4" in str(f)]) == 3 + 1 + 3 + 2

            for d in docs:
                d.doc_id = d.doc_id + "-dup"

            ds.execute()
            files = [f for f in Path(tmpdir).rglob("*")]
            assert len([f for f in files if "-dup" in str(f)]) == 4 + 4
            test4_files = [f for f in files if ".test4" in str(f)]
            assert len(test4_files) == 2 * (3 + 1 + 3 + 2)

            a._path["clean"] = True
            ds.execute()
            files = [f for f in Path(tmpdir).rglob("*")]
            assert len([f for f in files if ".test4" in str(f)]) == 3 + 1 + 3 + 2

    def test_source_mode(self):
        class NumCalls:
            x = 0

        # Note: This only makes sense in local mode, as the count is not thread safe
        def inc_counter(doc):
            NumCalls.x += 1
            return doc

        def check(a, docs):
            ctx = sycamore.init(exec_mode=ExecMode.LOCAL, rewrite_rules=[a])
            ds = ctx.read.document(docs).map(inc_counter)
            ds.execute()
            ds.filter(lambda d: d.doc_id != "doc_2").execute()

        docs = make_docs(3)
        with tempfile.TemporaryDirectory() as tmpdir:
            a = AutoMaterialize(tmpdir, source_mode=sycamore.MATERIALIZE_USE_STORED)
            check(a, docs)
            assert NumCalls.x == 3

        NumCalls.x = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            a = AutoMaterialize(tmpdir, source_mode=sycamore.MATERIALIZE_RECOMPUTE)
            check(a, docs)
            assert NumCalls.x == 6


def any_id(d):
    if isinstance(d, MetadataDocument):
        return str(d.metadata)
    else:
        return d.doc_id


def ids(docs):
    ret = []
    for d in docs:
        ret.append(any_id(d))
    ret.sort()
    return ret


class TestMaterializeRead(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exec_mode = ExecMode.LOCAL

    def test_materialize_read(self):
        ctx = sycamore.init(exec_mode=self.exec_mode)
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = make_docs(3)
            ds = (
                ctx.read.document(docs)
                .map(noop_fn)
                .materialize(path=tmpdir, source_mode=sycamore.MATERIALIZE_USE_STORED)
            )
            e1 = ds.take_all()
            assert e1 is not None
            e2 = ds.take_all()
            assert e2 is not None
            assert ids(e1) == ids(e2)

            # Does not recompute despite input changing because we read from cache
            extra_doc = Document({"doc_id": "doc_3"})
            docs.append(extra_doc)
            e3 = ds.take_all()
            assert ids(e1) == ids(e3)

            # Fake shove another doc into the cache dir.
            with open(Path(tmpdir) / "doc_3.pickle", "wb") as f:
                f.write(extra_doc.serialize())

            e4 = ds.take_all()
            assert ids(e1 + [extra_doc]) == ids(e4)

            # Remove the cache; should reconstruct to the faked cache
            shutil.rmtree(Path(tmpdir))
            e5 = ds.take_all()
            assert ids(e4) == ids(e5)
            assert (Path(tmpdir) / "materialize.success").exists()

            # Make a new docset based on the existing one
            ds = ctx.read.materialize(path=tmpdir)
            e6 = ds.take_all()
            assert ids(e6) == ids(e5)

            # Should still work when we delete the success file, but it can log
            (Path(tmpdir) / "materialize.success").unlink()
            e7 = ds.take_all()
            assert ids(e7) == ids(e5)

            # If we nuke all the files in the cache dir, it should raise a value error
            shutil.rmtree(Path(tmpdir))
            Path(tmpdir).mkdir()
            with pytest.raises(ValueError):
                ds.take_all()

            # And should also do so with no directory
            Path(tmpdir).rmdir()
            with pytest.raises(ValueError):
                ds.take_all()


class TestAllViaPyarrowFS(unittest.TestCase):
    def test_simple(self):
        fs = InMemPyArrowFileSystem()
        ctx = sycamore.init(exec_mode=ExecMode.LOCAL)
        docs = make_docs(3)
        path = {"root": "/no/such/path", "fs": fs}
        ctx.read.document(docs).materialize(path=path).execute()
        docs_out = ctx.read.materialize(path).take_all()
        assert docs.sort(key=any_id) == docs_out.sort(key=any_id)

    def test_clean(self):
        fs = InMemPyArrowFileSystem()
        ctx = sycamore.init(exec_mode=ExecMode.LOCAL)
        docs = make_docs(3)
        path = {"root": "/fake/inmem/no/such/path", "fs": fs}
        fs.open_output_stream(path["root"] + "/fake.pickle").close()
        ctx.read.document(docs).materialize(path=path).execute()
        docs_out = ctx.read.materialize(path).take_all()
        assert docs.sort(key=any_id) == docs_out.sort(key=any_id)

    def test_automaterialize(self):
        fs = InMemPyArrowFileSystem()
        path = {"root": "/fake/inmem/no/such/path", "fs": fs}
        ctx = sycamore.init(exec_mode=ExecMode.LOCAL, rewrite_rules=[AutoMaterialize(path)])
        docs = make_docs(3)
        docs_out = ctx.read.document(docs).materialize(path=path).take_all()
        assert docs.sort(key=any_id) == docs_out.sort(key=any_id)

    def test_fail_if_hierarchy(self):
        fs = InMemPyArrowFileSystem()
        ctx = sycamore.init(exec_mode=ExecMode.LOCAL)
        docs = make_docs(3)
        path = {"root": "/fake/inmem/no/such/path", "fs": fs}
        fs.open_output_stream(path["root"] + "/subdir/fake.pickle").close()
        from sycamore.materialize import _PyArrowFsHelper

        fsh = _PyArrowFsHelper(fs)

        # Fail with explicit materialize
        with pytest.raises(AssertionError):
            ctx.read.document(docs).materialize(path=path).execute()

        ctx = sycamore.init(exec_mode=ExecMode.LOCAL, rewrite_rules=[AutoMaterialize(path)])
        pipeline = ctx.read.document(docs).materialize(path=path)
        # Fail with auto-materialize
        with pytest.raises(AssertionError):
            pipeline.take_all()

        assert fsh.file_exists(path["root"] + "/subdir/fake.pickle")

        fs.open_output_stream(path["root"] + "/DocScan.0/subdir/fake.pickle").close()
        # Fail with auto-materialize and file in one of the real subdirs
        with pytest.raises(AssertionError):
            pipeline.take_all()

        assert fsh.file_exists(path["root"] + "/DocScan.0/subdir/fake.pickle")


class TestClearMaterialize(unittest.TestCase):
    def test_noop(self):
        ctx = sycamore.init(exec_mode=ExecMode.LOCAL)
        ds = ctx.read.document(make_docs(3)).map(noop_fn)
        with tempfile.TemporaryDirectory() as tmpdir:
            docx = Path(tmpdir) / "doc_x"
            docx.touch()
            assert docx.exists()
            ds.materialize(path=tmpdir).clear_materialize()
            assert not docx.exists()

    def maybe_clear_non_local(self, clear_non_local: bool):
        from pyarrow.fs import SubTreeFileSystem, LocalFileSystem

        ctx = sycamore.init(exec_mode=ExecMode.LOCAL)
        ds = ctx.read.document(make_docs(3)).map(noop_fn)
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = SubTreeFileSystem("/", LocalFileSystem())
            (Path(tmpdir) / "x").mkdir()
            (Path(tmpdir) / "y").mkdir()
            docx = Path(tmpdir) / "x/doc_x"
            docx.touch()
            docy = Path(tmpdir) / "y/doc_y"
            docy.touch()
            assert docx.exists()
            assert docy.exists()
            (
                ds.materialize(path=f"{tmpdir}/y")
                .materialize(path={"fs": fs, "root": f"{tmpdir}/x"})
                .clear_materialize(clear_non_local=clear_non_local)
            )
            if clear_non_local:
                assert not docx.exists()
            else:
                assert docx.exists()
            assert not docy.exists()

    def test_clear_non_local(self):
        self.maybe_clear_non_local(True)

    def test_no_clear_non_local(self):
        self.maybe_clear_non_local(False)

    def test_clear_matching(self):
        ctx = sycamore.init(exec_mode=ExecMode.LOCAL)
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = (
                ctx.read.document(make_docs(3))
                .map(noop_fn)
                .materialize(path=tmpdir + "/a")
                .materialize(path=tmpdir + "/b")
                .materialize(path=tmpdir + "/b2")
            )
            ds.execute()
            assert Path(f"{tmpdir}/a/materialize.success").exists()
            assert Path(f"{tmpdir}/b/materialize.success").exists()
            assert Path(f"{tmpdir}/b2/materialize.success").exists()

            # not matching removes nothing
            ds.clear_materialize("c*")
            assert Path(f"{tmpdir}/a/materialize.success").exists()
            assert Path(f"{tmpdir}/b/materialize.success").exists()
            assert Path(f"{tmpdir}/b2/materialize.success").exists()

            # match exact
            ds.clear_materialize("a")
            assert not Path(f"{tmpdir}/a/materialize.success").exists()
            assert Path(f"{tmpdir}/b/materialize.success").exists()
            assert Path(f"{tmpdir}/b2/materialize.success").exists()

            # match exact with path
            ds.clear_materialize(Path(tmpdir) / "b")
            assert not Path(f"{tmpdir}/a/materialize.success").exists()
            assert not Path(f"{tmpdir}/b/materialize.success").exists()
            assert Path(f"{tmpdir}/b2/materialize.success").exists()

            # match multiple, and as a path
            ds.execute()
            ds.clear_materialize(Path(tempfile.gettempdir()) / "*/b*")
            assert Path(f"{tmpdir}/a/materialize.success").exists()
            assert not Path(f"{tmpdir}/b/materialize.success").exists()
            assert not Path(f"{tmpdir}/b2/materialize.success").exists()


class TestErrorChecking(unittest.TestCase):
    def test_duplicate_root(self):
        ctx = sycamore.init(exec_mode=ExecMode.LOCAL)
        ds = ctx.read.document(make_docs(3))
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError):
                ds.materialize(path=tmpdir).materialize(path=tmpdir).execute()
