import logging
import os
from pathlib import Path
import pytest
import random
import tempfile
from unittest.mock import patch

import sycamore
from sycamore.connectors.opensearch.sync import OpenSearchSync
from sycamore.connectors.opensearch.opensearch_writer import OpenSearchWriterClientParams, OpenSearchWriterTargetParams
from sycamore.data.document import Document
from sycamore.data.docid import path_to_sha256_docid
from sycamore.materialize_config import MRRNameGroup

logging.getLogger("sycamore.connectors.opensearch.sync").setLevel(logging.DEBUG)


class FakeOpensearch:
    def __init__(self, inject_429_frac=0.0):
        self._indices = {}
        self._index_properties = {}
        self.written = []
        self.deleted = []
        self.inject_429_frac = inject_429_frac
        self.injected_429_count = 0
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def fake_os_client(self, x):
        return self

    def search(self, index, body, scroll, size):
        assert body == {"query": {"match_all": {}}, "_source": ["parent_id", "doc_mtime"]}
        if index not in self._indices:
            from opensearchpy.exceptions import NotFoundError

            raise NotFoundError(f"Fake notfound {index}")
        hits = []

        for k, v in self._indices[index].items():
            hits.append({"_id": k, "_source": v})

        return {"_scroll_id": None, "hits": {"hits": hits}}

    @property
    def indices(self):
        return self

    # indices.get
    def get(self, index_name):
        return {index_name: self._index_properties[index_name]}

    # indices.create
    def create(self, name, body):
        print(f"fake create {name} {body}")
        assert name not in self._indices
        self._indices[name] = {}
        self._index_properties[name] = body

    def clear_scroll(scroll_id):
        assert False

    def parallel_bulk(self, record_gen, **kwargs):
        ret = []
        single_op_type = None
        for r in record_gen:
            op_type = r.get("_op_type", "index")
            if single_op_type != op_type:
                # not an OS requirement, but our code should only generate bulk single op type
                assert single_op_type is None
                single_op_type = op_type
            index, id = r["_index"], r["_id"]
            assert index in self._indices  # oddly not a requirement for opensearch
            if random.random() < self.inject_429_frac:
                ret.append([False, {op_type: {"_id": id, "status": 429}}])
                self.injected_429_count += 1
            elif op_type == "index":
                self._indices[index][id] = r["_source"]
                ret.append([True, {"index": {"_id": r["_id"]}}])
                self.written.append([index, r["_id"]])
            elif op_type == "delete":
                del self._indices[index][id]
                ret.append([True, {"delete": {"_id": r["_id"]}}])
                self.deleted.append([index, r["_id"]])
            else:
                assert False, f"unimplimented {op_type}"

        return ret


class UnitTestOpenSearchSync(OpenSearchSync):
    def __init__(self, sources, client_params, target_params, fake_os=None):
        self.fake_os = fake_os or FakeOpensearch()
        super().__init__(sources, client_params, target_params)

    def os_client(self):
        return self.fake_os


@pytest.fixture()
def mat_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = [Document(doc_id=path_to_sha256_docid(str(i)), text_representation=str(i)) for i in range(5)]
        sycamore.init(exec_mode=sycamore.EXEC_LOCAL).read.document(d).materialize(
            {"root": f"{tmpdir}/xx", "name": MRRNameGroup}
        ).execute()
        d = [Document(doc_id=path_to_sha256_docid(str(i + 10)), text_representation=str(i + 10)) for i in range(5)]
        sycamore.init(exec_mode=sycamore.EXEC_LOCAL).read.document(d).materialize(
            {"root": f"{tmpdir}/yy", "name": MRRNameGroup}
        ).execute()
        yield tmpdir


def test_simple(mat_dirs):
    oss = UnitTestOpenSearchSync(
        [(f"{mat_dirs}/xx", lambda d: [d])],
        OpenSearchWriterClientParams(),
        OpenSearchWriterTargetParams(index_name="test_create"),
    )
    oss.sync()
    assert "test_create" in oss.fake_os._indices
    assert len(oss.fake_os.written) == 5
    oss.fake_os.written = []
    oss.sync()
    assert len(oss.fake_os.written) == 0


def test_add_source(mat_dirs):
    oss = UnitTestOpenSearchSync(
        [(f"{mat_dirs}/xx", lambda d: [d])],
        OpenSearchWriterClientParams(),
        OpenSearchWriterTargetParams(index_name="test_create"),
    )
    oss.sync()
    assert "test_create" in oss.fake_os._indices
    assert len(oss.fake_os.written) == 5
    oss.fake_os.written = []
    oss = UnitTestOpenSearchSync(
        [(f"{mat_dirs}/xx", lambda d: [d]), (f"{mat_dirs}/yy", lambda d: [d])],
        OpenSearchWriterClientParams(),
        OpenSearchWriterTargetParams(index_name="test_create"),
        fake_os=oss.fake_os,
    )
    oss.sync()
    assert len(oss.fake_os.written) == 5


def test_drop_in_opensearch(mat_dirs):
    oss = UnitTestOpenSearchSync(
        [(f"{mat_dirs}/xx", lambda d: [d])],
        OpenSearchWriterClientParams(),
        OpenSearchWriterTargetParams(index_name="test_create"),
    )
    oss.sync()
    assert "test_create" in oss.fake_os._indices
    assert len(oss.fake_os.written) == 5
    oss.fake_os.written = []

    doc_id_3 = "path-sha256-4e07408562bedb8b60ce05c1decfe3ad16b72230967de01f640b7e4729b49fce"
    del oss.fake_os._indices["test_create"][doc_id_3]

    oss.sync()
    assert len(oss.fake_os.deleted) == 0
    assert len(oss.fake_os.written) == 1
    assert oss.fake_os.written[0] == ["test_create", doc_id_3]


def test_drop_in_matdir(mat_dirs):
    oss = UnitTestOpenSearchSync(
        [(f"{mat_dirs}/xx", lambda d: [d])],
        OpenSearchWriterClientParams(),
        OpenSearchWriterTargetParams(index_name="test_create"),
    )
    oss.sync()
    assert "test_create" in oss.fake_os._indices
    assert len(oss.fake_os.written) == 5
    oss.fake_os.written = []

    doc_id_3 = "4e07408562bedb8b60ce05c1decfe3ad16b72230967de01f640b7e4729b49fce"
    oss_file = list(Path(f"{mat_dirs}/xx").glob(f"oss-{doc_id_3},*.md"))
    assert len(oss_file) == 1
    oss_file[0].unlink()

    print("--------------- re-sync after drop -------------")
    oss.sync()
    assert len(oss.fake_os.written) == 1
    assert oss.fake_os.written[0] == ["test_create", f"path-sha256-{doc_id_3}"]
    assert len(oss.fake_os.deleted) == 1
    assert oss.fake_os.deleted[0] == ["test_create", f"path-sha256-{doc_id_3}"]


def fake_splitter(doc):
    children = int(doc.text_representation)
    ret = [doc]
    for i in range(children):
        child_id = f"{doc.doc_id}.{i}"
        d = Document(doc_id="", parent_id=doc.doc_id, text_representation=str(child_id))
        ret.append(d)

    return ret


def test_drop_subdoc_in_opensearch(mat_dirs):
    # sync 5 docs
    oss = UnitTestOpenSearchSync(
        [(f"{mat_dirs}/xx", fake_splitter)],
        OpenSearchWriterClientParams(),
        OpenSearchWriterTargetParams(index_name="test_create"),
    )
    oss.sync()
    assert "test_create" in oss.fake_os._indices
    assert len(oss.fake_os.written) == 5 + 0 + 1 + 2 + 3 + 4  # main + subdocs
    assert len(oss.fake_os.deleted) == 0

    # resync after dropping doc 3 - root doc
    doc_id_3 = "path-sha256-4e07408562bedb8b60ce05c1decfe3ad16b72230967de01f640b7e4729b49fce"
    del oss.fake_os._indices["test_create"][doc_id_3]
    oss.fake_os.written = []
    oss.sync()
    assert len(oss.fake_os.written) == 4
    assert oss.fake_os.written[0] == ["test_create", doc_id_3]
    assert len(oss.fake_os.deleted) == 3  # 3 subdocs, main already dropped

    # resync unchanged
    oss.fake_os.written = []
    oss.fake_os.deleted = []
    oss.sync()
    assert len(oss.fake_os.written) == 0
    assert len(oss.fake_os.deleted) == 0

    # WARNING: There is an annoying effect that if you add a default None field to
    # DEFAULT_RECORD_PROPERTIES, when it is converted to a OpenSearchWriterRecord in split_doc,
    # the new field will end up changing the calculated content-based hash.  There is debug
    # code in sync.py (search for DROP_SUBDOC_TEST) to help find the new magic constant.
    # TODO: Consider removing None fields from the os record, that way changes like that don't
    # affect the hash.

    # resync after dropping doc 3 - first part
    doc_id_3_p1 = "splitdoc-V25ZNuGuBJo2gDh4uIoee8Aj_hYs0VPFqGxA04gG9aE="
    del oss.fake_os._indices["test_create"][doc_id_3_p1]
    oss.fake_os.written = []
    oss.sync()
    assert len(oss.fake_os.written) == 4
    assert oss.fake_os.written[0] == ["test_create", doc_id_3]
    assert len(oss.fake_os.deleted) == 3  # main + 2 subdocs

    # resync unchanged
    oss.fake_os.written = []
    oss.fake_os.deleted = []
    oss.sync()
    assert len(oss.fake_os.written) == 0
    assert len(oss.fake_os.deleted) == 0


def test_delete_source_file(mat_dirs):
    to_remove = "4e07408562bedb8b60ce05c1decfe3ad16b72230967de01f640b7e4729b49fce"
    xx_del = f"{mat_dirs}/xx-del"
    os.makedirs(xx_del)
    for i in Path(f"{mat_dirs}/xx").glob("*"):
        os.link(f"{mat_dirs}/xx/{i.name}", f"{xx_del}/{i.name}")

    oss = UnitTestOpenSearchSync(
        [(xx_del, fake_splitter)],
        OpenSearchWriterClientParams(),
        OpenSearchWriterTargetParams(index_name="test_delete"),
    )
    oss.sync()
    assert len(oss.fake_os.written) == 5 + 0 + 1 + 2 + 3 + 4  # main + subdocs
    assert len(oss.fake_os.deleted) == 0

    print("--------------------------- remove and resync ------------------------")
    os.unlink(f"{xx_del}/doc-path-sha256-{to_remove}.pickle")
    oss.fake_os.written = []
    oss.sync()
    assert len(oss.fake_os.written) == 0
    assert len(oss.fake_os.deleted) == 4


def test_update_source_file(mat_dirs):
    to_update = "4e07408562bedb8b60ce05c1decfe3ad16b72230967de01f640b7e4729b49fce"
    xx = f"{mat_dirs}/xx"
    oss = UnitTestOpenSearchSync(
        [(xx, fake_splitter)],
        OpenSearchWriterClientParams(),
        OpenSearchWriterTargetParams(index_name="test_update"),
    )
    oss.sync()
    assert len(oss.fake_os.written) == 5 + 0 + 1 + 2 + 3 + 4  # main + subdocs
    assert len(oss.fake_os.deleted) == 0

    old_md = list(Path(xx).glob(f"oss-{to_update}*"))
    assert len(old_md) == 1

    print("--------------------------- update and resync ------------------------")
    Path(f"{xx}/doc-path-sha256-{to_update}.pickle").touch()
    oss.fake_os.written = []
    oss.sync()
    assert len(oss.fake_os.written) == 4
    assert len(oss.fake_os.deleted) == 4
    new_md = list(Path(xx).glob(f"oss-{to_update}*"))
    assert len(new_md) == 1
    assert old_md != new_md


def test_spurious_os_doc(mat_dirs):
    to_update = "4e07408562bedb8b60ce05c1decfe3ad16b72230967de01f640b7e4729b49fce"
    oss = UnitTestOpenSearchSync(
        [(f"{mat_dirs}/xx", fake_splitter)],
        OpenSearchWriterClientParams(),
        OpenSearchWriterTargetParams(index_name="test_spurious"),
    )
    oss.sync()
    assert len(oss.fake_os.written) == 5 + 0 + 1 + 2 + 3 + 4  # main + subdocs
    assert len(oss.fake_os.deleted) == 0

    oss.fake_os._indices["test_spurious"]["splitdoc-garbageid"] = {"parent_id": f"path-sha256-{to_update}"}
    oss.fake_os.written = []
    oss.sync()
    assert len(oss.fake_os.written) == 4
    assert len(oss.fake_os.deleted) == 5


def test_mangled_oss_md(mat_dirs):
    to_mangle = "4e07408562bedb8b60ce05c1decfe3ad16b72230967de01f640b7e4729b49fce"
    xx = f"{mat_dirs}/xx"
    oss = UnitTestOpenSearchSync(
        [(xx, fake_splitter)],
        OpenSearchWriterClientParams(),
        OpenSearchWriterTargetParams(index_name="test_spurious"),
    )
    oss.sync()
    assert len(oss.fake_os.written) == 5 + 0 + 1 + 2 + 3 + 4  # main + subdocs
    assert len(oss.fake_os.deleted) == 0
    # A completely invalid oss-*.md file
    Path(f"{xx}/oss-{to_mangle},sadf213qdssd.md").touch()
    oss.sync()
    print(oss.stats)
    assert oss.stats.misformatted_oss_file == 1

    mtime_ns = Path(f"{xx}/doc-path-sha256-{to_mangle}.pickle").stat().st_mtime_ns
    # A syntactically valid md file with the wrong key
    Path(f"{xx}/oss-{to_mangle},{mtime_ns},abcdefghij.md").touch()
    oss.fake_os.written = []
    oss.fake_os.deleted = []
    oss.sync()
    assert oss.stats.missing_md_info == 1
    assert len(oss.fake_os.written) == 4
    assert len(oss.fake_os.deleted) == 4


def test_wrong_key_oss_md(mat_dirs):
    to_rekey = "4e07408562bedb8b60ce05c1decfe3ad16b72230967de01f640b7e4729b49fce"
    xx = f"{mat_dirs}/xx"
    oss = UnitTestOpenSearchSync(
        [(xx, fake_splitter)],
        OpenSearchWriterClientParams(),
        OpenSearchWriterTargetParams(index_name="test_spurious"),
    )
    oss.sync()
    assert len(oss.fake_os.written) == 5 + 0 + 1 + 2 + 3 + 4  # main + subdocs
    assert len(oss.fake_os.deleted) == 0

    existing = list(Path(xx).glob(f"oss-{to_rekey},*"))
    assert len(existing) == 1
    parts = existing[0].name.split(",")
    assert len(parts) == 3
    if parts[2][0] == "X":
        parts[2] = "Y" + parts[2][1:]
    else:
        parts[2] = "X" + parts[2][1:]
    existing[0].unlink()
    Path(f"{xx}/{parts[0]},{parts[1]},{parts[2]}").touch()
    oss.fake_os.written = []
    oss.fake_os.deleted = []
    oss.sync()
    assert oss.stats.mismatch_key == 1
    assert len(oss.fake_os.written) == 4
    assert len(oss.fake_os.deleted) == 4


def test_intermittent_429s(mat_dirs):
    fake_os = FakeOpensearch(inject_429_frac=0.1)
    with patch.object(UnitTestOpenSearchSync.ProcessBatch, "sleep", lambda a, b: True):
        for i in range(0, 5):
            oss = UnitTestOpenSearchSync(
                [(f"{mat_dirs}/xx", fake_splitter)],
                OpenSearchWriterClientParams(),
                OpenSearchWriterTargetParams(index_name=f"test_429_{i}"),
                fake_os=fake_os,
            )
            oss.fake_os.written = []
            oss.sync()
            assert len(oss.fake_os.written) == 5 + 0 + 1 + 2 + 3 + 4  # main + subdocs
            assert len(oss.fake_os.deleted) == 0
            if oss.fake_os.injected_429_count > 0:
                print(f"Succeeded with {oss.fake_os.injected_429_count} injected 429s on try {i}")
                break
            else:
                assert i < 4, "Too many tries to get any injected 429s"


def test_multiple_os_write_rounds(mat_dirs):
    def mega_splitter(doc):
        children = int(doc.text_representation)
        ret = [doc]
        for i in range(children * 20):
            child_id = f"{doc.doc_id}.{i}"
            d = Document(doc_id="", parent_id=doc.doc_id, text_representation=str(child_id))
            ret.append(d)

        return ret

    oss = UnitTestOpenSearchSync(
        [(f"{mat_dirs}/xx", mega_splitter)],
        OpenSearchWriterClientParams(),
        OpenSearchWriterTargetParams(index_name="test_spurious"),
    )
    oss.sync()
    assert len(oss.fake_os.written) == 5 + 0 + 1 * 20 + 2 * 20 + 3 * 20 + 4 * 20  # main + subdocs
    assert len(oss.fake_os.deleted) == 0
    oss.fake_os.written = []

    oss.sync()
    print(oss.stats)
    assert len(oss.fake_os.written) == 0


def test_tolerate_other_md(mat_dirs):
    xx = f"{mat_dirs}/xx"
    oss = UnitTestOpenSearchSync(
        [(xx, fake_splitter)],
        OpenSearchWriterClientParams(),
        OpenSearchWriterTargetParams(index_name="test_spurious"),
    )
    Path(f"{xx}/foo.md").touch()
    Path(f"{xx}/yy.test").touch()
    oss.sync()
    assert oss.stats.ignored_other_md == 1
    assert oss.stats.ignored_unrecognized == 1


def test_unrecognized_pickle_file_aborts(mat_dirs):
    xx = f"{mat_dirs}/xx"
    oss = UnitTestOpenSearchSync(
        [(xx, fake_splitter)],
        OpenSearchWriterClientParams(),
        OpenSearchWriterTargetParams(index_name="test_spurious"),
    )
    Path(f"{xx}/foo.pickle").touch()
    with pytest.raises(ValueError):
        oss.sync()
