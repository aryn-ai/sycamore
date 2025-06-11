import logging
import os
import pytest
import tempfile

import sycamore
from sycamore.connectors.opensearch.sync import OpenSearchSync
from sycamore.connectors.opensearch.opensearch_writer import OpenSearchWriterClientParams, OpenSearchWriterTargetParams
from sycamore.data.document import Document
from sycamore.data.docid import path_to_sha256_docid
from sycamore.materialize_config import MRRNameGroup

logging.getLogger("sycamore.connectors.opensearch.sync").setLevel(logging.DEBUG)


class FakeOpensearch:
    def __init__(self):
        self._indices = {}
        self._index_properties = {}
        self.written = []
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def fake_os_client(self, x):
        return self

    def search(self, index, body, scroll, size):
        assert body == {"query": {"match_all": {}}, "_source": ["parent_id"]}
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
        for r in record_gen:
            index, id = r["_index"], r["_id"]
            assert index in self._indices
            self._indices[index][id] = r["_source"]
            ret.append([True, {"index": {"_id": r["_id"]}}])
            self.written.append([index, r["_id"]])

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
    derived_key_3 = "SBNJTRN-FjG7owHVrKtue7eqdM4RhdRWVl71HXN2d7I="
    os.unlink(f"{mat_dirs}/xx/oss-{doc_id_3},{derived_key_3}.md")

    oss.sync()
    assert len(oss.fake_os.written) == 1
    assert oss.fake_os.written[0] == ["test_create", f"path-sha256-{doc_id_3}"]


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

    # resync after dropping doc 3 - root doc
    doc_id_3 = "path-sha256-4e07408562bedb8b60ce05c1decfe3ad16b72230967de01f640b7e4729b49fce"
    del oss.fake_os._indices["test_create"][doc_id_3]
    oss.fake_os.written = []
    oss.sync()
    assert len(oss.fake_os.written) == 4
    assert oss.fake_os.written[0] == ["test_create", doc_id_3]

    # resync unchanged
    oss.fake_os.written = []
    oss.sync()
    assert len(oss.fake_os.written) == 0

    # resync after dropping doc 3 - first part
    doc_id_3_p1 = "splitdoc-V25ZNuGuBJo2gDh4uIoee8Aj_hYs0VPFqGxA04gG9aE="
    del oss.fake_os._indices["test_create"][doc_id_3_p1]
    oss.fake_os.written = []
    oss.sync()
    assert len(oss.fake_os.written) == 4
    assert oss.fake_os.written[0] == ["test_create", doc_id_3]

    # resync unchanged
    oss.fake_os.written = []
    oss.sync()
    assert len(oss.fake_os.written) == 0
