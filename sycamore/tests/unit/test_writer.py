from sycamore import DocSet, Context
import sycamore
from sycamore.data import Document
from sycamore.plan_nodes import Node
from sycamore.writers import OpenSearchWriter

from pathlib import Path

from sycamore.writers.file_writer import default_filename, default_doc_to_bytes, json_properties_content


def generate_docs(num: int, type: str = "test", text=True, binary=False) -> list[Document]:
    docs = []
    for i in range(num):
        doc = Document(
            {
                "doc_id": f"doc_{i}",
                "type": "test",
                "elements": {"array": []},
                "properties": {"filename": f"doc_{i}.dat"},
            }
        )

        if text:
            doc["text_representation"] = f"This is text content {i}"
        if binary:
            doc["binary_representation"] = f"This is binary content {i}".encode("utf-8")

        docs.append(doc)

    return docs


def _check_doc_path(
    docs: list[Document], path: Path, filename_fn=default_filename, doc_to_bytes_fn=default_doc_to_bytes
):
    docs_by_filename = {filename_fn(doc): doc for doc in docs}
    paths = [p for p in path.iterdir() if p.is_file()]
    assert len(paths) == len(docs)

    for p in paths:
        with open(p, "rb") as infile:
            expected = doc_to_bytes_fn(docs_by_filename[p.name])
            actual = infile.read()

            assert expected == actual


def _test_filename(doc: Document) -> str:
    return doc.properties["filename"]


class TestDocSetWriter:
    def test_opensearch(self, mocker):
        context = mocker.Mock(spec=Context)
        docset = DocSet(context, mocker.Mock(spec=Node))
        execute = mocker.patch.object(OpenSearchWriter, "execute")
        docset.write.opensearch(os_client_args={}, index_name="index")
        execute.assert_called_once()

    def test_file_writer_text(self, tmp_path: Path):
        docs = generate_docs(5)
        context = sycamore.init()
        doc_set = context.read.document(docs)
        doc_set.write.files(str(tmp_path))
        _check_doc_path(docs, tmp_path)

    def test_file_writer_binary(self, tmp_path: Path):
        docs = generate_docs(5, text=False, binary=True)
        context = sycamore.init()
        doc_set = context.read.document(docs)
        doc_set.write.files(str(tmp_path))
        _check_doc_path(docs, tmp_path)

    def test_file_writer_filename_fn(self, tmp_path: Path):
        docs = generate_docs(5)
        context = sycamore.init()
        doc_set = context.read.document(docs)
        doc_set.write.files(str(tmp_path), filename_fn=_test_filename)
        _check_doc_path(docs, tmp_path, filename_fn=_test_filename)

    def test_file_writer_doc_to_bytes_fn(self, tmp_path: Path):
        docs = generate_docs(5)
        context = sycamore.init()
        doc_set = context.read.document(docs)
        doc_set.write.files(str(tmp_path), doc_to_bytes_fn=json_properties_content)
        _check_doc_path(docs, tmp_path, doc_to_bytes_fn=json_properties_content)
