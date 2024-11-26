import string
import random
import unittest

import sycamore
from sycamore import DocSet, ExecMode
from sycamore.data import Document, MetadataDocument


class TestSort(unittest.TestCase):
    NUM_DOCS = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exec_mode = ExecMode.LOCAL

    def docs(self) -> list[Document]:
        doc_list = [
            # text_representation is random 6 letter strings
            Document(
                text_representation=f"{''.join(random.choices(string.ascii_letters, k=6))}",
                properties={"document_number": random.randint(1, 10000), "even": i if i % 2 == 0 else None},
            )
            for i in range(self.NUM_DOCS)
        ]

        for doc in doc_list:
            if doc.properties["even"] is None:
                doc.properties.pop("even")
        return doc_list

    def docset(self) -> DocSet:
        context = sycamore.init(exec_mode=self.exec_mode)
        return context.read.document(self.docs())

    def test_sort_descending(self):
        sorted_docset = self.docset().sort(True, "text_representation")
        doc_list = sorted_docset.take_all()

        for i in range(1, len(doc_list)):
            assert str(doc_list[i].text_representation) <= str(doc_list[i - 1].text_representation)

    def test_sort_ascending(self):
        sorted_docset = self.docset().sort(False, "properties.document_number")
        doc_list = sorted_docset.take_all()

        for i in range(1, len(doc_list)):
            assert doc_list[i].properties["document_number"] >= doc_list[i - 1].properties["document_number"]

    def test_default_value(self):
        sorted_docset = self.docset().sort(False, "properties.even")
        doc_list = sorted_docset.take_all()

        for i in range(1, len(doc_list)):
            assert doc_list[i].properties.get("even", 0) >= doc_list[i - 1].properties.get("even", 0)

        assert len(doc_list) == self.NUM_DOCS / 2

        sorted_docset = self.docset().sort(False, "properties.even", 0)
        doc_list = sorted_docset.take_all()

        for i in range(1, len(doc_list)):
            assert doc_list[i].properties.get("even", 0) >= doc_list[i - 1].properties.get("even", 0)

    def test_metadata_document(self):
        doc_list = [
            Document(text_representation="Z"),
            Document(text_representation="B"),
            MetadataDocument(),
            Document(text_representation="C"),
            MetadataDocument(),
            Document(text_representation=None),
        ]

        context = sycamore.init(exec_mode=self.exec_mode)
        docset = context.read.document(doc_list)

        sorted_docset = docset.sort(False, "text_representation")
        sorted_doc_list = sorted_docset.take_all(include_metadata=True)
        assert len(sorted_doc_list) == 3
        assert sorted_doc_list[0].text_representation == "B"
        assert sorted_doc_list[1].text_representation == "C"
        assert sorted_doc_list[2].text_representation == "Z"

        sorted_docset = docset.sort(False, "text_representation", "A")
        sorted_doc_list = sorted_docset.take_all(include_metadata=True)

        for i in range(3):
            d = sorted_doc_list[i]
            assert isinstance(d, MetadataDocument) or d.text_representation is None

        assert sorted_doc_list[3].text_representation == "B"
        assert sorted_doc_list[4].text_representation == "C"
        assert sorted_doc_list[5].text_representation == "Z"

        sorted_docset = docset.sort(True, "text_representation", "A")
        sorted_doc_list = sorted_docset.take_all(include_metadata=True)

        assert sorted_doc_list[0].text_representation == "Z"
        assert sorted_doc_list[1].text_representation == "C"
        assert sorted_doc_list[2].text_representation == "B"
        for i in range(3):
            d = sorted_doc_list[i + 3]
            assert isinstance(d, MetadataDocument) or d.text_representation is None
