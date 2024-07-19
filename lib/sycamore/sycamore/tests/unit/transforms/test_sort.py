import string
import pytest
import random

import sycamore
from sycamore import DocSet
from sycamore.data import Document


class TestSort:
    @pytest.fixture()
    def docs(self) -> list[Document]:
        doc_list = [
            # text_representation is random 6 letter strings
            Document(
                text_representation=f"{''.join(random.choices(string.ascii_letters, k=6))}",
                doc_id={"even": i if i % 2 == 0 else None},
                properties={"document_number": random.randint(1, 10000)},
            )
            for i in range(10)
        ]

        for doc in doc_list:
            if doc.doc_id["even"] is None:
                doc.doc_id = {}
        return doc_list

    @pytest.fixture()
    def docset(self, docs: list[Document]) -> DocSet:
        context = sycamore.init()
        return context.read.document(docs)

    def test_sort_descending(self, docset: DocSet):
        sorted_docset = docset.sort(True, "text_representation")
        doc_list = sorted_docset.take_all()

        for i in range(1, len(doc_list)):
            assert doc_list[i].text_representation <= doc_list[i - 1].text_representation

    def test_sort_ascending(self, docset: DocSet):
        sorted_docset = docset.sort(False, "properties.document_number")
        doc_list = sorted_docset.take_all()

        for i in range(1, len(doc_list)):
            assert doc_list[i].properties["document_number"] >= doc_list[i - 1].properties["document_number"]

    def test_default_value(self, docset: DocSet):
        sorted_docset = docset.sort(False, "doc_id.even", 0)
        doc_list = sorted_docset.take_all()

        for i in range(1, len(doc_list)):
            assert doc_list[i].doc_id.get("even", 0) >= doc_list[i - 1].doc_id.get("even", 0)
