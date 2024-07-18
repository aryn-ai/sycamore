import string
import pytest
import random

import sycamore
from sycamore import DocSet
from sycamore.data import Document


class TestSort:
    @pytest.fixture()
    def docs(self) -> list[Document]:
        print("Generating docs")
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
        sorted = True
        doc_list = sorted_docset.take_all()

        for i in range(1, len(doc_list)):
            if doc_list[i].text_representation > doc_list[i - 1].text_representation:
                sorted = False
                break

        assert sorted

    def test_sort_ascending(self, docset: DocSet):
        sorted_docset = docset.sort(False, "properties.document_number")
        sorted = True
        doc_list = sorted_docset.take_all()

        for i in range(1, len(doc_list)):
            if doc_list[i].properties["document_number"] < doc_list[i - 1].properties["document_number"]:
                sorted = False
                break

        assert sorted

    def test_default_value(self, docset: DocSet):
        sorted_docset = docset.sort(False, "doc_id.even", 0)
        sorted = True
        doc_list = sorted_docset.take_all()

        for i in range(1, len(doc_list)):

            if "even" not in doc_list[i].doc_id or "even" not in doc_list[i - 1].doc_id:
                continue

            if doc_list[i].doc_id["even"] < doc_list[i - 1].doc_id["even"]:
                sorted = False
                break

        assert sorted
