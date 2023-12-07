import math

import pytest

import sycamore
from sycamore import DocSet
from sycamore.data import Document


class TestRandomSample:
    @pytest.fixture()
    def docs(self) -> list[Document]:
        print("Generating docs")
        return [
            Document(text_representation=f"Document {i}", doc_id=i, properties={"document_number": i})
            for i in range(100)
        ]

    @pytest.fixture()
    def docset(self, docs: list[Document]) -> DocSet:
        context = sycamore.init()
        return context.read.document(docs)

    def test_empty_sample(self, docset: DocSet):
        assert docset.random_sample(0).count() == 0

    def test_complete_sample(self, docset: DocSet):
        assert docset.random_sample(1).count() == 100

    def test_random_sample(self, docset: DocSet):
        actual = docset.random_sample(0.5).count()
        math.isclose(actual, 50, rel_tol=2, abs_tol=2)
