import pytest

import sycamore
from sycamore import DocSet
from sycamore.data import Document


class TestGroup:
    @pytest.fixture
    def fruits_docset(self) -> DocSet:
        doc_list = [
            Document(text_representation="apple", parent_id=8),
            Document(text_representation="banana", parent_id=7),
            Document(text_representation="apple", parent_id=8),
            Document(text_representation="banana", parent_id=7),
            Document(text_representation="cherry", parent_id=6),
            Document(text_representation="apple", parent_id=9),
        ]
        context = sycamore.init()
        return context.read.document(doc_list)

    def test_groupby_count(self, fruits_docset):
        aggregated = fruits_docset.groupby("text_representation").count()
        assert aggregated.count() == 3
