import pytest

import sycamore
from sycamore import DocSet
from sycamore.data import Document
from sycamore.grouped_data import AggregateCollect
from sycamore.plan_nodes import Node
from sycamore.transforms import Filter


class TestGroup:
    @pytest.fixture
    def fruits_docset(self) -> DocSet:
        doc_list = [
            Document(text_representation="apple", parent_id=8, properties={"name": "A"}),
            Document(text_representation="banana", parent_id=7, properties={"name": "B"}),
            Document(text_representation="apple", parent_id=8, properties={"name": "C"}),
            Document(text_representation="banana", parent_id=7, properties={"name": "D"}),
            Document(text_representation="cherry", parent_id=6, properties={"name": "E"}),
            Document(text_representation="apple", parent_id=9, properties={"name": "F"}),
        ]
        context = sycamore.init()
        return context.read.document(doc_list)

    def test_groupby_count(self, fruits_docset):
        aggregated = fruits_docset.groupby("text_representation").count()
        assert aggregated.count() == 3

    def test_groupby_collect(self, fruits_docset):
        aggregated = fruits_docset.groupby("text_representation", entity="properties.name").collect()
        assert aggregated.count() == 3

    def test_rewrite(self, fruits_docset):
        from sycamore.rules import Rule

        doc_list = [
            Document(text_representation="apple", parent_id=8, properties={"name": "A"}),
            Document(text_representation="banana", parent_id=7, properties={"name": "B"}),
            Document(text_representation="apple", parent_id=8, properties={"name": "C"}),
        ]

        class FilterRule(Rule):
            # if filter succeeds, the result count is 1, otherwise, it's 2
            def __call__(self, plan: Node) -> Node:
                if isinstance(plan, AggregateCollect):
                    assert plan.children[0] is not None
                    node = Filter(child=plan.children[0], f=lambda doc: doc.field_to_value("properties.name") == "A")
                    plan.children = [node]
                return plan

        context = sycamore.init()
        ds = context.read.document(doc_list).groupby("text_representation", entity="properties.name").collect()
        context.rewrite_rules.append(FilterRule())
        assert ds.count() == 1
