from typing import Callable, Dict, List, Any, Optional

import pytest

import sycamore
from sycamore.data import Document
from sycamore.docset import DocSet
from sycamore.llms import LLM
from sycamore.query.execution.operations import (
    join_operation,
    llm_generate_operation,
    math_operation,
    semantic_cluster,
    top_k_operation,
    SC_FORM_GROUPS_PROMPT,
    SC_ASSIGN_GROUPS_PROMPT,
)


class MockLLM(LLM):
    def __init__(self):
        super().__init__(model_name="mock_model")

    def generate(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None):
        if prompt_kwargs["messages"][0]["content"] == SC_FORM_GROUPS_PROMPT.format(
            field="text_representation", description="", text="1, 2, one, two, 1, 3"
        ):
            return '{"groups": ["group1", "group2", "group3"]}'
        elif prompt_kwargs["messages"][0]["content"] == SC_ASSIGN_GROUPS_PROMPT.format(
            field="text_representation", groups=["group1", "group2", "group3"]
        ):
            value = prompt_kwargs["messages"][1]["content"]
            if value == "1" or value == "one":
                return "group1"
            elif value == "2" or value == "two":
                return "group2"
            elif value == "3" or value == "three":
                return "group3"
        else:
            return ""

    def is_chat_mode(self):
        return True


class TestOperations:
    @pytest.fixture
    def generate_docset(self) -> Callable[[Dict[str, List[Any]]], DocSet]:
        def _generate(docs_info: Dict[str, List[Any]]) -> DocSet:
            # make sure same length
            keys = list(docs_info.keys())
            num_docs = len(docs_info[keys[0]])

            for k in docs_info:
                assert len(docs_info[k]) == num_docs

            doc_list = []

            for i in range(num_docs):
                doc_data = {key: docs_info[key][i] for key in keys}
                doc_list.append(Document(**doc_data))

            context = sycamore.init()
            return context.read.document(doc_list)

        return _generate

    @pytest.fixture
    def words_and_ids_docset(self, generate_docset) -> DocSet:
        texts = {
            "text_representation": ["submarine", None, "awesome", True, "unSubtle", "Sub", "sunny", "", 4],
            "doc_id": [1, 3, 5, 9, 3, 2, 4, 6, 7],
        }
        return generate_docset(texts)

    @pytest.fixture
    def number_docset(self, generate_docset) -> DocSet:
        return generate_docset(
            {"text_representation": ["1", "2", "one", "two", "1", "3"], "parent_id": [8, 1, 11, 17, 13, 5]},
        )

    # LLM Generate
    def test_llm_generate(words_and_ids_docset):
        response = llm_generate_operation(client=MockLLM(), question="", result_description="", result_data=[""])
        assert response == ""

        response = llm_generate_operation(
            client=MockLLM(), question="", result_description="", result_data=[words_and_ids_docset]
        )
        assert response == ""

    # Join
    def test_join(self, words_and_ids_docset, number_docset):
        joined_docset = join_operation(
            docset1=number_docset, docset2=words_and_ids_docset, field1="parent_id", field2="doc_id"
        )
        assert joined_docset.count() == 2

        for doc in joined_docset.take():
            assert doc.doc_id == 5 or doc.doc_id == 1

            if doc.doc_id == 5:
                assert doc.text_representation == "awesome"

            elif doc.doc_id == 1:
                assert doc.text_representation == "submarine"

        joined_docset_reverse = join_operation(
            docset1=words_and_ids_docset, docset2=number_docset, field1="doc_id", field2="parent_id"
        )

        assert joined_docset_reverse.count() == 2

        for doc in joined_docset_reverse.take():
            assert doc.parent_id == 5 or doc.parent_id == 1

            if doc.parent_id == 5:
                assert doc.text_representation == "3"

            elif doc.parent_id == 1:
                assert doc.text_representation == "2"

    # Math
    def test_math(self):
        assert math_operation(val1=1, val2=2, operator="add") == 3
        assert math_operation(val1=5, val2=3, operator="subtract") == 2
        assert math_operation(val1=4, val2=2, operator="divide") == 2
        assert math_operation(val1=3, val2=3, operator="multiply") == 9

    # Top K
    def test_top_k_discrete(self, generate_docset):
        docset = generate_docset({"text_representation": ["apple", "banana", "apple", "banana", "cherry", "apple"]})

        top_k_docset = top_k_operation(
            client=None,
            docset=docset,
            field="text_representation",
            k=2,
            description="Find 2 most frequent fruits",
            descending=True,
            use_llm=False,
        )
        assert top_k_docset.count() == 2

        top_k_list = top_k_docset.take()
        assert top_k_list[0].properties["key"] == "apple"
        assert top_k_list[0].properties["count"] == 3
        assert top_k_list[1].properties["key"] == "banana"
        assert top_k_list[1].properties["count"] == 2

    def test_top_k_use_llm(self, number_docset):
        top_k_docset = top_k_operation(
            client=MockLLM(),
            docset=number_docset,
            field="text_representation",
            k=2,
            description="",
            descending=True,
            use_llm=True,
        )
        assert top_k_docset.count() == 2

        top_k_list = top_k_docset.take()
        assert top_k_list[0].properties["key"] == "group1"
        assert top_k_list[0].properties["count"] == 3
        assert top_k_list[1].properties["key"] == "group2"
        assert top_k_list[1].properties["count"] == 2

    def test_semantic_cluster(self, number_docset):
        cluster_docset = semantic_cluster(
            client=MockLLM(), docset=number_docset, description="", field="text_representation"
        )
        for doc in cluster_docset.take():
            if doc.text_representation == "1" or doc.text_representation == "one":
                assert doc.properties["_autogen_ClusterAssignment"] == "group1"
            elif doc.text_representation == "2" or doc.text_representation == "two":
                assert doc.properties["_autogen_ClusterAssignment"] == "group2"
            elif doc.text_representation == "3" or doc.text_representation == "three":
                assert doc.properties["_autogen_ClusterAssignment"] == "group3"
