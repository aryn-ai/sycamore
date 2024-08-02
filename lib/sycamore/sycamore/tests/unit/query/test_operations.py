from typing import Callable, Dict, List, Any, Optional

import pytest

import sycamore
from sycamore.data import Document
from sycamore.docset import DocSet
from sycamore.functions.basic_filters import MatchFilter, RangeFilter
from sycamore.llms import LLM
from sycamore.llms.prompts.default_prompts import (
    LlmClusterEntityAssignGroupsMessagesPrompt,
    LlmClusterEntityFormGroupsMessagesPrompt,
)
from sycamore.query.execution.operations import (
    summarize_data,
    math_operation,
)


class MockLLM(LLM):
    def __init__(self):
        super().__init__(model_name="mock_model")

    def generate(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None):
        if (
            prompt_kwargs["messages"]
            == LlmClusterEntityFormGroupsMessagesPrompt(
                field="text_representation", description="", text="1, 2, one, two, 1, 3"
            ).get_messages_dict()
        ):
            return '{"groups": ["group1", "group2", "group3"]}'
        elif (
            prompt_kwargs["messages"][0]
            == LlmClusterEntityAssignGroupsMessagesPrompt(
                field="text_representation", groups=["group1", "group2", "group3"]
            ).get_messages_dict()[0]
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
    def test_summarize_data(words_and_ids_docset):
        response = summarize_data(llm=MockLLM(), question="", result_description="", result_data=[""])
        assert response == ""

        response = summarize_data(llm=MockLLM(), question="", result_description="", result_data=[words_and_ids_docset])
        assert response == ""

    # Math
    def test_math(self):
        assert math_operation(val1=1, val2=2, operator="add") == 3
        assert math_operation(val1=5, val2=3, operator="subtract") == 2
        assert math_operation(val1=4, val2=2, operator="divide") == 2
        assert math_operation(val1=3, val2=3, operator="multiply") == 9

    def test_match_filter_number(self, words_and_ids_docset):
        query = 3
        filtered_docset = words_and_ids_docset.filter(f=MatchFilter(query=query, field="doc_id"))

        assert filtered_docset.count() == 2
        for doc in filtered_docset.take():
            assert doc.doc_id == 3

    def test_match_filter_string(self, words_and_ids_docset):

        query = "sub"
        filtered_docset = words_and_ids_docset.filter(f=MatchFilter(query=query, field="text_representation"))

        assert filtered_docset.count() == 3

        filtered_texts = []
        for doc in filtered_docset.take():
            filtered_texts.append(doc.text_representation)
        assert filtered_texts == ["submarine", "unSubtle", "Sub"]

    def test_match_filter_string_case_sensititve(self, words_and_ids_docset):

        query = "sub"
        filtered_docset = words_and_ids_docset.filter(
            f=MatchFilter(query=query, field="text_representation", ignore_case=False)
        )

        assert filtered_docset.count() == 1

        filtered_texts = []
        for doc in filtered_docset.take():
            filtered_texts.append(doc.text_representation)
        assert filtered_texts == ["submarine"]

    def test_range_filter_number(self, words_and_ids_docset):
        start, end = 2, 4
        filtered_docset = words_and_ids_docset.filter(f=RangeFilter(field="doc_id", start=start, end=end))

        assert filtered_docset.count() == 4

        filtered_ids = []
        for doc in filtered_docset.take():
            filtered_ids.append(doc.doc_id)
        assert filtered_ids == [3, 3, 2, 4]

    def test_range_filter_one_sided(self, words_and_ids_docset):
        start = 5
        filtered_docset = words_and_ids_docset.filter(f=RangeFilter(field="doc_id", start=start, end=None))

        assert filtered_docset.count() == 4

        filtered_ids = []
        for doc in filtered_docset.take():
            filtered_ids.append(doc.doc_id)
        assert filtered_ids == [5, 9, 6, 7]

        end = 5
        filtered_docset = words_and_ids_docset.filter(f=RangeFilter(field="doc_id", start=None, end=end))

        assert filtered_docset.count() == 6

        filtered_ids = []
        for doc in filtered_docset.take():
            filtered_ids.append(doc.doc_id)
        assert filtered_ids == [1, 3, 5, 3, 2, 4]

    # only works if the entire field is strings
    def test_range_filter_string(self, generate_docset):
        start, end = "b", "t"
        docset = generate_docset(
            {"text_representation": ["a", "b", "bBc", "abc", "lmnop", "qq", "edgar", "", "t", "tense"]}
        )
        filtered_docset = docset.filter(f=RangeFilter(field="text_representation", start=start, end=end))

        assert filtered_docset.count() == 6

        filtered_ids = []
        for doc in filtered_docset.take():
            filtered_ids.append(doc.text_representation)
        assert filtered_ids == ["b", "bBc", "lmnop", "qq", "edgar", "t"]

    # only allow date formats recognized by DateUtil parser
    # e.g. "December 31, 2022, 12:30 Local" will not work
    def test_range_filter_date(self, generate_docset):
        start, end = "01-01-2022", "December 31, 2022"
        docset = generate_docset(
            {
                "text_representation": [
                    "January 1, 2022",
                    "2/4/20",
                    "2022-05-04",
                    "January 14, 2023",
                    "2023-01-29T12:30:00Z",
                    "12/12/2023",
                    "September 19, 2022",
                    "2022-06-07T03:47:00Z",
                    "April 15, 2023",
                ]
            }
        )
        filtered_docset = docset.filter(f=RangeFilter(field="text_representation", start=start, end=end, date=True))

        assert filtered_docset.count() == 4

        filtered_ids = []
        for doc in filtered_docset.take():
            filtered_ids.append(doc.text_representation)
        assert filtered_ids == ["January 1, 2022", "2022-05-04", "September 19, 2022", "2022-06-07T03:47:00Z"]
