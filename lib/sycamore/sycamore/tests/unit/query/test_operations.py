from datetime import datetime
from typing import Callable, Dict, List, Any, Optional

import pytest

import sycamore
from sycamore.data import Document
from sycamore.docset import DocSet
from sycamore.llms import LLM
from sycamore.query.execution.operations import (
    convert_string_to_date,
    field_to_value,
    join_operation,
    llm_extract_operation,
    llm_filter_operation,
    llm_generate_operation,
    match_filter_operation,
    math_operation,
    range_filter_operation,
    count_operation,
    semantic_cluster,
    top_k_operation,
    SC_FORM_GROUPS_PROMPT,
    SC_ASSIGN_GROUPS_PROMPT,
)


class MockLLM(LLM):
    def __init__(self):
        super().__init__(model_name="mock_model")

    def generate(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None):
        if prompt_kwargs == {"messages": [{"role": "user", "content": "test1"}]} and llm_kwargs == {}:
            return 4
        elif prompt_kwargs == {"messages": [{"role": "user", "content": "test2"}]} and llm_kwargs == {}:
            return 2
        elif prompt_kwargs["messages"][0]["content"] == SC_FORM_GROUPS_PROMPT.format(
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
    def test_docset(self, generate_docset) -> DocSet:
        return generate_docset({"text_representation": ["test1", "test2"]})

    @pytest.fixture
    def number_docset(self, generate_docset) -> DocSet:
        return generate_docset(
            {"text_representation": ["1", "2", "one", "two", "1", "3"], "parent_id": [8, 1, 11, 17, 13, 5]},
        )

    # Filters
    def test_llm_filter(self, test_docset):
        filtered_docset = llm_filter_operation(
            client=MockLLM(), docset=test_docset, field="text_representation", messages=[]
        )

        assert filtered_docset.count() == 1
        for doc in filtered_docset.take():
            assert doc.text_representation == "test1"
            assert int(doc.properties["LlmFilterOutput"]) == 4

        filtered_docset = llm_filter_operation(
            client=MockLLM(), docset=test_docset, field="text_representation", messages=[], threshold=2
        )

        assert filtered_docset.count() == 2

        for doc in filtered_docset.take():
            if doc.text_representation == "test1":
                assert int(doc.properties["LlmFilterOutput"]) == 4
            elif doc.text_representation == "test2":
                assert int(doc.properties["LlmFilterOutput"]) == 2

    def test_match_filter_number(self, words_and_ids_docset):
        query = 3
        filtered_docset = words_and_ids_docset.filter(
            lambda doc: match_filter_operation(doc, query=query, field="doc_id")
        )

        assert filtered_docset.count() == 2
        for doc in filtered_docset.take():
            assert doc.doc_id == 3

    def test_match_filter_string(self, words_and_ids_docset):

        query = "sub"
        filtered_docset = words_and_ids_docset.filter(
            lambda doc: match_filter_operation(doc, query=query, field="text_representation")
        )

        assert filtered_docset.count() == 3

        filtered_texts = []
        for doc in filtered_docset.take():
            filtered_texts.append(doc.text_representation)
        assert filtered_texts == ["submarine", "unSubtle", "Sub"]

    def test_match_filter_string_case_sensititve(self, words_and_ids_docset):

        query = "sub"
        filtered_docset = words_and_ids_docset.filter(
            lambda doc: match_filter_operation(doc, query=query, field="text_representation", ignore_case=False)
        )

        assert filtered_docset.count() == 1

        filtered_texts = []
        for doc in filtered_docset.take():
            filtered_texts.append(doc.text_representation)
        assert filtered_texts == ["submarine"]

    def test_range_filter_number(self, words_and_ids_docset):
        start, end = 2, 4
        filtered_docset = words_and_ids_docset.filter(
            lambda doc: range_filter_operation(doc, field="doc_id", start=start, end=end)
        )

        assert filtered_docset.count() == 4

        filtered_ids = []
        for doc in filtered_docset.take():
            filtered_ids.append(doc.doc_id)
        assert filtered_ids == [3, 3, 2, 4]

    def test_range_filter_one_sided(self, words_and_ids_docset):
        start = 5
        filtered_docset = words_and_ids_docset.filter(
            lambda doc: range_filter_operation(doc, field="doc_id", start=start, end=None)
        )

        assert filtered_docset.count() == 4

        filtered_ids = []
        for doc in filtered_docset.take():
            filtered_ids.append(doc.doc_id)
        assert filtered_ids == [5, 9, 6, 7]

        end = 5
        filtered_docset = words_and_ids_docset.filter(
            lambda doc: range_filter_operation(doc, field="doc_id", start=None, end=end)
        )

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
        filtered_docset = docset.filter(
            lambda doc: range_filter_operation(doc, field="text_representation", start=start, end=end)
        )
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
        filtered_docset = docset.filter(
            lambda doc: range_filter_operation(doc, field="text_representation", start=start, end=end, date=True)
        )

        assert filtered_docset.count() == 4

        filtered_ids = []
        for doc in filtered_docset.take():
            filtered_ids.append(doc.text_representation)
        assert filtered_ids == ["January 1, 2022", "2022-05-04", "September 19, 2022", "2022-06-07T03:47:00Z"]

    # LLM Extract
    def test_llm_extract(self, test_docset):
        extracted_docset = test_docset.map(
            lambda doc: llm_extract_operation(
                client=MockLLM(), doc=doc, new_field="new_field", field="text_representation", messages=[]
            )
        )

        assert extracted_docset.count() == 2
        for doc in extracted_docset.take():
            if doc.text_representation == "test1":
                assert int(doc.properties["new_field"]) == 4
            elif doc.text_representation == "test2":
                assert int(doc.properties["new_field"]) == 2

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

    # Count
    def test_count_normal(self, words_and_ids_docset):
        assert count_operation(words_and_ids_docset) == 9

    def test_count_primary_or_field(self, words_and_ids_docset):
        assert count_operation(words_and_ids_docset, field="doc_id", primaryField=None) == 8
        assert count_operation(words_and_ids_docset, field=None, primaryField="doc_id") == 8

    def test_count_unique_primary_and_field(self, words_and_ids_docset):
        assert count_operation(words_and_ids_docset, field="doc_id", primaryField="text_representation") == 8

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
                assert doc.properties["ClusterAssignment"] == "group1"
            elif doc.text_representation == "2" or doc.text_representation == "two":
                assert doc.properties["ClusterAssignment"] == "group2"
            elif doc.text_representation == "3" or doc.text_representation == "three":
                assert doc.properties["ClusterAssignment"] == "group3"

    # Helpers
    def test_field_to_value(self):
        doc = Document(
            text_representation="hello",
            doc_id=1,
            properties={"letter": "A", "animal": "panda", "math": {"pi": 3.14, "e": 2.72, "tanx": "sinx/cosx"}},
        )

        assert field_to_value(doc, "text_representation") == "hello"
        assert field_to_value(doc, "doc_id") == 1
        assert field_to_value(doc, "properties.letter") == "A"
        assert field_to_value(doc, "properties.animal") == "panda"
        assert field_to_value(doc, "properties.math.pi") == 3.14
        assert field_to_value(doc, "properties.math.e") == 2.72
        assert field_to_value(doc, "properties.math.tanx") == "sinx/cosx"

        with pytest.raises(KeyError):
            field_to_value(doc, "properties.math.log")

        with pytest.raises(Exception):
            field_to_value(doc, "document_id")

        with pytest.raises(AssertionError):
            field_to_value(doc, "text_representation.text")

    def test_convert_string_to_date(self):
        date_string = "2024-07-21"
        expected_date = datetime(2024, 7, 21)
        assert convert_string_to_date(date_string) == expected_date

        date_string = "2024-07-21 14:30:00"
        expected_date = datetime(2024, 7, 21, 14, 30, 0)
        assert convert_string_to_date(date_string) == expected_date

        date_string = "2024-07-21T14:30:00+00:00"
        expected_date = datetime(2024, 7, 21, 14, 30, 0)
        assert convert_string_to_date(date_string) == expected_date

        date_string = "21st July 2024"
        expected_date = datetime(2024, 7, 21)
        assert convert_string_to_date(date_string) == expected_date

        date_string = "Not a date"
        with pytest.raises(ValueError):
            convert_string_to_date(date_string)

        date_string = "2020-02-29"
        expected_date = datetime(2020, 2, 29)
        assert convert_string_to_date(date_string) == expected_date

        date_string = "2023-12-31 23:59:59"
        expected_date = datetime(2023, 12, 31, 23, 59, 59)
        assert convert_string_to_date(date_string) == expected_date

        date_string = "2024-07-21T14:30:00.123456"
        expected_date = datetime(2024, 7, 21, 14, 30, 0, 123456)
        assert convert_string_to_date(date_string) == expected_date
