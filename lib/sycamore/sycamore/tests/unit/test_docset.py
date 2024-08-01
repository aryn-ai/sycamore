import random
import string
from typing import Any, Callable, Optional

import pytest

import sycamore
from sycamore import DocSet, Context
from sycamore.data import Document, Element
from sycamore.transforms import (
    Embedder,
    Embed,
    Partitioner,
    Summarize,
    ExtractEntity,
    FlatMap,
    Map,
    MapBatch,
    Partition,
    ExtractSchema,
    ExtractBatchSchema,
    ExtractProperties,
    Query,
)

from sycamore.llms import LLM
from sycamore.transforms.base import get_name_from_callable
from sycamore.transforms.extract_entity import OpenAIEntityExtractor
from sycamore.transforms.extract_schema import SchemaExtractor
from sycamore.transforms import Filter
from sycamore.transforms.summarize import LLMElementTextSummarizer
from sycamore.transforms.query import QueryExecutor


class MockLLM(LLM):
    def __init__(self):
        super().__init__(model_name="mock_model")

    def generate(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None):
        if prompt_kwargs == {"messages": [{"role": "user", "content": "test1"}]} and llm_kwargs == {}:
            return 4
        elif prompt_kwargs == {"messages": [{"role": "user", "content": "test2"}]} and llm_kwargs == {}:
            return 2

    def is_chat_mode(self):
        return True


class TestDocSet:
    @pytest.fixture
    def generate_docset(self) -> Callable[[dict[str, list[Any]]], DocSet]:
        def _generate(docs_info: dict[str, list[Any]]) -> DocSet:
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

    def test_partition_pdf(self, mocker):
        context = mocker.Mock(spec=Context)
        partitioner = mocker.Mock(spec=Partitioner, device="cpu")
        docset = DocSet(context, None)
        docset = docset.partition(partitioner=partitioner)
        assert isinstance(docset.lineage(), Partition)

    def test_embedding(self, mocker):
        context = mocker.Mock(spec=Context)
        docset = DocSet(context, None)
        embedder = mocker.Mock(spec=Embedder, batch_size=1, device="cpu")
        docset = docset.embed(embedder=embedder)
        assert isinstance(docset.lineage(), Embed)

    def test_llm_extract_entity(self, mocker):
        context = mocker.Mock(spec=Context)
        llm = mocker.Mock(spec=LLM)
        docset = DocSet(context, None)
        docset = docset.extract_entity(entity_extractor=OpenAIEntityExtractor("title", llm=llm, prompt_template=""))
        assert isinstance(docset.lineage(), ExtractEntity)

    def test_query(self, mocker):
        context = mocker.Mock(spec=Context)
        query_executor = mocker.Mock(spec=QueryExecutor, query=lambda: None)
        docset = DocSet(context, None)
        docset = docset.query(query_executor=query_executor)
        assert isinstance(docset.lineage(), Query)

    def test_map_default_name(self, mocker):
        context = mocker.Mock(spec=Context)
        docset = DocSet(context, None)

        def f(doc):
            return doc

        docset = docset.map(f=f)
        assert isinstance(docset.lineage(), Map)
        assert docset.lineage()._name == get_name_from_callable(f)

    def test_map_custom_name(self, mocker):
        test_name = "test_map_1"
        context = mocker.Mock(spec=Context)
        docset = DocSet(context, None)
        docset = docset.map(f=lambda doc: doc, name=test_name)
        assert isinstance(docset.lineage(), Map)
        assert docset.lineage()._name == test_name

    def test_flat_map_default_name(self, mocker):
        context = mocker.Mock(spec=Context)
        docset = DocSet(context, None)

        def f(doc):
            return [doc]

        docset = docset.flat_map(f=f)
        assert isinstance(docset.lineage(), FlatMap)
        assert docset.lineage()._name == get_name_from_callable(f)

    def test_flat_map_custom_name(self, mocker):
        test_name = "test_flat_map_1"
        context = mocker.Mock(spec=Context)
        docset = DocSet(context, None)
        docset = docset.flat_map(f=lambda doc: [doc], name=test_name)
        assert isinstance(docset.lineage(), FlatMap)
        assert docset.lineage()._name == test_name

    def test_map_batch(self, mocker):
        context = mocker.Mock(spec=Context)
        docset = DocSet(context, None)
        docset = docset.map_batch(f=lambda doc: doc)
        assert isinstance(docset.lineage(), MapBatch)

    def test_summarize(self, mocker):
        context = mocker.Mock(spec=Context)
        llm = mocker.Mock(spec=LLM)
        docset = DocSet(context, None)
        docset = docset.summarize(llm=llm, summarizer=LLMElementTextSummarizer(llm))
        assert isinstance(docset.lineage(), Summarize)

    def test_filter(self, mocker):
        context = mocker.Mock(spec=Context)
        func = mocker.Mock(spec=Callable)
        docset = DocSet(context, None)
        docset = docset.filter(func)
        assert isinstance(docset.lineage(), Filter)

    def test_extract_schema(self, mocker):
        context = mocker.Mock(spec=Context)
        func = mocker.Mock(spec=Callable, extract_schema=lambda d: {})
        docset = DocSet(context, None)
        docset = docset.extract_schema(func)
        assert isinstance(docset.lineage(), ExtractSchema)

    def test_extract_batch_schema(self, mocker):
        context = mocker.Mock(spec=Context)
        func = mocker.Mock(spec=SchemaExtractor)
        docset = DocSet(context, None)
        docset = docset.extract_batch_schema(func)
        assert isinstance(docset.lineage(), ExtractBatchSchema)

    def test_extract_properties(self, mocker):
        context = mocker.Mock(spec=Context)
        func = mocker.Mock(spec=Callable, extract_properties=lambda d: {})
        docset = DocSet(context, None)
        docset = docset.extract_properties(func)
        assert isinstance(docset.lineage(), ExtractProperties)

    def test_take_all(self):
        num_docs = 30

        docs = []
        for i in range(num_docs):
            docs.append(Document(text_representation=f"Document {i}", doc_id=i, properties={"document_number": i}))

        context = sycamore.init()
        docset = context.read.document(docs)

        assert len(docset.take_all()) == num_docs

        with pytest.raises(ValueError):
            docset.take_all(limit=20)

    def random_string(self, min_size: int, max_size: int) -> str:
        k = random.randrange(min_size, max_size)
        return "".join(random.choices(string.ascii_letters, k=k))

    def text_len(self, doc: Document) -> int:
        if not doc.text_representation:
            return 0
        return len(doc.text_representation)

    def num_as(self, doc: Document) -> int:
        if not doc.text_representation:
            return 0

        count = 0
        for c in doc.text_representation:
            if c == "a":
                count += 1

        return count

    def test_with_property(self):
        texts = [self.random_string(min_size=20, max_size=100) for _ in range(10)]
        docs = [Document(text_representation=t, doc_id=i, properties={}) for i, t in enumerate(texts)]

        context = sycamore.init()

        docset = context.read.document(docs).with_property("text_size", self.text_len)

        expected = [len(t) for t in texts]
        actual = [d.properties["text_size"] for d in docset.take_all()]

        assert sorted(expected) == sorted(actual)

    def test_with_properties(self):
        texts = [self.random_string(min_size=20, max_size=100) for _ in range(10)]
        docs = [Document(text_representation=t, doc_id=i, properties={}) for i, t in enumerate(texts)]

        context = sycamore.init()
        docset = context.read.document(docs).with_properties({"text_size": self.text_len, "num_as": self.num_as})

        expected_length = [len(t) for t in texts]
        expected_as = [len(list(filter(lambda x: x == "a", t))) for t in texts]

        post_docs = docset.take_all()

        actual_length = [d.properties["text_size"] for d in post_docs]
        actual_as = [d.properties["num_as"] for d in post_docs]

        assert sorted(expected_length) == sorted(actual_length)
        assert sorted(expected_as) == sorted(actual_as)

    def double_element(self, elem: Element) -> Element:
        elem.properties["element_double"] = elem.properties["element_val"] * 2
        return elem

    def test_map_elements(self):
        docs = []
        for i in range(10):
            doc = Document(text_representation=f"Document {i}", doc_id=i, properties={"document_number": i})
            doc.elements = [
                Element(text_representation=f"Document {i} Element {j}", properties={"element_val": i})
                for j in range(10)
            ]
            docs.append(doc)

        context = sycamore.init()
        docset = context.read.document(docs).map_elements(self.double_element)

        all_docs = docset.take_all()
        for doc in all_docs:
            for elem in doc.elements:
                assert elem.properties["element_double"] == elem.properties["element_val"] * 2

    def test_filter_elements(self):
        docs = []
        for i in range(10):
            doc = Document(text_representation=f"Document {i}", doc_id=i, properties={"document_number": i})
            doc.elements = [
                Element(text_representation=f"Document {i} Element {j}", properties={"element_val": j})
                for j in range(10)
            ]
            docs.append(doc)

        context = sycamore.init()
        docset = context.read.document(docs).filter_elements(lambda e: e.properties["element_val"] % 2 == 0)

        all_docs = docset.take_all()
        assert len(all_docs) == len(docs)

        for doc in all_docs:
            for elem in doc.elements:
                assert elem.properties["element_val"] % 2 == 0

    def test_count(self):

        docs = []
        for i in range(10):
            docs.append(Document(text_representation=""))

        context = sycamore.init()
        docset = context.read.document(docs)
        assert docset.count() == 10

    def test_count_distinct(self):

        docs = []
        for i in range(10):
            if i == 8 or i == 9:
                num = 20
            else:
                num = i
            docs.append(Document(text_representation="", doc_id=num))

        context = sycamore.init()
        docset = context.read.document(docs)
        assert docset.count_distinct("doc_id") == 9

    def test_llm_filter(self):

        doc_list = [Document(text_representation="test1"), Document(text_representation="test2")]
        context = sycamore.init()
        docset = context.read.document(doc_list)
        new_field = "_autogen_LLMFilterOutput"

        filtered_docset = docset.llm_filter(
            llm=MockLLM(), new_field=new_field, prompt=[], field="text_representation", threshold=3
        )

        assert filtered_docset.count() == 1
        for doc in filtered_docset.take():
            assert doc.text_representation == "test1"
            assert int(doc.properties[new_field]) == 4

        filtered_docset = docset.llm_filter(
            llm=MockLLM(), new_field=new_field, prompt=[], field="text_representation", threshold=2
        )

        assert filtered_docset.count() == 2

        for doc in filtered_docset.take():
            if doc.text_representation == "test1":
                assert int(doc.properties[new_field]) == 4
            elif doc.text_representation == "test2":
                assert int(doc.properties[new_field]) == 2

    def test_match_filter_number(self, words_and_ids_docset):
        query = 3
        filtered_docset = words_and_ids_docset.match_filter(query=query, field="doc_id")

        assert filtered_docset.count() == 2
        for doc in filtered_docset.take():
            assert doc.doc_id == 3

    def test_match_filter_string(self, words_and_ids_docset):

        query = "sub"
        filtered_docset = words_and_ids_docset.match_filter(query=query, field="text_representation")

        assert filtered_docset.count() == 3

        filtered_texts = []
        for doc in filtered_docset.take():
            filtered_texts.append(doc.text_representation)
        assert filtered_texts == ["submarine", "unSubtle", "Sub"]

    def test_match_filter_string_case_sensititve(self, words_and_ids_docset):

        query = "sub"
        filtered_docset = words_and_ids_docset.match_filter(query=query, field="text_representation", ignore_case=False)

        assert filtered_docset.count() == 1

        filtered_texts = []
        for doc in filtered_docset.take():
            filtered_texts.append(doc.text_representation)
        assert filtered_texts == ["submarine"]

    def test_range_filter_number(self, words_and_ids_docset):
        start, end = 2, 4
        filtered_docset = words_and_ids_docset.range_filter(field="doc_id", start=start, end=end)

        assert filtered_docset.count() == 4

        filtered_ids = []
        for doc in filtered_docset.take():
            filtered_ids.append(doc.doc_id)
        assert filtered_ids == [3, 3, 2, 4]

    def test_range_filter_one_sided(self, words_and_ids_docset):
        start = 5
        filtered_docset = words_and_ids_docset.range_filter(field="doc_id", start=start, end=None)

        assert filtered_docset.count() == 4

        filtered_ids = []
        for doc in filtered_docset.take():
            filtered_ids.append(doc.doc_id)
        assert filtered_ids == [5, 9, 6, 7]

        end = 5
        filtered_docset = words_and_ids_docset.range_filter(field="doc_id", start=None, end=end)

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
        filtered_docset = docset.range_filter(field="text_representation", start=start, end=end)

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
        filtered_docset = docset.range_filter(field="text_representation", start=start, end=end, date=True)

        assert filtered_docset.count() == 4

        filtered_ids = []
        for doc in filtered_docset.take():
            filtered_ids.append(doc.text_representation)
        assert filtered_ids == ["January 1, 2022", "2022-05-04", "September 19, 2022", "2022-06-07T03:47:00Z"]
