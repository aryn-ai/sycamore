import random
import string
from typing import Callable, Optional

import pytest

import sycamore
from sycamore import DocSet, Context
from sycamore.context import OperationTypes
from sycamore.data import Document, Element
from sycamore.llms.prompts.default_prompts import (
    LlmClusterEntityAssignGroupsMessagesPrompt,
    LlmClusterEntityFormGroupsMessagesPrompt,
)
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
from sycamore.transforms.similarity import SimilarityScorer
from sycamore.transforms.sort import Sort
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

        elif (
            prompt_kwargs["messages"]
            == LlmClusterEntityFormGroupsMessagesPrompt(
                field="text_representation", instruction="", text="1, 2, one, two, 1, 3"
            ).as_messages()
        ):
            return '{"groups": ["group1", "group2", "group3"]}'
        elif (
            prompt_kwargs["messages"][0]
            == LlmClusterEntityAssignGroupsMessagesPrompt(
                field="text_representation", groups=["group1", "group2", "group3"]
            ).as_messages()[0]
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


class TestDocSet:
    @pytest.fixture
    def number_docset(self) -> DocSet:
        doc_list = [
            Document(text_representation="1", parent_id=8),
            Document(text_representation="2", parent_id=1),
            Document(text_representation="one", parent_id=11),
            Document(text_representation="two", parent_id=17),
            Document(text_representation="1", parent_id=13),
            Document(text_representation="3", parent_id=5),
        ]
        context = sycamore.init(params={"default": {"llm": MockLLM()}})
        return context.read.document(doc_list)

    @pytest.fixture
    def words_and_ids_docset(self) -> DocSet:
        doc_list = [
            Document(text_representation="submarine", doc_id=1),
            Document(text_representation=None, doc_id=3),
            Document(text_representation="awesome", doc_id=5),
            Document(text_representation=True, doc_id=9),
            Document(text_representation="unSubtle", doc_id=3),
            Document(text_representation="Sub", doc_id=2),
            Document(text_representation="sunny", doc_id=4),
            Document(text_representation="", doc_id=6),
            Document(text_representation=4, doc_id=7),
        ]
        context = sycamore.init()
        return context.read.document(doc_list)

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

    def test_sort(self, mocker):
        context = mocker.Mock(spec=Context)
        docset = DocSet(context, None)
        docset = docset.sort(None, None)
        assert isinstance(docset.lineage(), Sort)

    def test_rerank(self, mocker):
        docset = DocSet(Context(), None)
        similarity_scorer = mocker.Mock(spec=SimilarityScorer)
        docset = docset.rerank(similarity_scorer, "")
        assert isinstance(docset.lineage(), Sort)

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

        docset.take_all(limit=num_docs)
        with pytest.raises(ValueError):
            docset.take_all(limit=num_docs - 1)

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
        context = sycamore.init(params={OperationTypes.BINARY_CLASSIFIER: {"llm": MockLLM()}})
        docset = context.read.document(doc_list)
        new_field = "_autogen_LLMFilterOutput"

        filtered_docset = docset.llm_filter(new_field=new_field, prompt=[], field="text_representation", threshold=3)

        assert filtered_docset.count() == 1
        for doc in filtered_docset.take():
            assert doc.text_representation == "test1"
            assert int(doc.properties[new_field]) == 4

        filtered_docset = docset.llm_filter(new_field=new_field, prompt=[], field="text_representation", threshold=2)

        assert filtered_docset.count() == 2

        for doc in filtered_docset.take():
            if doc.text_representation == "test1":
                assert int(doc.properties[new_field]) == 4
            elif doc.text_representation == "test2":
                assert int(doc.properties[new_field]) == 2

    def test_groupby_count(self, fruits_docset):

        grouped_docset = fruits_docset.groupby_count(field="text_representation")
        assert grouped_docset.count() == 3
        for doc in grouped_docset.take():
            if doc.properties["key"] == "banana":
                assert doc.properties["count"] == 2
            if doc.properties["key"] == "apple":
                assert doc.properties["count"] == 3
            if doc.properties["key"] == "cherry":
                assert doc.properties["count"] == 1

    def test_groupby_count_unique_field(self, fruits_docset):

        grouped_docset = fruits_docset.groupby_count(field="text_representation", unique_field="parent_id")
        assert grouped_docset.count() == 3
        for doc in grouped_docset.take():
            if doc.properties["key"] == "banana":
                assert doc.properties["count"] == 1
            if doc.properties["key"] == "apple":
                assert doc.properties["count"] == 2
            if doc.properties["key"] == "cherry":
                assert doc.properties["count"] == 1

    def test_top_k_discrete(self, fruits_docset):

        top_k_docset = fruits_docset.top_k(
            llm=None,
            field="text_representation",
            k=2,
            descending=True,
            llm_cluster=False,
        )
        assert top_k_docset.count() == 2

        top_k_list = top_k_docset.take()
        assert top_k_list[0].properties["key"] == "apple"
        assert top_k_list[0].properties["count"] == 3
        assert top_k_list[1].properties["key"] == "banana"
        assert top_k_list[1].properties["count"] == 2

    def test_top_k_unique_field(self, fruits_docset):

        top_k_docset = fruits_docset.top_k(
            llm=None,
            field="text_representation",
            k=1,
            descending=True,
            llm_cluster=False,
            unique_field="parent_id",
            llm_cluster_instruction="Find 2 most frequent fruits",
        )
        assert top_k_docset.count() == 1

        top_k_list = top_k_docset.take()
        assert top_k_list[0].properties["key"] == "apple"
        assert top_k_list[0].properties["count"] == 2

    def test_top_k_llm_cluster(self, number_docset):
        top_k_docset = number_docset.top_k(
            field="text_representation",
            k=2,
            descending=True,
            llm_cluster=True,
            llm_cluster_instruction="",
        )
        assert top_k_docset.count() == 2

        top_k_list = top_k_docset.take()
        assert top_k_list[0].properties["key"] == "group1"
        assert top_k_list[0].properties["count"] == 3
        assert top_k_list[1].properties["key"] == "group2"
        assert top_k_list[1].properties["count"] == 2

    def test_llm_cluster_entity(self, number_docset):
        cluster_docset = number_docset.llm_cluster_entity(instruction="", field="text_representation")
        for doc in cluster_docset.take():
            if doc.text_representation == "1" or doc.text_representation == "one":
                assert doc.properties["_autogen_ClusterAssignment"] == "group1"
            elif doc.text_representation == "2" or doc.text_representation == "two":
                assert doc.properties["_autogen_ClusterAssignment"] == "group2"
            elif doc.text_representation == "3" or doc.text_representation == "three":
                assert doc.properties["_autogen_ClusterAssignment"] == "group3"

    def test_field_in(self, number_docset, words_and_ids_docset):

        joined_docset = words_and_ids_docset.field_in(docset2=number_docset, field1="doc_id", field2="parent_id")
        assert joined_docset.count() == 2

        for doc in joined_docset.take():
            assert doc.doc_id == 5 or doc.doc_id == 1

            if doc.doc_id == 5:
                assert doc.text_representation == "awesome"

            elif doc.doc_id == 1:
                assert doc.text_representation == "submarine"

        joined_docset_reverse = number_docset.field_in(
            docset2=words_and_ids_docset, field1="parent_id", field2="doc_id"
        )

        assert joined_docset_reverse.count() == 2

        for doc in joined_docset_reverse.take():
            assert doc.parent_id == 5 or doc.parent_id == 1

            if doc.parent_id == 5:
                assert doc.text_representation == "3"

            elif doc.parent_id == 1:
                assert doc.text_representation == "2"
