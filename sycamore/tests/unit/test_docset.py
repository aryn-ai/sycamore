import random
import string
from typing import Callable

import pytest

import sycamore
from sycamore import DocSet, Context
from sycamore.data import Document, Element
from sycamore.plan_nodes import Node
from sycamore.scans import BinaryScan
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
from sycamore.transforms.extract_entity import OpenAIEntityExtractor
from sycamore.llms import LLM
from sycamore.transforms import Filter
from sycamore.transforms.summarize import LLMElementTextSummarizer
from sycamore.transforms.query import QueryExecutor


class TestDocSet:
    def test_partition_pdf(self, mocker):
        context = mocker.Mock(spec=Context)
        scan = mocker.Mock(spec=BinaryScan)
        partitioner = mocker.Mock(spec=Partitioner)
        docset = DocSet(context, scan)
        docset = docset.partition(partitioner=partitioner)
        assert isinstance(docset.lineage(), Partition)

    def test_embedding(self, mocker):
        context = mocker.Mock(spec=Context)
        node = mocker.Mock(spec=Node)
        docset = DocSet(context, node)
        embedder = mocker.Mock(spec=Embedder)
        docset = docset.embed(embedder=embedder)
        assert isinstance(docset.lineage(), Embed)

    def test_llm_extract_entity(self, mocker):
        context = mocker.Mock(spec=Context)
        node = mocker.Mock(spec=Node)
        llm = mocker.Mock(spec=LLM)
        docset = DocSet(context, node)
        docset = docset.extract_entity(entity_extractor=OpenAIEntityExtractor("title", llm=llm, prompt_template=""))
        assert isinstance(docset.lineage(), ExtractEntity)

    def test_query(self, mocker):
        context = mocker.Mock(spec=Context)
        node = mocker.Mock(spec=Node)
        query_executor = mocker.Mock(spec=QueryExecutor)
        docset = DocSet(context, node)
        docset = docset.query(query_executor=query_executor)
        a = docset.lineage()
        assert isinstance(docset.lineage(), Query)

    def test_map(self, mocker):
        context = mocker.Mock(spec=Context)
        node = mocker.Mock(spec=Node)
        docset = DocSet(context, node)
        docset = docset.map(f=lambda doc: doc)
        assert isinstance(docset.lineage(), Map)

    def test_flat_map(self, mocker):
        context = mocker.Mock(spec=Context)
        node = mocker.Mock(spec=Node)
        docset = DocSet(context, node)
        docset = docset.flat_map(f=lambda doc: [doc])
        assert isinstance(docset.lineage(), FlatMap)

    def test_map_batch(self, mocker):
        context = mocker.Mock(spec=Context)
        node = mocker.Mock(spec=Node)
        docset = DocSet(context, node)
        docset = docset.map_batch(f=lambda doc: doc)
        assert isinstance(docset.lineage(), MapBatch)

    def test_summarize(self, mocker):
        context = mocker.Mock(spec=Context)
        node = mocker.Mock(spec=Node)
        llm = mocker.Mock(spec=LLM)
        docset = DocSet(context, node)
        docset = docset.summarize(llm=llm, summarizer=LLMElementTextSummarizer(llm))
        assert isinstance(docset.lineage(), Summarize)

    def test_filter(self, mocker):
        context = mocker.Mock(spec=Context)
        node = mocker.Mock(spec=Node)
        func = mocker.Mock(spec=Callable)
        docset = DocSet(context, node)
        docset = docset.filter(func)
        assert isinstance(docset.lineage(), Filter)

    def test_extract_schema(self, mocker):
        context = mocker.Mock(spec=Context)
        node = mocker.Mock(spec=Node)
        func = mocker.Mock(spec=Callable)
        docset = DocSet(context, node)
        docset = docset.extract_schema(func)
        assert isinstance(docset.lineage(), ExtractSchema)

    def test_extract_batch_schema(self, mocker):
        context = mocker.Mock(spec=Context)
        node = mocker.Mock(spec=Node)
        func = mocker.Mock(spec=Callable)
        docset = DocSet(context, node)
        docset = docset.extract_batch_schema(func)
        assert isinstance(docset.lineage(), ExtractBatchSchema)

    def test_extract_properties(self, mocker):
        context = mocker.Mock(spec=Context)
        node = mocker.Mock(spec=Node)
        func = mocker.Mock(spec=Callable)
        docset = DocSet(context, node)
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
        props = elem.properties
        props["element_double"] = props["element_val"] * 2
        elem.properties = props
        return elem

    def test_map_elements(self):
        docs = []
        for i in range(10):
            doc = Document(text_representation=f"Document {i}", doc_id=i, properties={"document_number": i})
            doc.elements = [
                Element(text_representation=f"Document {i} Element {j}", properties={"element_val": i})
                for j in range(10)
            ]

        context = sycamore.init()
        docset = context.read.document(docs).map_elements(self.double_element)

        all_docs = docset.take_all()
        for doc in all_docs:
            for elem in doc.elements:
                assert elem.properties["element_double"] == elem.properties["element_val"] * 2
