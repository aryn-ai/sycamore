from typing import Callable

from sycamore import DocSet, Context
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
)
from sycamore.transforms.extract_entity import OpenAIEntityExtractor
from sycamore.llms import LLM
from sycamore.transforms import Filter
from sycamore.transforms.summarize import LLMElementTextSummarizer


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
