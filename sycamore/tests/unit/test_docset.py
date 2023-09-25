from typing import Callable

from sycamore import DocSet, Context
from sycamore.execution import Node
from sycamore.execution.scans import BinaryScan
from sycamore.execution.transforms import (
    Embedder,
    Embed,
    Partitioner,
    Summarize,
    ExtractEntity,
    FlatMap,
    Map,
    MapBatch,
    Partition,
)
from sycamore.execution.transforms.entity_extraction import OpenAIEntityExtractor
from sycamore.execution.transforms.llms import LLM
from sycamore.execution.transforms.mapping import Filter
from sycamore.execution.transforms.summarize import LLMElementTextSummarizer


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
