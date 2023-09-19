from sycamore import DocSet, Context
from sycamore.execution import Node
from sycamore.execution.scans import BinaryScan
from sycamore.execution.transforms import (
    Partitioner,
    Summarize,
    ExtractEntity,
    SentenceTransformerEmbedding,
    FlatMap,
    Map,
    MapBatch,
    Partition,
)
from sycamore.execution.transforms.entity_extraction import OpenAIEntityExtractor
from sycamore.execution.transforms.llms import LLM
from sycamore.execution.transforms.summarize import LLMElementTextSummarizer


class TestDocSet:
    def test_partition_pdf(self, mocker):
        context = mocker.Mock(spec=Context)
        scan = mocker.Mock(spec=BinaryScan)
        partitioner = mocker.Mock(spec=Partitioner)
        docset = DocSet(context, scan)
        docset = docset.partition(partitioner=partitioner)
        assert isinstance(docset.lineage(), Partition)

    def test_sentence_transformer_embedding(self, mocker):
        context = mocker.Mock(spec=Context)
        node = mocker.Mock(spec=Node)
        docset = DocSet(context, node)
        docset = docset.sentence_transformer_embed(
            col_name="col_name", model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        assert isinstance(docset.lineage(), SentenceTransformerEmbedding)

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
