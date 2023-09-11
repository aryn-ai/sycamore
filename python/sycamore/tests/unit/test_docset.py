from sycamore import DocSet, Context
from sycamore.execution import Node
from sycamore.execution.scans import BinaryScan
from sycamore.execution.transforms import (
    SentenceTransformerEmbedding,
    FlatMap,
    Map,
    MapBatch,
    Partition,
    PdfPartitionerOptions,
    LLMExtractEntity,
    SummarizeText,
)
from sycamore.execution.transforms.llms import LLM


class TestDocSet:
    def test_partition_pdf(self, mocker):
        context = mocker.Mock(spec=Context)
        scan = mocker.Mock(spec=BinaryScan)
        docset = DocSet(context, scan)
        docset = docset.partition(PdfPartitionerOptions())
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
        docset = docset.llm_extract_entity(entity_to_extract="title", llm=llm)
        assert isinstance(docset.lineage(), LLMExtractEntity)

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
        docset = docset.summarize(llm=llm)
        assert isinstance(docset.lineage(), SummarizeText)
