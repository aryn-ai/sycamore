from shannon import (DocSet, Context)
from shannon.execution import Node
from shannon.execution.scans import BinaryScan
from shannon.execution.transforms import (
    SentenceTransformerEmbedding, FlatMap, Map, MapBatch,
    UnstructuredPartition)


class TestDocSet:

    def test_partition_pdf(self, mocker):
        context = mocker.Mock(spec=Context)
        scan = mocker.Mock(spec=BinaryScan)
        docset = DocSet(context, scan)
        docset = docset.unstructured_partition(col_name="bytes")
        assert (isinstance(docset.lineage(), UnstructuredPartition))

    def test_sentence_transformer_embedding(self, mocker):
        context = mocker.Mock(spec=Context)
        node = mocker.Mock(spec=Node)
        docset = DocSet(context, node)
        docset = docset.sentence_transformer_embed(
            col_name="col_name",
            model_name="sentence-transformers/all-MiniLM-L6-v2")
        assert (isinstance(docset.lineage(), SentenceTransformerEmbedding))

    def test_map(self, mocker):
        context = mocker.Mock(spec=Context)
        node = mocker.Mock(spec=Node)
        docset = DocSet(context, node)
        docset = docset.map(f=lambda doc: doc)
        assert (isinstance(docset.lineage(), Map))

    def test_flat_map(self, mocker):
        context = mocker.Mock(spec=Context)
        node = mocker.Mock(spec=Node)
        docset = DocSet(context, node)
        docset = docset.flat_map(f=lambda doc: [doc])
        assert (isinstance(docset.lineage(), FlatMap))

    def test_map_batch(self, mocker):
        context = mocker.Mock(spec=Context)
        node = mocker.Mock(spec=Node)
        docset = DocSet(context, node)
        docset = docset.map_batch(f=lambda doc: doc)
        assert (isinstance(docset.lineage(), MapBatch))
