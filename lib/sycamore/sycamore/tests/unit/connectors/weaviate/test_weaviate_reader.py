from sycamore.connectors.weaviate.weaviate_reader import (
    WeaviateReaderQueryParams,
    WeaviateReaderClient,
    WeaviateReaderQueryResponse,
)
from sycamore.data.document import DocumentSource

from weaviate.classes.query import Rerank, MetadataQuery
from sycamore.connectors.common import compare_docs
from sycamore.data.document import Document
import weaviate
from weaviate.collections.query import _QueryCollection


class WeaviateReturnObject(object):
    pass


class TestWeaviateClient:
    @staticmethod
    def mock_client(mocker):
        client = mocker.Mock(spec=weaviate.WeaviateClient)
        client.__enter__ = mocker.Mock()
        client.__enter__.return_value = client
        client.__exit__ = mocker.Mock()
        client.collections = mocker.Mock()
        collection = mocker.Mock(spec=_QueryCollection)
        client.collections.get.query.return_value = collection
        return client

    def test_check_existence(self, mocker):
        cn = "TestCheckExistence"
        wtp_a = WeaviateReaderQueryParams(collection_name=cn)
        wcl = TestWeaviateClient.mock_client(mocker)
        wcl.collections.exists = mocker.Mock()
        wcl.collections.create.return_value = True
        client = WeaviateReaderClient(wcl)
        client.check_target_presence(wtp_a)
        wcl.collections.exists.assert_called_once()

    def test_read_documents(self, mocker):
        cn = "TestReadDocuments"
        wtp_a = WeaviateReaderQueryParams(collection_name=cn)

        wcl = TestWeaviateClient.mock_client(mocker)
        wcl.collections.get(cn).iterator = mocker.Mock()
        client = WeaviateReaderClient(wcl)
        client.read_records(wtp_a)
        assert wcl.collections.get(cn).iterator.call_count == 1

    def test_read_documents_with_query(self, mocker):
        cn = "TestReadDocumentsQuery"
        wtp_a = WeaviateReaderQueryParams(collection_name=cn, query_kwargs={"fetch_objects": {"limit": 2}})
        wcl = TestWeaviateClient.mock_client(mocker)
        wcl.collections.get(cn).query.fetch_objects = mocker.Mock()
        client = WeaviateReaderClient(wcl)
        client.read_records(wtp_a)
        assert wcl.collections.get(cn).query.fetch_objects.call_count == 1

    def test_read_documents_with_search_query(self, mocker):
        cn = "TestReadDocumentsQueryBM25"
        wtp_a = WeaviateReaderQueryParams(collection_name=cn, query_kwargs={"bm25": {"query": "traffic"}})
        wcl = TestWeaviateClient.mock_client(mocker)
        wcl.collections.get(cn).query.bm25 = mocker.Mock()
        client = WeaviateReaderClient(wcl)
        client.read_records(wtp_a)
        assert wcl.collections.get(cn).query.bm25.call_count == 1

    def test_read_documents_with_reranking_multiple_parameters_query(self, mocker):
        cn = "TestReadDocumentsQueryNearText"
        wtp_a = WeaviateReaderQueryParams(
            collection_name=cn,
            query_kwargs={
                "near_text": {
                    "query": "traffic",
                    "limit": 2,
                    "rerank": Rerank(prop="question", query="publication"),
                    "return_metadata": MetadataQuery(score=True),
                }
            },
        )
        wcl = TestWeaviateClient.mock_client(mocker)
        wcl.collections.get(cn).query.near_text = mocker.Mock()
        client = WeaviateReaderClient(wcl)
        client.read_records(wtp_a)
        assert wcl.collections.get(cn).query.near_text.call_count == 1


class TestWeaviateQueryResponse:
    def test_to_doc(self):
        cn = "TestToDocInitial"
        record = WeaviateReaderQueryResponse(collection=[])
        wtp_a = WeaviateReaderQueryParams(
            collection_name=cn,
        )
        doc_list = WeaviateReaderQueryResponse.to_docs(record, wtp_a)
        assert len(doc_list) == 0

    def test_to_doc_flattened(self):
        cn = "TestToDocFlattened"
        wro = WeaviateReturnObject()
        wro.uuid = "00000000-0000-0000-0000-000000000000"
        wro.properties = {
            "properties__field": "value",
            "properties__nested__object": "value",
            "type": "text",
            "text_representation": "my first document",
        }
        record = WeaviateReaderQueryResponse(collection=[wro])
        wtp_a = WeaviateReaderQueryParams(
            collection_name=cn,
        )
        returned_doc = WeaviateReaderQueryResponse.to_docs(record, wtp_a)[0]
        doc = Document(
            {
                "doc_id": "aryn:d-00000000000000000000000",
                "properties": {"field": "value", "nested": {"object": "value"}, "_doc_source": DocumentSource.DB_QUERY},
                "type": "text",
                "text_representation": "my first document",
            }
        )
        assert compare_docs(doc, returned_doc)

    def test_to_doc_with_embeddings(self):
        cn = "TestToDocEmbedding"
        wro = WeaviateReturnObject()
        wro.uuid = "00000000-0000-0000-0000-000000000000"
        wro.properties = {"text_representation": "helloworld"}
        wro.vector = {"embedding": [0.4] * 19}
        record = WeaviateReaderQueryResponse(collection=[wro])
        wtp_a = WeaviateReaderQueryParams(
            collection_name=cn,
        )
        returned_doc = WeaviateReaderQueryResponse.to_docs(record, wtp_a)[0]
        print(returned_doc)
        doc = Document(
            {
                "doc_id": "aryn:d-00000000000000000000000",
                "text_representation": "helloworld",
                "properties": {"_doc_source": DocumentSource.DB_QUERY},
                "embedding": [0.4] * 19,
            }
        )
        assert compare_docs(doc, returned_doc)

    def test_to_doc_with_list_types(self):
        cn = "TestToDocEmbeddingAndList"
        wro = WeaviateReturnObject()
        wro.uuid = "00000000-0000-0000-0000-000000000000"
        wro.properties = {
            "text_representation": "my second document",
            "bbox": (0.1, 1.2, 2.3, 3.4),
            "shingles": [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
        }
        wro.vector = {"embedding": [0.4] * 19}
        record = WeaviateReaderQueryResponse(collection=[wro])
        wtp_a = WeaviateReaderQueryParams(
            collection_name=cn,
        )
        returned_doc = WeaviateReaderQueryResponse.to_docs(record, wtp_a)[0]
        doc = Document(
            {
                "doc_id": "aryn:d-00000000000000000000000",
                "text_representation": "my second document",
                "bbox": (0.1, 1.2, 2.3, 3.4),
                "properties": {"_doc_source": DocumentSource.DB_QUERY},
                "shingles": [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
                "embedding": [0.4] * 19,
            }
        )
        assert compare_docs(doc, returned_doc)
