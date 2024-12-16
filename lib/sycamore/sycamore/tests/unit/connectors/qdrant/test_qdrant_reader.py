from qdrant_client import QdrantClient, models
from sycamore.connectors.common import compare_docs
from sycamore.connectors.qdrant.qdrant_reader import (
    QdrantReaderClient,
    QdrantReaderQueryParams,
    QdrantReaderQueryResponse,
)
from sycamore.data.docid import uuid_to_docid
from sycamore.data.document import Document, DocumentSource


class QdrantReturnObject(object):
    pass


class TestQdrantClient:
    @staticmethod
    def mock_client(mocker):
        client = mocker.Mock(spec=QdrantClient)
        return client

    def test_check_existence(self, mocker):
        cn = "TestCheckExistence"
        query_params = QdrantReaderQueryParams(query_params={"collection_name": cn})
        qclient = TestQdrantClient.mock_client(mocker)
        qclient.collection_exists.return_value = True
        client = QdrantReaderClient(qclient)
        client.check_target_presence(query_params)
        qclient.collection_exists.assert_called_once()

    def test_read_documents(self, mocker):
        cn = "TestReadDocuments"
        query_params = QdrantReaderQueryParams(query_params={"collection_name": cn})

        qclient = TestQdrantClient.mock_client(mocker)
        client = QdrantReaderClient(qclient)
        client.read_records(query_params)

        qclient.query_points.assert_called_once()

    def test_read_documents_with_limit(self, mocker):
        cn = "TestReadDocuments"
        query_params = QdrantReaderQueryParams(query_params={"collection_name": cn, "limit": 7})

        qclient = TestQdrantClient.mock_client(mocker)
        client = QdrantReaderClient(qclient)
        client.read_records(query_params)

        qclient.query_points.assert_called_once_with(limit=7, collection_name=cn)

    def test_read_documents_with_query(self, mocker):
        cn = "TestReadDocuments"
        query_params = QdrantReaderQueryParams(query_params={"collection_name": cn, "query": [0.1, 0.2, 0.523]})

        qclient = TestQdrantClient.mock_client(mocker)
        client = QdrantReaderClient(qclient)
        client.read_records(query_params)

        qclient.query_points.assert_called_once_with(query=[0.1, 0.2, 0.523], collection_name=cn)


class TestQdrantQueryResponse:
    def test_to_doc(self):
        record = QdrantReaderQueryResponse(points=[])
        wtp_a = QdrantReaderQueryParams({})
        doc_list = QdrantReaderQueryResponse.to_docs(record, wtp_a)
        assert len(doc_list) == 0

    def test_point_to_doc(self):
        uu = "0e14ade4-7f2a-490e-844b-f063c92bdfbb"
        id = uuid_to_docid(uu)
        point = models.ScoredPoint(
            id=uu,
            vector=[0.1, 0.2, 0.3],
            payload={
                "properties__field": "value",
                "properties__nested__object": "value",
                "type": "text",
                "text_representation": "my first document",
            },
            version=1,
            score=0.1,
        )
        record = QdrantReaderQueryResponse(points=[point])
        query_params = QdrantReaderQueryParams({})
        returned_doc = QdrantReaderQueryResponse.to_docs(record, query_params)[0]
        doc = Document(
            {
                "doc_id": id,
                "properties": {"field": "value", "nested": {"object": "value"}, "_doc_source": DocumentSource.DB_QUERY},
                "type": "text",
                "text_representation": "my first document",
                "embedding": [0.1, 0.2, 0.3],
            }
        )
        assert compare_docs(doc, returned_doc)
