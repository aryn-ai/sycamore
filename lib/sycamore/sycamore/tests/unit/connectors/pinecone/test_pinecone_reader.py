import pytest
from unittest import mock
from sycamore.data import Document
from sycamore.connectors.base_reader import BaseDBReader
from sycamore.connectors.pinecone.pinecone_reader import (
    PineconeReaderClient,
    PineconeReaderClientParams,
    PineconeReaderQueryParams,
    PineconeReaderQueryResponse,
    PineconeReader,
)


@pytest.fixture
def client_params():
    return PineconeReaderClientParams(api_key="test_api_key")


@pytest.fixture
def query_params():
    return PineconeReaderQueryParams(index_name="test_index", namespace="test_namespace", query=None)


@pytest.fixture
def mock_pinecone_grpc():
    with mock.patch("pinecone.grpc.PineconeGRPC") as MockPineconeGRPC:
        yield MockPineconeGRPC


def test_pinecone_reader_client_init(client_params, mock_pinecone_grpc):
    client = PineconeReaderClient(client_params)
    mock_pinecone_grpc.assert_called_once_with(api_key="test_api_key", source_tag="Aryn")
    assert client._client is not None


def test_pinecone_reader_client_from_client_params(client_params):
    client = PineconeReaderClient.from_client_params(client_params)
    assert isinstance(client, PineconeReaderClient)


def test_pinecone_reader_client_read_records(query_params, mock_pinecone_grpc):
    mock_index = mock.Mock()
    mock_pinecone_grpc.return_value.Index.return_value = mock_index
    mock_index.list.return_value = [["id1", "id2"]]
    mock_index.fetch.return_value = {
        "vectors": {"id1": {"id": "id1", "values": [0.1, 0.2]}, "id2": {"id": "id2", "values": [0.3, 0.4]}}
    }

    client = PineconeReaderClient(PineconeReaderClientParams(api_key="test_api_key"))
    response = client.read_records(query_params)

    assert isinstance(response, PineconeReaderQueryResponse)
    assert len(response.output) == 2


def test_pinecone_reader_client_read_records_with_query(query_params, mock_pinecone_grpc):
    query_params.query = {"top_k": 2, "include_values": True}
    mock_index = mock.Mock()
    mock_pinecone_grpc.return_value.Index.return_value = mock_index
    mock_index.query.return_value = {
        "matches": [{"id": "id1", "values": [0.1, 0.2]}, {"id": "id2", "values": [0.3, 0.4]}]
    }

    client = PineconeReaderClient(PineconeReaderClientParams(api_key="test_api_key"))
    response = client.read_records(query_params)

    assert isinstance(response, PineconeReaderQueryResponse)
    assert len(response.output) == 2


def test_pinecone_reader_client_check_target_presence(query_params, mock_pinecone_grpc):
    mock_index = mock.Mock()
    mock_pinecone_grpc.return_value.Index.return_value = mock_index
    mock_index.describe_index_stats.return_value = {"namespaces": {"test_namespace": {}}}

    client = PineconeReaderClient(PineconeReaderClientParams(api_key="test_api_key"))
    presence = client.check_target_presence(query_params)

    assert presence is True


def test_pinecone_reader_client_check_target_presence_not_found(query_params, mock_pinecone_grpc):
    mock_index = mock.Mock()
    mock_pinecone_grpc.return_value.Index.return_value = mock_index
    mock_index.describe_index_stats.return_value = {"namespaces": {}}

    client = PineconeReaderClient(PineconeReaderClientParams(api_key="test_api_key"))
    presence = client.check_target_presence(query_params)

    assert presence is False


def test_pinecone_reader_query_response_to_docs():
    data = [
        mock.Mock(id="doc1", values=[0.1, 0.2], metadata={"key": "value"}, sparse_vector=None),
        mock.Mock(id="doc2#parent", values=[0.3, 0.4], metadata={}, sparse_vector=None),
    ]
    response = PineconeReaderQueryResponse(output=data)
    docs = response.to_docs(PineconeReaderQueryParams(index_name="test_index", namespace="test_namespace", query=None))

    assert len(docs) == 2
    assert isinstance(docs[0], Document)
    assert docs[0].doc_id == "doc1"
    assert docs[1].parent_id == "doc2"
    assert docs[1].doc_id == "parent"


def test_pinecone_reader_query_response_to_docs_with_sparse_vector():
    data = [
        mock.Mock(
            id="doc1",
            values=[0.1, 0.2],
            metadata={"key": "value"},
            sparse_vector=mock.Mock(indices=[0, 1], values=[0.5, 0.5]),
        ),
    ]
    response = PineconeReaderQueryResponse(output=data)
    docs = response.to_docs(PineconeReaderQueryParams(index_name="test_index", namespace="test_namespace", query=None))

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].doc_id == "doc1"
    assert docs[0].properties["term_frequency"] == {0: 0.5, 1: 0.5}
