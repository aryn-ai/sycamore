import pytest
from unittest import mock
from sycamore.data import Document
from sycamore.data.document import DocumentPropertyTypes, DocumentSource

from sycamore.connectors.elasticsearch.elasticsearch_reader import (
    ElasticsearchReaderClient,
    ElasticsearchReaderClientParams,
    ElasticsearchReaderQueryParams,
    ElasticsearchReaderQueryResponse,
)


@pytest.fixture
def mock_elasticsearch_client():
    return mock.Mock()


@pytest.fixture
def client_params():
    return ElasticsearchReaderClientParams(url="http://localhost:9200")


@pytest.fixture
def query_params():
    return ElasticsearchReaderQueryParams(index_name="test_index")


def test_elasticsearch_reader_client_initialization(mock_elasticsearch_client):
    client = ElasticsearchReaderClient(mock_elasticsearch_client)
    assert client._client == mock_elasticsearch_client


def test_elasticsearch_reader_client_from_client_params(client_params):
    with mock.patch("elasticsearch.Elasticsearch") as MockElasticsearch:
        client = ElasticsearchReaderClient.from_client_params(client_params)
        MockElasticsearch.assert_called_once_with(client_params.url, **client_params.es_client_args)
        assert isinstance(client, ElasticsearchReaderClient)


def test_elasticsearch_reader_client_read_records(mock_elasticsearch_client, query_params):
    mock_elasticsearch_client.open_point_in_time.return_value = {"id": "test_pit_id"}
    mock_elasticsearch_client.search.side_effect = [
        {
            "hits": {"hits": [{"_id": "1", "_source": {"properties": {}}, "sort": {"_shard_doc": "desc"}}]},
            "pit_id": "test_pit_id",
        },
        {"hits": {"hits": []}, "pit_id": "test_pit_id"},
    ]

    client = ElasticsearchReaderClient(mock_elasticsearch_client)
    response = client.read_records(query_params)

    assert isinstance(response, ElasticsearchReaderQueryResponse)
    assert len(response.output) == 1
    assert response.output[0]["_id"] == "1"
    mock_elasticsearch_client.close_point_in_time.assert_called_once_with(id="test_pit_id")


def test_elasticsearch_reader_client_read_records_no_hits(mock_elasticsearch_client, query_params):
    mock_elasticsearch_client.open_point_in_time.return_value = {"id": "test_pit_id"}
    mock_elasticsearch_client.search.side_effect = [
        {"hits": {"hits": [], "pit_id": "test_pit_id"}},
    ]

    client = ElasticsearchReaderClient(mock_elasticsearch_client)
    response = client.read_records(query_params)

    assert isinstance(response, ElasticsearchReaderQueryResponse)
    assert len(response.output) == 0
    mock_elasticsearch_client.close_point_in_time.assert_called_once_with(id="test_pit_id")


def test_elasticsearch_reader_client_check_target_presence(mock_elasticsearch_client, query_params):
    mock_elasticsearch_client.indices.exists.return_value = True

    client = ElasticsearchReaderClient(mock_elasticsearch_client)
    result = client.check_target_presence(query_params)

    assert result is True
    mock_elasticsearch_client.indices.exists.assert_called_once_with(index=query_params.index_name)


def test_elasticsearch_reader_client_check_target_absence(mock_elasticsearch_client, query_params):
    mock_elasticsearch_client.indices.exists.return_value = False

    client = ElasticsearchReaderClient(mock_elasticsearch_client)
    result = client.check_target_presence(query_params)

    assert result is False
    mock_elasticsearch_client.indices.exists.assert_called_once_with(index=query_params.index_name)


def test_elasticsearch_reader_query_response_to_docs():
    response = ElasticsearchReaderQueryResponse(output=[{"_id": "1", "_source": {"properties": {}}}])
    query_params = ElasticsearchReaderQueryParams(index_name="test_index")
    docs = response.to_docs(query_params)

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].properties[DocumentPropertyTypes.SOURCE] == DocumentSource.DB_QUERY


def test_elasticsearch_reader_query_response_to_docs_empty():
    response = ElasticsearchReaderQueryResponse(output=[])
    query_params = ElasticsearchReaderQueryParams(index_name="test_index")
    docs = response.to_docs(query_params)

    assert len(docs) == 0
