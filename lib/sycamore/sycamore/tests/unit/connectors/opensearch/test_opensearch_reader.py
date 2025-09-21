import pytest
from unittest import mock
from sycamore.data import Document
from sycamore.data.document import DocumentPropertyTypes, DocumentSource

from sycamore.connectors.opensearch.opensearch_reader import (
    OpenSearchReaderClient,
    OpenSearchReaderClientParams,
    OpenSearchReaderQueryParams,
    OpenSearchReaderQueryResponse,
)


@pytest.fixture
def mock_opensearch_client():
    return mock.Mock()


@pytest.fixture
def client_params():
    return OpenSearchReaderClientParams(os_client_args={"hosts": ["http://localhost:9200"]})


@pytest.fixture
def query_params():
    return OpenSearchReaderQueryParams(index_name="test_index", query={"match_all": {}})


def test_opensearch_reader_client_initialization(mock_opensearch_client):
    client = OpenSearchReaderClient(mock_opensearch_client)
    assert client._client == mock_opensearch_client


def test_opensearch_reader_client_from_client_params(client_params):
    with mock.patch("opensearchpy.OpenSearch") as MockOpenSearch:
        client = OpenSearchReaderClient.from_client_params(client_params)
        MockOpenSearch.assert_called_once_with(**client_params.os_client_args)
        assert isinstance(client, OpenSearchReaderClient)


def test_opensearch_reader_client_read_records(mock_opensearch_client, query_params):
    mock_opensearch_client.search.return_value = {
        "hits": {"hits": [{"_id": "1", "_source": {"properties": {}}}]}
    }

    client = OpenSearchReaderClient(mock_opensearch_client)
    response = client.read_records(query_params)

    assert isinstance(response, OpenSearchReaderQueryResponse)
    assert len(response.output) == 1
    assert response.output[0]["_id"] == "1"


def test_opensearch_reader_client_read_records_no_hits(mock_opensearch_client, query_params):
    mock_opensearch_client.search.return_value = {"hits": {"hits": []}}

    client = OpenSearchReaderClient(mock_opensearch_client)
    response = client.read_records(query_params)

    assert isinstance(response, OpenSearchReaderQueryResponse)
    assert len(response.output) == 0


def test_opensearch_reader_client_check_target_presence(mock_opensearch_client, query_params):
    mock_opensearch_client.indices.exists.return_value = True

    client = OpenSearchReaderClient(mock_opensearch_client)
    result = client.check_target_presence(query_params)

    assert result is True
    mock_opensearch_client.indices.exists.assert_called_once_with(index=query_params.index_name)


def test_opensearch_reader_client_check_target_absence(mock_opensearch_client, query_params):
    mock_opensearch_client.indices.exists.return_value = False

    client = OpenSearchReaderClient(mock_opensearch_client)
    result = client.check_target_presence(query_params)

    assert result is False
    mock_opensearch_client.indices.exists.assert_called_once_with(index=query_params.index_name)


def test_opensearch_reader_query_response_to_docs():
    response = OpenSearchReaderQueryResponse(output=[{"_id": "1", "_source": {"properties": {}}}])
    query_params = OpenSearchReaderQueryParams(index_name="test_index", query={"match_all": {}})
    docs = response.to_docs(query_params)

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].properties[DocumentPropertyTypes.SOURCE] == DocumentSource.DB_QUERY


def test_opensearch_reader_query_response_to_docs_empty():
    response = OpenSearchReaderQueryResponse(output=[])
    query_params = OpenSearchReaderQueryParams(index_name="test_index", query={"match_all": {}})
    docs = response.to_docs(query_params)

    assert len(docs) == 0


def test_opensearch_reader_query_params_compatible_with():
    params1 = OpenSearchReaderQueryParams(index_name="test_index", query={"match_all": {}})
    params2 = OpenSearchReaderQueryParams(index_name="test_index", query={"match_all": {}})
    assert params1.compatible_with(params2)

    params3 = OpenSearchReaderQueryParams(index_name="different_index", query={"match_all": {}})
    with pytest.raises(ValueError, match="Incompatible index names: Expected test_index, found different_index"):
        params1.compatible_with(params3)

    params4 = OpenSearchReaderQueryParams(index_name="test_index", query={"match": {"field": "value"}})
    with pytest.raises(ValueError, match="Incompatible queries: Expected {'match_all': {}}, found {'match': {'field': 'value'}}"):
        params1.compatible_with(params4)

    params5 = OpenSearchReaderQueryParams(index_name="test_index", kwargs={"key": "value"})
    with pytest.raises(ValueError, match="Incompatible kwargs: Expected {}, found {'key': 'value'}"):
        params1.compatible_with(params5)

    params6 = OpenSearchReaderQueryParams(index_name="test_index", reconstruct_document=True)
    with pytest.raises(ValueError, match="Incompatible reconstruct_document values: Expected False, found True"):
        params1.compatible_with(params6)

    params7 = OpenSearchReaderQueryParams(index_name="test_index", doc_reconstructor=mock.Mock())
    with pytest.raises(ValueError, match="Incompatible doc_reconstructor values: Expected None, found <Mock"):
        params1.compatible_with(params7)
