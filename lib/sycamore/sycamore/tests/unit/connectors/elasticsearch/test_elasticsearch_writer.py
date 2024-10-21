import pytest
from unittest import mock
from sycamore.data.document import Document

from sycamore.connectors.elasticsearch.elasticsearch_writer import (
    ElasticsearchWriterClient,
    ElasticsearchWriterClientParams,
    ElasticsearchWriterTargetParams,
    ElasticsearchWriterDocumentRecord,
)


@pytest.fixture
def mock_elasticsearch_client():
    with mock.patch("elasticsearch.Elasticsearch") as MockElasticsearch:
        yield MockElasticsearch()


@pytest.fixture
def client_params():
    return ElasticsearchWriterClientParams(url="http://localhost:9200")


@pytest.fixture
def target_params():
    return ElasticsearchWriterTargetParams(index_name="test_index")


@pytest.fixture
def document():
    return Document(
        doc_id="1",
        properties={"key": "value"},
        type="test_type",
        text_representation="test_text",
        bbox=None,
        shingles=["shingle1", "shingle2"],
        parent_id="parent1",
        embedding=[0.1, 0.2, 0.3],
    )


def test_elasticsearch_writer_client_init(mock_elasticsearch_client):
    client = ElasticsearchWriterClient(mock_elasticsearch_client)
    assert client._client == mock_elasticsearch_client


def test_elasticsearch_writer_client_from_client_params(client_params, mock_elasticsearch_client):
    client = ElasticsearchWriterClient.from_client_params(client_params)
    assert isinstance(client, ElasticsearchWriterClient)
    assert client._client == mock_elasticsearch_client


def test_elasticsearch_writer_client_write_many_records(mock_elasticsearch_client, target_params):
    client = ElasticsearchWriterClient(mock_elasticsearch_client)
    records = [
        ElasticsearchWriterDocumentRecord(
            doc_id="1",
            properties={"key": "value"},
            parent_id="parent1",
            embedding=[0.1, 0.2, 0.3],
        )
    ]
    with mock.patch("elasticsearch.helpers.parallel_bulk") as mock_parallel_bulk:
        client.write_many_records(records, target_params)
        assert mock_parallel_bulk.called


def test_elasticsearch_writer_client_write_many_records_empty(mock_elasticsearch_client, target_params):
    client = ElasticsearchWriterClient(mock_elasticsearch_client)
    records = []
    with mock.patch("elasticsearch.helpers.parallel_bulk") as mock_parallel_bulk:
        client.write_many_records(records, target_params)
        assert not mock_parallel_bulk.called


def test_elasticsearch_writer_client_create_target_idempotent(mock_elasticsearch_client, target_params):
    client = ElasticsearchWriterClient(mock_elasticsearch_client)
    client.create_target_idempotent(target_params)
    assert mock_elasticsearch_client.indices.create.called


def test_elasticsearch_writer_client_create_target_idempotent_existing(mock_elasticsearch_client, target_params):
    from elasticsearch import ApiError

    mock_elasticsearch_client.indices.create.side_effect = ApiError(400, "index_already_exists_exception", "body")
    client = ElasticsearchWriterClient(mock_elasticsearch_client)
    client.create_target_idempotent(target_params)
    assert mock_elasticsearch_client.indices.create.called


def test_elasticsearch_writer_client_get_existing_target_params(mock_elasticsearch_client, target_params):
    mock_elasticsearch_client.indices.get_mapping.return_value = {
        target_params.index_name: {"mappings": target_params.mappings}
    }
    mock_elasticsearch_client.indices.get_settings.return_value = {
        target_params.index_name: {"settings": target_params.settings}
    }
    client = ElasticsearchWriterClient(mock_elasticsearch_client)
    existing_params = client.get_existing_target_params(target_params)
    assert existing_params.index_name == target_params.index_name
    assert existing_params.mappings == target_params.mappings
    assert existing_params.settings == target_params.settings


def test_elasticsearch_writer_document_record_from_doc(document, target_params):
    record = ElasticsearchWriterDocumentRecord.from_doc(document, target_params)
    assert record.doc_id == document.doc_id
    assert record.properties["properties"] == document.properties
    assert record.parent_id == document.parent_id
    assert record.embedding == document.embedding


def test_elasticsearch_writer_document_record_from_doc_no_doc_id(target_params):
    document = Document(
        doc_id=None,
        properties={"key": "value"},
        type="test_type",
        text_representation="test_text",
        bbox=None,
        shingles=["shingle1", "shingle2"],
        parent_id="parent1",
        embedding=[0.1, 0.2, 0.3],
    )
    with pytest.raises(ValueError):
        ElasticsearchWriterDocumentRecord.from_doc(document, target_params)


def test_elasticsearch_writer_target_params_compatible_with():
    params1 = ElasticsearchWriterTargetParams(index_name="test_index")
    params2 = ElasticsearchWriterTargetParams(index_name="test_index")
    assert params1.compatible_with(params2) is True

    params3 = ElasticsearchWriterTargetParams(index_name="different_index")
    assert params1.compatible_with(params3) is False


def test_elasticsearch_writer_target_params_compatible_with_different_settings():
    params1 = ElasticsearchWriterTargetParams(index_name="test_index", settings={"number_of_shards": 1})
    params2 = ElasticsearchWriterTargetParams(index_name="test_index", settings={"number_of_shards": 2})
    assert params1.compatible_with(params2) is False


def test_elasticsearch_writer_target_params_compatible_with_different_mappings():
    params1 = ElasticsearchWriterTargetParams(
        index_name="test_index", mappings={"properties": {"field1": {"type": "text"}}}
    )
    params2 = ElasticsearchWriterTargetParams(
        index_name="test_index", mappings={"properties": {"field2": {"type": "text"}}}
    )
    assert params1.compatible_with(params2) is False
