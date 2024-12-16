import pytest
from unittest import mock
from sycamore.data.document import Document

from sycamore.connectors.pinecone.pinecone_writer import (
    PineconeWriterClient,
    PineconeWriterRecord,
    PineconeWriterTargetParams,
    PineconeWriterClientParams,
    _narrow_list_of_pinecone_records,
    wait_on_index,
)


@pytest.fixture
def mock_pinecone_grpc():
    with mock.patch("pinecone.grpc.PineconeGRPC") as mock_grpc:
        yield mock_grpc


@pytest.fixture
def mock_pinecone_api_exception():
    with mock.patch("pinecone.exceptions.PineconeApiException") as mock_exception:
        yield mock_exception


@pytest.fixture
def mock_pinecone_exception():
    with mock.patch("pinecone.exceptions.PineconeApiException") as mock_exception:
        yield mock_exception


def test_pinecone_writer_target_params_compatible_with():
    params1 = PineconeWriterTargetParams(index_name="index1", dimensions=128)
    params2 = PineconeWriterTargetParams(index_name="index1", dimensions=128)
    assert params1.compatible_with(params2)

    params3 = PineconeWriterTargetParams(index_name="index2", dimensions=128)
    assert not params1.compatible_with(params3)

    params4 = PineconeWriterTargetParams(index_name="index1", dimensions=64)
    assert not params1.compatible_with(params4)


def test_pinecone_writer_client_init(mock_pinecone_grpc):
    client = PineconeWriterClient(api_key="test_key", batch_size=100)
    mock_pinecone_grpc.assert_called_once_with(api_key="test_key", source_tag="Aryn")
    assert client._batch_size == 100


def test_pinecone_writer_client_from_client_params():
    params = PineconeWriterClientParams(api_key="test_key", batch_size=100)
    client = PineconeWriterClient.from_client_params(params)
    assert isinstance(client, PineconeWriterClient)
    assert client._batch_size == 100


def test_pinecone_writer_client_write_many_records(mock_pinecone_grpc):
    client = PineconeWriterClient(api_key="test_key", batch_size=2)
    target_params = PineconeWriterTargetParams(index_name="index1", dimensions=128)
    records = [
        PineconeWriterRecord(id="1", values=[0.1, 0.2], metadata={}, sparse_values=None),
        PineconeWriterRecord(id="2", values=[0.3, 0.4], metadata={}, sparse_values=None),
    ]
    client.write_many_records(records, target_params)
    assert mock_pinecone_grpc.return_value.Index.return_value.upsert.call_count == 1


def test_pinecone_writer_client_write_many_records_empty(mock_pinecone_grpc):
    client = PineconeWriterClient(api_key="test_key", batch_size=2)
    target_params = PineconeWriterTargetParams(index_name="index1", dimensions=128)
    records = []
    client.write_many_records(records, target_params)
    assert mock_pinecone_grpc.return_value.Index.return_value.upsert.call_count == 0


def test_pinecone_writer_client_create_target_idempotent(mock_pinecone_grpc, mock_pinecone_api_exception):
    client = PineconeWriterClient(api_key="test_key", batch_size=100)
    target_params = PineconeWriterTargetParams(index_name="index1", dimensions=128, index_spec={"spec": "value"})
    client.create_target_idempotent(target_params)
    mock_pinecone_grpc.return_value.create_index.assert_called_once_with(
        name="index1", dimension=128, spec={"spec": "value"}, metric="cosine"
    )


def test_pinecone_writer_client_create_target_idempotent_existing(mock_pinecone_grpc, mock_pinecone_api_exception):
    client = PineconeWriterClient(api_key="test_key", batch_size=100)
    target_params = PineconeWriterTargetParams(index_name="index1", dimensions=128, index_spec={"spec": "value"})
    mock_pinecone_grpc.return_value.create_index.side_effect = mock_pinecone_api_exception
    client.create_target_idempotent(target_params)
    mock_pinecone_grpc.return_value.create_index.assert_called_once()


def test_pinecone_writer_client_get_existing_target_params(mock_pinecone_grpc):
    client = PineconeWriterClient(api_key="test_key", batch_size=100)
    target_params = PineconeWriterTargetParams(index_name="index1", dimensions=128)
    mock_pinecone_grpc.return_value.describe_index.return_value.to_dict.return_value = {
        "name": "index1",
        "dimension": 128,
        "spec": {"spec": "value"},
        "metric": "cosine",
    }
    result = client.get_existing_target_params(target_params)
    assert result.index_name == "index1"
    assert result.dimensions == 128
    assert result.index_spec == {"spec": "value"}
    assert result.distance_metric == "cosine"


def test_pinecone_writer_record_from_doc():
    document = Document(
        doc_id="doc1",
        embedding=[0.1, 0.2],
        type="type1",
        text_representation="text",
        bbox=None,
        shingles=None,
        properties={},
    )
    target_params = PineconeWriterTargetParams(index_name="index1", dimensions=128)
    record = PineconeWriterRecord.from_doc(document, target_params)
    assert record.id == "doc1"
    assert record.values == [0.1, 0.2]
    assert record.metadata["type"] == "type1"


def test_pinecone_writer_record_from_doc_with_parent_id():
    document = Document(
        doc_id="doc1",
        parent_id="parent1",
        embedding=[0.1, 0.2],
        type="type1",
        text_representation="text",
        bbox=None,
        shingles=None,
        properties={},
    )
    target_params = PineconeWriterTargetParams(index_name="index1", dimensions=128)
    record = PineconeWriterRecord.from_doc(document, target_params)
    assert record.id == "parent1#doc1"


def test_pinecone_writer_record_from_doc_with_sparse_vector():
    document = Document(
        doc_id="doc1",
        embedding=[0.1, 0.2],
        type="type1",
        text_representation="text",
        bbox=None,
        shingles=None,
        properties={"term_frequency": {1: 0.5, 2: 0.3}},
    )
    target_params = PineconeWriterTargetParams(index_name="index1", dimensions=128)
    record = PineconeWriterRecord.from_doc(document, target_params)
    assert record.sparse_values is not None
    assert record.sparse_values["indices"] == [1, 2]
    assert record.sparse_values["values"] == [0.5, 0.3]


def test_narrow_list_of_pinecone_records():
    records = [
        PineconeWriterRecord(id="1", values=[0.1, 0.2], metadata={}, sparse_values=None),
        PineconeWriterRecord(id="2", values=[0.3, 0.4], metadata={}, sparse_values=None),
    ]
    assert _narrow_list_of_pinecone_records(records)


def test_wait_on_index(mock_pinecone_grpc):
    client = mock_pinecone_grpc.return_value
    client.describe_index.return_value = {"status": {"ready": True}}
    wait_on_index(client, "index1")
    client.describe_index.assert_called_with("index1")


def test_wait_on_index_timeout(mock_pinecone_grpc):
    client = mock_pinecone_grpc.return_value
    client.describe_index.return_value = {"status": {"ready": False}}
    with pytest.raises(RuntimeError, match="Pinecone failed to create index in 30 seconds"):
        wait_on_index(client, "index1")
